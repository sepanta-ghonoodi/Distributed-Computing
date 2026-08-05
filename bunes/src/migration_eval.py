"""Phase 6 — does better trajectory prediction actually reduce migration cost?

    python -m src.migration_eval --ckpt runs/ngsim_control/best.pt

Rolls the predictor out over the test set, maps every trajectory onto a chain of
RSU coverage zones, and scores the proactive migration policy against three
references:

    reactive           migrate on handover — the do-nothing baseline
    constant velocity  the policy you get for free, with no model at all
    PAS-LGT            the trained predictor
    oracle             perfect foresight — the floor the policy can reach

The constant-velocity row is the one that matters. Beating "reactive" only
shows that predicting anything helps; beating straight-line extrapolation is
what justifies the trajectory model inside a VEC system.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config, DataConfig, ModelConfig, PhysicsConfig, TrainConfig
from .data.dataset import load_splits
from .data.schema import NUM_FEATURES
from .engine import evaluate
from .map.highway import LaneCentreMap, make_link_projection_hook
from .models.seq2seq_transformer import build_model
from .vec.migration import migration_metrics, reactive_metrics
from .vec.rsu import RSUChain, first_handover, to_world

ROW_KEYS = [
    "mean_interruption_s",
    "interruption_reduction_pct",
    "zero_interruption_rate",
    "handover_detect_rate",
    "mean_eta_error_s",
    "mean_premature_s",
]


def config_from_dict(d: dict) -> Config:
    return Config(
        data=DataConfig(**d["data"]),
        model=ModelConfig(**d["model"]),
        train=TrainConfig(**d["train"]),
        physics=PhysicsConfig(**d.get("physics", {})),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ngsim_control/best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument(
        "--rsu-spacing", type=float, default=300.0,
        help="RSU coverage diameter along the highway [m]",
    )
    ap.add_argument(
        "--migration-time", type=float, nargs="+", default=[1.0, 3.0, 5.0, 10.0],
        help="service migration durations to sweep [s]",
    )
    ap.add_argument(
        "--margin", type=float, default=0.0,
        help="safety lead time applied in the migration-duration tables [s]",
    )
    ap.add_argument(
        "--margins", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0],
        help="safety lead times to sweep",
    )
    ap.add_argument("--snap", action="store_true", help="apply Link Projection")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ckpt["config"])
    device = torch.device(
        "cuda" if (cfg.train.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    dt = cfg.data.dt

    datasets, _ = load_splits(cfg.data)
    loader = DataLoader(datasets[args.split], batch_size=cfg.train.batch_size * 2, shuffle=False)

    model = build_model(NUM_FEATURES, cfg.model).to(device)
    model.load_state_dict(ckpt["model"])

    hook_factory = None
    if args.snap:
        lane_map = LaneCentreMap.from_parquet(cfg.data.unified_path).to(device)
        hook_factory = lambda b: make_link_projection_hook(  # noqa: E731
            lane_map, b["origin"], b["theta"], blend=1.0, start_step=2
        )

    rep = evaluate(
        model, loader, cfg.train, device,
        pred_len=cfg.data.pred_len, target_hz=cfg.data.target_hz,
        hook_factory=hook_factory, desc="rollout", return_predictions=True,
    )
    pred_pos, true_pos = rep["_pred_pos"], rep["_true_pos"]
    origin, theta = rep["_origin"], rep["_theta"]

    # The constant-velocity policy, materialised the same way as in the ablation.
    cv_delta = torch.cat([b["cv_delta"] for b in loader], dim=0).float()
    k = torch.arange(1, cfg.data.pred_len + 1, dtype=torch.float32).view(1, -1, 1)
    cv_pos = cv_delta.unsqueeze(1) * k

    # --- world coordinates -------------------------------------------------
    worlds = {
        "PAS-LGT": to_world(pred_pos, origin, theta)[..., 0],
        "constant velocity": to_world(cv_pos, origin, theta)[..., 0],
        "oracle": to_world(true_pos, origin, theta)[..., 0],
    }
    true_x = worlds["oracle"]

    chain = RSUChain(spacing=args.rsu_spacing, x0=float(origin[:, 0].min()))
    current_rsu = chain.index(origin[:, 0])

    t_true, occurred = first_handover(true_x, current_rsu, chain, dt)

    n_total = len(true_x)
    n_handover = int(occurred.sum())
    print(
        f"\nRSU chain | spacing {args.rsu_spacing:.0f} m over "
        f"{float(origin[:, 0].min()):.0f}..{float(true_x.max()):.0f} m of highway"
    )
    print(
        f"windows   | {n_total:,} total, {n_handover:,} ({100 * n_handover / n_total:.1f}%) "
        f"cross an RSU boundary within {cfg.data.pred_seconds:.0f} s"
    )
    if n_handover == 0:
        raise SystemExit(
            "No handovers in the horizon — lower --rsu-spacing for this road length."
        )

    # Only windows with a real handover are scorable: the others have no
    # migration event to be early or late for.
    handovers = {}
    for name, x in worlds.items():
        t_p, seen = first_handover(x, current_rsu, chain, dt)
        handovers[name] = (t_p[occurred], seen[occurred])
    t_true_h = t_true[occurred]

    # --- sweep migration durations -----------------------------------------
    results: dict[str, dict[str, dict[str, float]]] = {}
    width = 20
    col = 15
    for t_m in args.migration_time:
        rows = {"reactive": reactive_metrics(t_true_h, t_m)}
        for name, (t_p, seen) in handovers.items():
            rows[name] = migration_metrics(t_p, seen, t_true_h, t_m, margin=args.margin)
        results[f"{t_m:g}"] = rows

        header = f"{'policy':<{width}}" + "".join(f"{k[:col - 1]:>{col}}" for k in ROW_KEYS)
        print(
            f"\n=== migration time {t_m:g} s | margin {args.margin:g} s "
            f"| {n_handover:,} handovers ==="
        )
        print(header)
        print("-" * len(header))
        for name, m in rows.items():
            print(f"{name:<{width}}" + "".join(f"{m[k]:>{col}.3f}" for k in ROW_KEYS))

    # --- what does a safety margin buy? ------------------------------------
    # The cost is asymmetric: early only wastes residency, late interrupts. A
    # more accurate predictor should therefore need less margin to reach the
    # same interruption -- this sweep is where prediction accuracy either does
    # or does not become a system-level advantage.
    t_ref = args.migration_time[len(args.migration_time) // 2]
    margin_sweep: dict[str, dict[str, list[float]]] = {}
    print(f"\n=== safety-margin sweep at migration time {t_ref:g} s ===")
    print(f"{'margin [s]':<12}" + "".join(f"{n:>22}" for n in handovers))
    print(f"{'':<12}" + "".join(f"{'interrupt / wasted':>22}" for _ in handovers))
    print("-" * (12 + 22 * len(handovers)))
    for mg in args.margins:
        cells = ""
        for name, (t_p, seen) in handovers.items():
            m = migration_metrics(t_p, seen, t_true_h, t_ref, margin=mg)
            margin_sweep.setdefault(name, {"margin": [], "interruption": [], "premature": []})
            margin_sweep[name]["margin"].append(mg)
            margin_sweep[name]["interruption"].append(m["mean_interruption_s"])
            margin_sweep[name]["premature"].append(m["mean_premature_s"])
            cells += f"{m['mean_interruption_s']:>13.3f} /{m['mean_premature_s']:>7.2f}"
        print(f"{mg:<12g}{cells}")

    out_dir = Path(args.ckpt).parent
    tag = "_snap" if args.snap else ""
    with open(out_dir / f"migration_{args.split}{tag}.json", "w") as f:
        json.dump(
            {
                "rsu_spacing": args.rsu_spacing,
                "n_handovers": n_handover,
                "sweep": results,
                "margin_sweep": margin_sweep,
            },
            f, indent=2,
        )
    print(f"\nwrote {out_dir / f'migration_{args.split}{tag}.json'}")

    if args.plot:
        plot_sweep(
            results, args.migration_time, margin_sweep, t_ref,
            out_dir / f"migration_{args.split}{tag}.png",
        )


def plot_sweep(results, migration_times, margin_sweep, t_ref, out_path: Path) -> None:
    """Three panels: cost vs migration time, and what a safety margin buys."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = list(next(iter(results.values())).keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for p in policies:
        y = [results[f"{t:g}"][p]["mean_interruption_s"] for t in migration_times]
        axes[0].plot(migration_times, y, marker="o", lw=2, label=p)
    axes[0].set_xlabel("service migration time [s]")
    axes[0].set_ylabel("mean service interruption [s]")
    axes[0].set_title("cost vs migration duration", fontsize=10)
    axes[0].legend(fontsize=8)

    for name, d in margin_sweep.items():
        axes[1].plot(d["margin"], d["interruption"], marker="o", lw=2, label=name)
        # The trade-off the margin actually makes: interruption bought with
        # wasted residency on the next RSU.
        axes[2].plot(d["premature"], d["interruption"], marker="o", lw=2, label=name)

    axes[1].set_xlabel("safety margin [s]")
    axes[1].set_ylabel("mean service interruption [s]")
    axes[1].set_title(f"effect of a safety margin (T_m = {t_ref:g} s)", fontsize=10)

    axes[2].set_xlabel("wasted RSU residency [s]")
    axes[2].set_ylabel("mean service interruption [s]")
    axes[2].set_title("the trade-off: lower is better on both axes", fontsize=10)
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
