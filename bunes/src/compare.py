"""The ablation ladder: every variant, judged on trajectory *and* migration.

    python -m src.compare \
        --variants base=runs/ngsim_control/best.pt \
                   idm=runs/ngsim_idm/best.pt \
                   sched=runs/ngsim_ss/best.pt \
        --snap-each

Why both metric families in one table: on this data they already disagree. The
trained model beats constant velocity on ADE by 17% yet lost to it on service
interruption, because migration cost is asymmetric and ADE is not. A feature
that improves ADE may leave handover timing untouched, and one that barely
moves ADE may still sharpen the handover ETA. Judging a feature on ADE alone
would therefore have drawn the wrong conclusion at least once already, so every
rung of the ladder is scored on both.

Each variant is optionally evaluated twice, with and without Link Projection,
since that costs nothing but a second rollout.
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
from .engine import constant_velocity_baseline, evaluate
from .map.highway import LaneCentreMap, make_link_projection_hook, off_road_rate
from .models.seq2seq_transformer import build_model
from .vec.migration import migration_metrics, reactive_metrics
from .vec.rsu import RSUChain, first_handover, to_world

COLUMNS = [
    ("ade", "ADE", 8, 2),
    ("fde", "FDE", 8, 2),
    ("ade@10s", "ADE@10s", 9, 2),
    ("rmse_long", "long", 8, 2),
    ("rmse_lat", "lat", 7, 2),
    ("off_road", "offroad", 9, 4),
    ("eta_err", "ETAerr", 8, 2),
    ("interruption", "interrupt", 10, 3),
    ("premature", "wasted", 8, 2),
    ("detect", "detect", 8, 3),
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
    ap.add_argument(
        "--variants", nargs="+", required=True,
        help="name=path/to/best.pt pairs, in the order they should appear",
    )
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--snap-each", action="store_true",
                    help="also evaluate every variant with Link Projection")
    ap.add_argument("--rsu-spacing", type=float, default=300.0)
    ap.add_argument("--migration-time", type=float, default=5.0)
    ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--out", default=None, help="where to write the JSON (default: first ckpt dir)")
    args = ap.parse_args()

    variants, missing = [], []
    for spec in args.variants:
        if "=" not in spec:
            raise SystemExit(f"--variants expects name=path, got '{spec}'")
        name, path = spec.split("=", 1)
        # A run that has not finished yet should cost a warning and a shorter
        # table, not the whole comparison.
        (variants if Path(path).exists() else missing).append((name, path))

    if missing:
        print("\nskipping variants whose checkpoint does not exist yet:")
        for name, path in missing:
            print(f"  {name:<12} {path}")
    if not variants:
        raise SystemExit("None of the requested checkpoints exist — nothing to compare.")

    # Every variant must share a dataset for the comparison to mean anything.
    first_cfg = config_from_dict(
        torch.load(variants[0][1], map_location="cpu", weights_only=False)["config"]
    )
    cfg = first_cfg
    device = torch.device(
        "cuda" if (cfg.train.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    dt = cfg.data.dt

    datasets, _ = load_splits(cfg.data)
    loader = DataLoader(datasets[args.split], batch_size=cfg.train.batch_size * 2, shuffle=False)
    lane_map = LaneCentreMap.from_parquet(cfg.data.unified_path).to(device)

    # --- shared reference frame for the RSU chain ---------------------------
    origin = torch.cat([b["origin"] for b in loader], dim=0).float()
    theta = torch.cat([b["theta"] for b in loader], dim=0).float()
    cv_delta = torch.cat([b["cv_delta"] for b in loader], dim=0).float()
    true_pos = torch.cat([b["tgt_pos"] for b in loader], dim=0).float()

    chain = RSUChain(spacing=args.rsu_spacing, x0=float(origin[:, 0].min()))
    current_rsu = chain.index(origin[:, 0])
    true_x = to_world(true_pos, origin, theta)[..., 0]
    t_true, occurred = first_handover(true_x, current_rsu, chain, dt)
    t_true_h = t_true[occurred]

    print(
        f"\ndataset  | {cfg.data.unified_path}, {args.split} split, "
        f"{len(true_pos):,} windows"
    )
    print(
        f"RSU      | spacing {args.rsu_spacing:.0f} m, "
        f"{int(occurred.sum()):,} ({100 * float(occurred.float().mean()):.1f}%) cross a boundary"
    )
    print(
        f"policy   | migration time {args.migration_time:g} s, "
        f"safety margin {args.margin:g} s\n"
    )

    rows: dict[str, dict[str, float]] = {}

    def score(name: str, pred_pos: torch.Tensor, traj: dict) -> None:
        pred_x = to_world(pred_pos, origin, theta)[..., 0]
        t_p, seen = first_handover(pred_x, current_rsu, chain, dt)
        mig = migration_metrics(
            t_p[occurred], seen[occurred], t_true_h, args.migration_time, margin=args.margin
        )
        rows[name] = {
            **{k: traj[k] for k, *_ in COLUMNS if k in traj},
            "off_road": off_road_rate(pred_pos, origin, theta, lane_map.to("cpu")),
            "eta_err": mig["mean_eta_error_s"],
            "interruption": mig["mean_interruption_s"],
            "premature": mig["mean_premature_s"],
            "detect": mig["handover_detect_rate"],
        }
        lane_map.to(device)

    # --- references ---------------------------------------------------------
    k = torch.arange(1, cfg.data.pred_len + 1, dtype=torch.float32).view(1, -1, 1)
    score("constant velocity", cv_delta.unsqueeze(1) * k,
          constant_velocity_baseline(loader, cfg.data.target_hz, device))

    # --- each variant -------------------------------------------------------
    for name, path in variants:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        vcfg = config_from_dict(ckpt["config"])
        if vcfg.data.unified_path != cfg.data.unified_path:
            raise SystemExit(
                f"variant '{name}' was trained on {vcfg.data.unified_path}, "
                f"but the comparison is against {cfg.data.unified_path}"
            )
        model = build_model(NUM_FEATURES, vcfg.model).to(device)
        model.load_state_dict(ckpt["model"])

        passes = [(name, None)]
        if args.snap_each:
            passes.append((
                f"{name} + snap",
                lambda b: make_link_projection_hook(
                    lane_map, b["origin"], b["theta"], blend=1.0, start_step=2
                ),
            ))

        for label, factory in passes:
            rep = evaluate(
                model, loader, cfg.train, device,
                pred_len=cfg.data.pred_len, target_hz=cfg.data.target_hz,
                hook_factory=factory, desc=label, return_predictions=True,
            )
            score(label, rep.pop("_pred_pos"), rep)

    # --- print --------------------------------------------------------------
    width = max(len(n) for n in rows) + 2
    header = f"{'variant':<{width}}" + "".join(f"{h:>{w}}" for _, h, w, _ in COLUMNS)
    print("=" * len(header))
    print(header)
    print(f"{'':<{width}}" + f"{'--- trajectory ---':>44}{'--- migration ---':>44}")
    print("-" * len(header))
    for name, r in rows.items():
        cells = ""
        for key, _, w, prec in COLUMNS:
            v = r.get(key, float("nan"))
            cells += f"{v:>{w}.{prec}f}"
        print(f"{name:<{width}}{cells}")
    print("=" * len(header))
    print(
        "\nreactive migration (no prediction at all) costs "
        f"{reactive_metrics(t_true_h, args.migration_time)['mean_interruption_s']:.3f} s "
        "of interruption per handover."
    )

    out = Path(args.out) if args.out else Path(variants[0][1]).parent / f"compare_{args.split}.json"
    with open(out, "w") as f:
        json.dump(
            {
                "split": args.split,
                "rsu_spacing": args.rsu_spacing,
                "migration_time": args.migration_time,
                "margin": args.margin,
                "rows": rows,
            },
            f, indent=2,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
