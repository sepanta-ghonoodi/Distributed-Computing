"""Phase 1 vs Phase 2 ablation — the presentation table.

    python -m src.ablation --ckpt runs/phase1_colab/best.pt

Evaluates one trained checkpoint under several Link-Projection strengths and
prints them side by side. Because the metrics already split error into
longitudinal and lateral components, the table shows *which kind* of error each
component fixes, rather than a single ADE number that hides the mechanism.

Link Projection is a pure inference-time correction, so all rows come from the
same weights — no retraining, and the comparison is exactly like-for-like.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config, DataConfig, ModelConfig, TrainConfig
from .data.dataset import load_splits
from .data.schema import NUM_FEATURES
from .engine import constant_velocity_baseline, evaluate
from .map.highway import LaneCentreMap, make_link_projection_hook, off_road_rate
from .models.seq2seq_transformer import build_model

# Columns worth showing on a slide. `rmse_lat` is the one Link Projection is
# supposed to move; `rmse_long` is what Phase 3's IDM loss will target.
REPORT_KEYS = ["ade", "fde", "ade@10s", "ade@30s", "rmse_long", "rmse_lat", "miss_rate@5m"]


def config_from_dict(d: dict) -> Config:
    return Config(
        data=DataConfig(**d["data"]),
        model=ModelConfig(**d["model"]),
        train=TrainConfig(**d["train"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/phase1_colab/best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument(
        "--blends", type=float, nargs="+", default=[0.0, 0.5, 1.0],
        help="Link-Projection strengths to compare (0 = Phase 1 baseline)",
    )
    ap.add_argument(
        "--start-step", type=int, default=2,
        help="skip snapping for the first N steps (vehicle may be mid-lane-change)",
    )
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ckpt["config"])
    device = torch.device(
        "cuda" if (cfg.train.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    datasets, _ = load_splits(cfg.data)
    loader = DataLoader(datasets[args.split], batch_size=cfg.train.batch_size * 2, shuffle=False)

    model = build_model(NUM_FEATURES, cfg.model).to(device)
    model.load_state_dict(ckpt["model"])

    lane_map = LaneCentreMap.from_parquet(cfg.data.unified_path).to(device)
    print(
        f"lane map | {len(lane_map.centres)} centrelines at "
        f"{[round(float(c), 2) for c in lane_map.centres]} m, "
        f"mean spacing {lane_map.lane_width:.2f} m\n"
    )

    rows: dict[str, dict[str, float]] = {}
    preds_for_plot: dict[str, torch.Tensor] = {}
    truth = origin = theta = None

    for blend in args.blends:
        name = "Phase 1 (no snap)" if blend == 0.0 else f"+ LinkProj b={blend:g}"

        def factory(batch, b=blend):
            if b == 0.0:
                return None
            return make_link_projection_hook(
                lane_map, batch["origin"], batch["theta"],
                blend=b, start_step=args.start_step,
            )

        rep = evaluate(
            model, loader, cfg.train, device,
            pred_len=cfg.data.pred_len, target_hz=cfg.data.target_hz,
            hook_factory=factory, desc=name, return_predictions=True,
        )
        pred = rep.pop("_pred_pos")
        truth = rep.pop("_true_pos")
        origin, theta = rep.pop("_origin"), rep.pop("_theta")

        rep["off_road"] = off_road_rate(pred, origin, theta, lane_map.to("cpu"))
        lane_map.to(device)
        rows[name] = rep
        preds_for_plot[name] = pred

    rows["constant velocity"] = constant_velocity_baseline(loader, cfg.data.target_hz, device)

    # --- print ------------------------------------------------------------
    keys = REPORT_KEYS + ["off_road"]
    width = max(len(n) for n in rows) + 2
    header = f"{'model':<{width}}" + "".join(f"{k:>13}" for k in keys)
    print(f"\n=== {args.split} set ({len(datasets[args.split]):,} windows) ===")
    print(header)
    print("-" * len(header))
    for name, rep in rows.items():
        cells = "".join(f"{rep.get(k, float('nan')):>13.3f}" for k in keys)
        print(f"{name:<{width}}{cells}")

    out_dir = Path(args.ckpt).parent
    with open(out_dir / f"ablation_{args.split}.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out_dir / f'ablation_{args.split}.json'}")

    if args.plot:
        plot_overlay(preds_for_plot, truth, out_dir / f"ablation_{args.split}.png")


def plot_overlay(preds: dict[str, torch.Tensor], truth, out_path: Path, n: int = 5) -> None:
    """Overlay every ablation variant on the same ground-truth trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n, len(truth))
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.1 * n))
    axes = [axes] if n == 1 else axes

    for i, ax in enumerate(axes):
        ax.plot(truth[i, :, 0], truth[i, :, 1], "-", lw=2.5, color="k", label="ground truth")
        for name, p in preds.items():
            ax.plot(p[i, :, 0], p[i, :, 1], "--", lw=1.6, label=name)
        ax.scatter([0], [0], marker="o", s=28, color="k", zorder=3)
        ax.set_ylim(-6, 6)
        ax.set_ylabel("lateral [m]")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8, ncol=len(preds) + 1)
    axes[-1].set_xlabel("longitudinal [m] (agent frame)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
