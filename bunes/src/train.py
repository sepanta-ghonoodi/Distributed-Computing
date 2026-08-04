"""Phase 1 training entry point.

    python -m src.train --config configs/phase1_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data.dataset import load_splits
from .data.schema import NUM_FEATURES
from .engine import (
    build_scheduler,
    constant_velocity_baseline,
    evaluate,
    train_one_epoch,
)
from .models.seq2seq_transformer import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Phase 1 baseline predictor.")
    ap.add_argument("--config", default="configs/phase1_baseline.yaml")
    ap.add_argument("--no-cache", action="store_true", help="rebuild windows from parquet")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # --- data ---------------------------------------------------------------
    datasets, scaler = load_splits(cfg.data, use_cache=not args.no_cache)
    with open(out_dir / "scaler.json", "w") as f:
        json.dump(scaler.to_dict(), f, indent=2)

    print(
        f"windows | train {len(datasets['train']):,} "
        f"val {len(datasets['val']):,} test {len(datasets['test']):,} "
        f"| obs {cfg.data.obs_len} steps, pred {cfg.data.pred_len} steps "
        f"@ {cfg.data.target_hz} Hz"
    )

    common = dict(
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.train.num_workers > 0,
    )
    train_loader = DataLoader(
        datasets["train"], batch_size=cfg.train.batch_size, shuffle=True,
        drop_last=True, **common
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=cfg.train.batch_size * 2, shuffle=False, **common
    )

    # --- model --------------------------------------------------------------
    model = build_model(NUM_FEATURES, cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model  | {n_params/1e6:.2f}M trainable parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = build_scheduler(optimizer, cfg.train, len(train_loader))
    grad_scaler = torch.amp.GradScaler(
        device.type, enabled=cfg.train.amp and device.type == "cuda"
    )

    # --- reference point ----------------------------------------------------
    # Every epoch is compared against this. A learned model that loses to
    # straight-line extrapolation is broken, not merely undertrained, and that
    # needs to be obvious from the log rather than something you notice after
    # 60 epochs.
    cv = constant_velocity_baseline(val_loader, cfg.data.target_hz, device)
    print(f"const-velocity baseline | ADE {cv['ade']:.2f} m  FDE {cv['fde']:.2f} m")

    # --- training loop ------------------------------------------------------
    history, best_ade, bad_epochs = [], float("inf"), 0
    for epoch in range(1, cfg.train.epochs + 1):
        stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, grad_scaler,
            cfg.train, device, epoch,
            physics=cfg.physics, dt=cfg.data.dt,
        )

        if epoch % cfg.train.eval_every == 0:
            val = evaluate(
                model, val_loader, cfg.train, device,
                pred_len=cfg.data.pred_len, target_hz=cfg.data.target_hz, desc="val",
            )
            stats.update({f"val_{k}": v for k, v in val.items()})
            flag = "  <-- WORSE THAN CONSTANT VELOCITY" if val["ade"] > cv["ade"] else ""
            phy = (
                f" (data {stats['train_data_loss']:.3f} + IDM {stats['train_phy_loss']:.3f})"
                if cfg.physics.weight > 0
                else ""
            )
            print(
                f"epoch {epoch:03d} | train {stats['train_loss']:.3f}{phy} "
                f"| val ADE {val['ade']:.2f} m  FDE {val['fde']:.2f} m "
                f"| lat {val['rmse_lat']:.2f}  long {val['rmse_long']:.2f}{flag}"
            )

            if val["ade"] < best_ade - 1e-4:
                best_ade, bad_epochs = val["ade"], 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg.to_dict(),
                        "scaler": scaler.to_dict(),
                        "epoch": epoch,
                        "val_ade": best_ade,
                    },
                    out_dir / "best.pt",
                )
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.train.early_stop_patience:
                    print(f"early stopping at epoch {epoch} (best val ADE {best_ade:.2f} m)")
                    break

        history.append({"epoch": epoch, **stats})
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"done | best val ADE {best_ade:.2f} m | checkpoint {out_dir/'best.pt'}")


if __name__ == "__main__":
    main()
