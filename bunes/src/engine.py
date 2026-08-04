"""Training and evaluation loops.

Kept separate from `train.py` so Phases 3 and 5 can reuse `train_one_epoch` with
extra loss terms (IDM regulariser, adversarial loss) instead of duplicating the
optimiser/AMP/clipping boilerplate.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .metrics import full_report
from .models.seq2seq_transformer import StepHook, TrajectoryTransformer


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def trajectory_loss(
    pred_pos: torch.Tensor, tgt_pos: torch.Tensor, kind: str = "huber_position"
) -> torch.Tensor:
    """Phase 1 reconstruction loss on absolute agent-frame positions (metres).

    All three options are computed on positions rather than displacements,
    because position error is what ADE/FDE measure and what accumulates.

    'huber_position' is the default: plain MSE is dominated by the handful of
    hard-braking windows where the constant-velocity anchor is off by 100 m, and
    those outliers were drowning out the ordinary cruising cases. 'ade' is the
    evaluation metric used directly as a loss.
    """
    if kind == "mse_position":
        return nn.functional.mse_loss(pred_pos, tgt_pos)
    if kind == "huber_position":
        return nn.functional.smooth_l1_loss(pred_pos, tgt_pos, beta=1.0)
    if kind == "ade":
        return torch.linalg.vector_norm(pred_pos - tgt_pos, dim=-1).mean()
    raise ValueError(f"Unknown loss '{kind}'")


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    """Linear warmup then cosine decay."""
    total = max(1, cfg.epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: TrajectoryTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    cfg: TrainConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    running, n = 0.0, 0
    bar = tqdm(loader, desc=f"epoch {epoch:03d} [train]", leave=False)

    for batch in bar:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            pred_pos = model(batch["src"], batch["tgt_pos"], batch["cv_delta"])
            loss = trajectory_loss(pred_pos, batch["tgt_pos"], cfg.loss)

        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # GradScaler skips optimizer.step() on the iterations where it detects
        # inf/nan and halves the scale. Advancing the LR schedule on a skipped
        # step is what produces the "lr_scheduler.step() before optimizer.step()"
        # warning, so gate the scheduler on whether the step actually happened.
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:
            scheduler.step()

        bs = batch["src"].size(0)
        running += loss.item() * bs
        n += bs
        bar.set_postfix(loss=f"{running / max(1, n):.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return {"train_loss": running / max(1, n)}


@torch.no_grad()
def evaluate(
    model: TrajectoryTransformer,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    pred_len: int,
    target_hz: float,
    step_hook: StepHook | None = None,
    desc: str = "eval",
    return_predictions: bool = False,
) -> dict[str, Any]:
    """Autoregressive evaluation — no teacher forcing.

    This is the number that matters: teacher-forced validation loss looks great
    right up until the model is asked to consume its own output for 30 seconds.
    `step_hook` is the Phase 2 Link-Projection injection point.
    """
    model.eval()
    preds, trues = [], []

    for batch in tqdm(loader, desc=f"[{desc}]", leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model.rollout(batch["src"], pred_len, batch["cv_delta"], step_hook=step_hook)
        preds.append(out["pos"].float().cpu())
        trues.append(batch["tgt_pos"].float().cpu())

    pred_pos = torch.cat(preds, 0)
    true_pos = torch.cat(trues, 0)

    report: dict[str, Any] = full_report(pred_pos, true_pos, target_hz)
    if return_predictions:
        report["_pred_pos"] = pred_pos
        report["_true_pos"] = true_pos
    return report


@torch.no_grad()
def constant_velocity_baseline(
    loader: DataLoader, target_hz: float, device: torch.device
) -> dict[str, float]:
    """Sanity reference: extrapolate the last observed displacement forever.

    Any learned model that does not beat this is broken, and it is surprisingly
    strong on highways over short horizons. Reported alongside the model so the
    Phase 1 numbers are interpretable.
    """
    preds, trues = [], []
    for batch in loader:
        cv = batch["cv_delta"].to(device)                       # (B, 2)
        steps = batch["tgt_pos"].size(1)
        idx = torch.arange(1, steps + 1, device=device, dtype=cv.dtype)
        pos = cv.unsqueeze(1) * idx.view(1, -1, 1)              # (B, T, 2)
        preds.append(pos.float().cpu())
        trues.append(batch["tgt_pos"].float())
    return full_report(torch.cat(preds, 0), torch.cat(trues, 0), target_hz)
