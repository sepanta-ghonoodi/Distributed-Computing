"""Trajectory evaluation metrics.

All metrics operate on agent-frame positions in metres. Because the agent frame
is a rigid transform of the world frame, every distance reported here is
identical to the corresponding world-frame distance.
"""

from __future__ import annotations

import torch


def displacement_error(pred_pos: torch.Tensor, true_pos: torch.Tensor) -> torch.Tensor:
    """Per-sample, per-timestep Euclidean error. (B, T, 2) x2 -> (B, T)."""
    return torch.linalg.vector_norm(pred_pos - true_pos, dim=-1)


def ade(pred_pos: torch.Tensor, true_pos: torch.Tensor) -> torch.Tensor:
    """Average Displacement Error over the whole horizon [m]."""
    return displacement_error(pred_pos, true_pos).mean()


def fde(pred_pos: torch.Tensor, true_pos: torch.Tensor) -> torch.Tensor:
    """Final Displacement Error at the last predicted step [m]."""
    return displacement_error(pred_pos, true_pos)[:, -1].mean()


def miss_rate(pred_pos: torch.Tensor, true_pos: torch.Tensor, threshold: float = 5.0) -> torch.Tensor:
    """Fraction of samples whose final error exceeds `threshold` metres."""
    return (displacement_error(pred_pos, true_pos)[:, -1] > threshold).float().mean()


def full_report(
    pred_pos: torch.Tensor,
    true_pos: torch.Tensor,
    target_hz: float,
    horizons_s: tuple[float, ...] = (1.0, 3.0, 5.0, 10.0, 20.0, 30.0),
) -> dict[str, float]:
    """ADE/FDE plus a horizon breakdown and a longitudinal/lateral split.

    The longitudinal/lateral split is the diagnostic that matters for this
    project: long-horizon *longitudinal* error is dominated by speed
    misestimation (Phase 3's IDM loss target), while *lateral* error is
    dominated by lane-change and off-road drift (Phase 2's Link Projection
    target). A single ADE number hides which one is failing.
    """
    err = displacement_error(pred_pos, true_pos)          # (B, T)
    residual = pred_pos - true_pos                        # (B, T, 2)

    out: dict[str, float] = {
        "ade": ade(pred_pos, true_pos).item(),
        "fde": fde(pred_pos, true_pos).item(),
        "miss_rate@5m": miss_rate(pred_pos, true_pos, 5.0).item(),
        "rmse_long": residual[..., 0].pow(2).mean().sqrt().item(),
        "rmse_lat": residual[..., 1].pow(2).mean().sqrt().item(),
    }

    total_steps = err.size(1)
    for h in horizons_s:
        k = int(round(h * target_hz))
        if 0 < k <= total_steps:
            out[f"ade@{h:g}s"] = err[:, :k].mean().item()
            out[f"fde@{h:g}s"] = err[:, k - 1].mean().item()
    return out
