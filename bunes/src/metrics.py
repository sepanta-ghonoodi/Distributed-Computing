from __future__ import annotations
import torch

def displacement_error(pred_pos: torch.Tensor, true_pos: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(pred_pos - true_pos, dim=-1)

def ade(pred_pos: torch.Tensor, true_pos: torch.Tensor) -> torch.Tensor:
    return displacement_error(pred_pos, true_pos).mean()

def fde(pred_pos: torch.Tensor, true_pos: torch.Tensor) -> torch.Tensor:
    return displacement_error(pred_pos, true_pos)[:, -1].mean()

def miss_rate(pred_pos: torch.Tensor, true_pos: torch.Tensor, threshold: float=5.0) -> torch.Tensor:
    return (displacement_error(pred_pos, true_pos)[:, -1] > threshold).float().mean()

def full_report(pred_pos: torch.Tensor, true_pos: torch.Tensor, target_hz: float, horizons_s: tuple[float, ...]=(1.0, 3.0, 5.0, 10.0, 20.0, 30.0)) -> dict[str, float]:
    err = displacement_error(pred_pos, true_pos)
    residual = pred_pos - true_pos
    out: dict[str, float] = {'ade': ade(pred_pos, true_pos).item(), 'fde': fde(pred_pos, true_pos).item(), 'miss_rate@5m': miss_rate(pred_pos, true_pos, 5.0).item(), 'rmse_long': residual[..., 0].pow(2).mean().sqrt().item(), 'rmse_lat': residual[..., 1].pow(2).mean().sqrt().item()}
    total_steps = err.size(1)
    for h in horizons_s:
        k = int(round(h * target_hz))
        if 0 < k <= total_steps:
            out[f'ade@{h:g}s'] = err[:, :k].mean().item()
            out[f'fde@{h:g}s'] = err[:, k - 1].mean().item()
    return out
