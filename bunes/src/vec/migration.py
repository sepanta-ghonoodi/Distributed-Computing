from __future__ import annotations
import torch

def migration_metrics(t_pred: torch.Tensor, predicted: torch.Tensor, t_true: torch.Tensor, t_migration: float, margin: float=0.0) -> dict[str, float]:
    t_m = torch.full_like(t_true, t_migration)
    completes = torch.where(predicted, t_pred - margin, t_true + t_m)
    interruption = (completes - t_true).clamp(min=0.0, max=t_migration)
    premature = (t_true - completes).clamp(min=0.0)
    eta_error = torch.where(predicted, (t_pred - t_true).abs(), torch.full_like(t_true, float('nan')))
    return {'mean_interruption_s': float(interruption.mean()), 'zero_interruption_rate': float((interruption <= 1e-06).float().mean()), 'mean_premature_s': float(premature.mean()), 'handover_detect_rate': float(predicted.float().mean()), 'mean_eta_error_s': float(eta_error[predicted].abs().mean()) if bool(predicted.any()) else float('nan'), 'interruption_reduction_pct': float(100.0 * (1.0 - interruption.mean() / t_migration))}

def reactive_metrics(t_true: torch.Tensor, t_migration: float) -> dict[str, float]:
    return {'mean_interruption_s': t_migration, 'zero_interruption_rate': 0.0, 'mean_premature_s': 0.0, 'handover_detect_rate': 0.0, 'mean_eta_error_s': float('nan'), 'interruption_reduction_pct': 0.0}
