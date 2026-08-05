"""Phase 6 — proactive service migration policy and its cost model.

The policy is the one from Papers 3 and 8: predict when the vehicle will leave
its serving RSU, and begin copying the service state `T_migration` seconds
before that, so the copy lands just as the vehicle arrives.

Cost model, all times relative to the start of the prediction horizon:

* **Reactive** — migration is triggered by the handover itself, so it starts at
  ``t_true`` and finishes at ``t_true + T_m``. The service is unavailable for
  the whole of `T_m`. This is the baseline every proactive scheme must beat.

* **Proactive** — migration starts at ``t_pred - T_m`` and finishes at
  ``t_pred``. Two ways to be wrong:
    - *late* (``t_pred > t_true``): the copy is still in flight when the vehicle
      crosses, so the service is interrupted for ``t_pred - t_true``, capped at
      `T_m` (past that it is no better than reacting).
    - *early* (``t_pred < t_true``): the copy completes before the vehicle
      arrives. No interruption, but the state sits on the next RSU while the
      vehicle is still served by the current one — wasted residency, reported
      separately rather than hidden.

* **No predicted handover** — the trigger never fires and the scheme silently
  degrades to reactive. Counted, because a predictor that simply never predicts
  a handover would otherwise look flawless on interruption.

The point of reporting a constant-velocity predictor alongside the model is
that it is the policy you get for free. If the learned predictor cannot beat
straight-line extrapolation *on this metric*, none of the trajectory modelling
earns its place in the system.
"""

from __future__ import annotations

import torch


def migration_metrics(
    t_pred: torch.Tensor,
    predicted: torch.Tensor,
    t_true: torch.Tensor,
    t_migration: float,
    margin: float = 0.0,
) -> dict[str, float]:
    """Evaluate one predictor's migration decisions.

    All inputs are restricted to windows where a handover genuinely occurs.

    Args:
        t_pred:    (N,) predicted handover time [s]
        predicted: (N,) whether this predictor saw a handover at all
        t_true:    (N,) actual handover time [s]
        margin:    safety lead time [s]. The cost here is asymmetric — being
            early only wastes residency, being late interrupts the service — so
            a policy that simply triggers at the predicted time is leaving
            interruption on the table. Starting `margin` seconds earlier trades
            residency for interruption, and the right value depends on how
            accurate the predictor is: a predictor with lower ETA error needs
            less margin to reach the same interruption, which is how prediction
            accuracy is supposed to turn into system benefit.
    """
    t_m = torch.full_like(t_true, t_migration)

    # A missed handover means the trigger never fires: fall back to reacting at
    # the crossing itself. The margin cannot help there — there is nothing to
    # be early about.
    completes = torch.where(predicted, t_pred - margin, t_true + t_m)

    interruption = (completes - t_true).clamp(min=0.0, max=t_migration)
    premature = (t_true - completes).clamp(min=0.0)
    eta_error = torch.where(
        predicted, (t_pred - t_true).abs(), torch.full_like(t_true, float("nan"))
    )

    return {
        "mean_interruption_s": float(interruption.mean()),
        "zero_interruption_rate": float((interruption <= 1e-6).float().mean()),
        "mean_premature_s": float(premature.mean()),
        "handover_detect_rate": float(predicted.float().mean()),
        "mean_eta_error_s": float(eta_error[predicted].abs().mean()) if bool(predicted.any()) else float("nan"),
        "interruption_reduction_pct": float(
            100.0 * (1.0 - interruption.mean() / t_migration)
        ),
    }


def reactive_metrics(t_true: torch.Tensor, t_migration: float) -> dict[str, float]:
    """The do-nothing baseline: every handover costs a full migration time."""
    return {
        "mean_interruption_s": t_migration,
        "zero_interruption_rate": 0.0,
        "mean_premature_s": 0.0,
        "handover_detect_rate": 0.0,
        "mean_eta_error_s": float("nan"),
        "interruption_reduction_pct": 0.0,
    }
