"""Phase 3 — Intelligent Driver Model regulariser (PIT-IDM, Paper 2).

The ablation showed that ~99.8% of the remaining error is longitudinal, so the
next constraint has to act on speed rather than geometry. IDM is the standard
car-following model: it prescribes an acceleration from the ego speed, the gap
to the leader, and the approach rate. Rather than replacing the network with
it, we differentiate the predicted trajectory into an acceleration profile and
penalise it for disagreeing with what IDM would have done.

    L = L_data + phy_weight * L_IDM

Two modelling choices worth stating out loud:

* **The leader's future is unknown.** Paper 2 rolls the preceding vehicle
  forward at constant speed during multi-step prediction, and so do we. Over
  30 s that is a coarse approximation, which is exactly why this is a soft
  regulariser and not a hard constraint.

* **The IDM parameters here are deliberately not the simulator's.** The
  synthetic data is generated with IDM, so grading it against the *same*
  parameters would be circular — the loss would simply be handing the model the
  generating process. Using generic literature values instead keeps the test
  closer to "does a roughly-right physical prior help?", which is the question
  that transfers to real NGSIM/highD data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IDMParams:
    """Generic highway car-following parameters.

    Deliberately offset from the values in `src/data/synthetic.py`
    (a=1.4, b=2.0, s0=2.0, T=1.5) so the regulariser cannot simply recover the
    data-generating process.
    """

    a_max: float = 1.2       # maximum acceleration [m/s^2]
    b_comf: float = 2.2      # comfortable deceleration [m/s^2]
    s0: float = 2.5          # minimum bumper-to-bumper gap [m]
    t_headway: float = 1.4   # desired time headway [s]
    delta: float = 4.0       # acceleration exponent
    a_clip: float = 6.0      # clamp on the prescribed acceleration [m/s^2]


def idm_acceleration(
    v: torch.Tensor,
    v0: torch.Tensor,
    gap: torch.Tensor,
    dv: torch.Tensor,
    p: IDMParams,
) -> torch.Tensor:
    """Differentiable IDM acceleration.

    Args:
        v:   current speed [m/s]
        v0:  desired free-flow speed [m/s]
        gap: bumper-to-bumper distance to the leader [m]
        dv:  approach rate, v_ego - v_leader [m/s]
    """
    # The gap floor keeps the 1/gap^2 term finite when a rollout drives the ego
    # through its leader; those samples are penalised hard but not to infinity.
    gap = gap.clamp(min=0.5)
    v = v.clamp(min=0.0)

    s_star = p.s0 + torch.clamp(
        v * p.t_headway + (v * dv) / (2.0 * (p.a_max * p.b_comf) ** 0.5), min=0.0
    )
    free = 1.0 - (v / v0.clamp(min=1.0)) ** p.delta
    return (p.a_max * (free - (s_star / gap) ** 2)).clamp(-p.a_clip, p.a_clip)


def kinematics_from_positions(
    pred_pos: torch.Tensor, cv_delta: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate a predicted trajectory into speed and acceleration.

    Only the longitudinal channel is used: the agent frame's +x axis is the
    heading at the last observed step, and highway travel is along it.

    Returns (speed, accel), both (B, T). The step before the horizon starts is
    the last *observed* displacement, so the first acceleration is a genuine
    continuation of the history rather than an artefact of a zero initial state.
    """
    x = pred_pos[..., 0]                                        # (B, T)
    x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
    v = (x - x_prev) / dt

    v_last_observed = (cv_delta[:, 0] / dt).unsqueeze(1)        # (B, 1)
    v_prev = torch.cat([v_last_observed, v[:, :-1]], dim=1)
    a = (v - v_prev) / dt
    return v, a


def idm_physics_loss(
    pred_pos: torch.Tensor,
    cv_delta: torch.Tensor,
    leader_gap: torch.Tensor,
    leader_speed: torch.Tensor,
    desired_speed: torch.Tensor,
    dt: float,
    p: IDMParams,
    min_gap: float = 1.0,
    horizon_steps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalise predicted accelerations that disagree with IDM.

    Args:
        pred_pos:      (B, T, 2) predicted positions, agent frame [m]
        cv_delta:      (B, 2)    last observed displacement [m]
        leader_gap:    (B,)      bumper-to-bumper gap at the last observed step [m]
        leader_speed:  (B,)      leader speed at that step [m/s]
        desired_speed: (B,)      free-flow speed target for each ego vehicle
        min_gap:       windows with a smaller (or missing) gap are excluded —
                       they are usually a data artefact rather than real
                       tailgating, and the 1/gap^2 term would dominate.
        horizon_steps: apply the penalty only to the first N predicted steps.
                       The leader is rolled forward at constant speed, and that
                       assumption decays fast: at 30 s it can be tens of metres
                       out, so IDM is then prescribing an acceleration for a gap
                       the leader never had. Restricting the term to the part of
                       the horizon where the assumption still holds is the
                       difference between a regulariser and a source of noise.
                       None applies it over the whole horizon.

    Returns (loss, valid_fraction). The loss is 0 when no window in the batch
    has a usable leader.
    """
    b, t, _ = pred_pos.shape
    v, a = kinematics_from_positions(pred_pos, cv_delta, dt)

    # The leader is rolled forward at constant speed (Paper 2). Its longitudinal
    # displacement from the ego's origin at step k is gap0 + v_leader * k*dt.
    steps = torch.arange(1, t + 1, device=pred_pos.device, dtype=pred_pos.dtype) * dt
    leader_x = leader_gap.unsqueeze(1) + leader_speed.unsqueeze(1) * steps.view(1, -1)
    gap = leader_x - pred_pos[..., 0]
    dv = v - leader_speed.unsqueeze(1)

    a_idm = idm_acceleration(v, desired_speed.unsqueeze(1).expand_as(v), gap, dv, p)

    valid = (
        torch.isfinite(leader_gap)
        & torch.isfinite(leader_speed)
        & (leader_gap > min_gap)
        & (leader_speed > 0.5)
    )
    if not bool(valid.any()):
        zero = pred_pos.sum() * 0.0        # keeps the graph connected
        return zero, torch.zeros((), device=pred_pos.device)

    k = t if horizon_steps is None else min(horizon_steps, t)
    per_sample = (a[:, :k] - a_idm[:, :k]).pow(2).mean(dim=1)   # (B,)
    loss = per_sample[valid].mean()
    return loss, valid.float().mean()
