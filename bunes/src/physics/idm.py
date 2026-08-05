from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class IDMParams:
    a_max: float = 1.2
    b_comf: float = 2.2
    s0: float = 2.5
    t_headway: float = 1.4
    delta: float = 4.0
    a_clip: float = 6.0

def idm_acceleration(v: torch.Tensor, v0: torch.Tensor, gap: torch.Tensor, dv: torch.Tensor, p: IDMParams) -> torch.Tensor:
    gap = gap.clamp(min=0.5)
    v = v.clamp(min=0.0)
    s_star = p.s0 + torch.clamp(v * p.t_headway + v * dv / (2.0 * (p.a_max * p.b_comf) ** 0.5), min=0.0)
    free = 1.0 - (v / v0.clamp(min=1.0)) ** p.delta
    return (p.a_max * (free - (s_star / gap) ** 2)).clamp(-p.a_clip, p.a_clip)

def kinematics_from_positions(pred_pos: torch.Tensor, cv_delta: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = pred_pos[..., 0]
    x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
    v = (x - x_prev) / dt
    v_last_observed = (cv_delta[:, 0] / dt).unsqueeze(1)
    v_prev = torch.cat([v_last_observed, v[:, :-1]], dim=1)
    a = (v - v_prev) / dt
    return (v, a)

def idm_physics_loss(pred_pos: torch.Tensor, cv_delta: torch.Tensor, leader_gap: torch.Tensor, leader_speed: torch.Tensor, desired_speed: torch.Tensor, dt: float, p: IDMParams, min_gap: float=1.0, horizon_steps: int | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    (b, t, _) = pred_pos.shape
    (v, a) = kinematics_from_positions(pred_pos, cv_delta, dt)
    valid = torch.isfinite(leader_gap) & torch.isfinite(leader_speed) & (leader_gap > min_gap) & (leader_speed > 0.5)
    safe_gap = torch.where(valid, leader_gap, torch.full_like(leader_gap, 50.0))
    safe_vlead = torch.where(valid, leader_speed, torch.full_like(leader_speed, 20.0))
    safe_v0 = torch.where(torch.isfinite(desired_speed), desired_speed, torch.full_like(desired_speed, 25.0))
    steps = torch.arange(1, t + 1, device=pred_pos.device, dtype=pred_pos.dtype) * dt
    leader_x = safe_gap.unsqueeze(1) + safe_vlead.unsqueeze(1) * steps.view(1, -1)
    gap = leader_x - pred_pos[..., 0]
    dv = v - safe_vlead.unsqueeze(1)
    with torch.no_grad():
        a_idm = idm_acceleration(v.detach(), safe_v0.unsqueeze(1).expand_as(v), gap.detach(), dv.detach(), p)
    k = t if horizon_steps is None else min(horizon_steps, t)
    per_sample = (a[:, :k] - a_idm[:, :k]).pow(2).mean(dim=1)
    w = valid.to(per_sample.dtype)
    loss = (per_sample * w).sum() / w.sum().clamp(min=1.0)
    return (loss, w.mean())
