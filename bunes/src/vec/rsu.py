"""Phase 6 — RSU coverage model and handover detection.

A highway is effectively one-dimensional, so an RSU's coverage polygon reduces
to an interval along the longitudinal axis. That makes "which RSU serves this
vehicle" a floor division and "when does it hand over" the first index at which
that division changes — both batched over the whole test set on the GPU.

Everything here works in *world* coordinates. The predictor emits agent-frame
positions, so `to_world` undoes the per-window rotation and translation first;
RSUs are fixed infrastructure and cannot live in a frame that moves with each
vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..map.highway import from_agent_frame_t


def to_world(
    pos_agent: torch.Tensor, origin: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """(N, T, 2) agent-frame positions -> (N, T, 2) world positions."""
    n, t, _ = pos_agent.shape
    flat = pos_agent.reshape(n * t, 2)
    o = origin.repeat_interleave(t, dim=0)
    th = theta.repeat_interleave(t, dim=0)
    return from_agent_frame_t(flat, o, th).reshape(n, t, 2)


@dataclass
class RSUChain:
    """Roadside units placed at a uniform spacing along the highway.

    `spacing` is the coverage diameter, so RSU k serves
    ``x in [x0 + k*spacing, x0 + (k+1)*spacing)``. Uniform spacing is the
    standard idealisation for a highway corridor and keeps the handover point
    unambiguous, which matters because the whole evaluation is about *when* that
    point is crossed.
    """

    spacing: float = 300.0
    x0: float = 0.0

    def index(self, x: torch.Tensor) -> torch.Tensor:
        """Serving RSU id for each longitudinal coordinate."""
        return torch.floor((x - self.x0) / self.spacing).long()


def first_handover(
    x_world: torch.Tensor,
    current_rsu: torch.Tensor,
    chain: RSUChain,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Time of the first RSU change along each trajectory.

    Args:
        x_world:     (N, T) longitudinal world coordinates of the horizon.
        current_rsu: (N,)   RSU serving the vehicle at the start of the horizon.
        dt:          seconds per step.

    Returns:
        (t_handover, occurred) — seconds from the start of the horizon, and a
        mask of which trajectories hand over at all. `t_handover` is 0 where no
        handover occurs; read it only under `occurred`.
    """
    idx = chain.index(x_world)                       # (N, T)
    changed = idx != current_rsu.unsqueeze(1)        # (N, T)
    occurred = changed.any(dim=1)

    # argmax on a bool cast returns the first True. Step k holds the position
    # k+1 intervals after the anchor, hence the +1.
    step = changed.float().argmax(dim=1)
    t_handover = (step + 1).to(x_world.dtype) * dt
    return torch.where(occurred, t_handover, torch.zeros_like(t_handover)), occurred
