"""Phase 2 — highway geometry and Link Projection.

Paper 7's Link Projection snaps each intermediate prediction back onto the road
geometry *before* it is fed back into the decoder, so lateral error cannot
accumulate across the 30 s rollout.

Two things make this cheap on a highway. First, the geometry is a set of
parallel centrelines, so "nearest point on the road" reduces to "nearest lane
centre in y" — no Shapely, no KD-tree, and it stays differentiable-shaped and
batched on the GPU. Second, the model already carries `origin`/`theta` in every
batch, so converting a partial rollout to world coordinates and back is exact.

The snapping strength is deliberately a knob rather than hard-wired to 1.0.
Snapping all the way to the centreline is correct for a vehicle tracking its
lane but wrong in the middle of a lane change, where the true trajectory is
genuinely between two centres. `blend=1.0` gives the largest reduction in
lateral error; lower values trade some of that for keeping lane changes intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..data import schema as S


# ---------------------------------------------------------------------------
# Frame transforms (torch, batched over samples — the numpy pair in
# data/transforms.py is the same maths for whole windows at preprocessing time)
# ---------------------------------------------------------------------------
def to_agent_frame_t(
    points: torch.Tensor, origin: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """World -> agent frame. points/origin (B, 2), theta (B,)."""
    rel = points - origin
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack(
        [rel[:, 0] * c + rel[:, 1] * s, -rel[:, 0] * s + rel[:, 1] * c], dim=-1
    )


def from_agent_frame_t(
    points: torch.Tensor, origin: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """Agent frame -> world. Exact inverse of `to_agent_frame_t`."""
    c, s = torch.cos(theta), torch.sin(theta)
    return (
        torch.stack(
            [points[:, 0] * c - points[:, 1] * s, points[:, 0] * s + points[:, 1] * c],
            dim=-1,
        )
        + origin
    )


# ---------------------------------------------------------------------------
# Lane geometry
# ---------------------------------------------------------------------------
@dataclass
class LaneCentreMap:
    """Straight-highway geometry: a set of lane centrelines at fixed lateral y.

    Derived from the data rather than hard-coded, so the same class works for
    the synthetic simulator, NGSIM and highD without edits.
    """

    centres: torch.Tensor  # (L,) lateral position of each lane centre [m]

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "LaneCentreMap":
        """Estimate one centreline per lane id as the median lateral position."""
        centres = df.groupby(S.LANE_ID)[S.Y].median().to_numpy(dtype=np.float32)
        return LaneCentreMap(torch.from_numpy(np.sort(centres)))

    @staticmethod
    def from_parquet(path: str) -> "LaneCentreMap":
        return LaneCentreMap.from_dataframe(pd.read_parquet(path, columns=[S.LANE_ID, S.Y]))

    def to(self, device: torch.device | str) -> "LaneCentreMap":
        return LaneCentreMap(self.centres.to(device))

    def snap(self, xy: torch.Tensor, blend: float = 1.0) -> torch.Tensor:
        """Pull the lateral coordinate towards the nearest lane centre.

        Args:
            xy:    (B, 2) world coordinates. The longitudinal channel is left
                   untouched — a highway centreline runs along x, so projecting
                   onto it only ever changes y.
            blend: 0 = no correction, 1 = snap exactly onto the centreline.
        """
        y = xy[:, 1:2]                                   # (B, 1)
        d = (y - self.centres.view(1, -1)).abs()         # (B, L)
        nearest = self.centres[d.argmin(dim=1)]          # (B,)
        corrected = y.squeeze(1) + blend * (nearest - y.squeeze(1))
        return torch.stack([xy[:, 0], corrected], dim=-1)

    @property
    def lane_width(self) -> float:
        """Mean spacing between adjacent centrelines — used for off-road tests."""
        if len(self.centres) < 2:
            return float("nan")
        return float(torch.diff(self.centres).mean())


# ---------------------------------------------------------------------------
# The Phase 2 rollout hook
# ---------------------------------------------------------------------------
def make_link_projection_hook(
    lane_map: LaneCentreMap,
    origin: torch.Tensor,
    theta: torch.Tensor,
    blend: float = 1.0,
    start_step: int = 0,
):
    """Build the per-step callback consumed by `TrajectoryTransformer.rollout`.

    Args:
        origin, theta: the agent-frame pose of each sample in the batch.
        start_step: leave the first N steps uncorrected. Immediately after the
            observation window the vehicle may legitimately be mid-lane-change,
            and snapping there fights the model instead of helping it.

    Returns a closure `(pos_agent (B, 2), step) -> corrected pos_agent (B, 2)`.
    """

    def hook(pos_agent: torch.Tensor, step: int) -> torch.Tensor:
        if step < start_step or blend <= 0.0:
            return pos_agent
        world = from_agent_frame_t(pos_agent, origin, theta)
        world = lane_map.snap(world, blend=blend)
        return to_agent_frame_t(world, origin, theta)

    return hook


# ---------------------------------------------------------------------------
# Diagnostic used in the Phase 1 vs Phase 2 comparison
# ---------------------------------------------------------------------------
def off_road_rate(
    pred_pos: torch.Tensor,
    origin: torch.Tensor,
    theta: torch.Tensor,
    lane_map: LaneCentreMap,
    margin: float = 0.5,
) -> float:
    """Fraction of predicted points that fall outside the drivable width.

    "Outside" means beyond the outermost centreline by more than half a lane
    plus `margin`. This is the headline number Link Projection is supposed to
    drive to zero.
    """
    b, t, _ = pred_pos.shape
    flat = pred_pos.reshape(b * t, 2)
    o = origin.repeat_interleave(t, dim=0)
    th = theta.repeat_interleave(t, dim=0)
    world_y = from_agent_frame_t(flat, o, th)[:, 1]

    half = 0.5 * lane_map.lane_width + margin
    lo = lane_map.centres.min() - half
    hi = lane_map.centres.max() + half
    return float(((world_y < lo) | (world_y > hi)).float().mean())
