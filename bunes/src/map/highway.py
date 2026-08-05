from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from ..data import schema as S

def to_agent_frame_t(points: torch.Tensor, origin: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    rel = points - origin
    (c, s) = (torch.cos(theta), torch.sin(theta))
    return torch.stack([rel[:, 0] * c + rel[:, 1] * s, -rel[:, 0] * s + rel[:, 1] * c], dim=-1)

def from_agent_frame_t(points: torch.Tensor, origin: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    (c, s) = (torch.cos(theta), torch.sin(theta))
    return torch.stack([points[:, 0] * c - points[:, 1] * s, points[:, 0] * s + points[:, 1] * c], dim=-1) + origin

@dataclass
class LaneCentreMap:
    centres: torch.Tensor

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> 'LaneCentreMap':
        centres = df.groupby(S.LANE_ID)[S.Y].median().to_numpy(dtype=np.float32)
        return LaneCentreMap(torch.from_numpy(np.sort(centres)))

    @staticmethod
    def from_parquet(path: str) -> 'LaneCentreMap':
        return LaneCentreMap.from_dataframe(pd.read_parquet(path, columns=[S.LANE_ID, S.Y]))

    def to(self, device: torch.device | str) -> 'LaneCentreMap':
        return LaneCentreMap(self.centres.to(device))

    def snap(self, xy: torch.Tensor, blend: float=1.0) -> torch.Tensor:
        y = xy[:, 1:2]
        d = (y - self.centres.view(1, -1)).abs()
        nearest = self.centres[d.argmin(dim=1)]
        corrected = y.squeeze(1) + blend * (nearest - y.squeeze(1))
        return torch.stack([xy[:, 0], corrected], dim=-1)

    @property
    def lane_width(self) -> float:
        if len(self.centres) < 2:
            return float('nan')
        return float(torch.diff(self.centres).mean())

def make_link_projection_hook(lane_map: LaneCentreMap, origin: torch.Tensor, theta: torch.Tensor, blend: float=1.0, start_step: int=0):

    def hook(pos_agent: torch.Tensor, step: int) -> torch.Tensor:
        if step < start_step or blend <= 0.0:
            return pos_agent
        world = from_agent_frame_t(pos_agent, origin, theta)
        world = lane_map.snap(world, blend=blend)
        return to_agent_frame_t(world, origin, theta)
    return hook

def off_road_rate(pred_pos: torch.Tensor, origin: torch.Tensor, theta: torch.Tensor, lane_map: LaneCentreMap, margin: float=0.5) -> float:
    (b, t, _) = pred_pos.shape
    flat = pred_pos.reshape(b * t, 2)
    o = origin.repeat_interleave(t, dim=0)
    th = theta.repeat_interleave(t, dim=0)
    world_y = from_agent_frame_t(flat, o, th)[:, 1]
    half = 0.5 * lane_map.lane_width + margin
    lo = lane_map.centres.min() - half
    hi = lane_map.centres.max() + half
    return float(((world_y < lo) | (world_y > hi)).float().mean())
