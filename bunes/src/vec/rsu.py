from __future__ import annotations
from dataclasses import dataclass
import torch
from ..map.highway import from_agent_frame_t

def to_world(pos_agent: torch.Tensor, origin: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    (n, t, _) = pos_agent.shape
    flat = pos_agent.reshape(n * t, 2)
    o = origin.repeat_interleave(t, dim=0)
    th = theta.repeat_interleave(t, dim=0)
    return from_agent_frame_t(flat, o, th).reshape(n, t, 2)

@dataclass
class RSUChain:
    spacing: float = 300.0
    x0: float = 0.0

    def index(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor((x - self.x0) / self.spacing).long()

def first_handover(x_world: torch.Tensor, current_rsu: torch.Tensor, chain: RSUChain, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    idx = chain.index(x_world)
    changed = idx != current_rsu.unsqueeze(1)
    occurred = changed.any(dim=1)
    step = changed.float().argmax(dim=1)
    t_handover = (step + 1).to(x_world.dtype) * dt
    return (torch.where(occurred, t_handover, torch.zeros_like(t_handover)), occurred)
