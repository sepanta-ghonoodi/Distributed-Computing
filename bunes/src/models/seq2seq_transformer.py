from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn
from ..config import ModelConfig
from .positional_encoding import SinusoidalPositionalEncoding
StepHook = Callable[[torch.Tensor, int], torch.Tensor]

class TrajectoryTransformer(nn.Module):

    def __init__(self, input_dim: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.src_embed = nn.Linear(input_dim, cfg.d_model)
        self.tgt_embed = nn.Linear(2, cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.dropout)
        self.transformer = nn.Transformer(d_model=cfg.d_model, nhead=cfg.nhead, num_encoder_layers=cfg.num_encoder_layers, num_decoder_layers=cfg.num_decoder_layers, dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout, activation=cfg.activation, batch_first=True, norm_first=cfg.norm_first)
        self.head = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model // 2), nn.GELU(), nn.Linear(cfg.d_model // 2, 2))
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def prior_positions(self, cv_delta: torch.Tensor, steps: int) -> torch.Tensor:
        if not self.cfg.use_cv_prior:
            return torch.zeros(cv_delta.size(0), steps, 2, device=cv_delta.device, dtype=cv_delta.dtype)
        k = torch.arange(1, steps + 1, device=cv_delta.device, dtype=cv_delta.dtype)
        return cv_delta.unsqueeze(1) * k.view(1, -1, 1)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        h = self.pos_enc(self.src_embed(src))
        return self.transformer.encoder(h)

    def _decode(self, memory: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.size(1)
        h = self.pos_enc(self.tgt_embed(tokens / self.cfg.output_scale))
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=h.device)
        out = self.transformer.decoder(h, memory, tgt_mask=mask)
        return self.head(out) * self.cfg.output_scale

    @staticmethod
    def _shift_right(offsets: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.zeros_like(offsets[:, :1, :]), offsets[:, :-1, :]], dim=1)

    def forward(self, src: torch.Tensor, tgt_pos: torch.Tensor, cv_delta: torch.Tensor) -> torch.Tensor:
        steps = tgt_pos.size(1)
        prior = self.prior_positions(cv_delta, steps)
        memory = self.encode(src)
        tokens = self._shift_right(tgt_pos - prior)
        return prior + self._decode(memory, tokens)

    @torch.no_grad()
    def rollout(self, src: torch.Tensor, steps: int, cv_delta: torch.Tensor, step_hook: StepHook | None=None) -> dict[str, torch.Tensor]:
        b = src.size(0)
        memory = self.encode(src)
        prior = self.prior_positions(cv_delta, steps)
        tokens = torch.zeros(b, 1, 2, device=src.device, dtype=src.dtype)
        positions = []
        for t in range(steps):
            offset = self._decode(memory, tokens)[:, -1, :]
            pos = prior[:, t, :] + offset
            if step_hook is not None:
                pos = step_hook(pos, t)
                offset = pos - prior[:, t, :]
            positions.append(pos)
            tokens = torch.cat([tokens, offset.unsqueeze(1)], dim=1)
        pos_seq = torch.stack(positions, dim=1)
        delta = torch.diff(pos_seq, dim=1, prepend=torch.zeros_like(pos_seq[:, :1, :]))
        return {'pos': pos_seq, 'delta': delta}

def build_model(input_dim: int, cfg: ModelConfig) -> TrajectoryTransformer:
    return TrajectoryTransformer(input_dim, cfg)
