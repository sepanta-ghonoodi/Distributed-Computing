"""Phase 1 baseline: a Seq2Seq Transformer trajectory predictor.

Design notes (these are the decisions that matter for Phases 2-5):

1. **The model predicts per-step displacements, not absolute positions.**
   Positions are recovered by cumulative sum. Predicting positions directly
   makes the network spend capacity learning "stay near where you were".

2. **Constant-velocity residual.** The network output is added to a
   constant-velocity extrapolation of the last observed step. This is the
   "History Message" of the PIT-IDM paper, and it means an untrained model
   already produces a straight-line prediction rather than a collapsed one.
   Phase 3 extends the same fusion point with an IDM acceleration term.

3. **`rollout()` exposes a per-step hook.** Phase 2's Link Projection snaps each
   intermediate coordinate to the nearest lane centreline *before* it is fed
   back into the decoder. That hook is already in place here, unused — so
   Phase 2 is a new module plus a callback, not a rewrite of the decode loop.

4. **Pre-LN (`norm_first=True`)** — stable without a long LR warmup, which
   matters once the adversarial loss of Phase 5 is added.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from ..config import ModelConfig
from .positional_encoding import SinusoidalPositionalEncoding

# Signature of the Phase 2 Link-Projection hook:
#   (positions (B, 2) in the agent frame, step index) -> corrected positions (B, 2)
StepHook = Callable[[torch.Tensor, int], torch.Tensor]


class TrajectoryTransformer(nn.Module):
    """Encoder-decoder Transformer over highway trajectory windows.

    Shapes (batch-first throughout):
        src        (B, T_obs,  F)  scaled kinematic history
        tgt_in     (B, T_pred, 2)  previous-step displacements (teacher forcing)
        cv_delta   (B, 2)          last observed displacement, metres
        returns    (B, T_pred, 2)  predicted per-step displacements, metres
    """

    def __init__(self, input_dim: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim

        # --- embeddings ---------------------------------------------------
        self.src_embed = nn.Linear(input_dim, cfg.d_model)
        self.tgt_embed = nn.Linear(2, cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.dropout)

        # --- transformer ---------------------------------------------------
        # nn.Transformer is used rather than hand-rolled blocks because Phase 4
        # needs to inject an additive attention bias, and the cleanest way to do
        # that is to subclass the decoder layer here later.
        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation=cfg.activation,
            batch_first=True,
            norm_first=cfg.norm_first,
        )

        # --- output head ----------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 2),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Zero the final layer so the model starts *exactly* at the
        # constant-velocity prior and learns the correction from there.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    # -- helpers -------------------------------------------------------------
    def _prior(self, cv_delta: torch.Tensor, steps: int) -> torch.Tensor:
        """Constant-velocity displacement prior, (B, steps, 2)."""
        if not self.cfg.use_cv_prior:
            return torch.zeros(
                cv_delta.size(0), steps, 2, device=cv_delta.device, dtype=cv_delta.dtype
            )
        return cv_delta.unsqueeze(1).expand(-1, steps, -1)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Run the encoder once; the memory is reused across all decode steps."""
        h = self.pos_enc(self.src_embed(src))
        return self.transformer.encoder(h)

    def _decode(
        self, memory: torch.Tensor, tgt_residual: torch.Tensor
    ) -> torch.Tensor:
        """Decode the whole (causally masked) target prefix. Returns (B, T, 2)."""
        t = tgt_residual.size(1)
        h = self.pos_enc(self.tgt_embed(tgt_residual / self.cfg.delta_scale))
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=h.device)
        out = self.transformer.decoder(h, memory, tgt_mask=mask)
        return self.head(out) * self.cfg.delta_scale

    # -- training path (teacher forcing) -------------------------------------
    def forward(
        self, src: torch.Tensor, tgt_in: torch.Tensor, cv_delta: torch.Tensor
    ) -> torch.Tensor:
        """Teacher-forced forward pass. All timesteps in one shot.

        `tgt_in` is the ground truth shifted right: tgt_in[:, 0] is the zero
        start token and tgt_in[:, t] == true_delta[:, t-1]. The decoder is fed
        the *residual* of each of those w.r.t. the prior, so that the token
        stream is identical to the one `rollout` constructs at inference.
        """
        steps = tgt_in.size(1)
        prior = self._prior(cv_delta, steps)
        memory = self.encode(src)

        tokens = torch.cat(
            [
                torch.zeros_like(tgt_in[:, :1, :]),          # start token
                tgt_in[:, 1:, :] - prior[:, :-1, :],         # residual of step t-1
            ],
            dim=1,
        )
        residual = self._decode(memory, tokens)
        return prior + residual

    # -- inference path (autoregressive) --------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        src: torch.Tensor,
        steps: int,
        cv_delta: torch.Tensor,
        step_hook: StepHook | None = None,
    ) -> dict[str, torch.Tensor]:
        """Autoregressively generate `steps` future displacements.

        Args:
            step_hook: optional per-step correction applied to the cumulative
                agent-frame position before it is fed back (Phase 2 hook).
                Must return a tensor of the same shape it receives.

        Returns:
            dict with 'delta' (B, steps, 2) and 'pos' (B, steps, 2), agent frame.

        Note: this re-runs the decoder over the full prefix at every step
        (O(T^2) decoder calls). At T_pred = 60 that is not a bottleneck; if the
        horizon grows, add incremental KV caching here.
        """
        b = src.size(0)
        device = src.device
        memory = self.encode(src)
        prior = self._prior(cv_delta, steps)

        # Decoder input starts with a single zero token (== "no previous step").
        dec_in = torch.zeros(b, 1, 2, device=device, dtype=src.dtype)
        pos = torch.zeros(b, 2, device=device, dtype=src.dtype)  # agent-frame origin

        deltas, positions = [], []
        for t in range(steps):
            residual = self._decode(memory, dec_in)[:, -1, :]     # (B, 2)
            delta = prior[:, t, :] + residual
            pos = pos + delta

            if step_hook is not None:
                corrected = step_hook(pos, t)
                # Re-derive the displacement so that the token fed back is
                # consistent with the corrected position.
                delta = delta + (corrected - pos)
                pos = corrected

            deltas.append(delta)
            positions.append(pos)

            # Feed back the residual w.r.t. the prior, matching training.
            next_token = (delta - prior[:, t, :]).unsqueeze(1)
            dec_in = torch.cat([dec_in, next_token], dim=1)

        return {
            "delta": torch.stack(deltas, dim=1),
            "pos": torch.stack(positions, dim=1),
        }


def build_model(input_dim: int, cfg: ModelConfig) -> TrajectoryTransformer:
    return TrajectoryTransformer(input_dim, cfg)
