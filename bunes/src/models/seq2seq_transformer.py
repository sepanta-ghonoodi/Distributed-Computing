"""Phase 1 baseline: a Seq2Seq Transformer trajectory predictor.

Design notes (these are the decisions that matter for Phases 2-5):

1. **The model predicts an offset from a constant-velocity trajectory.**
   The prediction is ``pos_t = (t+1) * cv_delta + o_t``, where ``cv_delta`` is
   the last observed displacement and ``o_t`` is the network output. This is
   the "History Message" fusion of the PIT-IDM paper, and Phase 3 extends the
   same anchor with an IDM-integrated trajectory instead of a straight line.

   The first version of this file predicted *per-step displacements* and
   recovered position by ``cumsum``. That parameterisation is an integrator:
   under teacher forcing the cheapest solution is to copy the previous
   displacement token, which scores near-perfectly, but during free-running
   rollout the model then integrates its own copied error 60 times. The result
   was a feedback loop with gain ~1 -- validation ADE oscillated between 7 m
   and 44 m across epochs while training loss fell monotonically, and the model
   lost to a constant-velocity baseline. Anchoring each step to the prior
   removes the integrator: an error in ``o_t`` no longer propagates into
   ``o_{t+1}``, and the degenerate copy solution degrades gracefully to
   "constant velocity plus a fixed offset" instead of diverging.

2. **The head is zero-initialised**, so an untrained model is *exactly* a
   constant-velocity predictor and training starts from a sane reference.

3. **``rollout()`` exposes a per-step hook.** Phase 2's Link Projection snaps
   each intermediate coordinate to the nearest lane centreline *before* it is
   fed back into the decoder. The hook receives an absolute agent-frame
   position and returns a corrected one; the loop re-derives the offset token,
   so Phase 2 is a new module plus a callback, not a rewrite of this loop.

4. **Pre-LN (``norm_first=True``)** -- stable without a long LR warmup, which
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
        src       (B, T_obs,  F)  scaled kinematic history
        tgt_pos   (B, T_pred, 2)  ground-truth future positions, agent frame [m]
        cv_delta  (B, 2)          last observed displacement [m]
        returns   (B, T_pred, 2)  predicted future positions, agent frame [m]
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
    def prior_positions(self, cv_delta: torch.Tensor, steps: int) -> torch.Tensor:
        """Constant-velocity anchor trajectory, (B, steps, 2).

        Step t (0-indexed) sits t+1 displacements after the agent-frame origin.
        """
        if not self.cfg.use_cv_prior:
            return torch.zeros(
                cv_delta.size(0), steps, 2, device=cv_delta.device, dtype=cv_delta.dtype
            )
        k = torch.arange(1, steps + 1, device=cv_delta.device, dtype=cv_delta.dtype)
        return cv_delta.unsqueeze(1) * k.view(1, -1, 1)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Run the encoder once; the memory is reused across all decode steps."""
        h = self.pos_enc(self.src_embed(src))
        return self.transformer.encoder(h)

    def _decode(self, memory: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Decode a causally masked token prefix. (B, T, 2) offsets -> (B, T, 2)."""
        t = tokens.size(1)
        h = self.pos_enc(self.tgt_embed(tokens / self.cfg.output_scale))
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=h.device)
        out = self.transformer.decoder(h, memory, tgt_mask=mask)
        return self.head(out) * self.cfg.output_scale

    @staticmethod
    def _shift_right(offsets: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing token stream: [0, o_0, o_1, ..., o_{T-2}]."""
        return torch.cat([torch.zeros_like(offsets[:, :1, :]), offsets[:, :-1, :]], dim=1)

    # -- training path (teacher forcing) -------------------------------------
    def forward(
        self, src: torch.Tensor, tgt_pos: torch.Tensor, cv_delta: torch.Tensor
    ) -> torch.Tensor:
        """Teacher-forced forward pass over all timesteps at once.

        Returns predicted absolute positions in the agent frame.
        """
        steps = tgt_pos.size(1)
        prior = self.prior_positions(cv_delta, steps)
        memory = self.encode(src)

        # Ground-truth offsets w.r.t. the anchor, shifted right by one step so
        # that position t is predicted from offsets strictly before it. This is
        # exactly the token stream `rollout` builds at inference.
        tokens = self._shift_right(tgt_pos - prior)
        return prior + self._decode(memory, tokens)

    # -- inference path (autoregressive) --------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        src: torch.Tensor,
        steps: int,
        cv_delta: torch.Tensor,
        step_hook: StepHook | None = None,
    ) -> dict[str, torch.Tensor]:
        """Autoregressively generate `steps` future positions.

        Args:
            step_hook: optional per-step correction applied to the agent-frame
                position before it is fed back (the Phase 2 hook). Must return a
                tensor of the same shape it receives.

        Returns:
            dict with 'pos' (B, steps, 2) and 'delta' (B, steps, 2), agent frame.

        Note: this re-runs the decoder over the full prefix at every step
        (O(T^2) decoder work). At T_pred = 60 that is not a bottleneck; if the
        horizon grows, add incremental KV caching here.
        """
        b = src.size(0)
        memory = self.encode(src)
        prior = self.prior_positions(cv_delta, steps)

        # The decoder input starts with a single zero token ("no previous step").
        tokens = torch.zeros(b, 1, 2, device=src.device, dtype=src.dtype)

        positions = []
        for t in range(steps):
            offset = self._decode(memory, tokens)[:, -1, :]      # (B, 2)
            pos = prior[:, t, :] + offset

            if step_hook is not None:
                pos = step_hook(pos, t)
                # Re-derive the offset so the token fed back is consistent with
                # the corrected position.
                offset = pos - prior[:, t, :]

            positions.append(pos)
            tokens = torch.cat([tokens, offset.unsqueeze(1)], dim=1)

        pos_seq = torch.stack(positions, dim=1)
        delta = torch.diff(
            pos_seq, dim=1, prepend=torch.zeros_like(pos_seq[:, :1, :])
        )
        return {"pos": pos_seq, "delta": delta}


def build_model(input_dim: int, cfg: ModelConfig) -> TrajectoryTransformer:
    return TrajectoryTransformer(input_dim, cfg)
