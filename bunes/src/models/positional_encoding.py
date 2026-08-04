"""Sinusoidal positional encoding.

Fixed (non-learned) encodings are used deliberately: the decoder is rolled out
for `pred_len` steps at inference and we want the option, later, to extrapolate
to longer horizons than were seen in training without retraining an embedding
table.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        # Registered as a buffer so it moves with .to(device) and is saved in
        # the checkpoint, but is never optimised.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, L, D)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """x: (B, T, D). `offset` lets a single-step decode use the right index."""
        t = x.size(1)
        return self.dropout(x + self.pe[:, offset : offset + t, :])
