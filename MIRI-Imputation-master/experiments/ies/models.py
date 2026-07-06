"""MIRI velocity-field network specialised for 8-dimensional IES tabular data."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPIES(nn.Module):
    """MIRI velocity field for d-dimensional tabular IES data.

    Input  x shape: [batch, 3*d]   (Xti1 | Xti2 | M, as assembled by rectified_impute)
    Input  t shape: [batch, 1]
    Output   shape: [batch, d]

    The last ``d`` columns of ``x`` hold the mask M; the velocity is zeroed on
    observed entries (``(1 - m) * out``) so observed coordinates are never moved.
    """

    def __init__(self, d: int, hidden_dims=(256, 256, 256)):
        super().__init__()
        self.d = d
        input_dim = 3 * d + 1

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        layers.append(nn.Linear(prev, d))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        m = x[:, -self.d:]
        inp = torch.cat([x, t], dim=1)
        out = self.net(inp)
        return (1.0 - m) * out
