"""Step 9: zero-inflation gate branch (plan §5.5).

A light classifier, parallel to diffusion and independent of the diffusion step
t, that predicts per-hour cooling/heating activation logits from h_cond.
"""
from __future__ import annotations

import torch.nn as nn


class GateBranch(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, 2, kernel_size=3, padding=1),  # [cool, heat]
        )

    def forward(self, h_cond):
        return self.net(h_cond)   # [B,2,24] logits
