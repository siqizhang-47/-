"""Step 7: shared conditioning encoder (plan §5.2).

Concatenate predicted weather, calendar and point-forecast along the feature
axis, run a 1d conv over the 24h axis, then add a broadcast device-era embedding.
Output h_cond [B, d_model, 24] is shared by the gate branch and the denoiser.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConditioningEncoder(nn.Module):
    def __init__(self, n_weather=6, n_calendar=8, n_channels=5, d_model=128,
                 n_era=3, era_emb_dim=32):
        super().__init__()
        in_ch = n_weather + n_calendar + n_channels  # 6+8+5 = 19
        self.conv = nn.Conv1d(in_ch, d_model, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.era_emb = nn.Embedding(n_era, d_model)  # added (broadcast over time)
        self.d_model = d_model

    def forward(self, W, CAL, Yhat, era, use_era=True):
        # W:[B,6,24] CAL:[B,8,24] Yhat:[B,5,24] era:[B]
        side = torch.cat([W, CAL, Yhat], dim=1)        # [B,19,24]
        h = self.act(self.conv(side))                  # [B,D,24]
        if use_era:
            h = h + self.era_emb(era).unsqueeze(-1)    # broadcast era over hours
        return h
