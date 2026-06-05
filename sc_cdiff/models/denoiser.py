"""Step 8: denoising network eps_theta (plan §5.3).

CSDI-style two-axis attention: a temporal attention along the 24h axis (per
channel, learns intra-day shapes) and a channel attention along the 5-channel
axis (per hour, carries the cross-variable dependence), interleaved over L
residual blocks with a per-step / per-era FiLM modulation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(t, dim):
    # t: [B] (float or long). Returns [B, dim]
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1))
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FiLM(nn.Module):
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * d_model)

    def forward(self, x, cond):
        # x:[B,5,24,D]  cond:[B,cond_dim]
        ss = self.to_scale_shift(cond)                  # [B,2D]
        scale, shift = ss.chunk(2, dim=-1)
        scale = scale[:, None, None, :]
        shift = shift[:, None, None, :]
        return x * (1 + scale) + shift


class _Attn(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x):  # x:[N, L, D]
        h = self.norm(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        return x + out


class Block(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult, cond_dim):
        super().__init__()
        self.temporal = _Attn(d_model, n_heads)
        self.channel = _Attn(d_model, n_heads)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult), nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )
        self.film = FiLM(d_model, cond_dim)

    def forward(self, x, cond):
        B, C, T, D = x.shape
        # temporal: along T, per channel -> batch (B*C)
        xt = x.reshape(B * C, T, D)
        xt = self.temporal(xt)
        x = xt.reshape(B, C, T, D)
        # channel: along C, per hour -> batch (B*T)
        xc = x.permute(0, 2, 1, 3).reshape(B * T, C, D)
        xc = self.channel(xc)
        x = xc.reshape(B, T, C, D).permute(0, 2, 1, 3)
        # ffn + film
        x = x + self.ffn(self.ffn_norm(x))
        x = self.film(x, cond)
        return x


class Denoiser(nn.Module):
    def __init__(self, n_channels=5, d_model=128, n_blocks=4, n_heads=4,
                 ffn_mult=4, era_emb_dim=32, n_era=3):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(3, d_model)           # [cond_val, M, Yhat]
        self.chan_emb = nn.Embedding(n_channels, d_model)
        self.t_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                   nn.Linear(d_model, d_model))
        self.era_emb = nn.Embedding(n_era, era_emb_dim)
        cond_dim = d_model + era_emb_dim
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, ffn_mult, cond_dim) for _ in range(n_blocks)])
        self.out = nn.Linear(d_model, 1)
        self.register_buffer("chan_ids", torch.arange(n_channels), persistent=False)

    def forward(self, cond_val, t, M, Yhat, h_cond, era):
        # cond_val,M,Yhat:[B,5,24]  t:[B]  h_cond:[B,D,24]  era:[B]
        B, C, T = cond_val.shape
        cell = torch.stack([cond_val, M, Yhat], dim=-1)   # [B,5,24,3]
        x = self.input_proj(cell)                          # [B,5,24,D]
        x = x + self.chan_emb(self.chan_ids)[None, :, None, :]
        x = x + h_cond.permute(0, 2, 1)[:, None, :, :]     # broadcast over channels
        t_emb = self.t_mlp(sinusoidal_embedding(t, self.d_model))  # [B,D]
        cond = torch.cat([t_emb, self.era_emb(era)], dim=-1)        # [B, D+era]
        for blk in self.blocks:
            x = blk(x, cond)
        eps_hat = self.out(x).squeeze(-1)                  # [B,5,24]
        return eps_hat
