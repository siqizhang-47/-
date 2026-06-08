"""TimeGrad baseline (Rasul et al. 2021): autoregressive denoising diffusion.

Compact re-implementation: a GRU rolls over the 24 hours encoding past values +
per-hour conditions; at each hour a small conditional diffusion over the 5-channel
vector generates that hour conditioned on the GRU hidden state. Observed-prefix
hours are teacher-forced at sampling (gives the k-horizon conditioning)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.denoiser import sinusoidal_embedding
from ..models.diffusion import Diffusion


class TimeGrad(nn.Module):
    def __init__(self, cfg, normalizer):
        super().__init__()
        d = cfg["model"]["d_model"]
        self.d = d
        self.norm = normalizer
        self.cond_dim = cfg["n_weather"] + cfg["n_calendar"] + cfg["n_channels"]  # 19
        self.gru = nn.GRUCell(cfg["n_channels"] + self.cond_dim, d)
        self.era_emb = nn.Embedding(3, d)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.den = nn.Sequential(
            nn.Linear(cfg["n_channels"] + d + d, d), nn.GELU(),
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg["n_channels"]))
        self.diff = Diffusion(cfg["diffusion"]["n_steps"])
        self.pv_idx, self.pv_cap = cfg["pv_idx"], cfg["pv_cap"]
        self.C = cfg["n_channels"]
        self.x0_clip = cfg["sample"].get("x0_clip", 8.0)

    def to(self, *a, **k):
        super().to(*a, **k)
        if a:
            self.diff.to(a[0])
        return self

    def _cond_seq(self, batch):
        # [B, 24, 19]
        return torch.cat([batch["W"], batch["CAL"], batch["Yhat"]], dim=1).permute(0, 2, 1)

    def _eps(self, y_noised, h, t):
        te = self.t_mlp(sinusoidal_embedding(t, self.d))
        return self.den(torch.cat([y_noised, h, te], dim=-1))

    def loss(self, batch):
        Y0 = batch["Y"]; dev = Y0.device; B = Y0.shape[0]
        cond = self._cond_seq(batch)                         # [B,24,19]
        h = self.era_emb(batch["era"])                       # [B,d] init hidden
        y_prev = torch.zeros(B, self.C, device=dev)
        total, cnt = 0.0, 0
        for tau in range(24):
            h = self.gru(torch.cat([y_prev, cond[:, tau]], dim=-1), h)
            y_t = Y0[:, :, tau]                              # [B,5] target this hour
            t = torch.randint(0, self.diff.N, (B,), device=dev)
            eps = torch.randn_like(y_t)
            ab = self.diff.abar[t].unsqueeze(-1)
            y_noised = torch.sqrt(ab) * y_t + torch.sqrt(1 - ab) * eps
            eps_hat = self._eps(y_noised, h, t)
            total = total + F.mse_loss(eps_hat, eps)
            cnt += 1
            y_prev = y_t                                     # teacher forcing
        L = total / cnt
        return L, {"L": float(L.item())}

    @torch.no_grad()
    def sample(self, batch, n):
        dev = batch["W"].device
        rep = lambda x: x.repeat_interleave(n, dim=0)
        cond = self._cond_seq({k: rep(batch[k]) for k in ("W", "CAL", "Yhat")})
        era = rep(batch["era"]); M = rep(batch["M"]); Yobs = rep(batch["Y"])
        Yrawr = rep(batch["Yraw"]); irr = rep(batch["irr"])
        Bn = cond.shape[0]
        h = self.era_emb(era)
        y_prev = torch.zeros(Bn, self.C, device=dev)
        ys = []
        for tau in range(24):
            h = self.gru(torch.cat([y_prev, cond[:, tau]], dim=-1), h)
            observed = M[:, 0, tau] > 0.5                    # same prefix mask all channels
            # reverse diffusion for this hour (with x0 thresholding for stability)
            clip = self.x0_clip
            y = torch.randn(Bn, self.C, device=dev)
            for t in reversed(range(self.diff.N)):
                tb = torch.full((Bn,), t, device=dev, dtype=torch.long)
                eps_hat = self._eps(y, h, tb)
                ab = self.diff.abar[t]
                ab_prev = self.diff.abar[t - 1] if t > 0 else torch.ones_like(ab)
                a = self.diff.alphas[t]; beta = self.diff.betas[t]
                x0 = torch.clamp((y - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab), -clip, clip)
                mean = (torch.sqrt(ab_prev) * beta / (1 - ab)) * x0 \
                    + (torch.sqrt(a) * (1 - ab_prev) / (1 - ab)) * y
                if t > 0:
                    var = beta * (1 - ab_prev) / (1 - ab)
                    y = mean + torch.sqrt(var) * torch.randn_like(y)
                else:
                    y = mean
            # teacher-force observed prefix with the standardized truth
            y = torch.where(observed.unsqueeze(-1), Yobs[:, :, tau], y)
            ys.append(y)
            y_prev = y
        Y = torch.stack(ys, dim=-1)                          # [Bn,5,24]
        Yr = self.norm.denormalize_torch(Y, era)
        out = F.relu(Yr)
        out[:, self.pv_idx] = torch.clamp(out[:, self.pv_idx], 0, self.pv_cap)
        out = M * Yrawr + (1 - M) * out
        B = batch["W"].shape[0]
        return out.reshape(B, n, 5, 24)


def build_timegrad(cfg, normalizer):
    return TimeGrad(cfg, normalizer)
