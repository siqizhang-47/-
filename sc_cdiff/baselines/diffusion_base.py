"""Generic conditional (non-autoregressive) diffusion baseline with a pluggable
denoiser backbone and mask inpainting. Used for the SSSD baseline. No
zero-inflation gate, no era-aware structural gating, no correlation regularizer
-- those are SC-CDiff's contributions, deliberately absent here."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.conditioning import ConditioningEncoder
from ..models.diffusion import Diffusion


class ConditionalDiffusionBaseline(nn.Module):
    def __init__(self, cfg, normalizer, backbone):
        super().__init__()
        m = cfg["model"]
        self.norm = normalizer
        self.enc = ConditioningEncoder(cfg["n_weather"], cfg["n_calendar"], cfg["n_channels"],
                                       m["d_model"], era_emb_dim=m["era_emb_dim"])
        self.backbone = backbone
        self.diff = Diffusion(cfg["diffusion"]["n_steps"])
        self.pv_idx, self.c_idx, self.h_idx = cfg["pv_idx"], cfg["c_idx"], cfg["h_idx"]
        self.e_idx, self.hw_idx = cfg["e_idx"], cfg["hw_idx"]
        self.pv_cap = cfg["pv_cap"]

    def to(self, *a, **k):
        super().to(*a, **k)
        if a:
            self.diff.to(a[0])
        return self

    def loss(self, batch):
        Y0 = batch["Y"]; W, CAL, Yhat, era = batch["W"], batch["CAL"], batch["Yhat"], batch["era"]
        M = batch["M"]; dev = Y0.device; B = Y0.shape[0]
        h_cond = self.enc(W, CAL, Yhat, era)
        t = torch.randint(0, self.diff.N, (B,), device=dev)
        eps = torch.randn_like(Y0)
        Yt = self.diff.q_sample(Y0, t, eps)
        cond_val = M * Y0 + (1 - M) * Yt
        eps_hat = self.backbone(cond_val, t, h_cond, era)
        w = (1 - M)
        L = (w * (eps - eps_hat) ** 2).sum() / w.sum().clamp_min(1.0)
        return L, {"L": float(L.item())}

    @torch.no_grad()
    def sample(self, batch, n):
        dev = batch["W"].device
        rep = lambda x: x.repeat_interleave(n, dim=0)
        W, CAL, Yhat, era = rep(batch["W"]), rep(batch["CAL"]), rep(batch["Yhat"]), rep(batch["era"])
        M, irr, Yobs = rep(batch["M"]), rep(batch["irr"]), rep(batch["Y"])
        Yrawr = rep(batch["Yraw"])
        B = batch["W"].shape[0]
        h_cond = self.enc(W, CAL, Yhat, era)
        Y = torch.randn(B * n, 5, 24, device=dev)
        for t in reversed(range(self.diff.N)):
            cv = M * Yobs + (1 - M) * Y
            tb = torch.full((B * n,), t, device=dev, dtype=torch.long)
            Y = self.diff.p_step(Y, self.backbone(cv, tb, h_cond, era), t)
            Y = M * Yobs + (1 - M) * Y
        Yr = self.norm.denormalize_torch(Y, era)
        out = F.relu(Yr)                                    # non-negativity only (no gate)
        out[:, self.pv_idx] = torch.clamp(out[:, self.pv_idx], 0, self.pv_cap)
        out = M * Yrawr + (1 - M) * out
        return out.reshape(B, n, 5, 24)


# ----------------------- SSSD backbone (state-space) -----------------------

class S4DLayer(nn.Module):
    """Lightweight diagonal state-space (S4D-style) temporal mixer along 24h."""

    def __init__(self, d_model, n_state=16):
        super().__init__()
        self.d, self.n = d_model, n_state
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, n_state)))
        self.A_imag = nn.Parameter(torch.pi * torch.arange(n_state).float().repeat(d_model, 1))
        self.B = nn.Parameter(torch.randn(d_model, n_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, n_state, 2) * 0.1)
        self.D = nn.Parameter(torch.randn(d_model))

    def forward(self, u):
        # u: [B, d, L]
        B, d, L = u.shape
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag      # [d,n]
        # build SSM kernel of length L
        t = torch.arange(L, device=u.device).float()
        # K[l] = sum_n C_n * B_n * exp(A_n * l)
        powers = torch.exp(A.unsqueeze(-1) * t)                  # [d,n,L]
        Cc = self.C[..., 0] + 1j * self.C[..., 1]               # [d,n]
        K = torch.einsum("dn,dn,dnl->dl", Cc, self.B.to(Cc.dtype), powers).real  # [d,L]
        # causal conv via FFT
        fft_len = 2 * L
        Kf = torch.fft.rfft(K, n=fft_len)
        Uf = torch.fft.rfft(u, n=fft_len)
        y = torch.fft.irfft(Uf * Kf.unsqueeze(0), n=fft_len)[..., :L]
        return y + u * self.D.view(1, d, 1)


class SSSDBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]; d = m["d_model"]
        self.in_proj = nn.Conv1d(cfg["n_channels"], d, 1)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.era_emb = nn.Embedding(3, d)
        self.blocks = nn.ModuleList()
        for _ in range(m["n_blocks"]):
            self.blocks.append(nn.ModuleDict({
                "s4": S4DLayer(d),
                "mix": nn.Conv1d(d, d, 1),
                "norm": nn.GroupNorm(8, d),
            }))
        self.out = nn.Conv1d(d, cfg["n_channels"], 1)
        self.d = d

    def forward(self, cond_val, t, h_cond, era):
        from ..models.denoiser import sinusoidal_embedding
        x = self.in_proj(cond_val) + h_cond                     # [B,d,24]
        te = self.t_mlp(sinusoidal_embedding(t, self.d)).unsqueeze(-1)
        ee = self.era_emb(era).unsqueeze(-1)
        for blk in self.blocks:
            h = blk["norm"](x + te + ee)
            h = torch.relu(blk["s4"](h))
            x = x + blk["mix"](h)
        return self.out(x)


def build_sssd(cfg, normalizer):
    return ConditionalDiffusionBaseline(cfg, normalizer, SSSDBackbone(cfg))
