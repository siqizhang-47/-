"""Step 10: assemble SC-CDiff (encoder + denoiser + gate + diffusion) with the
training loss and the gated/inpainting sampler (plan §5.6-5.8)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import ConditioningEncoder
from .denoiser import Denoiser
from .diffusion import Diffusion
from .gate import GateBranch


def _corr_matrix(Fm, eps=1e-6):
    # Fm: [B,P] -> Pearson correlation [P,P]
    Fc = Fm - Fm.mean(dim=0, keepdim=True)
    sd = Fc.std(dim=0, keepdim=True) + eps
    Fn = Fc / sd
    return (Fn.t() @ Fn) / (Fm.shape[0] - 1 + eps)


def _daily_features(Y):
    # Y:[B,5,24] -> [B, 15]  (per-channel mean, max, ramp)
    mean = Y.mean(dim=2)
    mx = Y.amax(dim=2)
    ramp = Y[:, :, 1:].sub(Y[:, :, :-1]).abs().mean(dim=2)
    return torch.cat([mean, mx, ramp], dim=1)


def _season_from_cal(CAL):
    # CAL[:,2]=month_sin, CAL[:,3]=month_cos at hour 0 -> month in 1..12 -> season 0..3
    ms = CAL[:, 2, 0]
    mc = CAL[:, 3, 0]
    ang = torch.atan2(ms, mc)                     # month-1 angle
    month = torch.round(ang / (2 * torch.pi) * 12) % 12 + 1
    season = ((month % 12) // 3).long()           # 12,1,2->0 ; 3,4,5->1 ; ...
    return season


class SCCDiff(nn.Module):
    def __init__(self, cfg, normalizer):
        super().__init__()
        m = cfg["model"]
        self.cfg = cfg
        self.norm = normalizer
        self.enc = ConditioningEncoder(
            n_weather=cfg["n_weather"], n_calendar=cfg["n_calendar"],
            n_channels=cfg["n_channels"], d_model=m["d_model"], era_emb_dim=m["era_emb_dim"])
        self.denoiser = Denoiser(
            n_channels=cfg["n_channels"], d_model=m["d_model"], n_blocks=m["n_blocks"],
            n_heads=m["n_heads"], ffn_mult=m["ffn_mult"], era_emb_dim=m["era_emb_dim"])
        self.gate = GateBranch(d_model=m["d_model"])
        self.diff = Diffusion(cfg["diffusion"]["n_steps"])
        self.pv_idx, self.c_idx, self.h_idx = cfg["pv_idx"], cfg["c_idx"], cfg["h_idx"]
        self.e_idx, self.hw_idx = cfg["e_idx"], cfg["hw_idx"]
        self.pv_cap = cfg["pv_cap"]
        self.lw = cfg["train"]

    def to(self, *a, **k):
        super().to(*a, **k)
        if a:
            self.diff.to(a[0])
        return self

    # -------------------- training --------------------
    def loss(self, batch):
        Y0 = batch["Y"]                       # [B,5,24] standardized
        W, CAL, Yhat, era = batch["W"], batch["CAL"], batch["Yhat"], batch["era"]
        M, m = batch["M"], batch["m"]
        Gc, Gh = batch["Gc"], batch["Gh"]
        B = Y0.shape[0]
        dev = Y0.device

        h_cond = self.enc(W, CAL, Yhat, era)
        t = torch.randint(0, self.diff.N, (B,), device=dev)
        eps = torch.randn_like(Y0)
        Yt = self.diff.q_sample(Y0, t, eps)
        cond_val = M * Y0 + (1 - M) * Yt      # CSDI-style: clean truth at observed cells
        eps_hat = self.denoiser(cond_val, t, M, Yhat, h_cond, era)

        # diffusion loss only on "to-generate AND valid" cells
        w = (1 - M) * m
        denom = w.sum().clamp_min(1.0)
        L_diff = (w * (eps - eps_hat) ** 2).sum() / denom

        # gate loss (cool/heat) on to-generate hours only
        gate_logits = self.gate(h_cond)       # [B,2,24]
        gate_w = (1 - M[:, self.c_idx])       # [B,24] (same time-mask for both)
        gl = F.binary_cross_entropy_with_logits(
            gate_logits, torch.stack([Gc, Gh], dim=1), reduction="none")
        L_gate = (gl * gate_w[:, None, :]).sum() / gate_w.sum().clamp_min(1.0)

        # seasonal correlation regularizer (Tweedie one-step x0 estimate)
        x0_hat = self.diff.x0_from_eps(Yt, t, eps_hat)
        L_corr = self._seasonal_corr_loss(x0_hat, Y0, CAL)

        # physical penalty on denormalized x0
        L_phy = self._phys_loss(x0_hat, era)

        L = (L_diff + self.lw["lambda_gate"] * L_gate
             + self.lw["lambda_corr"] * L_corr + self.lw["lambda_phy"] * L_phy)
        logs = {"L_diff": L_diff.item(), "L_gate": L_gate.item(),
                "L_corr": float(L_corr.item()), "L_phy": float(L_phy.item()),
                "L": L.item()}
        return L, logs

    def _seasonal_corr_loss(self, gen, real, CAL):
        season = _season_from_cal(CAL)
        Fg, Fr = _daily_features(gen), _daily_features(real)
        total, nseas = gen.new_zeros(()), 0
        for s in range(4):
            idx = (season == s).nonzero(as_tuple=True)[0]
            if idx.numel() < 8:
                continue
            Rg = _corr_matrix(Fg[idx]); Rr = _corr_matrix(Fr[idx])
            total = total + (Rg - Rr).pow(2).sum()
            nseas += 1
        return total / max(nseas, 1)

    def _phys_loss(self, x0_hat, era):
        Yr = self.norm.denormalize_torch(x0_hat, era)   # [B,5,24] raw scale
        neg = F.relu(-Yr).mean()
        over = F.relu(Yr[:, self.pv_idx] - self.pv_cap).mean()
        return neg + over

    # -------------------- sampling --------------------
    @torch.no_grad()
    def sample(self, batch, n_scenarios):
        """Generate n scenarios per item. Returns raw-scale [B, n, 5, 24]."""
        dev = batch["W"].device
        W, CAL, Yhat, era = batch["W"], batch["CAL"], batch["Yhat"], batch["era"]
        M, irr = batch["M"], batch["irr"]
        Yobs = batch["Y"]                      # standardized observed truth (used where M=1)
        B = W.shape[0]
        n = n_scenarios

        # expand conditioning to [B*n, ...]
        def rep(x):
            return x.repeat_interleave(n, dim=0)
        Wr, CALr, Yhatr, erar = rep(W), rep(CAL), rep(Yhat), rep(era)
        Mr, irrr, Yobsr = rep(M), rep(irr), rep(Yobs)
        Yrawr = rep(batch["Yraw"])             # raw observed truth for the prefix
        h_cond = self.enc(Wr, CALr, Yhatr, erar)

        # gate -> sample Bernoulli activations once
        pi = torch.sigmoid(self.gate(h_cond))            # [B*n,2,24]
        Bc = torch.bernoulli(pi[:, 0])
        Bh = torch.bernoulli(pi[:, 1])

        Y = torch.randn(B * n, 5, 24, device=dev)
        for t in reversed(range(self.diff.N)):
            cond_val = Mr * Yobsr + (1 - Mr) * Y
            t_b = torch.full((B * n,), t, device=dev, dtype=torch.long)
            eps_hat = self.denoiser(cond_val, t_b, Mr, Yhatr, h_cond, erar)
            Y = self.diff.p_step(Y, eps_hat, t)
            Y = Mr * Yobsr + (1 - Mr) * Y                # inpaint observed each step

        Yr = self.norm.denormalize_torch(Y, erar)        # raw scale
        # structural gating + non-negativity (plan §5.6)
        irr_pos = (irrr > 0).float()
        out = torch.empty_like(Yr)
        out[:, self.pv_idx] = irr_pos * torch.clamp(Yr[:, self.pv_idx], 0, self.pv_cap)
        out[:, self.c_idx] = Bc * F.relu(Yr[:, self.c_idx])
        out[:, self.h_idx] = Bh * F.relu(Yr[:, self.h_idx])
        out[:, self.e_idx] = F.relu(Yr[:, self.e_idx])
        out[:, self.hw_idx] = F.relu(Yr[:, self.hw_idx])
        # keep raw observed truth on the observed prefix (M=1)
        out = Mr * Yrawr + (1 - Mr) * out
        return out.reshape(B, n, 5, 24)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
