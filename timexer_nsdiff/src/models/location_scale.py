"""Per-step conditional mean f_phi and conditional scale g_psi (sections 12-13).

Both heads consume the same per-hour, per-target input

    u_{h,k} = [ g_k' ; C_{h,k} ]  in R^{2d}

and share parameters across the 24 horizons and the 4 targets: target identity
is already carried by g_k' / C_target,k and time identity by E_h / R_h.  The two
heads never share weights with each other (section 13).

g_psi returns a positive standard deviation via Softplus + 1e-4 (24.8).  NsDiff
consumes a *variance*, so ``NsDiffConditioned`` squares it before entering the
diffusion process.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _mlp(in_dim, hidden, out_dim, n_layers, dropout, activation="gelu"):
    act = nn.GELU if activation == "gelu" else nn.ReLU
    layers = []
    dim = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(dim, hidden), act(), nn.Dropout(dropout)]
        dim = hidden
    layers += [nn.Linear(dim, out_dim)]
    return nn.Sequential(*layers)


class TemporalSmoother(nn.Module):
    """Optional light self-attention over the 24 horizons (section 12, "optional")."""

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u):
        # u: [B, H, K, D] -> attend along H independently per target
        B, H, K, D = u.shape
        x = u.permute(0, 2, 1, 3).reshape(B * K, H, D)
        out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + out)
        return x.reshape(B, K, H, D).permute(0, 2, 1, 3)


class PerStepLocationScaleHeads(nn.Module):
    def __init__(self, d_model: int = 128, hidden: int = 256, n_layers: int = 2,
                 dropout: float = 0.1, sigma_floor: float = 1e-4,
                 temporal_smoothing: bool = False):
        super().__init__()
        in_dim = 2 * d_model
        self.sigma_floor = sigma_floor
        self.smoother_mu = TemporalSmoother(in_dim, dropout=dropout) if temporal_smoothing else None
        self.smoother_sigma = TemporalSmoother(in_dim, dropout=dropout) if temporal_smoothing else None
        self.mean_head = _mlp(in_dim, hidden, 1, n_layers, dropout)
        self.scale_head = _mlp(in_dim, hidden, 1, n_layers, dropout)

    @staticmethod
    def build_input(target_tokens: torch.Tensor, cond_horizon: torch.Tensor) -> torch.Tensor:
        """[B, K, d] + [B, H, K, d] -> u [B, H, K, 2d]."""
        H = cond_horizon.shape[1]
        g = target_tokens.unsqueeze(1).expand(-1, H, -1, -1)
        return torch.cat([g, cond_horizon], dim=-1)

    def forward(self, target_tokens, cond_horizon, target_tokens_scale=None,
                cond_horizon_scale=None):
        """Returns mean [B, H, K] and sigma (std) [B, H, K].

        ``*_scale`` lets ablation A5 feed a *different* (exogenous-free) condition
        to g_psi than to f_phi.
        """
        u_mu = self.build_input(target_tokens, cond_horizon)
        u_sg = self.build_input(
            target_tokens if target_tokens_scale is None else target_tokens_scale,
            cond_horizon if cond_horizon_scale is None else cond_horizon_scale)
        if self.smoother_mu is not None:
            u_mu = self.smoother_mu(u_mu)
            u_sg = self.smoother_sigma(u_sg)
        mean = self.mean_head(u_mu).squeeze(-1)
        sigma = nn.functional.softplus(self.scale_head(u_sg)).squeeze(-1) + self.sigma_floor
        return mean, sigma


def gaussian_location_scale_loss(target, mean, sigma, detach_mean: bool = True):
    """L_sigma = mean[ (y-mu)^2 / sigma^2 + 2 log sigma ] / 2   (section 15.2)."""
    mu = mean.detach() if detach_mean else mean
    residual = target - mu
    return 0.5 * torch.mean(residual.pow(2) / sigma.pow(2) + 2.0 * torch.log(sigma))


def gaussian_nll_full(target, mean, sigma):
    """Full Gaussian NLL including the constant -- reported as a val metric."""
    residual = target - mean
    return 0.5 * torch.mean(math.log(2 * math.pi) + 2 * torch.log(sigma) + residual.pow(2) / sigma.pow(2))
