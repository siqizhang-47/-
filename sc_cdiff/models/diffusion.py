"""Step 6: diffusion noise schedule (cosine, N=200) and q_sample / p_step."""
from __future__ import annotations

import math

import torch


def cosine_beta_schedule(n_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = n_steps + 1
    x = torch.linspace(0, n_steps, steps)
    ac = torch.cos(((x / n_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clip(betas, 1e-4, 0.999)


class Diffusion:
    """Holds schedule tensors. Index convention: t in [0, N-1]."""

    def __init__(self, n_steps: int = 200, device="cpu"):
        self.N = n_steps
        betas = cosine_beta_schedule(n_steps)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.abar = abar.to(device)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.abar = self.abar.to(device)
        return self

    def q_sample(self, Y0, t, eps):
        """Forward noising. t: [B] long. Y0/eps: [B,5,24]."""
        ab = self.abar[t].view(-1, 1, 1)
        return torch.sqrt(ab) * Y0 + torch.sqrt(1 - ab) * eps

    def x0_from_eps(self, Yt, t, eps_hat):
        ab = self.abar[t].view(-1, 1, 1)
        return (Yt - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)

    def p_step(self, Y, eps_hat, t_scalar: int):
        """One reverse DDPM step at integer step t_scalar (same t for the batch)."""
        a = self.alphas[t_scalar]
        ab = self.abar[t_scalar]
        beta = self.betas[t_scalar]
        mean = (Y - (1 - a) / torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(a)
        if t_scalar > 0:
            noise = torch.randn_like(Y)
            return mean + torch.sqrt(beta) * noise
        return mean
