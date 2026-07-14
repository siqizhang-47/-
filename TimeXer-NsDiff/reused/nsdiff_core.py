"""
NsDiff diffusion schedule + denoiser wrapper, extracted verbatim from the
official NsDiff code (src/models/NsDiff.py) so the non-stationary location-scale
diffusion math is UNCHANGED. Only the torch_timeseries-specific history embedding
(whose output the denoiser ignores anyway) is dropped, making this file depend on
torch only.

The reverse/forward samplers live in reused/nsdiff_utils.py (also verbatim).
"""
import torch
import torch.nn as nn

from .nsdiff_utils import make_beta_schedule
from .denoise import ConditionalGuidedModel


# ---- alpha bookkeeping (verbatim from NsDiff.py) ---------------------------
def compute_gx_term(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.float()
    n = alpha.shape[0]
    gx_term = torch.zeros_like(alpha)
    for t in range(n):
        slice_t = alpha[:t + 1].flip(dims=[0])
        cprod = torch.cat([torch.tensor([1]).to(slice_t.device), torch.cumprod(slice_t, dim=0)])
        cprod = cprod[:-1] * ((1 - slice_t) ** 2)
        gx_term[t] = cprod.sum()
    return gx_term


def compute_tilde_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.float()
    n = alpha.shape[0]
    tilde_alpha = torch.zeros_like(alpha)
    for t in range(n):
        slice_t = alpha[:t + 1].flip(dims=[0])
        cprod = torch.cumprod(slice_t, dim=0)
        tilde_alpha[t] = cprod.sum()
    return tilde_alpha


def compute_hat_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.float()
    n = alpha.shape[0]
    hat_alpha = torch.zeros_like(alpha)
    for t in range(n):
        slice_t = alpha[:t + 1].flip(dims=[0])
        cprod = torch.cumprod(slice_t, dim=0)
        cprod = cprod * slice_t
        hat_alpha[t] = cprod.sum()
    return hat_alpha


class NsDiffCore(nn.Module):
    """Holds the NsDiff diffusion schedule and the conditional denoiser.

    forward(y_t, y_0_hat, gx, t) -> (eps_theta, sigma_theta), matching the
    original `ConditionalGuidedModel` (which conditions only on y_t, mu, sigma).
    """

    def __init__(self, num_timesteps, enc_in, device,
                 beta_schedule="linear", beta_start=1e-4, beta_end=1e-2):
        super().__init__()
        self.device = device
        self.num_timesteps = num_timesteps

        betas = make_beta_schedule(schedule=beta_schedule, num_timesteps=num_timesteps,
                                   start=beta_start, end=beta_end)
        betas = self.betas = betas.float().to(device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.to("cpu").cumprod(dim=0).to(device)
        self.alphas_cumprod = alphas_cumprod
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.betas_bar = 1 - self.alphas_cumprod
        self.alphas_cumprod_sum = compute_tilde_alpha(alphas).to(device)
        self.alphas_tilde = self.alphas_cumprod_sum
        self.alphas_hat = compute_hat_alpha(alphas).to(device)
        self.betas_tilde = self.alphas_tilde - self.alphas_hat
        self.gx_term = compute_gx_term(alphas).to(device)
        assert (self.betas_tilde >= 0).all()
        assert ((self.betas_bar - self.betas_tilde) >= 0).all()

        self.betas_tilde_m_1 = torch.cat([torch.ones(1, device=device), self.betas_tilde[:-1]], dim=0)
        self.betas_bar_m_1 = torch.cat([torch.ones(1, device=device), self.betas_bar[:-1]], dim=0)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_sum_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod_sum[:-1]], dim=0)
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.diffussion_model = ConditionalGuidedModel(num_timesteps, enc_in)

    def forward(self, x, x_mark, y_t, y_0_hat, gx, t):
        # x / x_mark are accepted for signature compatibility with the original
        # p_sample* helpers; ConditionalGuidedModel ignores the history encoding.
        return self.diffussion_model(None, y_t, y_0_hat, gx, t)
