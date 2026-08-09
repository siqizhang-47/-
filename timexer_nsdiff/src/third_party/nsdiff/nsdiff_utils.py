"""NsDiff forward / reverse process, taken from NsDiff-main/src/layer/nsdiff_utils.py.

All the diffusion algebra (beta schedule, ``q_sample``, the Sigma_1 / Sigma_2 /
gamma coefficients, the y0 reparameterisation and the reverse loop) is
reproduced unchanged -- the design document, section 11, explicitly requires the
diffusion derivation to stay identical.

Two kinds of edits were needed, each marked with ``# [MOD]``:

1. the denoiser is now called as ``model(y_t, y_0_hat, gx, t, cond)`` instead of
   ``model(x, x_mark, y_t, y_0_hat, gx, t)``, because the historical encoder
   lives inside the TimeXer condition encoder and the external condition
   ``cond_denoiser`` replaces the ``(x, x_mark)`` embedding input;
2. two ``clamp(min=...)`` guards around square roots.  The original code can
   take the square root of a marginally negative quantity (float32 round-off in
   ``lambda_1**2 - 4*lambda_0*lambda_2`` and in ``noise``), which produces NaNs
   after a few hundred sampling calls.  The guards do not change the maths, they
   only keep it inside its own domain.

The PE ("perfect estimate") variants of the original file are not vendored.
"""
import math
import torch
import numpy as np

EPS = 10e-8


def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                    math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta) for i in
             range(num_timesteps)])
        if schedule == "cosine_reverse":
            betas = betas.flip(0)
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
             range(num_timesteps)])
    else:
        raise ValueError(f"unknown beta schedule: {schedule}")
    return betas


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


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def cal_sigma12(alphas, alphas_cumprod, alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev,
                betas_tiled_m_1, betas_bar_m_1, gx, y_sigma, t):
    at = extract(alphas, t, gx)
    at_bar = extract(alphas_cumprod, t, gx)
    at_tilde = extract(alphas_cumprod_sum, t, gx)
    b_tilde_m_1 = extract(betas_tiled_m_1, t, gx)
    b_bar_m_1 = extract(betas_bar_m_1, t, gx)

    Sigma_1 = (1 - at) ** 2 * gx + at * (1 - at) * y_sigma
    Sigma_2 = (b_bar_m_1 - b_tilde_m_1) * gx + b_tilde_m_1 * y_sigma
    return at, at_bar, at_tilde, Sigma_1, Sigma_2


def cal_forward_noise(betas_tiled, betas_bar, gx, y_sigma, t):
    b_bar_t = extract(betas_bar, t, gx)
    b_tilded_t = extract(betas_tiled, t, gx)

    noise = (b_bar_t - b_tilded_t) * gx + b_tilded_t * y_sigma
    noise = noise.clamp(min=EPS)  # [MOD] numerical guard
    return noise


def cal_sigma_tilde(alphas, alphas_cumprod, alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev,
                    betas_tiled_m_1, betas_bar_m_1, gx, y_sigma, t):
    at, at_bar, at_tilde, Sigma_1, Sigma_2 = cal_sigma12(
        alphas, alphas_cumprod, alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev,
        betas_tiled_m_1, betas_bar_m_1, gx, y_sigma, t)
    sigma_tilde = (Sigma_1 * Sigma_2) / (at * Sigma_2 + Sigma_1)
    return sigma_tilde


def calc_gammas(alphas, alphas_cumprod, alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev,
                betas_tiled_m_1, betas_bar_m_1, gx, y_sigma, t):
    at, at_bar, at_tilde, Sigma_1, Sigma_2 = cal_sigma12(
        alphas, alphas_cumprod, alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev,
        betas_tiled_m_1, betas_bar_m_1, gx, y_sigma, t)

    alpha_bar_t_m_1 = extract(alpha_bar_prev, t, gx)
    sqrt_alpha_t = at.sqrt()
    sqrt_alpha_bar_t_m_1 = alpha_bar_t_m_1.sqrt()

    at_s1_s2 = at * Sigma_2 + Sigma_1

    gamma_0 = sqrt_alpha_bar_t_m_1 * Sigma_1 / at_s1_s2
    gamma_1 = sqrt_alpha_t * Sigma_2 / at_s1_s2
    gamma_2 = ((sqrt_alpha_t * (at - 1)) * Sigma_2 + (1 - sqrt_alpha_bar_t_m_1) * Sigma_1) / at_s1_s2
    return gamma_0, gamma_1, gamma_2


# ---------------------------------------------------------------- forward
def q_sample(y, y_0_hat, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=None):
    """
    y_0_hat: prediction of pre-trained guidance model; can be extended to represent
        any prior mean setting at timestep T.
    """
    if noise is None:
        noise = torch.randn_like(y).to(y.device)
    sqrt_alpha_bar_t = extract(alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    # q(y_t | y_0, x)
    y_t = sqrt_alpha_bar_t * y + (1 - sqrt_alpha_bar_t) * y_0_hat + noise
    return y_t


# ---------------------------------------------------------------- reverse
def _estimate_sigma_y0(alpha_t, betas_tiled_m_1, betas_bar_m_1, gx, sigma_theta):
    lambda_0 = alpha_t * (1 - alpha_t) * betas_tiled_m_1
    lambda_1 = (((1 - alpha_t) ** 2 * betas_tiled_m_1
                 + alpha_t * (1 - alpha_t) * (betas_bar_m_1 - betas_tiled_m_1)) * gx
                - sigma_theta * (alpha_t * betas_tiled_m_1 + alpha_t * (1 - alpha_t)))
    lambda_2 = (gx ** 2 * (1 - alpha_t) ** 2 * (betas_bar_m_1 - betas_tiled_m_1)
                - sigma_theta * gx * (alpha_t * betas_bar_m_1 - alpha_t * betas_tiled_m_1 + (1 - alpha_t) ** 2))
    disc = (lambda_1 ** 2 - 4 * lambda_0 * lambda_2).clamp(min=0.0)  # [MOD] numerical guard
    sigma_y0_hat = (-lambda_1 + disc.sqrt()) / (2 * lambda_0 + EPS)
    return sigma_y0_hat.clamp(min=EPS)  # [MOD]


def p_sample(model, cond, y, y_0_hat, gx, y_T_mean, t, alphas, one_minus_alphas_bar_sqrt, alphas_cumprod,
             alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_all, betas_bar_all,
             betas_tiled_m_1_all, betas_bar_m_1_all):
    """Reverse diffusion process sampling -- one time step."""
    device = y.device
    t = torch.tensor([t]).to(device)
    eps_theta, sigma_theta = model(y, y_0_hat, gx, t, cond)  # [MOD] condition-aware denoiser call

    eps_theta = eps_theta.to(device).detach()
    sigma_theta = sigma_theta.to(device).detach() + EPS

    z = torch.randn_like(y)
    alpha_t = extract(alphas, t, y)

    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).clamp(min=EPS).sqrt()

    betas_tiled_m_1 = extract(betas_tiled_m_1_all, t, y)
    betas_bar_m_1 = extract(betas_bar_m_1_all, t, y)
    betas_tiled = extract(betas_tiled_all, t, y)
    betas_bar = extract(betas_bar_all, t, y)

    sigma_y0_hat = _estimate_sigma_y0(alpha_t, betas_tiled_m_1, betas_bar_m_1, gx, sigma_theta)
    noise = ((betas_bar - betas_tiled) * gx + betas_tiled * sigma_y0_hat).clamp(min=EPS)  # [MOD]

    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (
            y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * torch.sqrt(noise))
    # posterior mean
    gamma_0, gamma_1, gamma_2 = calc_gammas(
        alphas, alphas_cumprod, alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev,
        betas_tiled_m_1_all, betas_bar_m_1_all, gx, sigma_y0_hat, t)
    y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y + gamma_2 * y_T_mean
    # posterior variance
    y_t_m_1 = y_t_m_1_hat + torch.sqrt(sigma_theta) * z
    return y_t_m_1


def p_sample_t_1to0(model, cond, y, y_0_hat, gx, y_T_mean, one_minus_alphas_bar_sqrt, alphas, alphas_cumprod,
                    alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled_all, betas_bar_all,
                    betas_tiled_m_1_all, betas_bar_m_1_all):
    device = y.device
    t = torch.tensor([0]).to(device)  # corresponding to timestep 1
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).clamp(min=EPS).sqrt()
    eps_theta, sigma_theta = model(y, y_0_hat, gx, t, cond)  # [MOD]

    eps_theta = eps_theta.to(device).detach()
    sigma_theta = sigma_theta.to(device).detach() + EPS
    alpha_t = extract(alphas, t, y)

    betas_tiled_m_1 = extract(betas_tiled_m_1_all, t, y)
    betas_bar_m_1 = extract(betas_bar_m_1_all, t, y)
    betas_tiled = extract(betas_tiled_all, t, y)
    betas_bar = extract(betas_bar_all, t, y)

    sigma_y0_hat = _estimate_sigma_y0(alpha_t, betas_tiled_m_1, betas_bar_m_1, gx, sigma_theta)
    noise = ((betas_bar - betas_tiled) * gx + betas_tiled * sigma_y0_hat).clamp(min=EPS)  # [MOD]

    y_0_reparam = 1 / sqrt_alpha_bar_t * (
            y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * torch.sqrt(noise))
    return y_0_reparam


def p_sample_loop(model, cond, y_0_hat, gx, y_T_mean, n_steps, alphas, one_minus_alphas_bar_sqrt, alphas_cumprod,
                  alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev, betas_tiled, betas_bar,
                  betas_tiled_m_1, betas_bar_m_1, return_all=False):
    z = torch.randn_like(y_T_mean)
    cur_y = torch.sqrt(gx) * z + y_T_mean  # sample y_T

    y_p_seq = [cur_y]
    for t in reversed(range(1, n_steps)):  # t from T to 2
        y_t = cur_y
        cur_y = p_sample(model, cond, y_t, y_0_hat, gx, y_T_mean, t, alphas, one_minus_alphas_bar_sqrt,
                         alphas_cumprod, alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev,
                         betas_tiled, betas_bar, betas_tiled_m_1, betas_bar_m_1)
        y_p_seq.append(cur_y)
    assert len(y_p_seq) == n_steps
    y_0 = p_sample_t_1to0(model, cond, y_p_seq[-1], y_0_hat, gx, y_T_mean, one_minus_alphas_bar_sqrt, alphas,
                          alphas_cumprod, alphas_cumprod_sum, alpha_bar_prev, alphas_cumprod_sum_prev,
                          betas_tiled, betas_bar, betas_tiled_m_1, betas_bar_m_1)
    y_p_seq.append(y_0)
    if return_all:
        return y_p_seq
    return y_0
