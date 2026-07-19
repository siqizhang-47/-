"""NsDiff diffusion mathematics, kept theoretically identical to the original
repository (make_beta_schedule / cal_forward_noise / cal_sigma_tilde /
calc_gammas / q_sample / p_sample_loop) with:

* explicit interfaces (the denoiser is passed as a closure);
* numerical stability clamps (spec 2.7), enabled for ALL models;
* torch.Generator support for reproducible sampling.
"""
import math

import torch

EPS = 1e-8
SIGMA_MAX = 1e6


def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule in ("cosine", "cosine_reverse"):
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [
                min(
                    1
                    - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2)
                    / (math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2),
                    max_beta,
                )
                for i in range(num_timesteps)
            ]
        )
        if schedule == "cosine_reverse":
            betas = betas.flip(0)
    else:
        raise ValueError(f"unknown beta schedule {schedule}")
    return betas


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def compute_tilde_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.float()
    n = alpha.shape[0]
    tilde_alpha = torch.zeros_like(alpha)
    for t in range(n):
        slice_t = alpha[: t + 1].flip(dims=[0])
        tilde_alpha[t] = torch.cumprod(slice_t, dim=0).sum()
    return tilde_alpha


def compute_hat_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.float()
    n = alpha.shape[0]
    hat_alpha = torch.zeros_like(alpha)
    for t in range(n):
        slice_t = alpha[: t + 1].flip(dims=[0])
        cprod = torch.cumprod(slice_t, dim=0) * slice_t
        hat_alpha[t] = cprod.sum()
    return hat_alpha


def compute_gx_term(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.float()
    n = alpha.shape[0]
    gx_term = torch.zeros_like(alpha)
    for t in range(n):
        slice_t = alpha[: t + 1].flip(dims=[0])
        cprod = torch.cat([torch.tensor([1.0], device=slice_t.device),
                           torch.cumprod(slice_t, dim=0)])
        cprod = cprod[:-1] * ((1 - slice_t) ** 2)
        gx_term[t] = cprod.sum()
    return gx_term


class DiffusionSchedule:
    """Precomputed schedule tensors, mirroring src/models/NsDiff.py."""

    def __init__(self, steps, schedule, beta_start, beta_end, device):
        self.num_timesteps = steps
        betas = make_beta_schedule(schedule, steps, beta_start, beta_end).float().to(device)
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas = alphas
        alphas_cumprod = alphas.cpu().cumprod(dim=0).to(device)
        self.alphas_cumprod = alphas_cumprod
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if schedule == "cosine":
            self.one_minus_alphas_bar_sqrt = self.one_minus_alphas_bar_sqrt * 0.9999
        self.betas_bar = 1 - alphas_cumprod
        self.alphas_cumprod_sum = compute_tilde_alpha(alphas).to(device)
        self.alphas_hat = compute_hat_alpha(alphas).to(device)
        self.betas_tilde = (self.alphas_cumprod_sum - self.alphas_hat).clamp_min(0.0)
        assert (self.betas_tilde >= 0).all()
        assert ((self.betas_bar - self.betas_tilde) >= -1e-6).all()
        self.betas_tilde_m_1 = torch.cat(
            [torch.ones(1, device=device), self.betas_tilde[:-1]], dim=0
        )
        self.betas_bar_m_1 = torch.cat(
            [torch.ones(1, device=device), self.betas_bar[:-1]], dim=0
        )
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_sum_prev = torch.cat(
            [torch.ones(1, device=device), self.alphas_cumprod_sum[:-1]], dim=0
        )
        self.device = device


def _sigma12(sched, gx, y_sigma, t):
    at = extract(sched.alphas, t, gx)
    b_tilde_m_1 = extract(sched.betas_tilde_m_1, t, gx)
    b_bar_m_1 = extract(sched.betas_bar_m_1, t, gx)
    Sigma_1 = (1 - at) ** 2 * gx + at * (1 - at) * y_sigma
    Sigma_2 = (b_bar_m_1 - b_tilde_m_1) * gx + b_tilde_m_1 * y_sigma
    return at, Sigma_1, Sigma_2


def cal_forward_noise(sched, gx, y_sigma, t):
    b_bar_t = extract(sched.betas_bar, t, gx)
    b_tilde_t = extract(sched.betas_tilde, t, gx)
    noise = (b_bar_t - b_tilde_t) * gx + b_tilde_t * y_sigma
    return noise.clamp_min(EPS)


def cal_sigma_tilde(sched, gx, y_sigma, t):
    at, Sigma_1, Sigma_2 = _sigma12(sched, gx, y_sigma, t)
    sigma_tilde = (Sigma_1 * Sigma_2) / (at * Sigma_2 + Sigma_1).clamp_min(EPS)
    return sigma_tilde.clamp_min(EPS)


def calc_gammas(sched, gx, y_sigma, t):
    at, Sigma_1, Sigma_2 = _sigma12(sched, gx, y_sigma, t)
    alpha_bar_t_m_1 = extract(sched.alphas_cumprod_prev, t, gx)
    sqrt_alpha_t = at.sqrt()
    sqrt_alpha_bar_t_m_1 = alpha_bar_t_m_1.sqrt()
    at_s1_s2 = (at * Sigma_2 + Sigma_1).clamp_min(EPS)
    gamma_0 = sqrt_alpha_bar_t_m_1 * Sigma_1 / at_s1_s2
    gamma_1 = sqrt_alpha_t * Sigma_2 / at_s1_s2
    gamma_2 = (
        (sqrt_alpha_t * (at - 1)) * Sigma_2 + (1 - sqrt_alpha_bar_t_m_1) * Sigma_1
    ) / at_s1_s2
    return gamma_0, gamma_1, gamma_2


def q_sample(y, y_0_hat, sched, t, noise):
    sqrt_alpha_bar_t = extract(sched.alphas_bar_sqrt, t, y)
    return sqrt_alpha_bar_t * y + (1 - sqrt_alpha_bar_t) * y_0_hat + noise


def _randn_like(x, generator=None):
    if generator is None:
        return torch.randn_like(x)
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)


def _estimate_sigma_y0(sched, gx, sigma_theta, t, y):
    """Solve the quadratic for Sigma_{Y0} with clamped discriminant (spec 2.7)."""
    alpha_t = extract(sched.alphas, t, y)
    betas_tilde_m_1 = extract(sched.betas_tilde_m_1, t, y)
    betas_bar_m_1 = extract(sched.betas_bar_m_1, t, y)
    lambda_0 = (alpha_t * (1 - alpha_t) * betas_tilde_m_1).clamp_min(EPS)
    lambda_1 = (
        (1 - alpha_t) ** 2 * betas_tilde_m_1
        + alpha_t * (1 - alpha_t) * (betas_bar_m_1 - betas_tilde_m_1)
    ) * gx - sigma_theta * (alpha_t * betas_tilde_m_1 + alpha_t * (1 - alpha_t))
    lambda_2 = gx ** 2 * (1 - alpha_t) ** 2 * (betas_bar_m_1 - betas_tilde_m_1) - sigma_theta * gx * (
        alpha_t * betas_bar_m_1 - alpha_t * betas_tilde_m_1 + (1 - alpha_t) ** 2
    )
    disc = (lambda_1.square() - 4 * lambda_0 * lambda_2).clamp_min(0.0)
    sigma_y0_hat = (-lambda_1 + disc.sqrt()) / (2 * lambda_0)
    return sigma_y0_hat.clamp(min=EPS, max=SIGMA_MAX)


def p_sample_step(denoise_fn, sched, y, y_0_hat, gx, y_T_mean, t_int, generator=None):
    """One reverse step t -> t-1 (t_int >= 1)."""
    device = y.device
    t = torch.tensor([t_int], device=device)
    eps_theta, sigma_theta = denoise_fn(y, t)
    eps_theta = eps_theta.detach()
    sigma_theta = sigma_theta.detach().clamp(min=EPS, max=SIGMA_MAX)

    z = _randn_like(y, generator)
    sqrt_one_minus_alpha_bar_t = extract(sched.one_minus_alphas_bar_sqrt, t, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()

    betas_tilde = extract(sched.betas_tilde, t, y)
    betas_bar = extract(sched.betas_bar, t, y)
    sigma_y0_hat = _estimate_sigma_y0(sched, gx, sigma_theta, t, y)
    noise = ((betas_bar - betas_tilde) * gx + betas_tilde * sigma_y0_hat).clamp_min(EPS)

    y_0_reparam = (
        y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * torch.sqrt(noise)
    ) / sqrt_alpha_bar_t.clamp_min(EPS)
    gamma_0, gamma_1, gamma_2 = calc_gammas(sched, gx, sigma_y0_hat, t)
    y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y + gamma_2 * y_T_mean
    return y_t_m_1_hat + torch.sqrt(sigma_theta) * z


def p_sample_t_1to0(denoise_fn, sched, y, y_0_hat, gx, y_T_mean):
    device = y.device
    t = torch.tensor([0], device=device)
    sqrt_one_minus_alpha_bar_t = extract(sched.one_minus_alphas_bar_sqrt, t, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    eps_theta, sigma_theta = denoise_fn(y, t)
    eps_theta = eps_theta.detach()
    sigma_theta = sigma_theta.detach().clamp(min=EPS, max=SIGMA_MAX)

    betas_tilde = extract(sched.betas_tilde, t, y)
    betas_bar = extract(sched.betas_bar, t, y)
    sigma_y0_hat = _estimate_sigma_y0(sched, gx, sigma_theta, t, y)
    noise = ((betas_bar - betas_tilde) * gx + betas_tilde * sigma_y0_hat).clamp_min(EPS)

    y_0 = (
        y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * torch.sqrt(noise)
    ) / sqrt_alpha_bar_t.clamp_min(EPS)
    return y_0


def p_sample_loop(denoise_fn, sched, y_0_hat, gx, y_T_mean, generator=None):
    """Full reverse chain; returns y_0 only (no per-step history kept)."""
    z = _randn_like(y_T_mean, generator)
    cur_y = torch.sqrt(gx.clamp_min(EPS)) * z + y_T_mean
    for t_int in reversed(range(1, sched.num_timesteps)):
        cur_y = p_sample_step(denoise_fn, sched, cur_y, y_0_hat, gx, y_T_mean, t_int, generator)
    return p_sample_t_1to0(denoise_fn, sched, cur_y, y_0_hat, gx, y_T_mean)
