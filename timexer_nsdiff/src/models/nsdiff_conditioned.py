"""TimeXer-conditioned NsDiff (design document, sections 11, 14, 16).

    Y = f_phi(X, C) + g_psi(X, C) * eps

The diffusion algebra is NsDiff's, untouched (see third_party/nsdiff).  What
changes is where the condition enters:

    f_phi     <- cond_horizon           [B, 24, 4, 128]
    g_psi     <- cond_horizon           [B, 24, 4, 128]   (independent parameters)
    denoiser  <- cond_denoiser          [B, 24, 512]      (plain concatenation)

NsDiff's ``gx`` is a *variance*.  ``g_psi`` returns a standard deviation, so the
model squares it on the way into the diffusion process.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..third_party.nsdiff.denoise import ConditionalGuidedModel
from ..third_party.nsdiff.nsdiff_utils import (EPS, cal_forward_noise, cal_sigma_tilde,
                                               compute_gx_term, compute_hat_alpha,
                                               compute_tilde_alpha, make_beta_schedule,
                                               p_sample_loop, q_sample)
from ..third_party.nsdiff.sigma import wv_sigma_trailing
from .location_scale import PerStepLocationScaleHeads
from .timexer_condition_encoder import TimeXerExogenousConditionEncoder


class DiffusionSchedule(nn.Module):
    """All NsDiff schedule tensors, computed exactly as in NsDiff-main/src/models/NsDiff.py."""

    def __init__(self, timesteps: int = 20, beta_schedule: str = "linear",
                 beta_start: float = 1e-4, beta_end: float = 1e-2):
        super().__init__()
        self.num_timesteps = timesteps
        betas = make_beta_schedule(schedule=beta_schedule, num_timesteps=timesteps,
                                   start=beta_start, end=beta_end).float()
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)

        alphas_cumprod_sum = compute_tilde_alpha(alphas)          # \tilde alpha
        alphas_hat = compute_hat_alpha(alphas)
        betas_bar = 1 - alphas_cumprod
        betas_tilde = alphas_cumprod_sum - alphas_hat

        assert (betas_tilde >= 0).all()
        assert ((betas_bar - betas_tilde) >= -1e-6).all()

        ones = torch.ones(1)
        buffers = {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "alphas_bar_sqrt": torch.sqrt(alphas_cumprod),
            "one_minus_alphas_bar_sqrt": torch.sqrt(1 - alphas_cumprod),
            "alphas_cumprod_prev": torch.cat([ones, alphas_cumprod[:-1]], dim=0),
            "alphas_cumprod_sum": alphas_cumprod_sum,
            "alphas_cumprod_sum_prev": torch.cat([ones, alphas_cumprod_sum[:-1]], dim=0),
            "betas_bar": betas_bar,
            "betas_tilde": betas_tilde,
            "betas_tilde_m_1": torch.cat([ones, betas_tilde[:-1]], dim=0),
            "betas_bar_m_1": torch.cat([ones, betas_bar[:-1]], dim=0),
            "gx_term": compute_gx_term(alphas),
        }
        for k, v in buffers.items():
            self.register_buffer(k, v)


class TimeXerNsDiff(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.window = cfg.window
        self.horizon = cfg.horizon
        self.n_targets = cfg.n_targets
        self.rolling_length = cfg.rolling_length

        needs_aux = cfg.use_exog and not (cfg.cond_to_mean and cfg.cond_to_scale
                                          and cfg.cond_to_denoiser)
        self.condition_encoder = TimeXerExogenousConditionEncoder(
            window=cfg.window, horizon=cfg.horizon, n_targets=cfg.n_targets, n_exo=cfg.n_exo,
            patch_len=cfg.patch_len, d_model=cfg.d_model, n_heads=cfg.n_heads,
            e_layers=cfg.e_layers, d_ff=cfg.d_ff, dropout=cfg.dropout,
            activation=cfg.activation, use_exog=cfg.use_exog,
            shared_global_token=cfg.shared_global_token, aux_history_branch=needs_aux,
        )
        self.location_scale = PerStepLocationScaleHeads(
            d_model=cfg.d_model, hidden=cfg.head_hidden, n_layers=cfg.head_layers,
            dropout=cfg.dropout, sigma_floor=cfg.sigma_floor,
            temporal_smoothing=cfg.temporal_smoothing,
        )
        self.schedule = DiffusionSchedule(cfg.diffusion_steps, cfg.beta_schedule,
                                          cfg.beta_start, cfg.beta_end)
        self.denoiser = ConditionalGuidedModel(
            diff_steps=cfg.diffusion_steps, enc_in=cfg.n_targets,
            cond_dim=cfg.n_targets * cfg.d_model, hidden=cfg.denoiser_hidden,
        )

    # ------------------------------------------------------------------ condition
    def encode(self, batch):
        return self.condition_encoder(batch["history_energy"], batch["future_calendar"],
                                      batch["future_weather"])

    def location_scale_forward(self, cond):
        cfg = self.cfg
        tok_mu = cond["target_tokens"] if cfg.cond_to_mean else cond["target_tokens_hist"]
        c_mu = cond["cond_horizon"] if cfg.cond_to_mean else cond["cond_horizon_hist"]
        tok_sg = cond["target_tokens"] if cfg.cond_to_scale else cond["target_tokens_hist"]
        c_sg = cond["cond_horizon"] if cfg.cond_to_scale else cond["cond_horizon_hist"]
        return self.location_scale(tok_mu, c_mu, target_tokens_scale=tok_sg, cond_horizon_scale=c_sg)

    def denoiser_condition(self, cond):
        return cond["cond_denoiser"] if self.cfg.cond_to_denoiser else cond["cond_denoiser_hist"]

    def forward(self, batch):
        cond = self.encode(batch)
        mean, sigma = self.location_scale_forward(cond)
        return cond, mean, sigma

    # ------------------------------------------------------------------ diffusion
    def empirical_target_variance(self, history_energy, future_energy):
        """y_sigma: trailing rolling variance over [history ; future] (NsDiff training signal)."""
        joint = torch.cat([history_energy, future_energy], dim=1)
        y_sigma = wv_sigma_trailing(joint, self.rolling_length)
        return y_sigma[:, -self.horizon:, :] + EPS

    def diffusion_loss(self, batch, cond=None, mean=None, sigma=None):
        """NsDiff training objective (NsDiff-main/src/experiments/NsDiff.py::_process_train_batch)."""
        s = self.schedule
        y = batch["future_energy"]
        n = y.shape[0]

        if cond is None:
            cond = self.encode(batch)
        if mean is None or sigma is None:
            mean, sigma = self.location_scale_forward(cond)

        gx = sigma.pow(2) + EPS                       # NsDiff consumes a variance
        y_sigma = self.empirical_target_variance(batch["history_energy"], y)

        t = torch.randint(low=0, high=s.num_timesteps, size=(n // 2 + 1,), device=y.device)
        t = torch.cat([t, s.num_timesteps - 1 - t], dim=0)[:n]

        y_T_mean = mean
        e = torch.randn_like(y)
        forward_noise = cal_forward_noise(s.betas_tilde, s.betas_bar, gx, y_sigma, t)
        noise = e * torch.sqrt(forward_noise)
        sigma_tilde = cal_sigma_tilde(s.alphas, s.alphas_cumprod, s.alphas_cumprod_sum,
                                      s.alphas_cumprod_prev, s.alphas_cumprod_sum_prev,
                                      s.betas_tilde_m_1, s.betas_bar_m_1, gx, y_sigma, t)

        y_t = q_sample(y, y_T_mean, s.alphas_bar_sqrt, s.one_minus_alphas_bar_sqrt, t, noise=noise)
        eps_pred, sigma_theta = self.denoiser(y_t, mean, gx, t, self.denoiser_condition(cond))
        sigma_theta = sigma_theta + EPS

        ratio = sigma_tilde / sigma_theta
        kl_loss = (e - eps_pred).square().mean() + ratio.mean() - torch.log(ratio).mean()
        return kl_loss, {"kl": float(kl_loss.detach())}

    @torch.no_grad()
    def sample(self, batch, n_samples: int = 100, chunk: int = 25, cond=None,
               mean=None, sigma=None):
        """Joint scenario generation -- returns [B, S, H, K] in the standardised space.

        All four targets are produced by the same reverse trajectory, so their
        cross-correlation is preserved (24.5).
        """
        s = self.schedule
        if cond is None:
            cond = self.encode(batch)
        if mean is None or sigma is None:
            mean, sigma = self.location_scale_forward(cond)
        gx = sigma.pow(2) + EPS
        cond_d = self.denoiser_condition(cond)

        B = mean.shape[0]
        outputs = []
        drawn = 0
        while drawn < n_samples:
            r = min(chunk, n_samples - drawn)
            mean_t = mean.repeat_interleave(r, dim=0)
            gx_t = gx.repeat_interleave(r, dim=0)
            cond_t = cond_d.repeat_interleave(r, dim=0)
            y0 = p_sample_loop(
                self.denoiser, cond_t, mean_t, gx_t, mean_t, s.num_timesteps,
                s.alphas, s.one_minus_alphas_bar_sqrt, s.alphas_cumprod, s.alphas_cumprod_sum,
                s.alphas_cumprod_prev, s.alphas_cumprod_sum_prev, s.betas_tilde, s.betas_bar,
                s.betas_tilde_m_1, s.betas_bar_m_1,
            )
            outputs.append(y0.reshape(B, r, self.horizon, self.n_targets))
            drawn += r
        return torch.cat(outputs, dim=1)

    # ------------------------------------------------------------------ freezing
    def set_trainable(self, encoder=True, mean_head=True, scale_head=True, denoiser=True):
        for p in self.condition_encoder.parameters():
            p.requires_grad_(encoder)
        for p in self.location_scale.mean_head.parameters():
            p.requires_grad_(mean_head)
        if self.location_scale.smoother_mu is not None:
            for p in self.location_scale.smoother_mu.parameters():
                p.requires_grad_(mean_head)
            for p in self.location_scale.smoother_sigma.parameters():
                p.requires_grad_(scale_head)
        for p in self.location_scale.scale_head.parameters():
            p.requires_grad_(scale_head)
        for p in self.denoiser.parameters():
            p.requires_grad_(denoiser)
