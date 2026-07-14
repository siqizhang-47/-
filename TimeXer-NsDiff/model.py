"""
TimeXer-conditioned NsDiff model.

Condition path (NEW, TimeXer): history energy + future calendar/weather
    -> cond_global -> f_phi (mu) and g_psi (sigma).
Diffusion path (ORIGINAL NsDiff, unchanged): the non-stationary location-scale
    forward/reverse process on the 4 standardised energy targets, driven by
    (mu, sigma). Implements the plan's §20 minimal version (condition enters
    f_phi and g_psi; the denoiser conditions on mu & sigma as in vanilla NsDiff).
"""
import torch
import torch.nn as nn

from reused.nsdiff_core import NsDiffCore
from reused.nsdiff_utils import q_sample, p_sample_loop, cal_forward_noise, cal_sigma_tilde
from reused.sigma import wv_sigma_trailing
from timexer_nsdiff_adapter import (TimeXerExogenousConditionEncoder, HistoryEncoder,
                                    ConditionedLocationScaleHeads, N_TARGETS)

EPS = 1e-8


class TimeXerNsDiff(nn.Module):
    def __init__(self, horizon=24, seq_len=168, d_model=128, patch_len=24,
                 n_heads=8, e_layers=2, d_ff=512, dropout=0.1,
                 d_x=128, diffusion_steps=20, rolling_length=96,
                 beta_schedule="linear", beta_start=1e-4, beta_end=1e-2, device="cpu"):
        super().__init__()
        self.horizon, self.rolling_length, self.device = horizon, rolling_length, device
        self.condition_encoder = TimeXerExogenousConditionEncoder(
            horizon=horizon, d_model=d_model, patch_len=patch_len,
            n_heads=n_heads, e_layers=e_layers, d_ff=d_ff, dropout=dropout)
        self.history_encoder = HistoryEncoder(n_vars=N_TARGETS, hidden=d_x, dropout=dropout)
        cond_dim = d_model * N_TARGETS
        self.heads = ConditionedLocationScaleHeads(d_x=d_x, cond_dim=cond_dim, horizon=horizon)
        self.diffusion = NsDiffCore(diffusion_steps, N_TARGETS, device,
                                    beta_schedule, beta_start, beta_end)
        self.num_timesteps = diffusion_steps

    # ---- shared: condition -> mu, sigma, attention -------------------------
    def _condition(self, history_energy, future_calendar, future_weather):
        cond = self.condition_encoder(history_energy, future_calendar, future_weather)
        hx = self.history_encoder(history_energy)
        mu, sigma = self.heads(hx, cond["cond_global"])
        return mu, sigma, cond

    # ---- training loss (verbatim NsDiff objective) -------------------------
    def train_loss(self, history_energy, future_calendar, future_weather, future_energy):
        y = future_energy
        mu, sigma, _ = self._condition(history_energy, future_calendar, future_weather)
        gx = sigma
        m = self.diffusion

        y_sigma = wv_sigma_trailing(torch.cat([history_energy, y], dim=1), self.rolling_length)
        y_sigma = y_sigma[:, -self.horizon:, :] + EPS

        n = y.size(0)
        t = torch.randint(0, self.num_timesteps, size=(n // 2 + 1,), device=y.device)
        t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)[:n]

        loss1 = (mu - y).square().mean()
        loss2 = (torch.sqrt(gx) - torch.sqrt(y_sigma)).square().mean()

        e = torch.randn_like(y)
        forward_noise = cal_forward_noise(m.betas_tilde, m.betas_bar, gx, y_sigma, t)
        noise = e * torch.sqrt(forward_noise)
        sigma_tilde = cal_sigma_tilde(m.alphas, m.alphas_cumprod, m.alphas_cumprod_sum,
                                      m.alphas_cumprod_prev, m.alphas_cumprod_sum_prev,
                                      m.betas_tilde_m_1, m.betas_bar_m_1, gx, y_sigma, t)
        y_t = q_sample(y, mu, m.alphas_bar_sqrt, m.one_minus_alphas_bar_sqrt, t, noise=noise)
        eps, sigma_theta = self.diffusion(None, None, y_t, mu, gx, t)
        sigma_theta = sigma_theta + EPS
        kl = ((e - eps)).square().mean() + (sigma_tilde / sigma_theta).mean() \
            - torch.log(sigma_tilde / sigma_theta).mean()
        loss = kl + loss1 + loss2
        return loss, {"kl": float(kl.detach()), "mean": float(loss1.detach()),
                      "scale": float(loss2.detach())}

    # ---- probabilistic sampling (verbatim NsDiff reverse process) ----------
    @torch.no_grad()
    def sample(self, history_energy, future_calendar, future_weather, n_samples=100,
               return_attention=False):
        mu, sigma, cond = self._condition(history_energy, future_calendar, future_weather)
        gx = sigma
        B, H, K = mu.shape
        S = n_samples
        # tile along a sample dimension -> (B*S, H, K), one vectorised reverse pass
        mu_t = mu.repeat_interleave(S, dim=0)
        gx_t = gx.repeat_interleave(S, dim=0)
        m = self.diffusion
        y_seq = p_sample_loop(self.diffusion, None, None, mu_t, gx_t, mu_t,
                              self.num_timesteps, m.alphas, m.one_minus_alphas_bar_sqrt,
                              m.alphas_cumprod, m.alphas_cumprod_sum, m.alphas_cumprod_prev,
                              m.alphas_cumprod_sum_prev, m.betas_tilde, m.betas_bar,
                              m.betas_tilde_m_1, m.betas_bar_m_1)
        y0 = y_seq[-1].reshape(B, S, H, K)                # [B, S, H, 4]
        if return_attention:
            return y0, mu, cond["attention"]
        return y0, mu
