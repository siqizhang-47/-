"""NsDiff-Exo: NsDiff with oracle future exogenous conditioning on all
three paths (mean f_phi, variance g_psi, denoiser eps/sigma_theta).
"""
from types import SimpleNamespace

import torch
import torch.nn as nn
from tqdm import tqdm

import src.layer.mu_backbone_exo as mu_backbone_exo
from src.layer.denoise_exo import ConditionalGuidedModelExo
from src.layer.g_backbone_exo import SigmaEstimationExo
from src.layer.masked_reduction import masked_mean
from src.models.diffusion_utils import (
    DiffusionSchedule,
    cal_forward_noise,
    cal_sigma_tilde,
    p_sample_loop,
    q_sample,
)
from src.utils.sigma import wv_sigma_trailing

EPS = 1e-8


class NsDiffExo(nn.Module):
    def __init__(self, cfg: dict, device):
        super().__init__()
        m = cfg["model"]
        d = cfg["diffusion"]
        data = cfg["data"]
        self.L = int(data["context_length"])
        self.H = int(data["prediction_length"])
        self.label_len = int(data.get("label_length", self.L // 2))
        self.enc_in = int(m.get("enc_in", 3))
        self.condition_dim = int(m.get("condition_dim", 10))
        self.rolling_length = int(d.get("rolling_length", 24))
        self.device = device

        args = SimpleNamespace(
            seq_len=self.L,
            pred_len=self.H,
            label_len=self.label_len,
            enc_in=self.enc_in,
            dec_in=self.enc_in,
            c_out=self.enc_in,
            condition_dim=self.condition_dim,
            d_model=int(m["d_model"]),
            n_heads=int(m["n_heads"]),
            e_layers=int(m["e_layers"]),
            d_layers=int(m["d_layers"]),
            d_ff=int(m["d_ff"]),
            factor=int(m["factor"]),
            dropout=float(m["dropout"]),
            activation=m.get("activation", "gelu"),
            p_hidden_dims=list(m.get("p_hidden_dims", [64, 64])),
            p_hidden_layers=int(m.get("p_hidden_layers", 2)),
        )
        self.args = args
        self.mean_model = mu_backbone_exo.Model(args)
        self.variance_model = SigmaEstimationExo(
            self.L, self.H, self.enc_in, self.condition_dim,
            hidden_size=int(m.get("g_hidden_size", 512)),
            kernel_size=self.rolling_length,
        )
        self.denoiser = ConditionalGuidedModelExo(
            int(d["steps"]), self.enc_in, self.condition_dim,
            history_embed_dim=int(m.get("history_embed_dim", 32)),
            context_dim=int(m.get("denoise_context_dim", 64)),
            hidden_dim=int(m.get("denoise_hidden_dim", 128)),
            dropout=float(m["dropout"]),
        )
        self.schedule = DiffusionSchedule(
            int(d["steps"]), d.get("beta_schedule", "linear"),
            float(d.get("beta_start", 1e-4)), float(d.get("beta_end", 1e-2)),
            device,
        )

    # ------------------------------------------------------------------ fwd
    def decoder_inputs(self, batch):
        history_target = batch["history_target"]
        decoder_target = torch.cat(
            [history_target[:, -self.label_len:, :],
             torch.zeros(history_target.size(0), self.H, self.enc_in,
                         device=history_target.device)],
            dim=1,
        )
        decoder_condition = torch.cat(
            [batch["history_condition"][:, -self.label_len:, :],
             batch["future_condition"]],
            dim=1,
        )
        return decoder_target, decoder_condition

    def forward_mean(self, batch):
        decoder_target, decoder_condition = self.decoder_inputs(batch)
        mu, future_hidden = self.mean_model(
            batch["history_target"], batch["history_condition"],
            decoder_target, decoder_condition,
        )
        return mu, future_hidden

    def forward_variance(self, batch):
        return self.variance_model(
            batch["history_target"], batch["history_condition"], batch["future_condition"]
        ) + EPS

    def denoise_fn(self, batch):
        def fn(y_t, t):
            return self.denoiser(
                batch["history_target"], batch["history_condition"],
                batch["future_condition"], y_t, batch["_y_0_hat"], batch["_gx"], t,
            )
        return fn

    # ---------------------------------------------------------------- losses
    def element_losses(self, batch, lambda_reverse_var=1.0):
        """Per-coordinate losses [B,H,D]; reduction/masking done by the caller."""
        y0 = batch["future_target"]
        B = y0.size(0)
        y_sigma = wv_sigma_trailing(
            torch.cat([batch["history_target"], y0], dim=1), self.rolling_length
        )[:, -self.H:, :] + EPS

        mu, future_hidden = self.forward_mean(batch)
        gx = self.forward_variance(batch)

        mean_elem = (mu - y0).square()
        variance_elem = (
            torch.sqrt(gx.clamp_min(EPS)) - torch.sqrt(y_sigma.clamp_min(EPS))
        ).square()

        # antithetic timestep sampling as in the original repo
        t = torch.randint(0, self.schedule.num_timesteps, size=(B // 2 + 1,),
                          device=y0.device)
        t = torch.cat([t, self.schedule.num_timesteps - 1 - t], dim=0)[:B]

        e = torch.randn_like(y0)
        forward_noise = cal_forward_noise(self.schedule, gx, y_sigma, t)
        noise = e * torch.sqrt(forward_noise)
        sigma_tilde = cal_sigma_tilde(self.schedule, gx, y_sigma, t)
        y_t = q_sample(y0, mu, self.schedule, t, noise)

        eps_pred, sigma_theta = self.denoiser(
            batch["history_target"], batch["history_condition"],
            batch["future_condition"], y_t, mu, gx, t,
        )
        sigma_theta = sigma_theta + EPS
        ratio = (sigma_tilde.clamp_min(EPS) / sigma_theta.clamp_min(EPS)).clamp(1e-6, 1e6)
        reverse_variance_elem = ratio - torch.log(ratio) - 1.0
        noise_elem = (e - eps_pred).square()
        diffusion_elem = noise_elem + lambda_reverse_var * reverse_variance_elem

        return {
            "mean_elem": mean_elem,
            "variance_elem": variance_elem,
            "diffusion_elem": diffusion_elem,
            "noise_elem": noise_elem,
            "reverse_variance_elem": reverse_variance_elem,
            "mu": mu,
            "gx": gx,
            "future_hidden": future_hidden,
        }

    def loss(self, batch, mask, loss_cfg):
        """NsDiff baseline continuous loss with a caller-provided mask
        (future_observed for NsDiff, magnitude mask for ZG)."""
        elems = self.element_losses(batch, float(loss_cfg.get("lambda_reverse_var", 1.0)))
        loss_mean = masked_mean(elems["mean_elem"], mask)
        loss_variance = masked_mean(elems["variance_elem"], mask)
        loss_diffusion = masked_mean(elems["diffusion_elem"], mask)
        total = (
            float(loss_cfg.get("lambda_mean", 1.0)) * loss_mean
            + float(loss_cfg.get("lambda_variance", 1.0)) * loss_variance
            + float(loss_cfg.get("lambda_diffusion", 1.0)) * loss_diffusion
        )
        logs = {
            "loss_mean": loss_mean.item(),
            "loss_variance": loss_variance.item(),
            "loss_diffusion": loss_diffusion.item(),
        }
        return total, logs, elems

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample_trajectories(self, batch, num_samples: int, chunk_size: int,
                            generator: torch.Generator = None, progress: bool = False):
        """Chunked reverse sampling -> samples [B, H, D, S] (model space)."""
        mu, _ = self.forward_mean(batch)
        gx = self.forward_variance(batch)
        B = mu.size(0)
        chunks = []
        rng = range(0, num_samples, chunk_size)
        it = tqdm(rng, desc="sampling", ncols=100, leave=False) if progress else rng
        for start in it:
            current = min(chunk_size, num_samples - start)
            rep = {
                "history_target": batch["history_target"].repeat_interleave(current, dim=0),
                "history_condition": batch["history_condition"].repeat_interleave(current, dim=0),
                "future_condition": batch["future_condition"].repeat_interleave(current, dim=0),
                "_y_0_hat": mu.repeat_interleave(current, dim=0),
                "_gx": gx.repeat_interleave(current, dim=0),
            }
            y0 = p_sample_loop(
                self.denoise_fn(rep), self.schedule,
                rep["_y_0_hat"], rep["_gx"], rep["_y_0_hat"], generator,
            )
            chunk = y0.reshape(B, current, self.H, self.enc_in).permute(0, 2, 3, 1)
            chunks.append(chunk.cpu())
        return torch.cat(chunks, dim=-1)  # [B,H,D,S]
