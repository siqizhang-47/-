"""DiffMTS adapted from CCDM-main/model.py.

Changes vs. the source:
- MW task: five-tuple batches (x, y0, w, x_mark, y_mark); the diffusion models
  only the 4 target channels, weather is a pure condition;
- window normalization acts on the target channels only (covariates keep their
  global standardization; weather has no long-term drift);
- tqdm progress bars for epochs / batches / validation;
- validation-CRPS model selection + EARLY STOPPING (patience on evaluations);
- best checkpoint saved to a single file instead of one file per epoch.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from backbone.diffusion import DDPM
from backbone.network import Denoiser
from utils.early_stopping import EarlyStopper


def cal_mse_loss(a, b):
    B = a.shape[0]
    inst_mse = F.mse_loss(a, b, reduction="none")
    return torch.mean(inst_mse.contiguous().view(B, -1), dim=1)  # (B, )


def ensemble_crps_np(scen, y):
    """scen (M, H, C), y (H, C) -> mean scalar CRPS (numpy)."""
    t1 = np.abs(scen - y[None]).mean(axis=0)
    t2 = np.abs(scen[None] - scen[:, None]).mean(axis=(0, 1))
    return float((t1 - 0.5 * t2).mean())


class DiffMTS:
    def __init__(self, configs, train_loader=None, val_loader=None):
        self.cfg = configs
        self.use_window_norm = configs.use_window_norm
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_steps = configs.n_steps
        self.device = configs.device
        self.n_epochs = configs.n_epochs
        self.num_feat = configs.num_feat
        self.num_target = configs.num_target
        self.cont_len = configs.cont_len
        self.pred_len = configs.pred_len
        self.parameterization = configs.parameterization
        self.step_dist = configs.step_dist
        self.use_contrast = configs.use_contrast
        self.n_negatives = configs.n_negatives
        self.contrast_weight = configs.contrast_weight
        self.temperature = configs.temperature
        self.init_lr = configs.init_lr

        self.denoiser = Denoiser(configs).to(self.device)
        self.diffusion = DDPM(self.denoiser, configs).to(self.device)
        self.optimizer = optim.Adam(self.denoiser.parameters(), lr=self.init_lr, weight_decay=1e-6)
        p1, p2 = int(0.75 * self.n_epochs), int(0.9 * self.n_epochs)
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[p1, p2], gamma=0.1)

    # ---------------- window normalization (targets only) ----------------
    def instance_normalization(self, x, y0=None):
        """Normalize the target channels with the context-window mean/std.

        x: (B, cont_len, num_feat) -- first num_target channels are targets.
        y0: (B, pred_len, num_target) or None.
        """
        x_t = x[:, :, :self.num_target]
        mean = x_t.mean(dim=1, keepdim=True)
        std = torch.sqrt(torch.var(x_t, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = torch.cat([(x_t - mean) / std, x[:, :, self.num_target:]], dim=-1)
        y0_norm = (y0 - mean) / std if y0 is not None else None
        return x_norm, y0_norm, mean, std

    def instance_denormalization(self, y0, mean, std):
        B = mean.shape[0]
        n_samples = y0.shape[0] // B
        std = torch.repeat_interleave(std, n_samples, dim=0).repeat(1, self.pred_len, 1)
        mean = torch.repeat_interleave(mean, n_samples, dim=0).repeat(1, self.pred_len, 1)
        return y0 * std + mean

    # ---------------- training losses (from CCDM, weather threaded through) ----------------
    def step_sampling(self, batch_size):
        if self.step_dist == "uniform":
            k_half0 = torch.randint(0, self.n_steps, (batch_size // 2,))
            k_half1 = self.n_steps - 1 - k_half0
            k = torch.cat([k_half0, k_half1], dim=0)
            if k.shape[0] < batch_size:  # odd batch
                k = torch.cat([k, torch.randint(0, self.n_steps, (batch_size - k.shape[0],))])
        else:
            raise NotImplementedError(f"step_dist {self.step_dist}")
        return k.to(self.device)

    def negative_sampling(self, y0, mode):
        B, y0_len = y0.shape[0], y0.shape[1]
        C = self.num_target
        if mode == "variation":
            patch_size = 8
            n_patches = y0_len // patch_size
            y0_patch = y0.view(B, n_patches, patch_size, C)
            neg_samples = torch.zeros((B * self.n_negatives, y0_len, C), device=y0.device)
            for i in range(self.n_negatives):
                idx_patch = torch.randperm(y0_patch.shape[1])
                y0_shuffle = y0_patch[:, idx_patch, :, :].reshape(B, y0_len, C)
                neg_indices = torch.arange(0, neg_samples.shape[0], self.n_negatives) + i
                neg_samples[neg_indices] = y0_shuffle
        elif mode == "scaling":
            scale_down = np.random.uniform(0.0, 0.5, (self.n_negatives // 2, 1, C))
            scale_up = np.random.uniform(1.5, 2.0, (self.n_negatives // 2, 1, C))
            scale = torch.from_numpy(np.concatenate([scale_up, scale_down], axis=0))
            scale_rep = scale.repeat(B, 1, 1).to(y0.device)
            y0_rep = torch.repeat_interleave(y0, self.n_negatives, dim=0)
            neg_samples = y0_rep * scale_rep
        else:
            raise NotImplementedError(f"No such negative sampling mode: {mode}!")
        return neg_samples.float()

    def cal_contrastive_loss(self, neg_samples, x, k, w, x_mark, y_mark):
        B = x.shape[0]
        n_negatives = neg_samples.shape[0] // B
        neg_noise = torch.randn_like(neg_samples)
        rep = lambda t: torch.repeat_interleave(t, n_negatives, dim=0) if t is not None else None
        neg_yk = self.diffusion.q_sample(neg_samples, rep(k), neg_noise)
        neg_pred = self.denoiser(rep(x), neg_yk, rep(k), rep(w), rep(x_mark), rep(y_mark))
        neg_pred = neg_pred.reshape(B * n_negatives, -1)
        neg_noise = neg_noise.reshape(B * n_negatives, -1)
        neg_loss = F.cosine_similarity(neg_pred, neg_noise, dim=1).view(B, n_negatives)
        return neg_loss

    def cal_train_loss(self, x, y0, w, x_mark, y_mark):
        B = x.shape[0]
        k = self.step_sampling(B)
        noise = torch.randn_like(y0)
        yk = self.diffusion.q_sample(y0, k, noise)
        pred_k = self.denoiser(x, yk, k, w, x_mark, y_mark)
        target = noise if self.parameterization == "noise" else y0
        denoise_loss = cal_mse_loss(pred_k, target)  # (B, )
        if self.use_contrast == "non-contrast":
            return torch.mean(denoise_loss), torch.zeros((1,), device=self.device)

        neg_variation = self.negative_sampling(y0, mode="variation")
        neg_scale = self.negative_sampling(y0, mode="scaling")
        neg_variation_loss = self.cal_contrastive_loss(neg_variation, x, k, w, x_mark, y_mark)
        neg_scale_loss = self.cal_contrastive_loss(neg_scale, x, k, w, x_mark, y_mark)

        pos_loss = F.cosine_similarity(pred_k.reshape(B, -1), target.reshape(B, -1), dim=1).unsqueeze(1)
        contrast = torch.concatenate([pos_loss, neg_variation_loss, neg_scale_loss], dim=1)
        infonce_loss = -torch.log(torch.softmax(contrast / self.temperature, dim=1)[:, 0])
        return torch.mean(denoise_loss), torch.mean(infonce_loss)

    def _prepare_batch(self, batch):
        x, y0, w, x_mark, y_mark = [b.float().to(self.device) for b in batch]
        norm_stats = None
        if self.use_window_norm:
            x, y0, mean, std = self.instance_normalization(x, y0)
            norm_stats = (mean, std)
        return x, y0, w, x_mark, y_mark, norm_stats

    # ---------------- training with early stopping ----------------
    def train(self, ckpt_path, log_path=None):
        cfg = self.cfg
        stopper = EarlyStopper(patience=getattr(cfg, "patience", 4))
        eval_every = getattr(cfg, "eval_every", 5)
        min_epochs = getattr(cfg, "min_epochs", 20)
        history = {"train_loss": [], "val_loss": [], "val_crps": []}
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        epoch_bar = tqdm(range(self.n_epochs), desc="epochs")
        for epoch_no in epoch_bar:
            self.denoiser.train()
            batch_losses = []
            start = time.time()
            for batch in tqdm(self.train_loader, desc=f"train e{epoch_no}", leave=False):
                self.optimizer.zero_grad()
                x, y0, w, x_mark, y_mark, _ = self._prepare_batch(batch)
                denoise_loss, contrast_loss = self.cal_train_loss(x, y0, w, x_mark, y_mark)
                total_loss = denoise_loss + self.contrast_weight * contrast_loss
                total_loss.backward()
                self.optimizer.step()
                batch_losses.append(total_loss.item())
            self.lr_scheduler.step()
            train_loss = float(np.mean(batch_losses))
            val_loss = self.validate_loss()
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            postfix = {"train": f"{train_loss:.4f}", "val": f"{val_loss:.4f}",
                       "t": f"{time.time()-start:.0f}s"}

            if (epoch_no + 1) % eval_every == 0 or epoch_no == self.n_epochs - 1:
                val_crps = self.validate_crps()
                history["val_crps"].append({"epoch": epoch_no, "crps": val_crps})
                postfix["val_crps"] = f"{val_crps:.4f}"
                should_stop = stopper.update(val_crps, epoch_no)
                if stopper.improved:
                    torch.save(self.denoiser.state_dict(), ckpt_path)
                    postfix["best"] = "*"
                if should_stop and epoch_no + 1 >= min_epochs:
                    epoch_bar.set_postfix(postfix)
                    print(f"\nEarly stopping at epoch {epoch_no} "
                          f"(best val CRPS {stopper.best:.5f} @ epoch {stopper.best_step})")
                    break
            epoch_bar.set_postfix(postfix)

        if stopper.best_step < 0:  # no CRPS evaluation ever ran; save last state
            torch.save(self.denoiser.state_dict(), ckpt_path)
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w") as f:
                json.dump({"history": history, "best_epoch": stopper.best_step,
                           "best_val_crps": stopper.best}, f, indent=2)
        print(f"Best checkpoint saved to {ckpt_path} "
              f"(epoch {stopper.best_step}, val CRPS {stopper.best:.5f})")
        return history

    @torch.no_grad()
    def validate_loss(self):
        """Mean denoising loss on the validation set (cheap, every epoch)."""
        self.denoiser.eval()
        losses = []
        for batch in tqdm(self.val_loader, desc="val loss", leave=False):
            x, y0, w, x_mark, y_mark, _ = self._prepare_batch(batch)
            B = x.shape[0]
            k = self.step_sampling(B)
            noise = torch.randn_like(y0)
            yk = self.diffusion.q_sample(y0, k, noise)
            pred_k = self.denoiser(x, yk, k, w, x_mark, y_mark)
            target = noise if self.parameterization == "noise" else y0
            losses.append(cal_mse_loss(pred_k, target).mean().item())
        return float(np.mean(losses))

    @torch.no_grad()
    def validate_crps(self):
        """Sample-based CRPS on a subset of validation windows (model selection)."""
        cfg = self.cfg
        n_windows = getattr(cfg, "crps_n_windows", 100)
        n_samples = getattr(cfg, "crps_n_samples", 32)
        dataset = self.val_loader.dataset
        # deterministic evenly-spaced day starts across the validation span
        idx_max = len(dataset)
        indices = np.linspace(0, idx_max - 1, min(n_windows, idx_max)).astype(int)
        self.denoiser.eval()
        crps_vals = []
        for idx in tqdm(indices, desc="val CRPS", leave=False):
            batch = [torch.from_numpy(np.asarray(b)).unsqueeze(0) for b in dataset[int(idx)]]
            x, y0, w, x_mark, y_mark, norm_stats = self._prepare_batch(batch)
            scen = self.diffusion.sampling(n_samples, x, w, x_mark, y_mark)
            if self.use_window_norm:
                scen = self.instance_denormalization(scen, *norm_stats)
                y0 = y0 * norm_stats[1].repeat(1, self.pred_len, 1) + \
                    norm_stats[0].repeat(1, self.pred_len, 1)
            crps_vals.append(ensemble_crps_np(scen.cpu().numpy(), y0[0].cpu().numpy()))
        return float(np.mean(crps_vals))

    # ---------------- inference ----------------
    def load_weights(self, ckpt_path):
        state = torch.load(ckpt_path, map_location=self.device)
        self.denoiser.load_state_dict(state)
        self.denoiser.eval()
        print(f"Loaded weights from {ckpt_path}")

    @torch.no_grad()
    def pred_sampling(self, x, w, n_samples, x_mark, y_mark):
        return self.diffusion.sampling(n_samples, x, w, x_mark, y_mark)
