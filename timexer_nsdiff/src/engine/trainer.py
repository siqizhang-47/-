"""Staged training driver (design document, section 15).

Stage 1  condition encoder + horizon adapter + f_phi        (MSE / Huber)
Stage 2  g_psi, mean frozen, encoder frozen for the first N epochs (Gaussian NLL)
Stage 3  joint location-scale fine-tuning                   (L_mu + lambda * L_sigma)
Stage 4  NsDiff denoiser, condition modules frozen          (NsDiff diffusion loss)
Stage 5  optional end-to-end fine-tuning at a very small LR (off by default)

Every stage has its own tqdm progress bar and its own early stopper.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from ..metrics.prob_metrics import crps_ensemble
from ..models.location_scale import gaussian_location_scale_loss, gaussian_nll_full
from ..utils.early_stop import EarlyStopping

BATCH_KEYS = ("history_energy", "future_energy", "future_calendar", "future_weather")


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def make_loader(dataset, batch_size, shuffle, num_workers, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=drop_last,
                      persistent_workers=num_workers > 0)


def build_optimizer(name, param_groups, lr, weight_decay):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"unsupported optimizer '{name}'")


def mean_loss_fn(kind):
    if kind == "huber":
        return nn.HuberLoss(delta=1.0)
    return nn.MSELoss()


def _clip(model, grad_clip):
    if grad_clip and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)


class Trainer:
    def __init__(self, model, cfg, device, train_set, val_set, run_dir, logger=print):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.run_dir = run_dir
        self.log = logger
        self.train_loader = make_loader(train_set, cfg.batch_size, True, cfg.num_workers, True)
        self.val_loader = make_loader(val_set, cfg.batch_size, False, cfg.num_workers)
        self.val_set = val_set
        self.history = {}
        os.makedirs(run_dir, exist_ok=True)

    # ================================================================= stage 1
    def train_mean(self):
        cfg = self.cfg
        if cfg.epochs_mean <= 0:
            return
        self.log("\n=== Stage 1/4 : condition encoder + horizon adapter + f_phi ===")
        self.model.set_trainable(encoder=True, mean_head=True, scale_head=False, denoiser=False)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optim = build_optimizer(cfg.optimizer, params, cfg.lr_mean, cfg.wd_mean)
        crit = mean_loss_fn(cfg.mean_loss)
        stopper = EarlyStopping(cfg.patience_mean, path=self.run_dir, name="stage1_mean")
        hist = []

        for epoch in tqdm(range(cfg.epochs_mean), desc="stage1", unit="ep", position=0):
            self.model.train()
            losses = []
            bar = tqdm(self.train_loader, desc=f"  ep{epoch:02d} train", leave=False,
                       unit="batch", position=1)
            for batch in bar:
                batch = to_device(batch, self.device)
                cond, mean, _ = self.model(batch)
                loss = crit(mean, batch["future_energy"])
                optim.zero_grad(set_to_none=True)
                loss.backward()
                _clip(self.model, cfg.grad_clip)
                optim.step()
                losses.append(loss.item())
                bar.set_postfix(loss=f"{loss.item():.4f}")

            val = self._val_mean(crit)
            tr = float(np.mean(losses))
            self.log(f"  [stage1] epoch {epoch:02d} train={tr:.5f} val={val:.5f}")
            hist.append({"epoch": epoch, "train": tr, "val": val})
            stopper(val, self.model, epoch)
            if stopper.early_stop:
                self.log(f"  [stage1] early stop at epoch {epoch}")
                break
        stopper.load_best(self.model)
        self.history["stage1"] = hist

    @torch.no_grad()
    def _val_mean(self, crit):
        self.model.eval()
        losses = []
        for batch in tqdm(self.val_loader, desc="  val", leave=False, unit="batch", position=1):
            batch = to_device(batch, self.device)
            _, mean, _ = self.model(batch)
            losses.append(crit(mean, batch["future_energy"]).item())
        return float(np.mean(losses))

    # ================================================================= stage 2
    def train_scale(self):
        cfg = self.cfg
        if cfg.epochs_scale <= 0:
            return
        self.log("\n=== Stage 2/4 : g_psi (mean frozen, Gaussian NLL) ===")
        stopper = EarlyStopping(cfg.patience_scale, path=self.run_dir, name="stage2_scale")
        hist = []

        for epoch in tqdm(range(cfg.epochs_scale), desc="stage2", unit="ep", position=0):
            unfreeze_encoder = epoch >= cfg.freeze_encoder_epochs
            self.model.set_trainable(encoder=unfreeze_encoder, mean_head=False,
                                     scale_head=True, denoiser=False)
            groups = [{"params": [p for p in self.model.location_scale.scale_head.parameters()],
                       "lr": cfg.lr_scale_head}]
            if unfreeze_encoder:
                groups.append({"params": [p for p in self.model.condition_encoder.parameters()],
                               "lr": cfg.lr_scale_encoder})
            optim = build_optimizer(cfg.optimizer, groups, cfg.lr_scale_head, cfg.wd_mean)

            self.model.train()
            losses = []
            bar = tqdm(self.train_loader, desc=f"  ep{epoch:02d} train", leave=False,
                       unit="batch", position=1)
            for batch in bar:
                batch = to_device(batch, self.device)
                cond, mean, sigma = self.model(batch)
                loss = gaussian_location_scale_loss(batch["future_energy"], mean, sigma,
                                                    detach_mean=True)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                _clip(self.model, cfg.grad_clip)
                optim.step()
                losses.append(loss.item())
                bar.set_postfix(nll=f"{loss.item():.4f}", enc="on" if unfreeze_encoder else "frozen")

            val = self._val_nll()
            tr = float(np.mean(losses))
            self.log(f"  [stage2] epoch {epoch:02d} train_nll={tr:.5f} val_nll={val:.5f} "
                     f"(encoder {'trainable' if unfreeze_encoder else 'frozen'})")
            hist.append({"epoch": epoch, "train": tr, "val": val})
            stopper(val, self.model, epoch)
            if stopper.early_stop:
                self.log(f"  [stage2] early stop at epoch {epoch}")
                break
        stopper.load_best(self.model)
        self.history["stage2"] = hist

    @torch.no_grad()
    def _val_nll(self):
        self.model.eval()
        losses = []
        for batch in tqdm(self.val_loader, desc="  val", leave=False, unit="batch", position=1):
            batch = to_device(batch, self.device)
            _, mean, sigma = self.model(batch)
            losses.append(gaussian_nll_full(batch["future_energy"], mean, sigma).item())
        return float(np.mean(losses))

    # ================================================================= stage 3
    def train_joint(self):
        cfg = self.cfg
        if cfg.epochs_joint <= 0:
            return
        self.log(f"\n=== Stage 3/4 : joint location-scale (lambda_sigma={cfg.lambda_sigma}) ===")
        self.model.set_trainable(encoder=True, mean_head=True, scale_head=True, denoiser=False)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optim = build_optimizer(cfg.optimizer, params, cfg.lr_joint, cfg.wd_mean)
        crit = mean_loss_fn(cfg.mean_loss)
        stopper = EarlyStopping(cfg.patience_joint, path=self.run_dir, name="stage3_joint")
        hist = []

        for epoch in tqdm(range(cfg.epochs_joint), desc="stage3", unit="ep", position=0):
            self.model.train()
            losses = []
            bar = tqdm(self.train_loader, desc=f"  ep{epoch:02d} train", leave=False,
                       unit="batch", position=1)
            for batch in bar:
                batch = to_device(batch, self.device)
                y = batch["future_energy"]
                _, mean, sigma = self.model(batch)
                l_mu = crit(mean, y)
                l_sigma = gaussian_location_scale_loss(y, mean, sigma, detach_mean=False)
                loss = l_mu + cfg.lambda_sigma * l_sigma
                optim.zero_grad(set_to_none=True)
                loss.backward()
                _clip(self.model, cfg.grad_clip)
                optim.step()
                losses.append(loss.item())
                bar.set_postfix(loss=f"{loss.item():.4f}", mu=f"{l_mu.item():.4f}")

            val = self._val_joint(crit)
            tr = float(np.mean(losses))
            self.log(f"  [stage3] epoch {epoch:02d} train={tr:.5f} val={val:.5f}")
            hist.append({"epoch": epoch, "train": tr, "val": val})
            stopper(val, self.model, epoch)
            if stopper.early_stop:
                self.log(f"  [stage3] early stop at epoch {epoch}")
                break
        stopper.load_best(self.model)
        self.history["stage3"] = hist

    @torch.no_grad()
    def _val_joint(self, crit):
        self.model.eval()
        losses = []
        for batch in tqdm(self.val_loader, desc="  val", leave=False, unit="batch", position=1):
            batch = to_device(batch, self.device)
            y = batch["future_energy"]
            _, mean, sigma = self.model(batch)
            losses.append((crit(mean, y)
                           + self.cfg.lambda_sigma * gaussian_nll_full(y, mean, sigma)).item())
        return float(np.mean(losses))

    # ================================================================= stage 4
    def train_diffusion(self):
        cfg = self.cfg
        if cfg.epochs_diffusion <= 0:
            return
        self.log("\n=== Stage 4/4 : NsDiff denoiser (condition modules frozen) ===")
        self.model.set_trainable(encoder=False, mean_head=False, scale_head=False, denoiser=True)
        params = [p for p in self.model.denoiser.parameters()]
        optim = build_optimizer(cfg.optimizer, params, cfg.lr_diffusion, 0.0)
        mode = cfg.diff_val_metric
        stopper = EarlyStopping(cfg.patience_diffusion, path=self.run_dir, name="stage4_diffusion")
        hist = []
        val_loader = self.val_loader
        if mode == "crps":
            n = min(cfg.val_subsample, len(self.val_set))
            idx = np.linspace(0, len(self.val_set) - 1, n).astype(int)
            val_loader = make_loader(Subset(self.val_set, idx.tolist()),
                                     cfg.eval_batch_size, False, cfg.num_workers)

        for epoch in tqdm(range(cfg.epochs_diffusion), desc="stage4", unit="ep", position=0):
            self.model.train()
            self.model.condition_encoder.eval()   # keep dropout off in the frozen condition path
            losses = []
            bar = tqdm(self.train_loader, desc=f"  ep{epoch:02d} train", leave=False,
                       unit="batch", position=1)
            for batch in bar:
                batch = to_device(batch, self.device)
                with torch.no_grad():
                    cond = self.model.encode(batch)
                    mean, sigma = self.model.location_scale_forward(cond)
                    cond = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in cond.items()}
                loss, _ = self.model.diffusion_loss(batch, cond=cond,
                                                    mean=mean.detach(), sigma=sigma.detach())
                optim.zero_grad(set_to_none=True)
                loss.backward()
                _clip(self.model, cfg.grad_clip)
                optim.step()
                losses.append(loss.item())
                bar.set_postfix(diff=f"{loss.item():.4f}")

            val = self._val_diffusion(val_loader, mode)
            tr = float(np.mean(losses))
            self.log(f"  [stage4] epoch {epoch:02d} train={tr:.5f} val_{mode}={val:.5f}")
            hist.append({"epoch": epoch, "train": tr, f"val_{mode}": val})
            stopper(val, self.model, epoch)
            if stopper.early_stop:
                self.log(f"  [stage4] early stop at epoch {epoch}")
                break
        stopper.load_best(self.model)
        self.history["stage4"] = hist

    @torch.no_grad()
    def _val_diffusion(self, loader, mode):
        self.model.eval()
        vals = []
        for batch in tqdm(loader, desc=f"  val({mode})", leave=False, unit="batch", position=1):
            batch = to_device(batch, self.device)
            if mode == "crps":
                samples = self.model.sample(batch, n_samples=self.cfg.num_samples,
                                            chunk=self.cfg.sample_chunk)
                ps = samples.permute(0, 2, 3, 1)          # [B, H, K, S]
                vals.append(crps_ensemble(ps, batch["future_energy"]).mean().item())
            else:
                loss, _ = self.model.diffusion_loss(batch)
                vals.append(loss.item())
        return float(np.mean(vals))

    # ================================================================= stage 5
    def finetune_end_to_end(self):
        cfg = self.cfg
        if cfg.epochs_finetune <= 0:
            return
        self.log("\n=== Stage 5 (optional) : end-to-end fine-tuning ===")
        # The historical patch transformer stays frozen (design document 15.5).
        self.model.set_trainable(encoder=True, mean_head=True, scale_head=True, denoiser=True)
        for p in self.model.condition_encoder.en_embedding.parameters():
            p.requires_grad_(False)
        for p in self.model.condition_encoder.encoder.parameters():
            p.requires_grad_(False)

        params = [p for p in self.model.parameters() if p.requires_grad]
        optim = build_optimizer(cfg.optimizer, params, cfg.lr_finetune, 0.0)
        stopper = EarlyStopping(cfg.patience_finetune, path=self.run_dir, name="stage5_finetune")
        hist = []

        for epoch in tqdm(range(cfg.epochs_finetune), desc="stage5", unit="ep", position=0):
            self.model.train()
            losses = []
            bar = tqdm(self.train_loader, desc=f"  ep{epoch:02d} train", leave=False,
                       unit="batch", position=1)
            for batch in bar:
                batch = to_device(batch, self.device)
                loss, _ = self.model.diffusion_loss(batch)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                _clip(self.model, cfg.grad_clip)
                optim.step()
                losses.append(loss.item())
                bar.set_postfix(diff=f"{loss.item():.4f}")

            val = self._val_diffusion(self.val_loader, "loss")
            tr = float(np.mean(losses))
            self.log(f"  [stage5] epoch {epoch:02d} train={tr:.5f} val_loss={val:.5f}")
            hist.append({"epoch": epoch, "train": tr, "val": val})
            stopper(val, self.model, epoch)
            if stopper.early_stop:
                break
        stopper.load_best(self.model)
        self.history["stage5"] = hist

    # ================================================================= all
    def run_all(self):
        t0 = time.time()
        self.train_mean()
        self.train_scale()
        self.train_joint()
        self.train_diffusion()
        self.finetune_end_to_end()
        self.log(f"\n[trainer] all stages finished in {(time.time() - t0) / 60:.1f} min")
        return self.history
