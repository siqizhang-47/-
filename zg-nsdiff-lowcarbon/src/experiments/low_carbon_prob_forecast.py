"""Shared training / validation / test driver for NsDiff-Exo and ZG-NsDiff
(spec section 9).

Per epoch: train + validate (mean normalized CRPS), keep best checkpoint.
After training: load best, test ONCE with the full sample budget, write the
unified prediction shards.  The test set is never used for model selection.
"""
import csv
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from src.baselines.prediction_contract import PredictionShardWriter
from src.data.low_carbon_datamodule import LowCarbonDataModule
from src.data.low_carbon_schema import ZERO_TARGET_INDICES, TARGET_NAMES
from src.evaluation.empirical_crps import crps_samples_torch
from src.models.NsDiffExo import NsDiffExo
from src.models.ZGNsDiff import ZGNsDiff, apply_gate_and_inverse, build_magnitude_mask
from src.utils.config import rng_state_dict, set_seed

EPS = 1e-8


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


class LowCarbonNsDiffTrainer:
    """variant: 'nsdiff' (baseline) or 'zg' (ZG-NsDiff, incl. ablations)."""

    def __init__(self, cfg: dict, variant: str, seed: int, device,
                 artifacts_root: str = "artifacts"):
        assert variant in ("nsdiff", "zg")
        self.cfg = cfg
        self.variant = variant
        self.seed = seed
        self.device = device
        self.artifacts_root = artifacts_root

        zg_cfg = cfg.get("zg", {})
        self.use_gate = bool(zg_cfg.get("use_gate", variant == "zg"))
        self.use_mask = bool(zg_cfg.get("use_mask", variant == "zg"))
        self.bernoulli_sampling = bool(
            cfg.get("sampling", {}).get("bernoulli_gate", variant == "zg")
        )
        self.model_name = cfg.get("model_name", "zg_nsdiff" if variant == "zg" else "nsdiff")

        set_seed(seed)
        self.dm = LowCarbonDataModule(
            cfg.get("data", {}).get("artifacts_dir", os.path.join(artifacts_root, "data", "low_carbon")),
            batch_size=int(cfg["training"]["batch_size"]),
            num_workers=int(cfg["training"].get("num_workers", 4)),
        )
        cfg["_gate_pos_weight"] = self.dm.stats["gate_pos_weight"]
        model_cls = ZGNsDiff if (variant == "zg" or self.use_gate) else NsDiffExo
        self.model = model_cls(cfg, device).to(device)
        self.transform = self.dm.target_transform
        self.train_scale = torch.tensor(self.dm.train_scale_raw, dtype=torch.float64)

        tr = cfg["training"]
        self.lr = float(tr["learning_rate"])
        self.clip = float(tr.get("gradient_clip_norm", 1.0))
        self.amp = bool(tr.get("amp", True)) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.loss_cfg = cfg.get("loss", {})

        ev = cfg.get("evaluation", {})
        self.val_num_samples = int(ev.get("validation_num_samples", 100))
        self.test_num_samples = int(ev.get("test_num_samples", 1000))
        self.chunk_size = int(cfg.get("sampling", {}).get("chunk_size", 20))
        self.val_max_batches = ev.get("val_max_batches")  # optional runtime cap
        self.shard_windows = int(ev.get("shard_windows", 256))

        self.run_dir = os.path.join(artifacts_root, "runs", self.model_name, f"seed_{seed}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.best_path = os.path.join(self.run_dir, "best_checkpoint.pt")

    # ------------------------------------------------------------- helpers
    def continuous_mask(self, batch):
        if self.use_mask:
            return build_magnitude_mask(batch["future_observed"], batch["future_active"])
        return batch["future_observed"].clone()

    def new_optimizer(self, params):
        name = self.cfg["training"].get("optimizer", "Adam")
        return getattr(torch.optim, name)(params, lr=self.lr)

    def _atomic_save(self, path, extra=None):
        state = {
            "mean_model": self.model.mean_model.state_dict(),
            "variance_model": self.model.variance_model.state_dict(),
            "denoiser": self.model.denoiser.state_dict(),
            "config": {k: v for k, v in self.cfg.items() if not k.startswith("_")},
            "preprocess_stats": self.dm.stats,
            "seed": self.seed,
        }
        if hasattr(self.model, "occurrence_head"):
            state["occurrence_head"] = self.model.occurrence_head.state_dict()
        state.update(rng_state_dict())
        if extra:
            state.update(extra)
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path, modules=("mean_model", "variance_model", "denoiser", "occurrence_head")):
        state = torch.load(path, map_location=self.device, weights_only=False)
        for name in modules:
            if name in state and hasattr(self.model, name):
                getattr(self.model, name).load_state_dict(state[name])
        return state

    # -------------------------------------------------------- pretrain F(+gate)
    def pretrain_f(self):
        epochs = int(self.cfg["training"].get("pretrain_f_epochs", 20))
        params = list(self.model.mean_model.parameters())
        if self.use_gate:
            params += list(self.model.occurrence_head.parameters())
        optim = self.new_optimizer(params)
        best = float("inf")
        path = os.path.join(self.run_dir, "pretrain_f.pt")
        for epoch in range(epochs):
            self.model.train()
            losses = []
            bar = tqdm(self.dm.train_loader(), desc=f"[F{'+gate' if self.use_gate else ''}] epoch {epoch+1}/{epochs}", ncols=110)
            for batch in bar:
                batch = to_device(batch, self.device)
                optim.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    loss = self._f_loss(batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, self.clip)
                self.scaler.step(optim)
                self.scaler.update()
                losses.append(loss.item())
                bar.set_postfix(loss=f"{np.mean(losses):.5f}")
            val_loss = self._f_val()
            print(f"  [F] epoch {epoch+1}: train {np.mean(losses):.5f} val {val_loss:.5f}")
            if val_loss < best:
                best = val_loss
                self._atomic_save(path, {"epoch": epoch, "best_validation_metric": best,
                                         "stage": "pretrain_f"})
        return path

    def _f_loss(self, batch):
        from src.layer.masked_reduction import masked_mean
        mu, future_hidden = self.model.forward_mean(batch)
        mask = self.continuous_mask(batch)
        loss = masked_mean((mu - batch["future_target"]).square(), mask)
        if self.use_gate:
            gate_logits = self.model.gate_logits_from_hidden(future_hidden)
            gate_observed = batch["future_observed"][..., ZERO_TARGET_INDICES]
            loss = loss + float(self.loss_cfg.get("lambda_gate", 0.5)) * self.model.gate_loss(
                gate_logits, batch["future_active"], gate_observed
            )
        return loss

    @torch.no_grad()
    def _f_val(self):
        self.model.eval()
        tot, n = 0.0, 0
        for batch in tqdm(self.dm.val_loader(), desc="  [F] val", ncols=110, leave=False):
            batch = to_device(batch, self.device)
            tot += self._f_loss(batch).item() * batch["future_target"].size(0)
            n += batch["future_target"].size(0)
        return tot / max(n, 1)

    # ------------------------------------------------------------ pretrain G
    def pretrain_g(self):
        from src.layer.masked_reduction import masked_mean
        from src.utils.sigma import wv_sigma_trailing
        epochs = int(self.cfg["training"].get("pretrain_g_epochs", 15))
        params = list(self.model.variance_model.parameters())
        optim = self.new_optimizer(params)
        best = float("inf")
        path = os.path.join(self.run_dir, "pretrain_g.pt")

        def g_loss(batch):
            y_sigma = wv_sigma_trailing(
                torch.cat([batch["history_target"], batch["future_target"]], dim=1),
                self.model.rolling_length,
            )[:, -self.model.H:, :] + EPS
            gx = self.model.forward_variance(batch)
            elem = (torch.sqrt(gx.clamp_min(EPS)) - torch.sqrt(y_sigma.clamp_min(EPS))).square()
            return masked_mean(elem, self.continuous_mask(batch))

        for epoch in range(epochs):
            self.model.train()
            losses = []
            bar = tqdm(self.dm.train_loader(), desc=f"[G] epoch {epoch+1}/{epochs}", ncols=110)
            for batch in bar:
                batch = to_device(batch, self.device)
                optim.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    loss = g_loss(batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, self.clip)
                self.scaler.step(optim)
                self.scaler.update()
                losses.append(loss.item())
                bar.set_postfix(loss=f"{np.mean(losses):.5f}")
            self.model.eval()
            with torch.no_grad():
                tot, n = 0.0, 0
                for batch in tqdm(self.dm.val_loader(), desc="  [G] val", ncols=110, leave=False):
                    batch = to_device(batch, self.device)
                    tot += g_loss(batch).item() * batch["future_target"].size(0)
                    n += batch["future_target"].size(0)
                val_loss = tot / max(n, 1)
            print(f"  [G] epoch {epoch+1}: train {np.mean(losses):.5f} val {val_loss:.5f}")
            if val_loss < best:
                best = val_loss
                self._atomic_save(path, {"epoch": epoch, "best_validation_metric": best,
                                         "stage": "pretrain_g"})
        return path

    # ------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample_raw(self, batch, num_samples, generator=None, progress=False):
        """Model-space sampling -> raw kW samples [B,H,4,S] (+ gate_prob or None)."""
        magnitude = self.model.sample_trajectories(
            batch, num_samples, self.chunk_size, generator, progress=progress
        )  # cpu [B,H,4,S]
        gate_prob = None
        if self.use_gate:
            gate_prob = self.model.predict_gate_prob(batch).cpu()      # [B,H,3]
            if self.bernoulli_sampling:
                gates = ZGNsDiff.sample_gates(gate_prob, num_samples)  # [B,H,3,S]
            else:
                gates = (gate_prob > 0.5).float().unsqueeze(-1).expand(
                    *gate_prob.shape, num_samples
                )
            raw = apply_gate_and_inverse(magnitude, gates, self.transform)
        else:
            raw = self.transform.inverse_torch(magnitude, target_dim=2)
        return raw, gate_prob

    # ------------------------------------------------------------ validation
    @torch.no_grad()
    def validate_ncrps(self):
        """Mean normalized CRPS on the validation split (gated for ZG)."""
        self.model.eval()
        crps_sum = torch.zeros(len(TARGET_NAMES), dtype=torch.float64)
        count = 0
        loader = self.dm.val_loader()
        bar = tqdm(loader, desc="  val NCRPS", ncols=110, leave=False)
        for bi, batch in enumerate(bar):
            if self.val_max_batches is not None and bi >= int(self.val_max_batches):
                break
            batch = to_device(batch, self.device)
            raw, _ = self.sample_raw(batch, self.val_num_samples)
            truth = batch["future_target_raw"].cpu().double()   # [B,H,4]
            c = crps_samples_torch(raw.double(), truth)         # [B,H,4]
            crps_sum += c.sum(dim=(0, 1))
            count += c.shape[0] * c.shape[1]
        crps_var = crps_sum / max(count, 1)
        ncrps = (crps_var / self.train_scale).mean().item()
        return ncrps, {f"crps_{n}": crps_var[i].item() for i, n in enumerate(TARGET_NAMES)}

    # ------------------------------------------------------------------ joint
    def train_joint(self, load_pretrain=True):
        tr = self.cfg["training"]
        epochs = int(tr.get("joint_epochs", 50))
        patience = int(tr.get("patience", 10))
        if load_pretrain:
            f_path = os.path.join(self.run_dir, "pretrain_f.pt")
            g_path = os.path.join(self.run_dir, "pretrain_g.pt")
            if os.path.exists(f_path):
                self.load_checkpoint(f_path, modules=("mean_model", "occurrence_head"))
                print(f"loaded pretrain F: {f_path}")
            if os.path.exists(g_path):
                self.load_checkpoint(g_path, modules=("variance_model",))
                print(f"loaded pretrain G: {g_path}")

        params = list(self.model.parameters())
        optim = self.new_optimizer([
            {"params": self.model.mean_model.parameters()},
            {"params": self.model.variance_model.parameters()},
            {"params": self.model.denoiser.parameters()},
        ] + ([{"params": self.model.occurrence_head.parameters()}] if self.use_gate else []))

        best = float("inf")
        bad_epochs = 0
        t_train0 = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(epochs):
            set_seed(self.seed + epoch)  # resumable reproducibility as in original
            self.model.train()
            losses, sub_logs = [], {}
            bar = tqdm(self.dm.train_loader(),
                       desc=f"[joint:{self.model_name}] epoch {epoch+1}/{epochs}", ncols=110)
            for batch in bar:
                batch = to_device(batch, self.device)
                optim.zero_grad(set_to_none=True)
                mask = self.continuous_mask(batch)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    loss, logs, _ = self.model.loss(batch, mask, self.loss_cfg)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, self.clip)
                self.scaler.step(optim)
                self.scaler.update()
                losses.append(loss.item())
                for k, v in logs.items():
                    sub_logs.setdefault(k, []).append(v)
                bar.set_postfix(loss=f"{np.mean(losses):.4f}",
                                **{k: f"{np.mean(v):.4f}" for k, v in sub_logs.items()})

            self.model.eval()
            ncrps, per_var = self.validate_ncrps()
            print(f"epoch {epoch+1}: train_loss {np.mean(losses):.5f} val_NCRPS {ncrps:.5f} "
                  + " ".join(f"{k}={v:.4f}" for k, v in per_var.items()))
            if ncrps < best:
                best = ncrps
                bad_epochs = 0
                self._atomic_save(self.best_path, {
                    "epoch": epoch, "best_validation_metric": best, "stage": "joint",
                })
                print(f"  saved best checkpoint (val NCRPS {best:.5f})")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"early stopping after {patience} epochs without improvement")
                    break

        train_seconds = time.time() - t_train0
        peak_mem = (torch.cuda.max_memory_allocated(self.device) / 2**20
                    if self.device.type == "cuda" else 0.0)
        return {"training_seconds": train_seconds, "peak_gpu_memory_mb": peak_mem,
                "best_val_ncrps": best}

    # -------------------------------------------------------------------- test
    @torch.no_grad()
    def test_and_export(self, efficiency_extra=None):
        """Load best checkpoint, run the single final test pass, write shards."""
        self.load_checkpoint(self.best_path)
        self.model.eval()
        pred_dir = os.path.join(self.artifacts_root, "predictions",
                                self.model_name, f"seed_{self.seed}")
        writer = PredictionShardWriter(pred_dir, {
            "model": self.model_name,
            "seed": self.seed,
            "context_length": self.model.L,
            "prediction_length": self.model.H,
            "num_samples": self.test_num_samples,
            "future_weather_mode": "oracle_observed",
            "checkpoint": os.path.abspath(self.best_path),
            "config_hash": self.cfg.get("_config_hash", ""),
        })
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed * 1000 + 7)

        buf = {"samples": [], "truth": [], "timestamps": [], "fsi": [], "gate": []}
        n_buf = 0
        t0 = time.time()
        n_traj = 0
        loader = self.dm.test_loader()
        for batch in tqdm(loader, desc=f"[test:{self.model_name}]", ncols=110):
            batch = to_device(batch, self.device)
            raw, gate_prob = self.sample_raw(batch, self.test_num_samples, progress=True)
            buf["samples"].append(raw.float().numpy())
            buf["truth"].append(batch["future_target_raw"].cpu().numpy())
            buf["timestamps"].append(batch["timestamps"].cpu().numpy())
            buf["fsi"].append(batch["forecast_start_index"].cpu().numpy())
            if gate_prob is not None:
                buf["gate"].append(gate_prob.numpy())
            n_buf += raw.shape[0]
            n_traj += raw.shape[0] * self.test_num_samples
            if n_buf >= self.shard_windows:
                self._flush(writer, buf)
                n_buf = 0
        if buf["samples"]:
            self._flush(writer, buf)
        writer.close()
        sampling_seconds = time.time() - t0

        eff = {
            "model": self.model_name,
            "seed": self.seed,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "additional_gate_parameters": (
                sum(p.numel() for p in self.model.occurrence_head.parameters())
                if hasattr(self.model, "occurrence_head") else 0
            ),
            "checkpoint_size_mb": os.path.getsize(self.best_path) / 2**20,
            "sampling_seconds": sampling_seconds,
            "milliseconds_per_trajectory": 1000.0 * sampling_seconds / max(n_traj, 1),
            "samples_per_second": n_traj / max(sampling_seconds, 1e-9),
        }
        if efficiency_extra:
            eff.update(efficiency_extra)
        self._append_efficiency(eff)
        print(f"predictions written to {pred_dir}")
        return pred_dir

    def _flush(self, writer, buf):
        writer.write(
            np.concatenate(buf["samples"]),
            np.concatenate(buf["truth"]),
            np.concatenate(buf["timestamps"]),
            np.concatenate(buf["fsi"]),
            np.concatenate(buf["gate"]) if buf["gate"] else None,
        )
        for k in buf:
            buf[k] = []

    def _append_efficiency(self, row):
        os.makedirs("results", exist_ok=True)
        path = os.path.join("results", "efficiency.csv")
        exists = os.path.exists(path)
        fields = ["model", "seed", "total_parameters", "trainable_parameters",
                  "additional_gate_parameters", "checkpoint_size_mb",
                  "training_seconds", "peak_gpu_memory_mb", "sampling_seconds",
                  "milliseconds_per_trajectory", "samples_per_second", "best_val_ncrps"]
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerow(row)
