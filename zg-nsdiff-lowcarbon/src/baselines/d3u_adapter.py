"""D3U-style baseline adapter (spec section 12).

Deterministic-forecast + residual-diffusion decomposition:
  * deterministic module: the same exogenous NS-Transformer mean backbone,
    conditioned on past target + past weather + oracle future weather/calendar;
  * residual diffusion: a standard DDPM over the residual in model space,
    conditioned on the deterministic forecast and the oracle future conditions.

samples = deterministic_forecast + residual_samples   (spec 12.3)

This is a faithful self-contained re-implementation of the D3U idea
(deterministic backbone + diffusion over the point-forecast residual);
see third_party/VERSIONS.md for provenance notes.

python -m src.baselines.d3u_adapter --config configs/d3u_low_carbon.yaml --seeds 1 2 3
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import src.layer.mu_backbone_exo as mu_backbone_exo
from src.baselines.prediction_contract import PredictionShardWriter
from src.data.low_carbon_datamodule import LowCarbonDataModule
from src.experiments.low_carbon_prob_forecast import to_device
from src.models.NsDiffExo import NsDiffExo
from src.utils.config import get_device, load_config, set_seed

EPS = 1e-8


class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        ang = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.mlp(emb)


class ResidualDenoiser(nn.Module):
    """epsilon net over residual sequences [B,H,D], conditioned on the
    deterministic forecast and the oracle future conditions."""

    def __init__(self, enc_in, condition_dim, hidden_dim=128):
        super().__init__()
        self.t_embed = TimestepEmbedding(hidden_dim)
        in_dim = enc_in * 2 + condition_dim  # r_t, det_forecast, future_condition
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.block1 = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.block2 = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, enc_in)

    def forward(self, r_t, det_forecast, future_condition, t):
        h = self.inp(torch.cat([r_t, det_forecast, future_condition], dim=-1))
        temb = self.t_embed(t)[:, None, :]
        h = h + temb
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.out(h)


class D3UAdapter(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        base = NsDiffExo(cfg, device)  # reuse arg plumbing for the mean backbone
        self.mean_model = base.mean_model
        self.L, self.H, self.enc_in = base.L, base.H, base.enc_in
        self.label_len = base.label_len
        self._decoder_inputs = base.decoder_inputs
        d = cfg["diffusion"]
        self.steps = int(d.get("steps", 50))
        betas = torch.linspace(float(d.get("beta_start", 1e-4)),
                               float(d.get("beta_end", 0.05)), self.steps).to(device)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.denoiser = ResidualDenoiser(
            self.enc_in, int(cfg["model"].get("condition_dim", 11)),
            hidden_dim=int(cfg["model"].get("denoise_hidden_dim", 128)),
        )

    def deterministic(self, batch):
        dec_t, dec_c = self._decoder_inputs(batch)
        mu, _ = self.mean_model(batch["history_target"], batch["history_condition"],
                                dec_t, dec_c)
        return mu

    def diffusion_loss(self, batch, det):
        r0 = batch["future_target"] - det.detach()
        B = r0.size(0)
        t = torch.randint(0, self.steps, (B,), device=r0.device)
        a_bar = self.alphas_cumprod[t][:, None, None]
        e = torch.randn_like(r0)
        r_t = a_bar.sqrt() * r0 + (1 - a_bar).sqrt() * e
        e_pred = self.denoiser(r_t, det.detach(), batch["future_condition"], t)
        return (e - e_pred).square().mean()

    @torch.no_grad()
    def sample_residuals(self, batch, det, num_samples, chunk_size, progress=False):
        B = det.size(0)
        chunks = []
        rng = range(0, num_samples, chunk_size)
        it = tqdm(rng, desc="d3u sampling", ncols=100, leave=False) if progress else rng
        for start in it:
            cur = min(chunk_size, num_samples - start)
            det_r = det.repeat_interleave(cur, dim=0)
            cond_r = batch["future_condition"].repeat_interleave(cur, dim=0)
            r = torch.randn_like(det_r)
            for ti in reversed(range(self.steps)):
                t = torch.full((r.size(0),), ti, device=r.device, dtype=torch.long)
                a = self.alphas[ti]
                a_bar = self.alphas_cumprod[ti]
                e_pred = self.denoiser(r, det_r, cond_r, t)
                coef = (1 - a) / (1 - a_bar).sqrt().clamp_min(EPS)
                mean = (r - coef * e_pred) / a.sqrt()
                if ti > 0:
                    a_bar_prev = self.alphas_cumprod[ti - 1]
                    var = self.betas[ti] * (1 - a_bar_prev) / (1 - a_bar)
                    r = mean + var.sqrt() * torch.randn_like(r)
                else:
                    r = mean
            chunks.append(r.reshape(B, cur, self.H, self.enc_in).permute(0, 2, 3, 1).cpu())
        return torch.cat(chunks, dim=-1)  # [B,H,D,S]


def run_seed(cfg, seed, device, artifacts_root="artifacts", skip_train=False):
    set_seed(seed)
    dm = LowCarbonDataModule(
        cfg.get("data", {}).get("artifacts_dir", os.path.join(artifacts_root, "data", "low_carbon")),
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"].get("num_workers", 4)),
        test_stride=int(cfg.get("evaluation", {}).get("test_stride", 1)),
    )
    model = D3UAdapter(cfg, device).to(device)
    tr = cfg["training"]
    ev = cfg.get("evaluation", {})
    run_dir = os.path.join(artifacts_root, "runs", "d3u", f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best_checkpoint.pt")

    if skip_train:
        if not os.path.exists(best_path):
            raise SystemExit(f"checkpoint not found: {best_path}")
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.mean_model.load_state_dict(state["mean_model"])
        if "denoiser" not in state:
            raise SystemExit(f"{best_path} has no trained denoiser — stage 2 never ran")
        model.denoiser.load_state_dict(state["denoiser"])
        print(f"loaded {best_path}, exporting test predictions only")
        export(cfg, model, dm, seed, device, artifacts_root)
        return

    # ---- stage 1: deterministic forecaster --------------------------------
    opt = torch.optim.Adam(model.mean_model.parameters(), lr=float(tr["learning_rate"]))
    best = float("inf")
    for epoch in range(int(tr.get("det_epochs", 20))):
        model.train()
        losses = []
        bar = tqdm(dm.train_loader(), desc=f"[d3u det] epoch {epoch+1}", ncols=110)
        for batch in bar:
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            det = model.deterministic(batch)
            mask = batch["future_observed"].float()
            loss = ((det - batch["future_target"]).square() * mask).sum() / mask.sum().clamp_min(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.mean_model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            bar.set_postfix(loss=f"{np.mean(losses):.5f}")
        model.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for batch in tqdm(dm.val_loader(), desc="  val", ncols=110, leave=False):
                batch = to_device(batch, device)
                det = model.deterministic(batch)
                mask = batch["future_observed"].float()
                tot += (((det - batch["future_target"]).square() * mask).sum()
                        / mask.sum().clamp_min(1)).item() * batch["future_target"].size(0)
                n += batch["future_target"].size(0)
        val = tot / max(n, 1)
        print(f"[d3u det] epoch {epoch+1} val {val:.5f}")
        if val < best:
            best = val
            torch.save({"mean_model": model.mean_model.state_dict()}, best_path)
    model.mean_model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=False)["mean_model"])

    # ---- stage 2: residual diffusion ---------------------------------------
    opt = torch.optim.Adam(model.denoiser.parameters(), lr=float(tr["learning_rate"]))
    best = float("inf")
    for epoch in range(int(tr.get("diffusion_epochs", 30))):
        model.train()
        losses = []
        bar = tqdm(dm.train_loader(), desc=f"[d3u diff] epoch {epoch+1}", ncols=110)
        for batch in bar:
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                det = model.deterministic(batch)
            loss = model.diffusion_loss(batch, det)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            bar.set_postfix(loss=f"{np.mean(losses):.5f}")
        model.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for batch in tqdm(dm.val_loader(), desc="  val", ncols=110, leave=False):
                batch = to_device(batch, device)
                det = model.deterministic(batch)
                tot += model.diffusion_loss(batch, det).item() * batch["future_target"].size(0)
                n += batch["future_target"].size(0)
        val = tot / max(n, 1)
        print(f"[d3u diff] epoch {epoch+1} val {val:.5f}")
        if val < best:
            best = val
            torch.save({"mean_model": model.mean_model.state_dict(),
                        "denoiser": model.denoiser.state_dict()}, best_path)
    state = torch.load(best_path, map_location=device, weights_only=False)
    model.mean_model.load_state_dict(state["mean_model"])
    model.denoiser.load_state_dict(state["denoiser"])

    # ---- final test export ---------------------------------------------------
    export(cfg, model, dm, seed, device, artifacts_root)


@torch.no_grad()
def export(cfg, model, dm, seed, device, artifacts_root):
    ev = cfg.get("evaluation", {})
    num_samples = int(ev.get("test_num_samples", 1000))
    chunk = int(cfg.get("sampling", {}).get("chunk_size", 20))
    shard_windows = int(ev.get("shard_windows", 256))
    transform = dm.target_transform
    pred_dir = os.path.join(artifacts_root, "predictions", "d3u", f"seed_{seed}")
    writer = PredictionShardWriter(pred_dir, {
        "model": "d3u", "seed": seed,
        "context_length": model.L, "prediction_length": model.H,
        "num_samples": num_samples, "future_weather_mode": "oracle_observed",
        "config_hash": cfg.get("_config_hash", ""),
    })
    model.eval()
    buf = {"s": [], "t": [], "ts": [], "fsi": []}
    n_buf = 0
    for batch in tqdm(dm.test_loader(), desc="[d3u test]", ncols=110):
        batch = to_device(batch, device)
        det = model.deterministic(batch)
        res = model.sample_residuals(batch, det, num_samples, chunk, progress=True)
        samples_model = det.unsqueeze(-1).cpu() + res            # [B,H,D,S]
        raw = transform.inverse_torch(samples_model, target_dim=2)
        buf["s"].append(raw.float().numpy())
        buf["t"].append(batch["future_target_raw"].cpu().numpy())
        buf["ts"].append(batch["timestamps"].cpu().numpy())
        buf["fsi"].append(batch["forecast_start_index"].cpu().numpy())
        n_buf += raw.shape[0]
        if n_buf >= shard_windows:
            writer.write(np.concatenate(buf["s"]), np.concatenate(buf["t"]),
                         np.concatenate(buf["ts"]), np.concatenate(buf["fsi"]))
            buf = {"s": [], "t": [], "ts": [], "fsi": []}
            n_buf = 0
    if buf["s"]:
        writer.write(np.concatenate(buf["s"]), np.concatenate(buf["t"]),
                     np.concatenate(buf["ts"]), np.concatenate(buf["fsi"]))
    writer.close()
    print(f"predictions written to {pred_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--skip_train", action="store_true", default=False,
                    help="export test predictions from an existing checkpoint")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device(args.device, args.gpu)
    print(f"device: {device}")
    for seed in args.seeds:
        print(f"===== D3U seed {seed} =====")
        run_seed(cfg, seed, device, skip_train=args.skip_train)


if __name__ == "__main__":
    main()
