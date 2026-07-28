"""WaveStitch-style baseline adapter (spec section 13).

Forecasting is mapped to conditional completion of the length L+H target
sequence: rows [0, L) observed, rows [L, L+H) to generate.  Exogenous
conditions (weather + calendar, incl. oracle future weather) are visible over
the WHOLE sequence; the future target is never an input.  Sampling uses
RePaint-style inpainting: at every reverse step the observed region is
re-imposed via q_sample of the known history.

The information set is identical to all other models (spec 13.2).
See third_party/VERSIONS.md for provenance notes.

python -m src.baselines.wavestitch_adapter --config configs/wavestitch_low_carbon.yaml --seeds 1 2 3
"""
import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.baselines.prediction_contract import PredictionShardWriter
from src.data.low_carbon_datamodule import LowCarbonDataModule
from src.data.low_carbon_schema import NUM_TARGETS
from src.experiments.low_carbon_prob_forecast import to_device
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
        return self.mlp(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))


class ConvBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(8, ch)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.norm(self.conv(x)))


class SequenceDenoiser(nn.Module):
    """Dilated temporal ConvNet over [B, L+H, D] conditioned on the full
    condition sequence, the observed-target channel and the timestep."""

    def __init__(self, enc_in, condition_dim, hidden=128, n_blocks=6):
        super().__init__()
        in_ch = enc_in * 2 + condition_dim + 1  # y_t, observed target, cond, mask
        self.inp = nn.Conv1d(in_ch, hidden, kernel_size=1)
        self.t_embed = TimestepEmbedding(hidden)
        self.blocks = nn.ModuleList(
            [ConvBlock(hidden, 2 ** (i % 4)) for i in range(n_blocks)]
        )
        self.out = nn.Conv1d(hidden, enc_in, kernel_size=1)

    def forward(self, y_t, observed_target, condition, target_mask, t):
        x = torch.cat([y_t, observed_target, condition, target_mask], dim=-1)
        h = self.inp(x.permute(0, 2, 1))
        h = h + self.t_embed(t)[:, :, None]
        for blk in self.blocks:
            h = blk(h)
        return self.out(h).permute(0, 2, 1)


class WaveStitchAdapter(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        data = cfg["data"]
        self.L = int(data["context_length"])
        self.H = int(data["prediction_length"])
        self.enc_in = NUM_TARGETS
        d = cfg["diffusion"]
        self.steps = int(d.get("steps", 50))
        betas = torch.linspace(float(d.get("beta_start", 1e-4)),
                               float(d.get("beta_end", 0.05)), self.steps).to(device)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.denoiser = SequenceDenoiser(
            self.enc_in, int(cfg["model"].get("condition_dim", 11)),
            hidden=int(cfg["model"].get("hidden_dim", 128)),
            n_blocks=int(cfg["model"].get("n_blocks", 6)),
        )

    def _inputs(self, batch):
        full_target = torch.cat([batch["history_target"], batch["future_target"]], dim=1)
        condition = torch.cat([batch["history_condition"], batch["future_condition"]], dim=1)
        B = full_target.size(0)
        mask = torch.zeros(B, self.L + self.H, 1, device=full_target.device)
        mask[:, : self.L] = 1.0
        # observed conditioning channel: history only, future zeroed
        observed_target = full_target * mask
        return full_target, observed_target, condition, mask

    def loss(self, batch):
        full_target, observed_target, condition, mask = self._inputs(batch)
        B = full_target.size(0)
        t = torch.randint(0, self.steps, (B,), device=full_target.device)
        a_bar = self.alphas_cumprod[t][:, None, None]
        e = torch.randn_like(full_target)
        y_t = a_bar.sqrt() * full_target + (1 - a_bar).sqrt() * e
        e_pred = self.denoiser(y_t, observed_target, condition, mask, t)
        # only the unobserved (future) region is scored — the future target
        # must never leak through the loss on observed rows
        gen_region = 1.0 - mask
        return ((e - e_pred).square() * gen_region).sum() / gen_region.sum().clamp_min(1) / self.enc_in

    @torch.no_grad()
    def sample(self, batch, num_samples, chunk_size, progress=False):
        full_target, observed_target, condition, mask = self._inputs(batch)
        history = observed_target  # future part already zeroed
        B = full_target.size(0)
        chunks = []
        rng = range(0, num_samples, chunk_size)
        it = tqdm(rng, desc="wavestitch sampling", ncols=100, leave=False) if progress else rng
        for start in it:
            cur = min(chunk_size, num_samples - start)
            hist_r = history.repeat_interleave(cur, dim=0)
            cond_r = condition.repeat_interleave(cur, dim=0)
            mask_r = mask.repeat_interleave(cur, dim=0)
            y = torch.randn_like(hist_r)
            for ti in reversed(range(self.steps)):
                t = torch.full((y.size(0),), ti, device=y.device, dtype=torch.long)
                a = self.alphas[ti]
                a_bar = self.alphas_cumprod[ti]
                # RePaint: impose the known history at the current noise level
                e_known = torch.randn_like(hist_r)
                y_known = a_bar.sqrt() * hist_r + (1 - a_bar).sqrt() * e_known
                y = mask_r * y_known + (1 - mask_r) * y
                e_pred = self.denoiser(y, hist_r, cond_r, mask_r, t)
                coef = (1 - a) / (1 - a_bar).sqrt().clamp_min(EPS)
                mean = (y - coef * e_pred) / a.sqrt()
                if ti > 0:
                    a_bar_prev = self.alphas_cumprod[ti - 1]
                    var = self.betas[ti] * (1 - a_bar_prev) / (1 - a_bar)
                    y = mean + var.sqrt() * torch.randn_like(y)
                else:
                    y = mean
            future = y[:, -self.H:, :]
            chunks.append(future.reshape(B, cur, self.H, self.enc_in)
                          .permute(0, 2, 3, 1).cpu())
        return torch.cat(chunks, dim=-1)  # [B,H,D,S]


def run_seed(cfg, seed, device, artifacts_root="artifacts", skip_train=False):
    set_seed(seed)
    dm = LowCarbonDataModule(
        cfg.get("data", {}).get("artifacts_dir", os.path.join(artifacts_root, "data", "heew")),
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"].get("num_workers", 4)),
        test_stride=int(cfg.get("evaluation", {}).get("test_stride", 1)),
    )
    model = WaveStitchAdapter(cfg, device).to(device)
    tr = cfg["training"]
    run_dir = os.path.join(artifacts_root, "runs", "wavestitch", f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best_checkpoint.pt")

    if skip_train:
        if not os.path.exists(best_path):
            raise SystemExit(f"checkpoint not found: {best_path}")
        model.denoiser.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=False)["denoiser"])
        print(f"loaded {best_path}, exporting test predictions only")
        export(cfg, model, dm, seed, device, artifacts_root)
        return

    opt = torch.optim.Adam(model.parameters(), lr=float(tr["learning_rate"]))
    best = float("inf")
    for epoch in range(int(tr.get("epochs", 40))):
        model.train()
        losses = []
        bar = tqdm(dm.train_loader(), desc=f"[wavestitch] epoch {epoch+1}", ncols=110)
        for batch in bar:
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            bar.set_postfix(loss=f"{np.mean(losses):.5f}")
        model.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for batch in tqdm(dm.val_loader(), desc="  val", ncols=110, leave=False):
                batch = to_device(batch, device)
                tot += model.loss(batch).item() * batch["future_target"].size(0)
                n += batch["future_target"].size(0)
        val = tot / max(n, 1)
        print(f"[wavestitch] epoch {epoch+1} val {val:.5f}")
        if val < best:
            best = val
            torch.save({"denoiser": model.denoiser.state_dict()}, best_path)
    model.denoiser.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=False)["denoiser"])
    export(cfg, model, dm, seed, device, artifacts_root)


@torch.no_grad()
def export(cfg, model, dm, seed, device, artifacts_root):
    ev = cfg.get("evaluation", {})
    num_samples = int(ev.get("test_num_samples", 1000))
    chunk = int(cfg.get("sampling", {}).get("chunk_size", 20))
    shard_windows = int(ev.get("shard_windows", 256))
    transform = dm.target_transform
    pred_dir = os.path.join(artifacts_root, "predictions", "wavestitch", f"seed_{seed}")
    writer = PredictionShardWriter(pred_dir, {
        "model": "wavestitch", "seed": seed,
        "context_length": model.L, "prediction_length": model.H,
        "num_samples": num_samples, "future_weather_mode": "oracle_observed",
        "config_hash": cfg.get("_config_hash", ""),
    })
    model.eval()
    buf = {"s": [], "t": [], "ts": [], "fsi": []}
    n_buf = 0
    for batch in tqdm(dm.test_loader(), desc="[wavestitch test]", ncols=110):
        batch = to_device(batch, device)
        samples_model = model.sample(batch, num_samples, chunk, progress=True)
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
        print(f"===== WaveStitch seed {seed} =====")
        run_seed(cfg, seed, device, skip_train=args.skip_train)


if __name__ == "__main__":
    main()
