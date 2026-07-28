"""DeepVAR baseline adapter.

An autoregressive LSTM producing a full multivariate Gaussian at every step
(Cholesky-parameterised covariance over the 3 targets), in the spirit of
GluonTS DeepVAR. The network consumes the previous target value plus the
exogenous conditions of the current step (weather + calendar; oracle future
weather at prediction time), unrolled with teacher forcing during training
and with sampled feedback at prediction time.

Output follows the shared prediction contract: samples [N,H,3,S] in raw units.

python -m src.baselines.deepvar_adapter --config configs/deepvar_heew.yaml --seeds 1
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.baselines.prediction_contract import PredictionShardWriter
from src.data.low_carbon_datamodule import LowCarbonDataModule
from src.data.low_carbon_schema import NUM_TARGETS
from src.experiments.low_carbon_prob_forecast import to_device
from src.utils.config import get_device, load_config, set_seed

EPS = 1e-6


class DeepVAR(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        data = cfg["data"]
        m = cfg.get("model", {})
        self.L = int(data["context_length"])
        self.H = int(data["prediction_length"])
        self.D = NUM_TARGETS
        self.condition_dim = int(m.get("condition_dim", 10))
        hidden = int(m.get("hidden_dim", 128))
        layers = int(m.get("num_layers", 2))
        dropout = float(m.get("dropout", 0.1))
        self.lstm = nn.LSTM(self.D + self.condition_dim, hidden, layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.mean_head = nn.Linear(hidden, self.D)
        self.diag_head = nn.Linear(hidden, self.D)
        n_lower = self.D * (self.D - 1) // 2
        self.lower_head = nn.Linear(hidden, n_lower)
        tril = torch.tril_indices(self.D, self.D, offset=-1)
        self.register_buffer("tril_rows", tril[0])
        self.register_buffer("tril_cols", tril[1])

    def _dist_params(self, h):
        """h [B,T,hidden] -> mu [B,T,D], scale_tril [B,T,D,D]."""
        mu = self.mean_head(h)
        diag = F.softplus(self.diag_head(h)) + EPS
        lower = self.lower_head(h)
        B, T, _ = mu.shape
        L = torch.zeros(B, T, self.D, self.D, device=h.device, dtype=h.dtype)
        L[..., torch.arange(self.D), torch.arange(self.D)] = diag
        L[..., self.tril_rows, self.tril_cols] = lower
        return mu, L

    def _inputs_teacher_forced(self, batch):
        """Concatenate history+future; input at step t is (y_{t-1}, cond_t)."""
        y = torch.cat([batch["history_target"], batch["future_target"]], dim=1)
        cond = torch.cat([batch["history_condition"], batch["future_condition"]], dim=1)
        y_prev = torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], dim=1)
        return torch.cat([y_prev, cond], dim=-1), y

    def loss(self, batch):
        """Negative log-likelihood, scored on the future H steps only."""
        x, y = self._inputs_teacher_forced(batch)
        h, _ = self.lstm(x)
        mu, Ltri = self._dist_params(h)
        dist = torch.distributions.MultivariateNormal(
            loc=mu[:, -self.H:], scale_tril=Ltri[:, -self.H:])
        return -dist.log_prob(y[:, -self.H:]).mean()

    @torch.no_grad()
    def sample(self, batch, num_samples, chunk_size, progress=False):
        """Autoregressive sampling -> [B,H,D,S] in model space."""
        hist_y = batch["history_target"]
        hist_c = batch["history_condition"]
        fut_c = batch["future_condition"]
        B = hist_y.size(0)
        chunks = []
        rng = range(0, num_samples, chunk_size)
        it = tqdm(rng, desc="deepvar sampling", ncols=100, leave=False) if progress else rng
        for start in it:
            cur = min(chunk_size, num_samples - start)
            y_r = hist_y.repeat_interleave(cur, dim=0)
            c_r = hist_c.repeat_interleave(cur, dim=0)
            f_r = fut_c.repeat_interleave(cur, dim=0)
            y_prev = torch.cat([torch.zeros_like(y_r[:, :1]), y_r[:, :-1]], dim=1)
            _, state = self.lstm(torch.cat([y_prev, c_r], dim=-1))
            last = y_r[:, -1]
            outs = []
            for t in range(self.H):
                x_t = torch.cat([last, f_r[:, t]], dim=-1).unsqueeze(1)
                h, state = self.lstm(x_t, state)
                mu, Ltri = self._dist_params(h)
                dist = torch.distributions.MultivariateNormal(
                    loc=mu[:, 0], scale_tril=Ltri[:, 0])
                last = dist.sample()
                outs.append(last)
            traj = torch.stack(outs, dim=1)  # [B*cur,H,D]
            chunks.append(traj.reshape(B, cur, self.H, self.D)
                          .permute(0, 2, 3, 1).cpu())
        return torch.cat(chunks, dim=-1)


def run_seed(cfg, seed, device, artifacts_root="artifacts", skip_train=False):
    set_seed(seed)
    dm = LowCarbonDataModule(
        cfg.get("data", {}).get("artifacts_dir", os.path.join(artifacts_root, "data", "heew")),
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"].get("num_workers", 4)),
        test_stride=int(cfg.get("evaluation", {}).get("test_stride", 1)),
    )
    model = DeepVAR(cfg, device).to(device)
    tr = cfg["training"]
    run_dir = os.path.join(artifacts_root, "runs", "deepvar", f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best_checkpoint.pt")

    if skip_train:
        if not os.path.exists(best_path):
            raise SystemExit(f"checkpoint not found: {best_path}")
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=False)["model"])
        print(f"loaded {best_path}, exporting test predictions only")
        export(cfg, model, dm, seed, device, artifacts_root)
        return

    opt = torch.optim.Adam(model.parameters(), lr=float(tr["learning_rate"]))
    best = float("inf")
    patience = int(tr.get("patience", 5))
    bad = 0
    for epoch in range(int(tr.get("epochs", 40))):
        model.train()
        losses = []
        bar = tqdm(dm.train_loader(), desc=f"[deepvar] epoch {epoch+1}", ncols=110)
        for batch in bar:
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            bar.set_postfix(nll=f"{np.mean(losses):.4f}")
        model.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for batch in tqdm(dm.val_loader(), desc="  val", ncols=110, leave=False):
                batch = to_device(batch, device)
                tot += model.loss(batch).item() * batch["future_target"].size(0)
                n += batch["future_target"].size(0)
        val = tot / max(n, 1)
        print(f"[deepvar] epoch {epoch+1} val NLL {val:.5f}")
        if val < best:
            best = val
            bad = 0
            torch.save({"model": model.state_dict()}, best_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"[deepvar] early stop at epoch {epoch+1} (best {best:.5f})")
                break
    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=False)["model"])
    export(cfg, model, dm, seed, device, artifacts_root)


@torch.no_grad()
def export(cfg, model, dm, seed, device, artifacts_root):
    ev = cfg.get("evaluation", {})
    num_samples = int(ev.get("test_num_samples", 1000))
    chunk = int(cfg.get("sampling", {}).get("chunk_size", 50))
    shard_windows = int(ev.get("shard_windows", 256))
    transform = dm.target_transform
    pred_dir = os.path.join(artifacts_root, "predictions", "deepvar", f"seed_{seed}")
    writer = PredictionShardWriter(pred_dir, {
        "model": "deepvar", "seed": seed,
        "context_length": model.L, "prediction_length": model.H,
        "num_samples": num_samples, "future_weather_mode": "oracle_observed",
        "config_hash": cfg.get("_config_hash", ""),
    })
    model.eval()
    buf = {"s": [], "t": [], "ts": [], "fsi": []}
    n_buf = 0
    for batch in tqdm(dm.test_loader(), desc="[deepvar test]", ncols=110):
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
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--skip_train", action="store_true", default=False)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device(args.device, args.gpu)
    print(f"device: {device}")
    for seed in args.seeds:
        print(f"===== DeepVAR seed {seed} =====")
        run_seed(cfg, seed, device, skip_train=args.skip_train)


if __name__ == "__main__":
    main()
