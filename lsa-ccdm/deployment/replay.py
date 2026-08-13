"""Adapter experiment replayer on cached frozen-backbone scenarios (Principle A).

All adapter-side experiments (proposed, baselines 4-8, ablations) share the
SAME cached scenarios; a full 3-year replay runs in minutes on CPU.

Strict no-look-ahead ordering per deployment day t (Principle B):
    1. build context from information <= t-1 only
    2. adapt scenarios (no_grad) and LOG METRICS
    3. push day-t truth into the context ("arrives next day")
    4. run adapter gradient updates on the buffer

Adapter optimization runs in the train-scaler-normalized space (well
conditioned); metrics are always computed in physical units. A positive
affine map commutes with the per-carrier standardization, so this is
equivalent to adapting in physical units with rescaled Delta.
"""
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from adapters.context import ResidualContext
from backbone.data_loader import TARGET_COLS, data_intervals
from evaluation import metrics as M

TUNE_CUTOFF = pd.Timestamp("2020-07-01")


class ScenarioCache:
    def __init__(self, cache_dir):
        self.dir = Path(cache_dir)
        if not self.dir.exists():
            raise FileNotFoundError(f"scenario cache not found: {self.dir}")

    def dates(self):
        return sorted(pd.Timestamp(p.stem) for p in self.dir.glob("*.npz"))

    def load(self, date):
        with np.load(self.dir / f"{pd.Timestamp(date).date()}.npz") as z:
            return z["scenarios"].astype(np.float32), z["y_true"].astype(np.float32)


class MetricsLogger:
    """Records daily metrics; the log() call MUST precede any same-day update."""

    def __init__(self, eval_dependence=False, corr_window_days=30):
        self.rows = []
        self.eval_dependence = eval_dependence
        self.resid_window = deque(maxlen=corr_window_days)

    def log(self, date, scen_frozen, scen_adapted, y_true, extra=None):
        row = {"date": pd.Timestamp(date)}
        row.update(M.daily_metrics(scen_adapted, y_true))
        row["frozen_crps"] = float(M.ensemble_crps(scen_frozen, y_true).mean())
        if self.eval_dependence and len(self.resid_window) >= 5:
            resid = np.concatenate(list(self.resid_window), axis=0)  # (days*H, C)
            for method in ("pearson", "spearman"):
                r_real = M.residual_corr(resid, method)
                r_gen = M.scenario_corr(scen_adapted, method)
                row[f"cme_{method}"] = M.cme(r_real, r_gen)
        if extra:
            row.update(extra)
        self.rows.append(row)
        # residual window uses information that becomes available only after
        # logging -- updated here, i.e. affecting FUTURE days only
        self.resid_window.append(y_true - scen_frozen.mean(axis=0))

    def to_frame(self):
        return pd.DataFrame(self.rows)


class ReplayRunner:
    def __init__(self, cfg, method, adapter, loss_fn, ctx_fn, cache_dir,
                 tune_mode=False, adapted_dir=None):
        """
        adapter: nn.Module or None (frozen)
        loss_fn(adapter, scen, y, ctx) -> scalar loss     (None for frozen)
        ctx_fn(context: ResidualContext) -> tensor passed to the adapter
        adapted_dir: if set, the ADAPTED scenarios of every deployment day are
            saved there as npz (needed by scripts/plot_model_figures.py)
        """
        self.cfg = cfg
        self.method = method
        self.adapter = adapter
        self.loss_fn = loss_fn
        self.ctx_fn = ctx_fn
        self.cache = ScenarioCache(cache_dir)
        self.tune_mode = tune_mode
        self.adapted_dir = Path(adapted_dir) if adapted_dir else None
        if self.adapted_dir:
            self.adapted_dir.mkdir(parents=True, exist_ok=True)

        self.K = cfg.get("K", 7)
        self.adapt_steps = cfg.get("adapt_steps", 5)
        self.adapt_every = cfg.get("adapt_every", 1)
        self.grad_clip = cfg.get("grad_clip", 0.1)
        self.context = ResidualContext(K=self.K)
        self.train_buffer = deque(maxlen=self.K)  # (scen_norm, y_norm) pairs
        self.logger = MetricsLogger(eval_dependence=cfg.get("eval_dependence", False))

        if adapter is not None:
            self.optimizer = torch.optim.Adam(
                [p for p in adapter.parameters() if p.requires_grad],
                lr=cfg.get("adapter_lr", 1e-3))
        else:
            self.optimizer = None

        # per-carrier standardization fitted on the TRAINING rows only
        df = pd.read_csv(cfg.data_csv)
        train_end = data_intervals[cfg.data_name][0]
        self.scaler = StandardScaler().fit(df[TARGET_COLS].values[:train_end])
        self.center = torch.tensor(self.scaler.mean_, dtype=torch.float32)
        self.scale = torch.tensor(self.scaler.scale_, dtype=torch.float32)

        self.w_c = torch.ones(4)
        self.param_track = []

    # ---------------- normalization helpers ----------------
    def _norm(self, a):
        return (torch.as_tensor(a, dtype=torch.float32) - self.center) / self.scale

    def _denorm_np(self, t):
        return (t * self.scale + self.center).numpy().astype(np.float32)

    # ---------------- cold start ----------------
    def prefill(self, warmup_dates):
        """Fill context/buffer with 2019 validation-tail days (legal info) and
        freeze the carrier weights w_c = 1 / warmup CRPS (dimension balance)."""
        crps_c = torch.zeros(4)
        for date in warmup_dates:
            scen, y = self.cache.load(date)
            scen_n, y_n = self._norm(scen), self._norm(y)
            self.context.push(y_n, scen_n.mean(0), scen_n.std(0))
            self.train_buffer.append((scen_n, y_n))
            crps_c += torch.tensor(
                M.ensemble_crps(scen_n.numpy(), y_n.numpy()).mean(axis=0))
        crps_c /= max(len(warmup_dates), 1)
        w = 1.0 / (crps_c + 1e-8)
        self.w_c = (w / w.mean()).detach()

    # ---------------- main loop ----------------
    def run(self, deploy_dates, desc=None):
        for day_idx, date in enumerate(tqdm(deploy_dates, desc=desc or self.method)):
            if self.tune_mode:
                assert pd.Timestamp(date) < TUNE_CUTOFF, \
                    f"tune mode must not load caches after {TUNE_CUTOFF.date()} (got {date})"
            scen, y_true = self.cache.load(date)
            scen_n = self._norm(scen)
            y_n = self._norm(y_true)

            # 1-2. adapt with <= t-1 information, then log BEFORE any update
            extra = {}
            if self.adapter is None:
                adapted = scen
            else:
                ctx = self.ctx_fn(self.context)
                with torch.no_grad():
                    adapted_n = self.adapter(scen_n, ctx)
                    if hasattr(self.adapter, "compute_params"):
                        delta, s = self.adapter.compute_params(ctx)
                        extra.update({f"s_{name}": float(s[c]) for c, name in
                                      enumerate(M.CARRIERS)})
                        extra.update({f"delta_{name}": float(delta[:, c].mean())
                                      for c, name in enumerate(M.CARRIERS)})
                adapted = self._denorm_np(adapted_n)
            self.logger.log(date, scen, adapted, y_true, extra=extra)
            if self.adapted_dir:
                np.savez_compressed(
                    self.adapted_dir / f"{pd.Timestamp(date).date()}.npz",
                    scenarios=adapted, y_true=y_true)

            # 3. day-t truth arrives (usable from t+1 onward)
            self.context.push(y_n, scen_n.mean(0), scen_n.std(0))
            self.train_buffer.append((scen_n, y_n))

            # 4. online update (ablation F: only every adapt_every days)
            if self.adapter is not None and (day_idx + 1) % self.adapt_every == 0:
                for _ in range(self.adapt_steps):
                    ctx_new = self.ctx_fn(self.context)
                    loss = torch.stack([
                        self.loss_fn(self.adapter, scen_d, y_d, ctx_new)
                        for scen_d, y_d in self.train_buffer]).mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.grad_clip)
                    self.optimizer.step()
        return self.logger.to_frame()

    # ---------------- outputs ----------------
    def save_results(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = self.logger.to_frame()
        df.to_parquet(out_dir / "daily_metrics.parquet", index=False)
        df["year"] = df["date"].dt.year
        num_cols = [c for c in df.columns if c not in ("date", "year")]
        yearly = df.groupby("year")[num_cols].mean()
        yearly.to_csv(out_dir / "yearly_metrics.csv")
        monthly = df.groupby(df["date"].dt.to_period("M"))[num_cols].mean()
        monthly.to_csv(out_dir / "monthly_metrics.csv")
        print(f"[{self.method}] results saved to {out_dir}")
        return df
