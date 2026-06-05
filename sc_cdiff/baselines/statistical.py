"""Statistical baselines (plan §8.4): Historical Sampling, Weather-kNN,
Gaussian Copula. Fast, fit on TRAIN only, same scenario format as SC-CDiff.

  historical : draw scenarios from real training days of the same calendar month
               (naturally non-negative, real structural zeros -- a strong baseline)
  weather_knn: nearest training days by daily predicted-weather features
  copula     : per-season Gaussian copula on the flattened daily vector
               (empirical marginals + Gaussian dependence)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from ..data.build_dataset import _load_cfg
from .common import load_full, month_of, run_and_save, season_of


def _train_day_pool(cfg):
    """Return raw train-day tensors and per-day month/season + daily weather feats."""
    npz, norm = load_full(cfg)
    Yfull = npz["Yfull"]; Wfull = npz["Wfull"]; FILL = npz["FILLEDfull"]
    train_end = int(npz["train_end_hour"])
    w_mu = np.asarray(norm.stats["W"][0], np.float32)
    w_sd = np.asarray(norm.stats["W"][1], np.float32)
    days, wfeat, starts = [], [], []
    for s in range(0, train_end - 24 + 1, 24):
        if FILL[s:s + 24].any():
            continue
        days.append(Yfull[s:s + 24].T)                       # [5,24]
        wn = (Wfull[s:s + 24] - w_mu) / w_sd                 # normalized weather [24,6]
        wfeat.append(wn.mean(axis=0))                        # daily feat [6]
        starts.append(s)
    Y = np.stack(days).astype(np.float32)                    # [Nd,5,24]
    return Y, np.stack(wfeat).astype(np.float32), np.array(starts)


class HistoricalSampler:
    def __init__(self, cfg):
        self.Y, _, self.starts = _train_day_pool(cfg)
        self.month = month_of(self.starts)

    def __call__(self, batch, n):
        m_test = month_of(batch["start"])
        B = len(m_test)
        out = np.empty((B, n, 5, 24), np.float32)
        for i, m in enumerate(m_test):
            pool = np.where(self.month == m)[0]
            if len(pool) == 0:
                pool = np.arange(len(self.Y))
            idx = np.random.choice(pool, size=n, replace=True)
            out[i] = self.Y[idx]
        return out


class WeatherKNN:
    def __init__(self, cfg, k_neighbors=150):
        self.Y, self.wfeat, _ = _train_day_pool(cfg)
        self.k = k_neighbors

    def __call__(self, batch, n):
        # test daily weather feature from the (normalized) predicted weather W
        Wt = batch["W"].cpu().numpy()                        # [B,6,24]
        feat = Wt.mean(axis=2)                               # [B,6]
        B = feat.shape[0]
        out = np.empty((B, n, 5, 24), np.float32)
        for i in range(B):
            d = np.linalg.norm(self.wfeat - feat[i][None, :], axis=1)
            nn = np.argsort(d)[: self.k]
            idx = np.random.choice(nn, size=n, replace=True)
            out[i] = self.Y[idx]
        return out


class GaussianCopula:
    """Per-season Gaussian copula over the flattened 120-dim daily vector."""

    def __init__(self, cfg):
        Y, _, starts = _train_day_pool(cfg)
        self.D = 5 * 24
        seas = season_of(starts)
        self.models = {}
        for s in range(4):
            X = Y[seas == s].reshape((seas == s).sum(), -1)   # [Ns,120]
            if X.shape[0] < 30:
                continue
            # empirical marginals: store sorted columns; Gaussian scores via rank
            order = np.argsort(X, axis=0)
            ranks = np.argsort(order, axis=0).astype(np.float64)
            U = (ranks + 0.5) / X.shape[0]
            from scipy.stats import norm as snorm
            Z = snorm.ppf(np.clip(U, 1e-4, 1 - 1e-4))         # [Ns,120]
            corr = np.corrcoef(Z, rowvar=False)
            corr = np.nan_to_num(corr) + 1e-3 * np.eye(self.D)
            self.models[s] = {"sorted": np.sort(X, axis=0), "chol": np.linalg.cholesky(corr)}

    def _sample_season(self, s, n):
        from scipy.stats import norm as snorm
        mdl = self.models.get(s) or next(iter(self.models.values()))
        z = np.random.randn(n, self.D) @ mdl["chol"].T
        u = np.clip(snorm.cdf(z), 1e-4, 1 - 1e-4)
        sorted_cols = mdl["sorted"]
        Ns = sorted_cols.shape[0]
        pos = u * (Ns - 1)
        lo = np.floor(pos).astype(int); hi = np.minimum(lo + 1, Ns - 1)
        frac = pos - lo
        out = np.empty((n, self.D), np.float32)
        for j in range(self.D):
            out[:, j] = sorted_cols[lo[:, j], j] * (1 - frac[:, j]) + sorted_cols[hi[:, j], j] * frac[:, j]
        return out.reshape(n, 5, 24)

    def __call__(self, batch, n):
        seas = season_of(batch["start"])
        B = len(seas)
        out = np.empty((B, n, 5, 24), np.float32)
        for i in range(B):
            out[i] = np.clip(self._sample_season(int(seas[i]), n), 0, None)  # non-neg clip
        return out


METHODS = {"historical": HistoricalSampler, "weather_knn": WeatherKNN, "copula": GaussianCopula}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts
    cfg["method_tag"] = args.method
    gen = METHODS[args.method](cfg)
    run_and_save(cfg, gen, split=args.split, device="cpu", desc=args.method)


if __name__ == "__main__":
    main()
