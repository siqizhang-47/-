"""Probabilistic-reliability metrics (plan §8.5): PICP, PINAW, CWC, ensemble
mean MAE/RMSE, rank histogram."""
from __future__ import annotations

import numpy as np


def reliability_metrics(scenarios, truth, alpha=0.9):
    """scenarios:[N,M,5,24], truth:[N,5,24]."""
    N, M, C, T = scenarios.shape
    lo = (1 - alpha) / 2
    hi = 1 - lo
    res = {}
    for ch in range(C):
        S = scenarios[:, :, ch, :]                 # [N,M,T]
        y = truth[:, ch, :]                        # [N,T]
        ql = np.quantile(S, lo, axis=1)            # [N,T]
        qh = np.quantile(S, hi, axis=1)
        inside = (y >= ql) & (y <= qh)
        picp = float(inside.mean())
        rng = (S.max() - S.min()) + 1e-9
        pinaw = float((qh - ql).mean() / rng)
        # CWC
        eta, mu = 50.0, alpha
        gamma = 0.0 if picp >= mu else 1.0
        cwc = float(pinaw * (1 + gamma * np.exp(-eta * (picp - mu))))
        mean_pred = S.mean(axis=1)
        mae = float(np.abs(mean_pred - y).mean())
        rmse = float(np.sqrt(((mean_pred - y) ** 2).mean()))
        # rank histogram (uniformity)
        ranks = (S < y[:, None, :]).sum(axis=1)    # [N,T] rank 0..M
        hist, _ = np.histogram(ranks.reshape(-1), bins=np.arange(M + 2))
        hist = hist / hist.sum()
        flatness = float(np.std(hist))             # 0 = perfectly flat
        res[ch] = {"picp": picp, "pinaw": pinaw, "cwc": cwc,
                   "mae": mae, "rmse": rmse, "rank_flatness": flatness}
    return res
