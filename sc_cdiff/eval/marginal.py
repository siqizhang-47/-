"""Marginal-distribution metrics (plan §8.5)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def _pool(scenarios, ch):
    # scenarios:[N,M,5,24] -> pooled samples of channel ch
    return scenarios[:, :, ch, :].reshape(-1)


def _pool_truth(truth, ch):
    return truth[:, ch, :].reshape(-1)


def crps_ensemble(scenarios, truth):
    """Mean CRPS per channel via the ensemble formula.
    scenarios:[N,M,5,24], truth:[N,5,24]. Returns dict per channel index."""
    out = {}
    N, M, C, T = scenarios.shape
    for ch in range(C):
        X = scenarios[:, :, ch, :].reshape(N * T, M)
        y = truth[:, ch, :].reshape(N * T)
        t1 = np.abs(X - y[:, None]).mean(axis=1)
        t2 = np.abs(X[:, :, None] - X[:, None, :]).mean(axis=(1, 2))
        out[ch] = float(np.mean(t1 - 0.5 * t2))
    return out


def marginal_metrics(scenarios, truth):
    C = scenarios.shape[2]
    res = {}
    for ch in range(C):
        g = _pool(scenarios, ch)
        r = _pool_truth(truth, ch)
        ks = stats.ks_2samp(g, r).statistic
        w1 = stats.wasserstein_distance(g, r)
        res[ch] = {
            "ks": float(ks),
            "wasserstein": float(w1),
            "mean_err": float(g.mean() - r.mean()),
            "std_err": float(g.std() - r.std()),
            "q90_err": float(np.quantile(g, 0.9) - np.quantile(r, 0.9)),
        }
    res["crps"] = crps_ensemble(scenarios, truth)
    return res
