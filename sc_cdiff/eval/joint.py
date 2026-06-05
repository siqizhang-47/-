"""Joint-distribution metrics (plan §8.5, main judge): Energy Score, Variogram
Score, seasonal correlation error, high/low-temperature coverage."""
from __future__ import annotations

import numpy as np


def energy_score(scenarios, truth):
    """Energy Score for one forecast case.
    scenarios: [M, D] ensemble members (flatten 5*24).  truth: [D].
    ES = E||X - y|| - 0.5 E||X - X'||.  Lower is better.
    """
    X = scenarios.reshape(scenarios.shape[0], -1)
    y = truth.reshape(-1)
    M = X.shape[0]
    term1 = np.linalg.norm(X - y[None, :], axis=1).mean()
    # pairwise distances (subsample if large)
    idx = np.arange(M)
    d = np.linalg.norm(X[idx][:, None, :] - X[idx][None, :, :], axis=2)
    term2 = d.sum() / (M * M)
    return float(term1 - 0.5 * term2)


def variogram_score(scenarios, truth, p=0.5):
    """Variogram score of order p. scenarios:[M,D], truth:[D]."""
    X = scenarios.reshape(scenarios.shape[0], -1)
    y = truth.reshape(-1)
    D = y.shape[0]
    # subsample dimension pairs for speed if large
    diff_true = np.abs(y[:, None] - y[None, :]) ** p          # [D,D]
    diff_gen = (np.abs(X[:, :, None] - X[:, None, :]) ** p).mean(axis=0)  # [D,D]
    return float(((diff_true - diff_gen) ** 2).sum())


def mean_energy_score(scenarios, truth):
    """scenarios:[N,M,5,24], truth:[N,5,24] -> mean ES over N cases."""
    N = scenarios.shape[0]
    vals = [energy_score(scenarios[i], truth[i]) for i in range(N)]
    return float(np.mean(vals))


def mean_variogram_score(scenarios, truth):
    N = scenarios.shape[0]
    vals = [variogram_score(scenarios[i], truth[i]) for i in range(N)]
    return float(np.mean(vals))


def seasonal_corr_error(scenarios, truth, seasons, c_idx=2, h_idx=3):
    """Frobenius error of per-channel-mean correlation matrix, averaged over seasons.
    scenarios:[N,M,5,24], truth:[N,5,24], seasons:[N] in 0..3."""
    def corr_of(feat):
        fc = feat - feat.mean(0, keepdims=True)
        sd = fc.std(0, keepdims=True) + 1e-6
        fn = fc / sd
        return fn.T @ fn / (feat.shape[0] - 1 + 1e-6)
    errs = []
    gen_mean = scenarios.mean(axis=1).mean(axis=2)   # [N,5] ensemble-mean daily mean
    real_mean = truth.mean(axis=2)                   # [N,5]
    for s in range(4):
        idx = np.where(seasons == s)[0]
        if len(idx) < 8:
            continue
        Rg = corr_of(gen_mean[idx]); Rr = corr_of(real_mean[idx])
        errs.append(np.sqrt(((Rg - Rr) ** 2).sum()))
    return float(np.mean(errs)) if errs else float("nan")
