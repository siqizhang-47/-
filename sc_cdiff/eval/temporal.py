"""Temporal-structure metrics (plan §8.5): ACF error, ramp error, peak-hour error."""
from __future__ import annotations

import numpy as np


def _acf(x, nlags=12):
    x = x - x.mean()
    v = np.var(x) + 1e-9
    return np.array([np.mean(x[: len(x) - k] * x[k:]) / v for k in range(nlags + 1)])


def temporal_metrics(scenarios, truth, nlags=12):
    """scenarios:[N,M,5,24], truth:[N,5,24]."""
    N, M, C, T = scenarios.shape
    res = {}
    for ch in range(C):
        # ACF averaged over cases; compare ensemble-mean acf vs real acf
        acf_real = np.mean([_acf(truth[i, ch], nlags) for i in range(N)], axis=0)
        gen_acfs = []
        for i in range(N):
            gen_acfs.append(np.mean([_acf(scenarios[i, j, ch], nlags) for j in range(M)], axis=0))
        acf_gen = np.mean(gen_acfs, axis=0)
        e_acf = float(np.linalg.norm(acf_real - acf_gen))

        ramp_real = np.abs(np.diff(truth[:, ch], axis=1)).mean()
        ramp_gen = np.abs(np.diff(scenarios[:, :, ch], axis=2)).mean()
        peak_real = truth[:, ch].argmax(axis=1)
        peak_gen = scenarios[:, :, ch].argmax(axis=2)            # [N,M]
        peak_err = float(np.abs(peak_gen - peak_real[:, None]).mean())
        res[ch] = {"acf_err": e_acf, "ramp_err": float(ramp_gen - ramp_real),
                   "peak_hour_err": peak_err}
    return res
