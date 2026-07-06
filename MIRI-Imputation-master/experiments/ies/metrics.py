"""Evaluation metrics for the IES imputation benchmark.

Point error       : MAE, RMSE (on artificially-missing test positions).
Distribution      : masked-value MMD, joint MMD (RBF kernel).
Joint structure   : correlation-matrix Frobenius error + key physical pairs.
"""

from __future__ import annotations

import numpy as np

# Fixed variable indices (order of ``feature_cols`` in the config).
ELECTRICITY = 0
COOLING = 1
HEATING = 2
PV = 3
IRRADIATION = 4
TEMP = 5
HUMIDITY = 6
WIND = 7


def masked_mae_rmse(X_true, X_imp, M_test):
    """MAE / RMSE over all artificially-missing entries (M_test == 0)."""
    miss = M_test == 0
    err = X_imp[miss] - X_true[miss]
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    return float(mae), float(rmse)


def per_variable_errors(X_true, X_imp, M_test, feature_cols):
    """Per-variable MAE / RMSE on that variable's missing positions."""
    out = {}
    for j, name in enumerate(feature_cols):
        miss = M_test[:, j] == 0
        if miss.sum() == 0:
            out[name] = {"mae": None, "rmse": None, "n_missing": 0}
            continue
        err = X_imp[miss, j] - X_true[miss, j]
        out[name] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "n_missing": int(miss.sum()),
        }
    return out


def rbf_mmd(X, Y, max_samples=5000, seed=0, sigma=None):
    """RBF-kernel Maximum Mean Discrepancy between two point sets.

    Sub-samples each set to ``max_samples`` rows and uses the median pairwise
    distance as the bandwidth when ``sigma`` is None.
    """
    rng = np.random.default_rng(seed)

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    if X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]
    if Y.shape[0] > max_samples:
        idx = rng.choice(Y.shape[0], max_samples, replace=False)
        Y = Y[idx]

    import torch
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)

    XX = torch.cdist(Xt, Xt)
    XY = torch.cdist(Xt, Yt)
    YY = torch.cdist(Yt, Yt)

    if sigma is None:
        sigma = torch.median(XY).item()
        sigma = max(sigma, 1e-6)

    kxx = torch.exp(-(XX ** 2) / (2 * sigma ** 2)).mean()
    kxy = torch.exp(-(XY ** 2) / (2 * sigma ** 2)).mean()
    kyy = torch.exp(-(YY ** 2) / (2 * sigma ** 2)).mean()

    return float(kxx + kyy - 2 * kxy)


def masked_value_mmd(X_true, X_imp, M_test, max_samples=5000, seed=0, sigma=None):
    """Average of per-variable 1-D MMD between true and imputed missing values."""
    vals = []
    for j in range(X_true.shape[1]):
        miss = M_test[:, j] == 0
        if miss.sum() < 10:
            continue
        x = X_true[miss, j:j + 1]
        y = X_imp[miss, j:j + 1]
        vals.append(rbf_mmd(x, y, max_samples=max_samples, seed=seed + j, sigma=sigma))
    return float(np.mean(vals)) if vals else None


def corr_error(X_true, X_imp):
    """Frobenius norm of the correlation-matrix difference."""
    C_true = np.corrcoef(X_true, rowvar=False)
    C_imp = np.corrcoef(X_imp, rowvar=False)

    C_true = np.nan_to_num(C_true)
    C_imp = np.nan_to_num(C_imp)

    return float(np.linalg.norm(C_true - C_imp, ord="fro"))


def corr_pair(X, i, j):
    c = np.corrcoef(X[:, i], X[:, j])[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def key_relation_errors(X_true, X_imp):
    """Absolute error of key physical pairwise correlations."""
    pairs = {
        "pv_irradiation_corr_error": (PV, IRRADIATION),
        "temp_cooling_corr_error": (TEMP, COOLING),
        "temp_heating_corr_error": (TEMP, HEATING),
        "electricity_cooling_corr_error": (ELECTRICITY, COOLING),
        "electricity_heating_corr_error": (ELECTRICITY, HEATING),
    }
    out = {}
    for name, (i, j) in pairs.items():
        out[name] = abs(corr_pair(X_true, i, j) - corr_pair(X_imp, i, j))
    return out
