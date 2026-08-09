"""Deterministic and probabilistic metrics (design document, section 21).

Everything is computed on the ORIGINAL scale, after inverse standardisation.

Deterministic (per target, using the scenario mean):
    MAE, RMSE, sMAPE
Probabilistic:
    CRPS, QICE, PICP@{50,80,90,95} and the matching mean interval width,
    Energy Score, Variogram Score
Conditional slices:
    PV day / night, Cooling on hot hours, Heat on cold hours,
    Electricity on weekdays / weekends

All estimators are pure torch and run on the sample tensor
``pred [W, S, H, K]`` against ``true [W, H, K]``.
"""
from __future__ import annotations

import numpy as np
import torch

TARGET_NAMES = ["Electricity", "PV", "Cooling", "Heat"]


# --------------------------------------------------------------------------- deterministic
def mae(pred_mean, true):
    return (pred_mean - true).abs().mean().item()


def rmse(pred_mean, true):
    return torch.sqrt((pred_mean - true).pow(2).mean()).item()


def smape(pred_mean, true, eps: float = 1e-8):
    """Symmetric MAPE in percent. The eps floor keeps PV's near-zero night hours finite."""
    denom = (pred_mean.abs() + true.abs()).clamp(min=eps)
    return (200.0 * (pred_mean - true).abs() / denom).mean().item()


# --------------------------------------------------------------------------- CRPS
def crps_ensemble(pred, true):
    """Fair (unbiased-spread) empirical CRPS.

        CRPS = E|X - y| - 1/2 E|X - X'|

    ``pred``: [..., S], ``true``: [...].  Computed in O(S log S) via the sorted
    identity  sum_ij |x_i - x_j| = 2 * sum_i (2i - S + 1) * x_(i).
    """
    S = pred.shape[-1]
    term1 = (pred - true.unsqueeze(-1)).abs().mean(dim=-1)
    xs, _ = torch.sort(pred, dim=-1)
    idx = torch.arange(S, device=pred.device, dtype=pred.dtype)
    weights = 2.0 * idx - S + 1.0
    pair_sum = 2.0 * (xs * weights).sum(dim=-1)
    term2 = pair_sum / (2.0 * S * S)
    return term1 - term2


# --------------------------------------------------------------------------- calibration
def picp_and_width(pred, true, level: float):
    """Coverage and mean width of the central `level` interval. pred [..., S]."""
    lo_q = (1.0 - level) / 2.0
    hi_q = 1.0 - lo_q
    qs = torch.quantile(pred.float(),
                        torch.tensor([lo_q, hi_q], device=pred.device, dtype=torch.float32),
                        dim=-1)
    lo, hi = qs[0], qs[1]
    covered = ((true >= lo) & (true <= hi)).float()
    return covered.mean().item(), (hi - lo).mean().item()


def qice(pred, true, n_bins: int = 10):
    """Quantile Interval Coverage Error: mean |1/M - empirical bin frequency|."""
    flat_pred = pred.reshape(-1, pred.shape[-1]).float()
    flat_true = true.reshape(-1).float()
    qlist = torch.linspace(0, 1, n_bins + 1, device=pred.device)[1:-1]
    edges = torch.quantile(flat_pred, qlist, dim=-1)             # [n_bins-1, N]
    membership = (flat_true.unsqueeze(0) > edges).sum(dim=0)     # in [0, n_bins-1]
    counts = torch.bincount(membership, minlength=n_bins).float()
    ratio = counts / counts.sum()
    return torch.abs(torch.full_like(ratio, 1.0 / n_bins) - ratio).mean().item()


# --------------------------------------------------------------------------- multivariate
def energy_score(pred, true, chunk: int = 64):
    """ES = mean_s ||x_s - y|| - 1/(2 S^2) sum_{s,s'} ||x_s - x_s'||.

    ``pred`` [W, S, D], ``true`` [W, D]; the D axis is the flattened (H, K) vector,
    so the score judges the *joint* 24h x 4-target scenario.
    """
    W, S, D = pred.shape
    total = 0.0
    for i in range(0, W, chunk):
        p = pred[i:i + chunk].float()
        y = true[i:i + chunk].float()
        t1 = torch.linalg.vector_norm(p - y.unsqueeze(1), dim=-1).mean(dim=1)
        pdist = torch.cdist(p, p)
        t2 = pdist.sum(dim=(1, 2)) / (2.0 * S * S)
        total += (t1 - t2).sum().item()
    return total / W


def variogram_score(pred, true, p: float = 0.5, chunk: int = 64):
    """VS_p = sum_{i,j} ( |y_i - y_j|^p - mean_s |x_si - x_sj|^p )^2.

    ``pred`` [W, S, D], ``true`` [W, D].
    """
    W, S, D = pred.shape
    total = 0.0
    for i in range(0, W, chunk):
        p_ = pred[i:i + chunk].float()
        y_ = true[i:i + chunk].float()
        vy = (y_.unsqueeze(-1) - y_.unsqueeze(-2)).abs().pow(p)                  # [b, D, D]
        vx = (p_.unsqueeze(-1) - p_.unsqueeze(-2)).abs().pow(p).mean(dim=1)      # [b, D, D]
        total += (vy - vx).pow(2).sum(dim=(1, 2)).sum().item()
    return total / W


# --------------------------------------------------------------------------- driver
def compute_all_metrics(pred, true, group_info=None, n_bins: int = 10,
                        levels=(0.5, 0.8, 0.9, 0.95), variogram_mode: str = "cross_var",
                        variogram_p: float = 0.5, target_names=None):
    """pred [W, S, H, K], true [W, H, K], group_info [W, H, 3] (hour, is_weekend, temp_degF)."""
    target_names = target_names or TARGET_NAMES
    W, S, H, K = pred.shape
    pred_mean = pred.mean(dim=1)
    out = {"n_windows": W, "n_samples": S}

    # ---- per target
    per_target = {}
    pred_swapped = pred.permute(0, 2, 3, 1)   # [W, H, K, S]
    for k, name in enumerate(target_names):
        pm, tt, ps = pred_mean[..., k], true[..., k], pred_swapped[..., k, :]
        m = {
            "MAE": mae(pm, tt),
            "RMSE": rmse(pm, tt),
            "sMAPE": smape(pm, tt),
            "CRPS": crps_ensemble(ps, tt).mean().item(),
            "QICE": qice(ps, tt, n_bins),
        }
        for lv in levels:
            cov, width = picp_and_width(ps, tt, lv)
            m[f"PICP@{int(lv * 100)}"] = cov
            m[f"Width@{int(lv * 100)}"] = width
        per_target[name] = m
    out["per_target"] = per_target

    # ---- aggregate over targets
    out["overall"] = {
        key: float(np.mean([per_target[n][key] for n in target_names]))
        for key in per_target[target_names[0]]
    }

    # ---- multivariate scores (joint over the 24 x 4 scenario)
    flat_pred = pred.reshape(W, S, H * K)
    flat_true = true.reshape(W, H * K)
    out["overall"]["EnergyScore"] = energy_score(flat_pred, flat_true)

    if variogram_mode == "full":
        out["overall"]["VariogramScore"] = variogram_score(flat_pred, flat_true, variogram_p)
    else:
        # cross-variable structure at each horizon step, averaged over the horizon
        vs = [variogram_score(pred[:, :, h, :], true[:, h, :], variogram_p) for h in range(H)]
        out["overall"]["VariogramScore"] = float(np.mean(vs))
    out["overall"]["variogram_mode"] = variogram_mode

    # ---- conditional slices
    if group_info is not None:
        out["conditional"] = _conditional_metrics(pred_swapped, pred_mean, true, group_info,
                                                  target_names)
    return out


def _slice_metrics(pred_s, pred_m, true, mask):
    if mask.sum() == 0:
        return None
    ps, pm, tt = pred_s[mask], pred_m[mask], true[mask]
    res = {
        "n": int(mask.sum().item()),
        "MAE": mae(pm, tt),
        "RMSE": rmse(pm, tt),
        "CRPS": crps_ensemble(ps, tt).mean().item(),
    }
    for lv in (0.8, 0.9):
        cov, width = picp_and_width(ps, tt, lv)
        res[f"PICP@{int(lv * 100)}"] = cov
        res[f"Width@{int(lv * 100)}"] = width
    return res


def _conditional_metrics(pred_swapped, pred_mean, true, group_info, target_names,
                         hot_degf: float = 95.0, cold_degf: float = 55.0):
    hour = group_info[..., 0]
    is_weekend = group_info[..., 1]
    temp = group_info[..., 2]

    idx = {n: i for i, n in enumerate(target_names)}
    day = (hour >= 7) & (hour <= 18)
    out = {}

    def add(tag, target, mask):
        k = idx[target]
        r = _slice_metrics(pred_swapped[..., k, :], pred_mean[..., k], true[..., k], mask)
        if r is not None:
            out[tag] = r

    add("PV_day", "PV", day)
    add("PV_night", "PV", ~day)
    add("Cooling_hot", "Cooling", temp >= hot_degf)
    add("Cooling_mild", "Cooling", temp < hot_degf)
    add("Heat_cold", "Heat", temp <= cold_degf)
    add("Heat_warm", "Heat", temp > cold_degf)
    add("Electricity_weekday", "Electricity", is_weekend < 0.5)
    add("Electricity_weekend", "Electricity", is_weekend >= 0.5)
    return out
