"""Evaluation metrics.

All metrics are computed on the **inverse-normalized** target (real load) so
that errors are reported in the original physical units. Three reporting
buckets are produced:

- Overall: MAE / RMSE / sMAPE on all test windows
- Normal: only windows where 1-12 event labels are all zero in the prediction interval
- Event: only windows where any 1-12 label is set
- Event families: temp (1-4), wind (5-7), typhoon (8-12)
- High-risk: 95th-percentile absolute error (P95AE) and peak-error MAE

A sample ``record_metrics_table`` is provided to render the spec tables.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return _safe_mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    if y_true.size == 0:
        return float("nan")
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / denom))


def p95_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.percentile(np.abs(y_true - y_pred), 95))


def peak_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE on the per-window peak (max over the horizon)."""
    if y_true.size == 0:
        return float("nan")
    peak_true = np.max(y_true, axis=1)
    peak_pred = np.max(y_pred, axis=1)
    return float(np.mean(np.abs(peak_true - peak_pred)))


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    masks: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Return all the metric buckets required by the spec.

    ``y_true`` / ``y_pred`` are ``[N, H]`` in original units, ``masks[name]``
    is a boolean array of length N indicating window membership.
    """
    out: Dict[str, float] = {}
    out.update({f"overall_{k}": v for k, v in basic_metrics(y_true, y_pred).items()})

    for bucket in ("normal", "event"):
        m = masks[bucket]
        ms = basic_metrics(y_true[m], y_pred[m])
        for k, v in ms.items():
            out[f"{bucket}_{k}"] = v

    for fam in ("temp", "wind", "typhoon"):
        m = masks[fam]
        ms = basic_metrics(y_true[m], y_pred[m])
        for k, v in ms.items():
            out[f"{fam}_{k}"] = v

    event_mask = masks["event"]
    out["overall_P95AE"] = p95_absolute_error(y_true, y_pred)
    out["event_P95AE"] = p95_absolute_error(y_true[event_mask], y_pred[event_mask])
    out["overall_MAE_peak"] = peak_mae(y_true, y_pred)
    out["event_MAE_peak"] = peak_mae(y_true[event_mask], y_pred[event_mask])
    return out
