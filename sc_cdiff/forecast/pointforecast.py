"""Step 3: sample-out-of-sample point forecast Yhat (a condition, not a target).

CRITICAL (plan §8.2): never use noised ground truth. We fit a climatology
point forecaster on TRAIN rows only, keyed by (era, month, hour), and emit a
continuous Yhat tape [n_hours, 5] that any window / aligned day can index.

For cooling/heating the plain (era,month,hour) mean over all hours already
equals activated_mean * activation_frequency, which is the desired soft
expectation of a zero-inflated variable.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from ..data.build_dataset import _load_cfg


def fit_predict(cfg: dict) -> np.ndarray:
    paths = cfg["paths"]
    npz = np.load(os.path.join(paths["artifacts"], paths["dataset_npz"]))
    Yfull = npz["Yfull"].astype(np.float64)        # [n,5]
    ERA = npz["ERAfull"]
    train_end = int(npz["train_end_hour"])
    n = Yfull.shape[0]

    # rebuild month/hour from the index using the known start timestamp
    dates = pd.date_range("2001-06-01 00:00:00", periods=n, freq="h")
    month = dates.month.to_numpy()
    hour = dates.hour.to_numpy()

    train_mask = np.arange(n) < train_end
    key_full = ERA.astype(np.int64) * 10000 + month * 100 + hour
    key2 = month * 100 + hour

    yhat = np.zeros_like(Yfull)
    for ch in range(Yfull.shape[1]):
        vals = Yfull[:, ch]
        # primary table keyed by (era,month,hour)
        tab1, tab2 = {}, {}
        for k in np.unique(key_full[train_mask]):
            tab1[int(k)] = float(vals[train_mask & (key_full == k)].mean())
        for k in np.unique(key2[train_mask]):
            tab2[int(k)] = float(vals[train_mask & (key2 == k)].mean())
        gmean = float(vals[train_mask].mean())
        yhat[:, ch] = [tab1.get(int(k1), tab2.get(int(k2), gmean))
                       for k1, k2 in zip(key_full, key2)]

    out = os.path.join(paths["artifacts"], paths["yhat_npy"])
    np.save(out, yhat.astype(np.float32))
    print(f"[pointforecast] wrote {out} shape={yhat.shape}")
    # quick sanity: out-of-sample electricity MAE on val/test
    val_end = int(npz["val_end_hour"])
    te = np.arange(n) >= val_end
    mae_e = np.abs(yhat[te, 1] - Yfull[te, 1]).mean()
    print(f"[pointforecast] test electricity MAE={mae_e:.1f} kW (climatology, expected ~100-200)")
    return yhat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--artifacts", default=None)
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts
    fit_predict(cfg)


if __name__ == "__main__":
    main()
