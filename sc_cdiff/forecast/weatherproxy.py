"""Step 4: predicted-weather proxy What (main task) + true weather (oracle).

Plan §8.2 / §4: the main task conditions on PREDICTED weather, not the truth,
to avoid leaking the PV / cooling answer. Without an independent NWP feed we
build a (month, hour) climatology proxy fit on TRAIN rows -- this carries a
realistic, non-zero error. We also dump the true weather tape unchanged for the
oracle upper-bound ablation only.

Limitation to state in the paper: a climatology proxy lacks day-to-day forecast
skill, so it is a conservative (pessimistic) stand-in for a real NWP forecast;
the oracle ablation brackets the achievable upper bound.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from ..data.build_dataset import _load_cfg


def build(cfg: dict):
    paths = cfg["paths"]
    npz = np.load(os.path.join(paths["artifacts"], paths["dataset_npz"]))
    Wfull = npz["Wfull"].astype(np.float64)        # [n,6]
    train_end = int(npz["train_end_hour"])
    n = Wfull.shape[0]

    dates = pd.date_range("2001-06-01 00:00:00", periods=n, freq="h")
    month = dates.month.to_numpy()
    hour = dates.hour.to_numpy()
    key = month * 100 + hour
    train_mask = np.arange(n) < train_end

    what = np.zeros_like(Wfull)
    for f in range(Wfull.shape[1]):
        tab = {}
        for k in np.unique(key[train_mask]):
            tab[int(k)] = float(Wfull[train_mask & (key == k), f].mean())
        gmean = float(Wfull[train_mask, f].mean())
        what[:, f] = [tab.get(int(k), gmean) for k in key]

    out = os.path.join(paths["artifacts"], paths["what_npy"])
    np.save(out, what.astype(np.float32))
    np.save(os.path.join(paths["artifacts"], paths["woracle_npy"]), Wfull.astype(np.float32))
    print(f"[weatherproxy] wrote {out} (proxy) and {paths['woracle_npy']} (oracle) shape={what.shape}")
    # sanity: temperature proxy RMSE on test
    val_end = int(npz["val_end_hour"])
    te = np.arange(n) >= val_end
    rmse_t = float(np.sqrt(np.mean((what[te, 1] - Wfull[te, 1]) ** 2)))
    print(f"[weatherproxy] test temperature proxy RMSE={rmse_t:.2f} C")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--artifacts", default=None)
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts
    build(cfg)


if __name__ == "__main__":
    main()
