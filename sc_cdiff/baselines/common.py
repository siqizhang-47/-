"""Shared harness for baseline methods.

Every baseline produces scenarios in the SAME format as SC-CDiff
(scenarios_<tag>_<split>_k<k>.npz with scenarios[N,M,5,24], truth, irr, start),
so the existing eval suite and the results collector work unchanged.

A baseline is just a callable `gen_fn(batch, n) -> raw [B, n, 5, 24]` where
`batch` is the dict yielded by AlignedDays. `run_and_save` iterates aligned days
for each horizon k, enforces prefix inpainting (first k hours = observed truth)
for fairness, and writes the npz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..artifacts import scen_path
from ..data.dataset import AlignedDays

BASE_TS = pd.Timestamp("2001-06-01 00:00:00")


def month_of(start):
    """start: array/tensor of absolute-hour indices -> calendar month 1..12."""
    if torch.is_tensor(start):
        start = start.cpu().numpy()
    return np.array([(BASE_TS + pd.Timedelta(hours=int(s))).month for s in start])


def season_of(start):
    m = month_of(start)
    return (m % 12) // 3        # 0 winter,1 spring,2 summer,3 autumn


def collate(sub):
    return {k: torch.stack([s[k] for s in sub]) for k in sub[0]}


def run_and_save(cfg, gen_fn, split="test", device="cpu", desc="baseline"):
    n_scen = cfg["sample"]["n_scenarios"]
    bs = cfg["sample"]["batch_days"]
    for k in cfg["sample"]["ks"]:
        ds = AlignedDays(cfg, split, k=k)
        scen_all, true_all, irr_all, start_all = [], [], [], []
        for i in range(0, len(ds), bs):
            sub = [ds[j] for j in range(i, min(i + bs, len(ds)))]
            batch = collate(sub)
            batch = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in batch.items()}
            scen = gen_fn(batch, n_scen)                 # raw [B,n,5,24] tensor or np
            if torch.is_tensor(scen):
                scen = scen.detach().cpu().numpy()
            scen = scen.astype(np.float32)
            # prefix inpainting (fairness): observed hours = truth
            if k > 0:
                yraw = batch["Yraw"].cpu().numpy()       # [B,5,24]
                scen[:, :, :, :k] = yraw[:, None, :, :k]
            scen_all.append(scen)
            true_all.append(batch["Yraw"].cpu().numpy())
            irr_all.append(batch["irr"].cpu().numpy())
            start_all.append(batch["start"].cpu().numpy())
        scen = np.concatenate(scen_all, 0)
        truth = np.concatenate(true_all, 0)
        irr = np.concatenate(irr_all, 0)
        out = scen_path(cfg, split, k)
        np.savez_compressed(out, scenarios=scen, truth=truth, irr=irr,
                            start=np.concatenate(start_all, 0))
        night = (irr <= 0)[:, None, :]
        npv = float(((scen[:, :, 0, :] > 1e-6) & night).sum() / max(night.sum() * n_scen, 1))
        print(f"[{desc}] {split} k={k} scenarios={scen.shape} night_pv_viol={npv:.4f} -> {out}")


def load_full(cfg):
    """Load the continuous tape + forecasts + normalizer for statistical fitting."""
    import os
    from ..normalize import Normalizer
    paths = cfg["paths"]
    npz = np.load(os.path.join(paths["artifacts"], paths["dataset_npz"]))
    norm = Normalizer.load(os.path.join(paths["artifacts"], paths["norm_stats"]))
    return npz, norm
