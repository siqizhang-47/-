"""Step 12: sample M scenarios per test day for each horizon k."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .data.build_dataset import _load_cfg
from .data.dataset import AlignedDays
from .models.sc_cdiff import SCCDiff
from .normalize import Normalizer
from .train import pick_device


@torch.no_grad()
def sample_split(cfg, device, split="test", use_ema=True):
    paths = cfg["paths"]
    norm = Normalizer.load(os.path.join(paths["artifacts"], paths["norm_stats"]))
    model = SCCDiff(cfg, norm).to(device)
    ckpt = torch.load(os.path.join(paths["artifacts"], paths["ckpt_best"]), map_location=device)
    model.load_state_dict(ckpt["ema"] if use_ema else ckpt["model"])
    model.eval()
    model.norm = norm

    n_scen = cfg["sample"]["n_scenarios"]
    bs = cfg["sample"]["batch_days"]
    for k in cfg["sample"]["ks"]:
        ds = AlignedDays(cfg, split, k=k)
        scen_all, true_all, irr_all, start_all = [], [], [], []
        for i in range(0, len(ds), bs):
            sub = [ds[j] for j in range(i, min(i + bs, len(ds)))]
            batch = {kk: torch.stack([s[kk] for s in sub]).to(device) for kk in sub[0]}
            scen = model.sample(batch, n_scen).cpu().numpy()
            scen_all.append(scen.astype(np.float32))
            true_all.append(batch["Yraw"].cpu().numpy())
            irr_all.append(batch["irr"].cpu().numpy())
            start_all.append(batch["start"].cpu().numpy())
        scen = np.concatenate(scen_all, 0)
        out = os.path.join(paths["artifacts"], f"scenarios_{split}_k{k}.npz")
        np.savez_compressed(out, scenarios=scen,
                            truth=np.concatenate(true_all, 0),
                            irr=np.concatenate(irr_all, 0),
                            start=np.concatenate(start_all, 0))
        # sanity: night-PV violation rate
        truth = np.concatenate(true_all, 0); irr = np.concatenate(irr_all, 0)
        night = (irr <= 0)[:, None, :]
        npv = float(((scen[:, :, 0, :] > 1e-6) & night).sum() / max(night.sum() * n_scen, 1))
        print(f"[sample] {split} k={k} scenarios={scen.shape} night_pv_viol={npv:.4f} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts
    if args.device:
        cfg["device"] = args.device
    device = pick_device(cfg["device"])
    sample_split(cfg, device, args.split)


if __name__ == "__main__":
    main()
