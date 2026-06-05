"""Collect every eval_<tag>_<split>_k<k>.json in the artifacts dir into one
master results table (plan §13). Prints a markdown table and writes a CSV."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from ..data.build_dataset import _load_cfg

PAT = re.compile(r"eval_(?P<tag>.+)_(?P<split>train|val|test)_k(?P<k>\d+)\.json$")


def _avg(d, field):
    vals = [d[str(ch)][field] for ch in range(5) if str(ch) in d]
    return float(np.mean(vals)) if vals else float("nan")


def collect(cfg, split="test", k=0):
    art = cfg["paths"]["artifacts"]
    rows = []
    for f in sorted(glob.glob(os.path.join(art, "eval_*.json"))):
        m = PAT.search(os.path.basename(f))
        if not m or m["split"] != split or int(m["k"]) != k:
            continue
        r = json.load(open(f))
        phys = r.get("physical", {})
        rel = r.get("reliability", {})
        mar = r.get("marginal", {})
        row = {
            "method": m["tag"],
            "EnergyScore": r.get("energy_score"),
            "VariogramScore": r.get("variogram_score"),
            "SeasCorrErr": r.get("seasonal_corr_err"),
            "NightPV": phys.get("night_pv_viol_rate"),
            "NegRate": phys.get("neg_rate"),
            "CoolOff": phys.get("cool_offseason_rate"),
            "HeatOff": phys.get("heat_offseason_rate"),
            "PICP": _avg(rel, "picp"),
            "PINAW": _avg(rel, "pinaw"),
            "CRPS": float(np.mean(list(mar.get("crps", {}).values()))) if mar.get("crps") else float("nan"),
        }
        rows.append(row)
    if not rows:
        print(f"[collect] no eval files for split={split} k={k}")
        return None
    df = pd.DataFrame(rows).sort_values("EnergyScore").reset_index(drop=True)
    out = os.path.join(art, f"master_results_{split}_k{k}.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"\n=== Master results  split={split}  k={k}  (sorted by Energy Score) ===")
    try:
        print(df.to_markdown(index=False, floatfmt=".4g"))
    except ImportError:
        print(df.to_string(index=False))
    print(f"\n[collect] wrote {out}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=0)
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts
    collect(cfg, args.split, args.k)


if __name__ == "__main__":
    main()
