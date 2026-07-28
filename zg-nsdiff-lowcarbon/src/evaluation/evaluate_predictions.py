"""Evaluate all prediction directories into per-seed metric rows.

Metrics (Table I of the paper): per variable MAPE / AW / PICP
(95% central interval, quantiles 0.025 / 0.975).

python -m src.evaluation.evaluate_predictions \
    --prediction_root artifacts/predictions --output results/per_seed_metrics.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.baselines.prediction_contract import (
    assert_alignment,
    load_manifest,
    shard_paths,
)
from src.data.low_carbon_schema import TARGET_NAMES
from src.evaluation.interval_metrics import StreamingIntervalMetrics
from src.evaluation.point_metrics import StreamingPointMetrics


def evaluate_dir(pred_dir):
    point = {n: StreamingPointMetrics() for n in TARGET_NAMES}
    interval = {n: StreamingIntervalMetrics(levels=[0.95]) for n in TARGET_NAMES}
    for p in tqdm(shard_paths(pred_dir), desc=f"eval {os.path.relpath(pred_dir)}", ncols=110):
        with np.load(p) as z:
            samples = z["samples"]      # [n,H,D,S]
            truth = z["truth"]          # [n,H,D]
        pred_mean = samples.mean(axis=-1)
        for d, name in enumerate(TARGET_NAMES):
            point[name].update(pred_mean[..., d], truth[..., d])
            interval[name].update(samples[..., d, :], truth[..., d])
    row = {}
    for name in TARGET_NAMES:
        pm = point[name].compute()
        im = interval[name].compute()
        row[f"{name}_mape"] = pm["mape_pos"]   # all HEEW targets are > 0
        row[f"{name}_aw"] = im["aw_95"]
        row[f"{name}_picp"] = im["picp_95"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction_root", default="artifacts/predictions")
    ap.add_argument("--output", default="results/per_seed_metrics.csv")
    ap.add_argument("--skip_alignment_check", action="store_true")
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.prediction_root, "*", "seed_*")))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "manifest.json"))]
    if not dirs:
        raise SystemExit(f"no prediction directories under {args.prediction_root}")
    print(f"found {len(dirs)} prediction dirs")

    if not args.skip_alignment_check and len(dirs) > 1:
        print("checking cross-model alignment of forecast_start_index/timestamps/truth ...")
        assert_alignment(dirs)

    rows = []
    for d in dirs:
        man = load_manifest(d)
        row = evaluate_dir(d)
        row["model"] = man["model"]
        row["seed"] = man["seed"]
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["model", "seed"] + [c for c in df.columns if c not in ("model", "seed")]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df[cols].to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
