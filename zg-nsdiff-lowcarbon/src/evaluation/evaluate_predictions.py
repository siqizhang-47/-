"""Evaluate all prediction directories into per-seed metric rows (spec 15).

python -m src.evaluation.evaluate_predictions \
    --prediction_root artifacts/predictions --output results/per_seed_metrics.csv
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.baselines.prediction_contract import (
    assert_alignment,
    iter_shards,
    load_manifest,
    shard_paths,
)
from src.data.low_carbon_schema import TARGET_NAMES, ZERO_TARGET_NAMES, ZERO_TARGET_INDICES
from src.evaluation.empirical_crps import crps_samples
from src.evaluation.interval_metrics import StreamingIntervalMetrics
from src.evaluation.joint_metrics import StreamingJointMetrics
from src.evaluation.point_metrics import StreamingPointMetrics
from src.evaluation.zero_metrics import StreamingZeroMetrics, active_prob_from_samples


def truth_ranges(pred_dir):
    """Per-variable truth min/max over the whole test stream (shared PINAW denom)."""
    tmin = np.full(len(TARGET_NAMES), np.inf)
    tmax = np.full(len(TARGET_NAMES), -np.inf)
    for shard in iter_shards(pred_dir):
        t = shard["truth"]
        tmin = np.minimum(tmin, t.min(axis=(0, 1)))
        tmax = np.maximum(tmax, t.max(axis=(0, 1)))
    return tmax - tmin


def evaluate_dir(pred_dir, stats, truth_range, reliability_out=None):
    train_scale = np.asarray(stats["train_scale_raw"], dtype=np.float64)
    point = {n: StreamingPointMetrics(positive_scale=float(stats["positive_scale"][d]) if d in ZERO_TARGET_INDICES else train_scale[d])
             for d, n in enumerate(TARGET_NAMES)}
    interval = {n: StreamingIntervalMetrics() for n in TARGET_NAMES}
    zero = {n: StreamingZeroMetrics() for n in ZERO_TARGET_NAMES}
    joint = StreamingJointMetrics(train_scale)
    crps_sum = np.zeros(len(TARGET_NAMES))
    crps_n = 0

    paths = shard_paths(pred_dir)
    for p in tqdm(paths, desc=f"eval {os.path.relpath(pred_dir)}", ncols=110):
        with np.load(p) as z:
            samples = z["samples"]      # [n,H,4,S]
            truth = z["truth"]          # [n,H,4]
            gate_prob = z["gate_prob"] if "gate_prob" in z.files else None
        pred_mean = samples.mean(axis=-1)
        for d, name in enumerate(TARGET_NAMES):
            point[name].update(pred_mean[..., d], truth[..., d])
            interval[name].update(samples[..., d, :], truth[..., d])
        crps_var = np.stack(
            [crps_samples(samples[..., d, :], truth[..., d]).mean() * truth[..., d].size
             for d in range(len(TARGET_NAMES))]
        )
        crps_sum += crps_var
        crps_n += truth[..., 0].size
        for gi, name in enumerate(ZERO_TARGET_NAMES):
            d = ZERO_TARGET_INDICES[gi]
            if gate_prob is not None:
                prob = gate_prob[..., gi]
            else:
                prob = active_prob_from_samples(samples[..., d, :])
            zero[name].update(prob, truth[..., d] > 0.0)
        joint.update(samples, truth)

    row = {}
    for d, name in enumerate(TARGET_NAMES):
        pm = point[name].compute()
        im = interval[name].compute(truth_range=float(truth_range[d]))
        crps = crps_sum[d] / max(crps_n, 1)
        row[f"{name}_mape_pos"] = pm["mape_pos"]
        row[f"{name}_mape_eps"] = pm["mape_eps"]
        row[f"{name}_mae"] = pm["mae"]
        row[f"{name}_rmse"] = pm["rmse"]
        row[f"{name}_crps"] = crps
        row[f"{name}_ncrps"] = crps / train_scale[d]
        for k, v in im.items():
            row[f"{name}_{k}"] = v
    row["electricity_mape"] = row["electricity_mape_pos"]  # elec truth has no zeros
    row["ncrps_mean"] = float(np.mean([row[f"{n}_ncrps"] for n in TARGET_NAMES]))
    for name in ZERO_TARGET_NAMES:
        zm = zero[name].compute()
        for k, v in zm.items():
            row[f"{name}_{k}"] = v
        if reliability_out is not None:
            reliability_out[name] = zero[name].reliability_bins()
    row.update(joint.compute())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction_root", default="artifacts/predictions")
    ap.add_argument("--stats", default="artifacts/data/low_carbon/preprocess_stats.json")
    ap.add_argument("--output", default="results/per_seed_metrics.csv")
    ap.add_argument("--skip_alignment_check", action="store_true")
    args = ap.parse_args()

    with open(args.stats, encoding="utf-8") as f:
        stats = json.load(f)

    dirs = sorted(glob.glob(os.path.join(args.prediction_root, "*", "seed_*")))
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "manifest.json"))]
    if not dirs:
        raise SystemExit(f"no prediction directories under {args.prediction_root}")
    print(f"found {len(dirs)} prediction dirs")

    if not args.skip_alignment_check and len(dirs) > 1:
        print("checking cross-model alignment of forecast_start_index/timestamps/truth ...")
        assert_alignment(dirs)

    rng = truth_ranges(dirs[0])
    rows = []
    os.makedirs("results", exist_ok=True)
    for d in dirs:
        man = load_manifest(d)
        reliability = {}
        row = evaluate_dir(d, stats, rng, reliability_out=reliability)
        row["model"] = man["model"]
        row["seed"] = man["seed"]
        rows.append(row)
        rel_path = os.path.join("results", f"reliability_{man['model']}_seed{man['seed']}.json")
        with open(rel_path, "w", encoding="utf-8") as f:
            json.dump(reliability, f, indent=2)

    df = pd.DataFrame(rows)
    cols = ["model", "seed"] + [c for c in df.columns if c not in ("model", "seed")]
    df = df[cols]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
