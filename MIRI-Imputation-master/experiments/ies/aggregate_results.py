"""Aggregate per-run JSON results into mean +/- std summary tables.

Reads ``results/ies/raw/*.json`` and writes:
    summary_by_seed.csv      one row per (method, mechanism, rate, seed)
    summary_mean_std.csv     mean & std over seeds
    rank_by_joint_mmd.csv    summary sorted by joint_mmd (lower is better)
    rank_by_corr_error.csv   summary sorted by corr_error (lower is better)

All ranked metrics (MAE / RMSE / MMD / corr_error) are lower-is-better.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd


METRIC_COLS = [
    "mae_std",
    "rmse_std",
    "masked_mmd",
    "joint_mmd",
    "corr_error",
    "runtime_sec",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/ies/raw")
    parser.add_argument("--output", default="results/ies/summaries")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    rows = []
    n_failed = 0
    for path in sorted(glob.glob(os.path.join(args.input, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if obj.get("status") != "ok":
            n_failed += 1
            continue
        row = {
            "method": obj["method"],
            "mechanism": obj["mechanism"],
            "missing_rate": obj["missing_rate"],
            "seed": obj["seed"],
            "runtime_sec": obj.get("runtime_sec"),
        }
        row.update(obj["metrics"])
        rows.append(row)

    if not rows:
        print(f"No successful results found in {args.input} (failed/skipped: {n_failed}).")
        return

    df = pd.DataFrame(rows)
    by_seed_path = os.path.join(args.output, "summary_by_seed.csv")
    df.to_csv(by_seed_path, index=False)

    metric_cols = [c for c in METRIC_COLS if c in df.columns]
    summary = df.groupby(["method", "mechanism", "missing_rate"])[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary_path = os.path.join(args.output, "summary_mean_std.csv")
    summary.to_csv(summary_path, index=False)

    if "joint_mmd_mean" in summary.columns:
        rank_joint = summary.sort_values(["mechanism", "missing_rate", "joint_mmd_mean"])
        rank_joint.to_csv(os.path.join(args.output, "rank_by_joint_mmd.csv"), index=False)
    if "corr_error_mean" in summary.columns:
        rank_corr = summary.sort_values(["mechanism", "missing_rate", "corr_error_mean"])
        rank_corr.to_csv(os.path.join(args.output, "rank_by_corr_error.csv"), index=False)

    print(f"Aggregated {len(df)} successful runs (failed/skipped: {n_failed}).")
    print(f"  -> {by_seed_path}")
    print(f"  -> {summary_path}")
    print(f"  -> {os.path.join(args.output, 'rank_by_joint_mmd.csv')}")
    print(f"  -> {os.path.join(args.output, 'rank_by_corr_error.csv')}")


if __name__ == "__main__":
    main()
