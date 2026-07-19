"""Aggregate per-seed metrics into mean ± sample std per model (spec 15.9).

python -m src.evaluation.aggregate_seeds \
    --input results/per_seed_metrics.csv --output results/summary_metrics.csv
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/per_seed_metrics.csv")
    ap.add_argument("--output", default="results/summary_metrics.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    metric_cols = [c for c in df.columns if c not in ("model", "seed")]
    g = df.groupby("model")[metric_cols]
    mean = g.mean().add_suffix("_mean")
    std = g.std(ddof=1).fillna(0.0).add_suffix("_std")
    n = g.size().rename("n_seeds")
    out = pd.concat([mean, std, n], axis=1).reset_index()
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(out)} models)")


if __name__ == "__main__":
    main()
