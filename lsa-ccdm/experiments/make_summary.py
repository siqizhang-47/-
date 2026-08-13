"""Final deliverable: results/summary_table.csv (method x year x metric) with
Diebold-Mariano significance vs. the `proposed` method (CRPS & Winkler daily
losses, HAC lag 7).

Usage: python experiments/make_summary.py [--methods frozen cosa ... proposed]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config, repo_root
from utils.seed import set_seed

KEY_METRICS = ["mae", "rmse", "bias", "crps", "crps_sum", "picp90", "ce90",
               "mpiw90", "pinaw90", "winkler90", "energy_score", "variogram_score"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = repo_root()
    parser.add_argument("--base-config", default=str(root / "configs/base.yaml"))
    parser.add_argument("--methods", nargs="*", default=None,
                        help="default: every results/<dir> containing daily_metrics.parquet")
    parser.add_argument("--reference", default="proposed")
    args = parser.parse_args()

    cfg = load_config(args.base_config)
    set_seed(cfg.seed)
    from evaluation.dm_test import dm_test

    results_dir = root / cfg.results_dir
    methods = args.methods or sorted(
        p.parent.name for p in results_dir.glob("*/daily_metrics.parquet"))
    frames = {m: pd.read_parquet(results_dir / m / "daily_metrics.parquet")
              for m in methods}

    rows = []
    ref = frames.get(args.reference)
    for method, df in frames.items():
        df = df.copy()
        df["year"] = pd.to_datetime(df["date"]).dt.year
        for year, g in df.groupby("year"):
            row = {"method": method, "year": year}
            for metric in KEY_METRICS:
                if metric in g:
                    row[metric] = float(g[metric].mean())
            if ref is not None and method != args.reference:
                merged = pd.merge(g[["date", "crps", "winkler90"]],
                                  ref[["date", "crps", "winkler90"]],
                                  on="date", suffixes=("_m", "_ref"))
                merged = merged[pd.to_datetime(merged["date"]).dt.year == year]
                if len(merged) > 10:
                    _, p_crps = dm_test(merged["crps_m"], merged["crps_ref"])
                    _, p_wink = dm_test(merged["winkler90_m"], merged["winkler90_ref"])
                    row["dm_p_crps_vs_ref"] = p_crps
                    row["dm_p_winkler_vs_ref"] = p_wink
            rows.append(row)

    out = pd.DataFrame(rows).sort_values(["method", "year"])
    out_path = results_dir / "summary_table.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
