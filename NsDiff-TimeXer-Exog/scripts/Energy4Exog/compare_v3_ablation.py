#!/usr/bin/env python3
"""Aggregate the V3 ablation ladder into one table (mean ± std over seeds).

Columns: CRPS, QICE(%), MAE, RMSE, SlopeMAE, PeakTime(h), PICP/MPIW 50/80/95,
PV negative-sample rate and nighttime 95 % PI width (original units).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

COLUMNS = [
    ("crps", "CRPS"), ("qice", "QICE(%)"), ("mae", "MAE"), ("rmse", "RMSE"),
    ("slope_mae", "SlopeMAE"), ("peak_time", "PeakTime(h)"),
    ("picp_50", "PICP50"), ("picp_80", "PICP80"), ("picp_95", "PICP95"),
    ("mpiw_50", "MPIW50"), ("mpiw_80", "MPIW80"), ("mpiw_95", "MPIW95"),
    ("pv_neg", "PV neg. rate"), ("pv_night_w95", "PV night W95"),
    ("direct_mae", "Direct-mean MAE"),
]


def collect(path: Path) -> dict:
    m = json.loads(path.read_text(encoding="utf-8"))
    s, d = m["standardized_metrics"], m.get("diagnostics") or {}
    peak = d.get("peak_timing_error_hours_mc", {})
    return {
        "crps": s["crps"], "qice": 100.0 * s["qice"], "mae": s["mae"], "rmse": float(np.sqrt(s["mse"])),
        "slope_mae": d.get("slope_mae_mc", np.nan),
        "peak_time": float(np.mean(list(peak.values()))) if peak else np.nan,
        **{f"picp_{l}": d.get(f"picp_{l}", np.nan) for l in ("50", "80", "95")},
        **{f"mpiw_{l}": d.get(f"mpiw_{l}", np.nan) for l in ("50", "80", "95")},
        "pv_neg": d.get("pv", {}).get("negative_sample_rate_raw_ensemble", np.nan),
        "pv_night_w95": d.get("pv", {}).get("nighttime_95pi_width_original_unit", np.nan),
        "direct_mae": d.get("direct_mean_mae", np.nan),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./energy4_exog_outputs/v3_ablation")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--variants", nargs="+", default=[f"V{i}" for i in range(8)])
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for v in args.variants:
        per_seed = []
        for seed in args.seeds:
            p = root / v / f"seed_{seed}" / "metrics.json"
            if p.exists():
                per_seed.append(collect(p))
        if not per_seed:
            print(f"skip {v}: no results")
            continue
        row = {"variant": v, "n_seeds": len(per_seed)}
        for key, _ in COLUMNS:
            vals = np.array([r[key] for r in per_seed], dtype=float)
            row[key] = float(np.nanmean(vals))
            row[key + "_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    if not rows:
        return
    out_csv = root / "ablation_table.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n_seeds"] + [c for _, c in COLUMNS] + [c + "_std" for _, c in COLUMNS])
        for r in rows:
            w.writerow([r["variant"], r["n_seeds"]] + [f"{r[k]:.5f}" for k, _ in COLUMNS] + [f"{r[k + '_std']:.5f}" for k, _ in COLUMNS])
    md = ["| Variant | " + " | ".join(c for _, c in COLUMNS) + " |", "|---|" + "---:|" * len(COLUMNS)]
    for r in rows:
        md.append("| " + r["variant"] + " | " + " | ".join(f"{r[k]:.4f}±{r[k + '_std']:.4f}" if r["n_seeds"] > 1 else f"{r[k]:.4f}" for k, _ in COLUMNS) + " |")
    (root / "ablation_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"Saved {out_csv} and {root / 'ablation_table.md'}")


if __name__ == "__main__":
    main()
