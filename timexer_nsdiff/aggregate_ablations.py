#!/usr/bin/env python
"""Collect results/<ablation>/summary.json into the ablation table of section 20.

    python aggregate_ablations.py                      # markdown + csv to results/
    python aggregate_ablations.py --metric CRPS RMSE
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from src.config import ABLATIONS

TARGETS = ["Electricity", "PV", "Cooling", "Heat"]


def load(results_dir):
    rows = {}
    for name in ABLATIONS:
        path = os.path.join(results_dir, name, "summary.json")
        if not os.path.exists(path):
            # a single-seed run writes only the per-seed metrics file
            single = os.path.join(results_dir, name, "seed1", "metrics_test.json")
            if os.path.exists(single):
                with open(single) as f:
                    m = json.load(f)
                rows[name] = {
                    "per_target": {t: {k: {"mean": v, "std": 0.0} for k, v in m["per_target"][t].items()}
                                   for t in m["per_target"]},
                    "overall": {k: ({"mean": v, "std": 0.0} if not isinstance(v, str) else v)
                                for k, v in m["overall"].items()},
                    "seeds": [m["seed"]],
                }
            continue
        with open(path) as f:
            rows[name] = json.load(f)
    return rows


def fmt(cell, prec=3):
    if cell is None:
        return "-"
    return f"{cell['mean']:.{prec}f}±{cell['std']:.{prec}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--metric", nargs="+", default=["CRPS", "MAE", "RMSE", "PICP@90", "QICE"])
    args = ap.parse_args()

    rows = load(args.results_dir)
    if not rows:
        raise SystemExit(f"no results found under {args.results_dir}/")

    lines, csv_rows = [], []
    lines.append("# Ablation results (Oracle Weather)\n")
    for metric in args.metric:
        lines.append(f"\n## {metric}\n")
        header = ["model", "description"] + TARGETS + ["mean"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for name in ABLATIONS:
            if name not in rows:
                continue
            r = rows[name]
            cells = [fmt(r["per_target"][t].get(metric)) for t in TARGETS]
            overall = fmt(r["overall"].get(metric))
            seeds = ",".join(str(s) for s in r.get("seeds", []))
            lines.append(f"| {name} (seeds {seeds}) | {ABLATIONS[name]['desc']} | "
                         + " | ".join(cells) + f" | {overall} |")
            csv_rows.append([metric, name] + [r["per_target"][t][metric]["mean"] for t in TARGETS]
                            + [r["overall"][metric]["mean"]])

    lines.append("\n## Multivariate scores\n")
    lines.append("| model | EnergyScore | VariogramScore |")
    lines.append("|---|---|---|")
    for name in ABLATIONS:
        if name not in rows:
            continue
        o = rows[name]["overall"]
        lines.append(f"| {name} | {fmt(o.get('EnergyScore'), 4)} | {fmt(o.get('VariogramScore'), 4)} |")

    md = "\n".join(lines)
    print(md)
    with open(os.path.join(args.results_dir, "ablation_table.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(args.results_dir, "ablation_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "model"] + TARGETS + ["mean"])
        w.writerows(csv_rows)
    print(f"\n[written] {args.results_dir}/ablation_table.md and .csv")


if __name__ == "__main__":
    main()
