#!/usr/bin/env python3
"""Aggregate 3-seed ExoMV-D results into publication-ready table files.

Outputs:
  summary/three_seed_metrics.csv        per-seed raw metrics
  summary/experiment_table.csv          mean ± std row + baseline placeholders
  summary/experiment_table.md
  summary/experiment_table.tex
  summary/experiment_table.png
  summary/fig1_prediction_intervals.png selected seed figure
  summary/fig2_marginal_pdfs.png        selected seed figure

QICE is shown in percentage points in the publication table (raw JSON remains 0..1).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

METRICS = ["crps", "qice", "es", "vs", "mae", "vmae"]
DISPLAY = ["CRPS↓", "QICE↓", "ES↓", "VS↓", "MAE↓", "VMAE↓"]
BASELINES = ["TimeGrad", "CSDI", "TimeDiff", "TMDM", "NsDiff"]


def fmt(mean: float, std: float, metric: str) -> str:
    if metric == "qice":
        mean *= 100.0
        std *= 100.0
    return f"{mean:.4f} ± {std:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./energy4_exog_outputs/exomvd_v2_calibrated")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--model-name", default="ExoMV-D")
    args = ap.parse_args()

    root = Path(args.root)
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        path = root / f"seed_{seed}" / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        m = payload["standardized_metrics"]
        row = {"seed": seed}
        for k in METRICS:
            if k not in m:
                raise KeyError(f"{path} does not contain standardized_metrics.{k}")
            row[k] = float(m[k])
        rows.append(row)

    # Per-seed raw values.
    with (summary / "three_seed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed"] + METRICS)
        w.writeheader()
        w.writerows(rows)

    means = {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
    stds = {k: float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0 for k in METRICS}
    values = [fmt(means[k], stds[k], k) for k in METRICS]

    table_rows = [[name] + ["—"] * len(METRICS) for name in BASELINES]
    table_rows.append([args.model_name] + values)

    with (summary / "experiment_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model"] + DISPLAY)
        w.writerows(table_rows)

    md = ["| Model | " + " | ".join(DISPLAY) + " |",
          "|---|" + "|".join(["---:"] * len(DISPLAY)) + "|"]
    for row in table_rows:
        md.append("| " + " | ".join(row) + " |")
    (summary / "experiment_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Simple LaTeX table; baseline placeholders can be replaced later.
    tex = ["\\begin{tabular}{l" + "c" * len(METRICS) + "}",
           "\\toprule",
           "Model & " + " & ".join([x.replace("↓", "$\\downarrow$") for x in DISPLAY]) + " \\\\",
           "\\midrule"]
    for row in table_rows:
        tex.append(" & ".join(row) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (summary / "experiment_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # Render PNG table.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12.5, 3.4))
        ax.axis("off")
        tbl = ax.table(cellText=table_rows, colLabels=["Model"] + DISPLAY,
                       cellLoc="center", colLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10.5)
        tbl.scale(1, 1.65)
        # Left-align model names and bold our row.
        for r in range(len(table_rows) + 1):
            tbl[(r, 0)].get_text().set_ha("left")
        last = len(table_rows)
        for c in range(len(METRICS) + 1):
            tbl[(last, c)].get_text().set_weight("bold")
        fig.tight_layout()
        fig.savefig(summary / "experiment_table.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"Table PNG skipped: {exc}")

    # Copy figures from the seed with the best CRPS; this choice is recorded.
    best = min(rows, key=lambda r: r["crps"])
    best_seed = int(best["seed"])
    src_dir = root / f"seed_{best_seed}"
    for fn in ["fig1_prediction_intervals.png", "fig2_marginal_pdfs.png", "fig3_pearson_correlation.png"]:
        src = src_dir / fn
        if src.exists():
            shutil.copy2(src, summary / fn)
    (summary / "selected_figure_seed.txt").write_text(
        f"seed={best_seed}\npolicy=lowest CRPS among requested seeds\n",
        encoding="utf-8",
    )

    print("\nThree-seed ExoMV-D summary")
    print("Seeds:", args.seeds)
    for k, label in zip(METRICS, DISPLAY):
        print(f"{label:7s} {fmt(means[k], stds[k], k)}")
    print(f"\nSaved publication files to: {summary}")
    print(f"Figures copied from seed {best_seed} (lowest CRPS); metrics are always 3-seed mean ± std.")


if __name__ == "__main__":
    main()
