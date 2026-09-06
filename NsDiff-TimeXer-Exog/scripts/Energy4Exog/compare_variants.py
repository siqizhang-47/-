#!/usr/bin/env python3
"""Compare Energy4Exog estimator variants after the corresponding runs finish."""
import csv
import json
from pathlib import Path

ROOT = Path("energy4_exog_outputs")
paths = {
    "mean_only": ROOT / "mean_only" / "seed_42" / "metrics.json",
    "mean_and_var_v1": ROOT / "mean_and_var" / "seed_42" / "metrics.json",
    "mean_and_var_v2": ROOT / "mean_and_var_v2" / "seed_42" / "metrics.json",
}

rows = {}
for name, path in paths.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the corresponding experiment first.")
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    rows[name] = payload["standardized_metrics"]

metrics = ["crps", "qice", "mae", "mse"]
out = ROOT / "comparison.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "metric",
        "mean_only",
        "mean_and_var_v1",
        "mean_and_var_v2",
        "v1_minus_mean_only",
        "v2_minus_mean_only",
        "v2_minus_v1",
    ])
    for metric in metrics:
        m = float(rows["mean_only"][metric])
        v1 = float(rows["mean_and_var_v1"][metric])
        v2 = float(rows["mean_and_var_v2"][metric])
        w.writerow([metric.upper(), m, v1, v2, v1 - m, v2 - m, v2 - v1])

print(f"Saved {out}")
for metric in metrics:
    m = float(rows["mean_only"][metric])
    v1 = float(rows["mean_and_var_v1"][metric])
    v2 = float(rows["mean_and_var_v2"][metric])
    print(
        f"{metric.upper():4s}: mean_only={m:.6f}  V1={v1:.6f}  V2={v2:.6f}  "
        f"V2-V1={v2-v1:+.6f}  V2-mean_only={v2-m:+.6f}"
    )

# Lower is better for all four metrics; a visual comparison is convenient for ablations.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["mean_only", "V1", "V2"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric in zip(axes.flat, metrics):
        vals = [
            float(rows["mean_only"][metric]),
            float(rows["mean_and_var_v1"][metric]),
            float(rows["mean_and_var_v2"][metric]),
        ]
        bars = ax.bar(labels, vals)
        ax.set_title(f"{metric.upper()} (lower is better)")
        ax.grid(axis="y", alpha=0.2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.4g}",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Energy4Exog variance-estimator ablation")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = ROOT / "comparison_metrics.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")
except Exception as exc:
    print(f"Metric CSV was saved, but comparison plot was skipped: {exc}")
