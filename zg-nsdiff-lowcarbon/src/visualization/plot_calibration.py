"""Gate reliability diagrams (18.4) and multi-nominal coverage calibration (18.5).

python -m src.visualization.plot_calibration --per_seed results/per_seed_metrics.csv
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.low_carbon_schema import TARGET_NAMES, ZERO_TARGET_NAMES

MODEL_LABEL = {"nsdiff": "NsDiff", "d3u": "D3U", "wavestitch": "WaveStitch",
               "zg_nsdiff": "ZG-NsDiff"}
LEVELS = [50, 80, 90, 95]


def plot_gate_reliability(results_dir, output_dir, seed):
    for name in ZERO_TARGET_NAMES:
        fig, ax = plt.subplots(figsize=(6, 5))
        any_data = False
        for f in sorted(glob.glob(os.path.join(results_dir, f"reliability_*_seed{seed}.json"))):
            model = os.path.basename(f)[len("reliability_"):-len(f"_seed{seed}.json")]
            with open(f, encoding="utf-8") as fh:
                rel = json.load(fh)
            if name not in rel:
                continue
            bins = rel[name]
            conf = [b["confidence"] for b in bins]
            acc = [b["accuracy"] for b in bins]
            ax.plot(conf, acc, marker="o", label=MODEL_LABEL.get(model, model))
            any_data = True
        if not any_data:
            plt.close(fig)
            continue
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.set_xlabel("predicted active probability")
        ax.set_ylabel("empirical active frequency")
        ax.set_title(f"Gate reliability — {name} (10 equal-frequency bins)")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(output_dir, f"gate_reliability_{name}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"wrote {path}")


def plot_coverage_calibration(per_seed_csv, output_dir):
    df = pd.read_csv(per_seed_csv)
    agg = df.groupby("model").mean(numeric_only=True)
    fig, axes = plt.subplots(1, len(TARGET_NAMES), figsize=(4.2 * len(TARGET_NAMES), 4),
                             sharey=True)
    for ax, name in zip(axes, TARGET_NAMES):
        for model in agg.index:
            emp = [agg.loc[model, f"{name}_picp_{lv}"] for lv in LEVELS]
            ax.plot([lv / 100 for lv in LEVELS], emp, marker="o",
                    label=MODEL_LABEL.get(model, model))
        ax.plot([0.4, 1.0], [0.4, 1.0], "k--", lw=1)
        ax.set_title(name)
        ax.set_xlabel("nominal coverage")
    axes[0].set_ylabel("empirical coverage")
    axes[0].legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "coverage_calibration.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--per_seed", default="results/per_seed_metrics.csv")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output_dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plot_gate_reliability(args.results_dir, args.output_dir, args.seed)
    if os.path.exists(args.per_seed):
        plot_coverage_calibration(args.per_seed, args.output_dir)


if __name__ == "__main__":
    main()
