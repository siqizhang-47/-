"""Correlation heatmaps (spec 18.1).

python -m src.visualization.plot_correlations --output_dir figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.low_carbon_schema import TARGET_NAMES, WEATHER_NAMES


def heatmap(matrix, xlabels, ylabels, title, path):
    fig, ax = plt.subplots(figsize=(1.1 * len(xlabels) + 2, 1.1 * len(ylabels) + 1.5))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", default="artifacts/data/low_carbon")
    ap.add_argument("--output_dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    target = np.load(os.path.join(args.artifacts_dir, "target_raw.npy"))
    weather = np.load(os.path.join(args.artifacts_dir, "weather_raw.npy"))
    both = np.concatenate([target, weather], axis=1)
    valid = np.isfinite(both).all(axis=1)
    both = both[valid]

    R = np.corrcoef(both, rowvar=False)
    heatmap(R[:4, :4], TARGET_NAMES, TARGET_NAMES,
            "Target correlations (all valid observations)",
            os.path.join(args.output_dir, "target_correlation_heatmap.png"))
    heatmap(R[:4, 4:], WEATHER_NAMES, TARGET_NAMES,
            "Target–weather correlations (all valid observations)",
            os.path.join(args.output_dir, "target_weather_correlation_heatmap.png"))


if __name__ == "__main__":
    main()
