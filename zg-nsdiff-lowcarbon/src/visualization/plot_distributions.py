"""figure3: Actual vs. predicted load distribution.

Per-variable subplots (a)(b)(c): KDE of the actual test values (black) and of
each model's predicted samples.

python -m src.visualization.plot_distributions --prediction_root artifacts/predictions
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from src.baselines.prediction_contract import iter_shards
from src.data.low_carbon_schema import TARGET_NAMES

MODEL_LABEL = {"deepvar": "DeepVAR", "d3u": "WCRD", "wavestitch": "WaveStitch",
               "nsdiff": "NsDiff"}
MODEL_ORDER = ["deepvar", "wavestitch", "nsdiff", "d3u"]
VAR_LABEL = {"electricity": "Electricity", "cooling": "Cooling", "heat": "Heat"}
SUB = "abc"


def kde_curve(values, xs):
    """gaussian_kde with a fallback for degenerate (near-constant) data."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or np.std(values) < 1e-9:
        density = np.zeros_like(xs)
        if len(values):
            idx = int(np.argmin(np.abs(xs - values.mean())))
            width = max(xs[1] - xs[0], 1e-9) if len(xs) > 1 else 1.0
            density[idx] = 1.0 / width
        return density
    try:
        return gaussian_kde(values)(xs)
    except np.linalg.LinAlgError:
        hist, edges = np.histogram(values, bins=min(50, max(len(xs) // 6, 10)),
                                   range=(xs.min(), xs.max()), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        return np.interp(xs, centers, hist, left=0, right=0)


def collect(pred_dir, d, max_values=200_000, rng=None):
    rng = rng or np.random.default_rng(0)
    truth, gen = [], []
    for shard in iter_shards(pred_dir):
        truth.append(shard["truth"][..., d].ravel())
        flat = shard["samples"][..., d, :].ravel()
        if len(flat) > max_values:
            flat = rng.choice(flat, max_values, replace=False)
        gen.append(flat)
    return np.concatenate(truth), np.concatenate(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction_root", default="artifacts/predictions")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output", default="figures/figure3_load_distribution.png")
    args = ap.parse_args()

    model_dirs = {os.path.basename(os.path.dirname(d)): d
                  for d in sorted(glob.glob(os.path.join(
                      args.prediction_root, "*", f"seed_{args.seed}")))}
    if not model_dirs:
        raise SystemExit("no predictions found")
    models = [m for m in MODEL_ORDER if m in model_dirs]
    models += [m for m in model_dirs if m not in models]

    fig, axes = plt.subplots(1, len(TARGET_NAMES), figsize=(5.2 * len(TARGET_NAMES), 4.2))
    for d, (name, ax) in enumerate(zip(TARGET_NAMES, axes)):
        data = {m: collect(model_dirs[m], d) for m in models}
        truth = next(iter(data.values()))[0]
        lo = min(float(truth.min()), 0.0)
        hi = float(np.quantile(truth, 0.999))
        xs = np.linspace(lo, hi if hi > lo else lo + 1.0, 300)
        ax.plot(xs, kde_curve(truth, xs), lw=2, color="k", label="Actual")
        for m in models:
            ax.plot(xs, kde_curve(data[m][1], xs), lw=1.5, label=MODEL_LABEL.get(m, m))
        ax.set_xlabel("kW")
        ax.set_ylabel("Density")
        ax.set_title(f"({SUB[d]}) {VAR_LABEL[name]}")
        ax.legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
