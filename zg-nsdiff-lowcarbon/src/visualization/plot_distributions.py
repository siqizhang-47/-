"""Truth vs predicted distributions per variable (spec 18.2).

electricity: KDE of truth + each model's samples.
zero-inflated: dual panel — zero-rate bars + positive-only KDE.

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
from src.data.low_carbon_schema import TARGET_NAMES, ZERO_TARGET_INDICES

MODEL_LABEL = {"nsdiff": "NsDiff", "d3u": "D3U", "wavestitch": "WaveStitch",
               "zg_nsdiff": "ZG-NsDiff"}


def collect(pred_dir, d, max_values=200_000, rng=None):
    rng = rng or np.random.default_rng(0)
    truth, gen = [], []
    for shard in iter_shards(pred_dir):
        truth.append(shard["truth"][..., d].ravel())
        s = shard["samples"][..., d, :]
        flat = s.ravel()
        if len(flat) > max_values:
            flat = rng.choice(flat, max_values, replace=False)
        gen.append(flat)
    return np.concatenate(truth), np.concatenate(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction_root", default="artifacts/predictions")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output_dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_dirs = {}
    for d in sorted(glob.glob(os.path.join(args.prediction_root, "*", f"seed_{args.seed}"))):
        model = os.path.basename(os.path.dirname(d))
        model_dirs[model] = d
    if not model_dirs:
        raise SystemExit("no predictions found")

    for d, name in enumerate(TARGET_NAMES):
        data = {m: collect(p, d) for m, p in model_dirs.items()}
        truth = next(iter(data.values()))[0]
        if d in ZERO_TARGET_INDICES:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            labels = ["truth"] + [MODEL_LABEL.get(m, m) for m in data]
            zr = [float((truth == 0).mean())] + [float((g == 0).mean()) for _, g in data.values()]
            ax1.bar(labels, zr)
            ax1.set_ylabel("P(Y = 0)")
            ax1.set_title(f"{name}: zero rate")
            ax1.tick_params(axis="x", rotation=30)
            pos_t = truth[truth > 0]
            xs = np.linspace(0, np.quantile(pos_t, 0.995), 300)
            ax2.plot(xs, gaussian_kde(pos_t)(xs), label="truth", lw=2, color="k")
            for m, (_, g) in data.items():
                gp = g[g > 0]
                if len(gp) > 100:
                    ax2.plot(xs, gaussian_kde(gp)(xs), label=MODEL_LABEL.get(m, m))
            ax2.set_title(f"{name}: p(Y | Y > 0)")
            ax2.set_xlabel("kW")
            ax2.legend()
        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            xs = np.linspace(truth.min(), np.quantile(truth, 0.999), 300)
            ax.plot(xs, gaussian_kde(truth)(xs), label="truth", lw=2, color="k")
            for m, (_, g) in data.items():
                ax.plot(xs, gaussian_kde(g)(xs), label=MODEL_LABEL.get(m, m))
            ax.set_title(f"{name}: distribution (test)")
            ax.set_xlabel("kW")
            ax.legend()
        fig.tight_layout()
        path = os.path.join(args.output_dir, f"distribution_{name}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
