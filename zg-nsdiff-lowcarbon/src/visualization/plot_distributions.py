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

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "Microsoft YaHei",
                                   "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from src.baselines.prediction_contract import iter_shards
from src.data.low_carbon_schema import TARGET_NAMES, ZERO_TARGET_INDICES

MODEL_LABEL = {"nsdiff": "NsDiff", "d3u": "D3U", "wavestitch": "WaveStitch",
               "zg_nsdiff": "ZG-NsDiff"}


def kde_curve(values, xs):
    """gaussian_kde with a fallback for degenerate (near-constant) data:
    returns a normalized histogram density instead of crashing."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or np.std(values) < 1e-9:
        density = np.zeros_like(xs)
        if len(values):
            idx = int(np.argmin(np.abs(xs - values.mean())))
            width = max(xs[1] - xs[0], 1e-9) if len(xs) > 1 else 1.0
            density[idx] = 1.0 / width  # point mass rendered as a spike
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
        s = shard["samples"][..., d, :]
        flat = s.ravel()
        if len(flat) > max_values:
            flat = rng.choice(flat, max_values, replace=False)
        gen.append(flat)
    return np.concatenate(truth), np.concatenate(gen)


VAR_LABEL = {"electricity": "电负荷", "cooling": "冷负荷", "heating": "热负荷", "pv": "光伏出力"}
SUB = "abcd"


def plot_combined(model_dirs, model, output_dir):
    """Template figure: one 2x2 panel, per-variable truth vs predicted KDE."""
    pred_dir = model_dirs[model]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for d, (name, ax) in enumerate(zip(TARGET_NAMES, axes.ravel())):
        truth, gen = collect(pred_dir, d)
        hi = np.quantile(truth, 0.999)
        lo = min(truth.min(), 0.0)
        xs = np.linspace(lo, hi if hi > lo else lo + 1.0, 300)
        ax.plot(xs, kde_curve(truth, xs), label="真实分布", lw=2, color="k")
        ax.plot(xs, kde_curve(gen, xs), label="预测分布", lw=2)
        ax.set_xlabel("kW")
        ax.set_ylabel("概率密度")
        ax.set_title(f"({SUB[d]}) {VAR_LABEL[name]}")
        ax.legend()
    fig.suptitle(f"真实分布与预测分布对比（{MODEL_LABEL.get(model, model)}）")
    fig.tight_layout()
    path = os.path.join(output_dir, "distribution_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction_root", default="artifacts/predictions")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default=None,
                    help="model for the combined template figure (default: zg_nsdiff if present)")
    ap.add_argument("--output_dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_dirs = {}
    for d in sorted(glob.glob(os.path.join(args.prediction_root, "*", f"seed_{args.seed}"))):
        model = os.path.basename(os.path.dirname(d))
        model_dirs[model] = d
    if not model_dirs:
        raise SystemExit("no predictions found")

    combined_model = args.model or ("zg_nsdiff" if "zg_nsdiff" in model_dirs
                                    else next(iter(model_dirs)))
    plot_combined(model_dirs, combined_model, args.output_dir)

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
            ax2.plot(xs, kde_curve(pos_t, xs), label="truth", lw=2, color="k")
            for m, (_, g) in data.items():
                gp = g[g > 0]
                if len(gp) > 100:
                    ax2.plot(xs, kde_curve(gp, xs), label=MODEL_LABEL.get(m, m))
            ax2.set_title(f"{name}: p(Y | Y > 0)")
            ax2.set_xlabel("kW")
            ax2.legend()
        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            xs = np.linspace(truth.min(), np.quantile(truth, 0.999), 300)
            ax.plot(xs, kde_curve(truth, xs), label="truth", lw=2, color="k")
            for m, (_, g) in data.items():
                ax.plot(xs, kde_curve(g, xs), label=MODEL_LABEL.get(m, m))
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
