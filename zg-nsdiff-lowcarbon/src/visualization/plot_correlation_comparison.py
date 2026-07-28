"""figure2: The Pearson correlation matrices of the real and generated sample.

Left : Variable Correlation Matrix (Generated Data) — model samples
Right: Variable Correlation Matrix (Real Data)      — test truth

Streamed shard-by-shard so memory stays flat with 1000 samples per window.

python -m src.visualization.plot_correlation_comparison \
    --model nsdiff --seed 1 --output figures/figure2_correlation_matrices.png
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.baselines.prediction_contract import shard_paths
from src.data.low_carbon_schema import NUM_TARGETS

VAR_LABEL = ["Electricity", "Cooling", "Heat"]
D = NUM_TARGETS


class StreamingCorr:
    def __init__(self, dim):
        self.n = 0
        self.s1 = np.zeros(dim, dtype=np.float64)
        self.s2 = np.zeros((dim, dim), dtype=np.float64)

    def update(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.n += X.shape[0]
        self.s1 += X.sum(axis=0)
        self.s2 += X.T @ X

    def corr(self):
        mean = self.s1 / self.n
        cov = self.s2 / self.n - np.outer(mean, mean)
        std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        return np.clip(cov / np.outer(std, std), -1.0, 1.0)


def compute_matrices(pred_dir, samples_per_shard=20, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    gen_acc, real_acc = StreamingCorr(D), StreamingCorr(D)
    paths = shard_paths(pred_dir)
    if not paths:
        raise SystemExit(f"no prediction shards under {pred_dir}")
    for p in tqdm(paths, desc="correlation", ncols=100):
        with np.load(p) as z:
            samples = z["samples"]        # [n,H,D,S]
            truth = z["truth"]            # [n,H,D]
        S = samples.shape[-1]
        pick = rng.choice(S, min(samples_per_shard, S), replace=False)
        gen = samples[..., pick]
        gen_acc.update(np.moveaxis(gen, 2, -1).reshape(-1, D))
        real_acc.update(truth.reshape(-1, D))
    return gen_acc.corr(), real_acc.corr()


def draw_panel(ax, R, title):
    im = ax.imshow(R, vmin=-1, vmax=1, cmap="RdBu")
    ax.set_xticks(range(D), VAR_LABEL, fontsize=12)
    ax.set_yticks(range(D), VAR_LABEL, fontsize=12, rotation=90, va="center")
    for i in range(D):
        for j in range(D):
            color = "white" if abs(R[i, j]) >= 0.45 else "black"
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, color=color)
    ax.set_title(title, fontsize=13)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction_root", default="artifacts/predictions")
    ap.add_argument("--model", default=None,
                    help="model whose samples form the Generated panel "
                         "(default: nsdiff if present, else first found)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--samples_per_shard", type=int, default=20)
    ap.add_argument("--output", default="figures/figure2_correlation_matrices.png")
    args = ap.parse_args()

    model_dirs = {os.path.basename(os.path.dirname(d)): d
                  for d in sorted(glob.glob(os.path.join(
                      args.prediction_root, "*", f"seed_{args.seed}")))}
    if not model_dirs:
        raise SystemExit(f"no predictions under {args.prediction_root}")
    model = args.model or ("nsdiff" if "nsdiff" in model_dirs
                           else next(iter(model_dirs)))
    if model not in model_dirs:
        raise SystemExit(f"model '{model}' not found; available: {list(model_dirs)}")

    R_gen, R_real = compute_matrices(model_dirs[model], args.samples_per_shard)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for ax, R, title in [
        (axes[0], R_gen, "Variable Correlation Matrix (Generated Data)"),
        (axes[1], R_real, "Variable Correlation Matrix (Real Data)"),
    ]:
        im = draw_panel(ax, R, title)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"wrote {args.output}  (generated panel model: {model})")


if __name__ == "__main__":
    main()
