"""Generate the report figures for the IES benchmark.

Produces:
    bar_mmd_mcar.png                 joint MMD under MCAR 20/40/60 (Figure 1)
    corr_matrix_triptych.png         true vs MIRI vs best-baseline corr (Figure 2)
    scatter_pv_irradiation_compare.png   PV vs irradiation (Figure 3)
    scatter_temp_load_compare.png    temp vs cooling/heating load (Figure 4)

All figures are read from the saved summary CSV and the per-run ``.npz`` files,
so no experiment is re-run here.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METHOD_ORDER = ["mean", "knn", "mice", "missforest", "gain", "hyperimpute", "miri"]

# short display names for the correlation-matrix / scatter axes
SHORT_NAMES = [
    "Elec load", "Cooling", "Heating", "PV gen",
    "Irradiation", "Temp", "Humidity", "Wind",
]

PV, IRRADIATION, TEMP, COOLING, HEATING = 3, 4, 5, 1, 2


def _rate_tag(rate: float) -> str:
    return f"{rate:g}".replace(".", "")


# --------------------------------------------------------------------------- #
# Figure 1: MCAR MMD bar chart
# --------------------------------------------------------------------------- #
def plot_bar_mmd_mcar(summary_csv: str, output_path: str, metric: str = "joint_mmd"):
    df = pd.read_csv(summary_csv)
    df = df[df["mechanism"] == "mcar"].copy()
    if df.empty:
        print("plot_bar_mmd_mcar: no MCAR rows in summary; skipping.")
        return

    rates = sorted(df["missing_rate"].unique())
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    mean_col, std_col = f"{metric}_mean", f"{metric}_std"

    x = np.arange(len(rates))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for k, method in enumerate(methods):
        means, stds = [], []
        for r in rates:
            sub = df[(df["method"] == method) & (df["missing_rate"] == r)]
            means.append(float(sub[mean_col].iloc[0]) if len(sub) else np.nan)
            stds.append(float(sub[std_col].iloc[0]) if len(sub) and std_col in sub else 0.0)
        ax.bar(x + k * width, means, width, yerr=stds, capsize=3,
               label=method, color=cmap(k % 10))

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([f"{int(r*100)}%" for r in rates])
    ax.set_xlabel("MCAR missing rate")
    ax.set_ylabel(f"{metric.replace('_', ' ')} (lower is better)")
    ax.set_title(f"{metric.replace('_', ' ').title()} under MCAR Missingness")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  -> {output_path}")


# --------------------------------------------------------------------------- #
# Best-baseline selection & npz loading
# --------------------------------------------------------------------------- #
def select_best_baseline(summary_csv: str, mechanism: str, missing_rate: float) -> str | None:
    """Non-MIRI method with lowest joint_mmd_mean (tie-break: corr_error_mean)."""
    df = pd.read_csv(summary_csv)
    sub = df[(df["mechanism"] == mechanism)
             & (np.isclose(df["missing_rate"], missing_rate))
             & (df["method"] != "miri")].copy()
    if sub.empty:
        return None
    sort_cols = [c for c in ["joint_mmd_mean", "corr_error_mean"] if c in sub.columns]
    sub = sub.sort_values(sort_cols)
    return str(sub["method"].iloc[0])


def load_imputation_npz(imputation_path: str) -> dict:
    data = np.load(imputation_path, allow_pickle=True)
    return {
        "X_test_true": data["X_test_true"],
        "X_test_imp": data["X_test_imp"],
        "M_test": data["M_test"],
        "feature_cols": list(data["feature_cols"]),
        "method": str(data["method"]),
        "mechanism": str(data["mechanism"]),
        "missing_rate": float(data["missing_rate"]),
        "seed": int(data["seed"]),
    }


def _npz_path(imp_dir, method, mechanism, rate, seed):
    tag = f"{method}_{mechanism}_r{_rate_tag(rate)}_seed{seed}"
    return os.path.join(imp_dir, f"{tag}.npz")


# --------------------------------------------------------------------------- #
# Figure 2: correlation matrices
# --------------------------------------------------------------------------- #
def _corr(X):
    C = np.corrcoef(X, rowvar=False)
    return np.nan_to_num(C)


def plot_corr_matrices(true_npz, miri_npz, baseline_npz, output_path, baseline_name="baseline"):
    d_true = load_imputation_npz(true_npz)
    d_miri = load_imputation_npz(miri_npz)
    d_base = load_imputation_npz(baseline_npz)

    mats = [
        ("True", _corr(d_true["X_test_true"])),
        ("MIRI", _corr(d_miri["X_test_imp"])),
        (f"Best baseline ({baseline_name})", _corr(d_base["X_test_imp"])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    im = None
    for ax, (title, C) in zip(axes, mats):
        im = ax.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xticks(range(len(SHORT_NAMES)))
        ax.set_yticks(range(len(SHORT_NAMES)))
        ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(SHORT_NAMES, fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.suptitle("Correlation matrices: True vs MIRI vs best baseline")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


# --------------------------------------------------------------------------- #
# Figures 3 & 4: scatter comparisons
# --------------------------------------------------------------------------- #
def _sample_idx(n, k, seed=0):
    rng = np.random.default_rng(seed)
    if n <= k:
        return np.arange(n)
    return rng.choice(n, k, replace=False)


def plot_pv_irradiation_scatter(true_npz, miri_npz, baseline_npz, output_path,
                                baseline_name="baseline", n_points=4000):
    d_true = load_imputation_npz(true_npz)
    d_miri = load_imputation_npz(miri_npz)
    d_base = load_imputation_npz(baseline_npz)

    Xt = d_true["X_test_true"]
    idx = _sample_idx(Xt.shape[0], n_points)

    panels = [
        ("True", Xt),
        ("MIRI", d_miri["X_test_imp"]),
        (f"Best baseline ({baseline_name})", d_base["X_test_imp"]),
    ]
    xlim = (Xt[:, IRRADIATION].min(), Xt[:, IRRADIATION].max())
    ylim = (Xt[:, PV].min(), Xt[:, PV].max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)
    for ax, (title, X) in zip(axes, panels):
        ax.scatter(X[idx, IRRADIATION], X[idx, PV], s=4, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("Horizontal solar irradiation (std)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    axes[0].set_ylabel("PV generation (std)")
    fig.suptitle("PV generation vs solar irradiation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  -> {output_path}")


def plot_temp_load_scatter(true_npz, miri_npz, baseline_npz, output_path,
                           baseline_name="baseline", n_points=4000):
    d_true = load_imputation_npz(true_npz)
    d_miri = load_imputation_npz(miri_npz)
    d_base = load_imputation_npz(baseline_npz)

    Xt = d_true["X_test_true"]
    idx = _sample_idx(Xt.shape[0], n_points)
    panels = [
        ("True", Xt),
        ("MIRI", d_miri["X_test_imp"]),
        (f"Best baseline ({baseline_name})", d_base["X_test_imp"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    for row, (load_idx, load_name) in enumerate([(COOLING, "Cooling load"), (HEATING, "Heating load")]):
        ylim = (Xt[:, load_idx].min(), Xt[:, load_idx].max())
        for col, (title, X) in enumerate(panels):
            ax = axes[row, col]
            ax.scatter(X[idx, TEMP], X[idx, load_idx], s=4, alpha=0.3)
            if row == 0:
                ax.set_title(title)
            if row == 1:
                ax.set_xlabel("Outdoor air temperature (std)")
            if col == 0:
                ax.set_ylabel(f"{load_name} (std)")
            ax.set_ylim(ylim)
    fig.suptitle("Temperature vs cooling / heating load")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  -> {output_path}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/ies/summaries/summary_mean_std.csv")
    parser.add_argument("--raw-dir", default="results/ies/raw")
    parser.add_argument("--imputation-dir", default="results/ies/imputations")
    parser.add_argument("--output-dir", default="results/ies/figures")
    parser.add_argument("--focus-mechanism", default="mcar")
    parser.add_argument("--focus-rate", type=float, default=0.4)
    parser.add_argument("--focus-seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.summary):
        print(f"Summary CSV not found: {args.summary}. Run aggregate_results.py first.")
        return

    # Figure 1
    plot_bar_mmd_mcar(args.summary, os.path.join(args.output_dir, "bar_mmd_mcar.png"),
                      metric="joint_mmd")
    plot_bar_mmd_mcar(args.summary, os.path.join(args.output_dir, "bar_masked_mmd_mcar.png"),
                      metric="masked_mmd")

    # Figures 2-4 need the imputation npz files
    mech, rate, seed = args.focus_mechanism, args.focus_rate, args.focus_seed
    baseline = select_best_baseline(args.summary, mech, rate)
    if baseline is None:
        print("No baseline available for correlation/scatter figures; skipping Figures 2-4.")
        return

    true_npz = _npz_path(args.imputation_dir, "miri", mech, rate, seed)  # true array lives in every npz
    miri_npz = _npz_path(args.imputation_dir, "miri", mech, rate, seed)
    base_npz = _npz_path(args.imputation_dir, baseline, mech, rate, seed)

    missing = [p for p in [miri_npz, base_npz] if not os.path.exists(p)]
    if missing:
        print(f"Missing npz files for Figures 2-4 (need MIRI + {baseline} at "
              f"{mech} r{rate} seed{seed}): {missing}. Skipping.")
        return

    plot_corr_matrices(true_npz, miri_npz, base_npz,
                       os.path.join(args.output_dir, "corr_matrix_triptych.png"),
                       baseline_name=baseline)
    plot_pv_irradiation_scatter(true_npz, miri_npz, base_npz,
                                os.path.join(args.output_dir, "scatter_pv_irradiation_compare.png"),
                                baseline_name=baseline)
    plot_temp_load_scatter(true_npz, miri_npz, base_npz,
                           os.path.join(args.output_dir, "scatter_temp_load_compare.png"),
                           baseline_name=baseline)


if __name__ == "__main__":
    main()
