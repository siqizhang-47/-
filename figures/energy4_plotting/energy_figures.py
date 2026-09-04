"""Shared figure code for the Energy4 / Energy4Exog experiments.

All three figures are drawn in per-unit (p.u.) scale: every variable is divided
by a fixed base value (by default the peak of that variable in the training
split), so the four panels share the same y-axis label ``Value (p.u.)``.

The functions only take numpy arrays, so the figures can be regenerated from
the ``figure_arrays.npz`` file written next to ``metrics.json`` without
re-running the model (see ``scripts/Energy4Exog/replot_figures.py``).
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np

FEATURE_NAMES = ["Electricity", "PV", "Cooling", "Heat"]

# Panel titles requested for the publication figures.
PANEL_TITLES: Dict[str, str] = {
    "Electricity": "Electrical load",
    "PV": "PV Power",
    "Cooling": "Cooling load",
    "Heat": "Heating load",
}

Y_LABEL = "Value (p.u.)"
X_LABEL_HORIZON = "Forecast horizon (hour)"


def panel_title(name: str) -> str:
    return PANEL_TITLES.get(name, name)


def compute_pu_base(data: np.ndarray, train_ratio: float = 0.7) -> np.ndarray:
    """Per-variable p.u. base = peak value of the training split (>= tiny eps)."""
    data = np.asarray(data, dtype=np.float64)
    n_train = max(int(round(data.shape[0] * train_ratio)), 1)
    base = np.max(np.abs(data[:n_train]), axis=0)
    return np.where(base > 1e-12, base, 1.0)


def save_figure_arrays(path: str, example, truth_pool, sample_pool, pu_base, feature_names=FEATURE_NAMES):
    """Persist the raw arrays behind the three figures so they can be re-plotted."""
    payload = {
        "feature_names": np.asarray(feature_names),
        "pu_base": np.asarray(pu_base, dtype=np.float64),
    }
    if example is not None:
        payload["example_history"] = np.asarray(example["history"])
        payload["example_truth"] = np.asarray(example["truth"])
        payload["example_samples"] = np.asarray(example["samples"])
        payload["example_selection_policy"] = np.asarray(example.get("selection_policy", "unspecified"))
    if truth_pool is not None:
        payload["truth_pool"] = np.asarray(truth_pool)
    if sample_pool is not None:
        payload["sample_pool"] = np.asarray(sample_pool)
    np.savez_compressed(path, **payload)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_prediction_intervals(
    example,
    path: str,
    pu_base: Sequence[float],
    feature_names: Sequence[str] = FEATURE_NAMES,
    dpi: int = 180,
    title: Optional[str] = None,
):
    """Figure 1: 24-hour probabilistic forecast, per-unit scale."""
    plt = _mpl()
    base = np.asarray(pu_base, dtype=np.float64)
    truth = np.asarray(example["truth"], dtype=np.float64) / base[None, :]
    samples = np.asarray(example["samples"], dtype=np.float64) / base[None, :, None]  # (O,N,S)
    mean = samples.mean(axis=-1)
    q025, q10, q25, q75, q90, q975 = np.quantile(samples, [0.025, 0.10, 0.25, 0.75, 0.90, 0.975], axis=-1)
    fx = np.arange(1, truth.shape[0] + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.8))
    for j, ax in enumerate(axes.flat):
        ax.fill_between(fx, q025[:, j], q975[:, j], color="#A6CEE3", alpha=0.28, label="95% PI")
        ax.fill_between(fx, q10[:, j], q90[:, j], color="#4EA3D8", alpha=0.30, label="80% PI")
        ax.fill_between(fx, q25[:, j], q75[:, j], color="#1F78B4", alpha=0.24, label="50% PI")
        ax.plot(fx, mean[:, j], color="#1F4E79", lw=2.0, label="Predictive mean")
        ax.plot(fx, truth[:, j], color="#D62728", lw=2.0, label="Ground truth")
        ax.set_title(panel_title(feature_names[j]), fontsize=13)
        ax.set_xlabel(X_LABEL_HORIZON)
        ax.set_ylabel(Y_LABEL)
        ax.set_xlim(1, truth.shape[0])
        ax.set_xticks(np.arange(0, truth.shape[0] + 1, 4)[1:] if truth.shape[0] >= 8 else fx)
        ax.grid(alpha=0.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    if title is None:
        policy = example.get("selection_policy", "unspecified")
        title = f"24-hour probabilistic forecast (display window: {policy})"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pdfs(
    truth_pool,
    sample_pool,
    path: str,
    pu_base: Sequence[float],
    feature_names: Sequence[str] = FEATURE_NAMES,
    dpi: int = 180,
    seed: int = 2026,
    title: str = "Marginal PDFs of real and calibrated NsDiff-generated samples",
):
    """Figure 2: marginal PDFs (shared absolute KDE bandwidth), per-unit scale."""
    plt = _mpl()
    from scipy.stats import gaussian_kde

    base = np.asarray(pu_base, dtype=np.float64)
    truth_pool = np.asarray(truth_pool, dtype=np.float64)
    sample_pool = np.asarray(sample_pool, dtype=np.float64)
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for j, ax in enumerate(axes.flat):
        real = truth_pool[:, :, j].reshape(-1) / base[j]
        gen = sample_pool[:, :, j, :].reshape(-1) / base[j]
        if real.size > 50000:
            real = rng.choice(real, 50000, replace=False)
        if gen.size > 50000:
            gen = rng.choice(gen, 50000, replace=False)

        pooled = np.concatenate([real, gen])
        pooled_std = float(np.std(pooled, ddof=1)) + 1e-12
        scott = max(int(pooled.size), 2) ** (-1.0 / 5.0)
        abs_bw = max(scott * pooled_std, 1e-8)
        rstd = float(np.std(real, ddof=1)) + 1e-12
        gstd = float(np.std(gen, ddof=1)) + 1e-12
        real_kde = gaussian_kde(real, bw_method=abs_bw / rstd)
        gen_kde = gaussian_kde(gen, bw_method=abs_bw / gstd)

        lo = float(min(np.quantile(real, 0.001), np.quantile(gen, 0.001)))
        hi = float(max(np.quantile(real, 0.999), np.quantile(gen, 0.999)))
        pad = 0.04 * (hi - lo + 1e-12)
        xs = np.linspace(lo - pad, hi + pad, 450)
        yr, yg = real_kde(xs), gen_kde(xs)
        ax.fill_between(xs, yr, alpha=0.42, color="#FF6B6B", label="Real sample")
        ax.plot(xs, yr, color="#FF4D4D", lw=1.6)
        ax.fill_between(xs, yg, alpha=0.40, color="#9E9E9E", label="Generated sample")
        ax.plot(xs, yg, color="#666666", lw=1.6)
        ax.set_title(panel_title(feature_names[j]), fontsize=13)
        ax.set_xlabel(Y_LABEL)
        ax.set_ylabel("Probability density")
        ax.legend()
        ax.grid(alpha=0.15)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_correlations(
    truth_pool,
    sample_pool,
    path: str,
    feature_names: Sequence[str] = FEATURE_NAMES,
    dpi: int = 180,
    seed: int = 2026,
    title: str = "Pearson correlation matrices of real and NsDiff-generated samples",
):
    """Figure 3: Pearson correlation matrices (scale-invariant, no p.u. needed)."""
    plt = _mpl()
    n = len(feature_names)
    rng = np.random.default_rng(seed)
    real = np.asarray(truth_pool, dtype=np.float64).reshape(-1, n)
    gen = np.asarray(sample_pool, dtype=np.float64).transpose(0, 1, 3, 2).reshape(-1, n)
    if real.shape[0] > 100000:
        real = real[rng.choice(real.shape[0], 100000, replace=False)]
    if gen.shape[0] > 100000:
        gen = gen[rng.choice(gen.shape[0], 100000, replace=False)]
    corr_gen = np.corrcoef(gen, rowvar=False)
    corr_real = np.corrcoef(real, rowvar=False)
    labels = [panel_title(f) for f in feature_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, corr, sub in [(axes[0], corr_gen, "Generated data"), (axes[1], corr_real, "Real data")]:
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu")
        ax.set_xticks(range(n), labels, rotation=30, ha="right")
        ax.set_yticks(range(n), labels)
        ax.set_title(f"Pearson correlation matrix ({sub})")
        for r in range(n):
            for c in range(n):
                ax.text(c, r, f"{corr[r, c]:.2f}", ha="center", va="center",
                        color="white" if abs(corr[r, c]) > 0.55 else "black", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_all(outdir: str, example, truth_pool, sample_pool, pu_base, feature_names=FEATURE_NAMES,
               dpi: int = 180, seed: int = 2026, fig1_title: Optional[str] = None,
               fig2_title: Optional[str] = None, fig3_title: Optional[str] = None):
    os.makedirs(outdir, exist_ok=True)
    if example is not None:
        plot_prediction_intervals(example, os.path.join(outdir, "fig1_prediction_intervals.png"),
                                  pu_base, feature_names, dpi, title=fig1_title)
    if truth_pool is not None and sample_pool is not None:
        kw2 = {} if fig2_title is None else {"title": fig2_title}
        kw3 = {} if fig3_title is None else {"title": fig3_title}
        plot_pdfs(truth_pool, sample_pool, os.path.join(outdir, "fig2_marginal_pdfs.png"),
                  pu_base, feature_names, dpi, seed, **kw2)
        plot_correlations(truth_pool, sample_pool, os.path.join(outdir, "fig3_pearson_correlation.png"),
                          feature_names, dpi, seed, **kw3)
