"""Figures for the TimeXer-NsDiff experiment.

fig1_pdf_kde.png     predicted vs actual marginal distribution (histogram + KDE)
                     per target  -- style of the reference (a)/(b)/(c) plots.
fig2_timeseries.png  real vs generated time-series (median + 90% band) over a
                     stretch of consecutive test days, per target.
fig3_correlation.png Pearson correlation matrices among the 4 targets,
                     generated vs real (4x4 heat-maps).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _kde(values, grid):
    try:
        from scipy.stats import gaussian_kde
        return gaussian_kde(values)(grid)
    except Exception:
        h, edges = np.histogram(values, bins=80, density=True)
        c = 0.5 * (edges[1:] + edges[:-1])
        return np.interp(grid, c, h)


def fig_pdf_kde(pred_pool, actual_pool, target_names, out_path, units=None):
    """pred_pool/actual_pool: dict{target_index: 1d array of values}."""
    K = len(target_names)
    ncol = 2
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for k in range(K):
        ax = axes[k]
        pv, av = pred_pool[k], actual_pool[k]
        lo = min(pv.min(), av.min()); hi = max(pv.max(), av.max())
        pad = 0.03 * (hi - lo + 1e-9)
        bins = np.linspace(lo - pad, hi + pad, 60)
        grid = np.linspace(lo - pad, hi + pad, 400)
        ax.hist(pv, bins=bins, density=True, color="#6f8fe0", alpha=0.55, label="Predicted Distribution")
        ax.hist(av, bins=bins, density=True, color="#e8837a", alpha=0.55, label="Actual Distribution")
        ax.plot(grid, _kde(pv, grid), color="#1f3fa0", lw=1.6, label="Predicted KDE")
        ax.plot(grid, _kde(av, grid), color="#c0271c", lw=1.6, label="Actual KDE")
        ax.set_xlabel(f"{target_names[k]}  (normalised [0,1])"); ax.set_ylabel("Density")
        ax.set_title(f"({chr(97+k)})  {target_names[k]}")
        ax.legend(fontsize=7)
    for k in range(K, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Predicted vs actual distribution — normalised [0,1] (Oracle Weather)")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def fig_timeseries(ts_samples, ts_truth, target_names, out_path):
    """ts_samples: [D,S,H,K] consecutive days; ts_truth: [D,H,K]. Stitched timeline."""
    D, S, H, K = ts_samples.shape
    med = np.percentile(ts_samples, 50, axis=1).reshape(D * H, K)     # [D*H,K]
    lo = np.percentile(ts_samples, 5, axis=1).reshape(D * H, K)
    hi = np.percentile(ts_samples, 95, axis=1).reshape(D * H, K)
    truth = ts_truth.reshape(D * H, K)
    x = np.arange(D * H)

    fig, axes = plt.subplots(K, 1, figsize=(11, 2.2 * K), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for k in range(K):
        ax = axes[k]
        ax.fill_between(x, lo[:, k], hi[:, k], color="#e8837a", alpha=0.35, label="90% interval")
        ax.plot(x, med[:, k], color="#b3271c", lw=1.0, label="Generated median")
        ax.plot(x, truth[:, k], color="#333333", lw=1.0, label="Actual")
        ax.set_ylabel(f"{target_names[k]} [0,1]")
        ax.set_title(f"({chr(97+k)})  {target_names[k]}", fontsize=9, loc="left")
        if k == 0:
            ax.legend(fontsize=7, ncol=3, loc="upper right")
    axes[-1].set_xlabel("Time (hours over consecutive test days)")
    fig.suptitle("Real vs generated time series — normalised [0,1] (Oracle Weather)")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def _corr(x):                        # x: [P, K] -> [K,K]
    return np.corrcoef(x, rowvar=False)


def fig_correlation(gen_mean, truth, target_names, out_path):
    """gen_mean/truth: [P, K]. Two 4x4 Pearson correlation heat-maps."""
    cg, cr = _corr(gen_mean), _corr(truth)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, mat, name in zip(axes, [cg, cr], ["Generated", "Real"]):
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(target_names))); ax.set_yticks(range(len(target_names)))
        ax.set_xticklabels(target_names); ax.set_yticklabels(target_names)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if abs(mat[i, j]) > 0.5 else "black", fontsize=9)
        ax.set_title(f"Variable Correlation Matrix ({name})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Correlation among the four targets: generated vs real")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
