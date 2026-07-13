"""
Figures (Fig.3 / Fig.4 / Fig.5) and the metric table
(CRPS, PSD, APD, Avg Std, DTW, MAE, RMSE, ACF) for the IES experiment.

All functions work on plain numpy arrays so they can be tested without torch:

    real   : (M, H, T)      ground-truth target windows          (per-unit)
    gmean  : (M, H, T)      generated sample mean                 (per-unit)
    gsamp  : (M, H, T, S)   S generated samples per window        (per-unit)

M = number of test windows, H = horizon (pred_len, e.g. 24h), T = 4 targets,
S = number of diffusion samples per window.

Metric definitions (documented so they can be adjusted to a paper's exact form):
  CRPS    : sample-based Continuous Ranked Probability Score,
            CRPS = E|X-y| - 0.5 E|X-X'|, averaged over all points & targets.
  PSD     : mean |logPSD_real - logPSD_gen| over frequencies (rFFT along H),
            averaged over targets  -> spectral-shape error.
  APD     : Absolute Pearson-correlation Difference = mean off-diagonal
            |Corr_real - Corr_gen| of the TxT inter-variable matrices (Fig.4).
  Avg Std : mean std across the S samples -> generation diversity / spread.
  DTW     : mean Dynamic-Time-Warping distance between real & gen-mean windows.
  MAE/RMSE: point error between real and gen-mean.
  ACF     : mean |ACF_real - ACF_gen| over lags (autocorrelation-shape error).
"""
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_NAMES = ["Electrical", "Cooling", "Heating", "PV"]
FIG_TITLES = ["Electrical Load", "Cooling Load", "Heating Load", "PV Power"]


# ----------------------------------------------------------------------------- metrics
def _mae(real, gmean):
    return float(np.mean(np.abs(real - gmean)))


def _rmse(real, gmean):
    return float(np.sqrt(np.mean((real - gmean) ** 2)))


def _crps(real, gsamp):
    """Sample based CRPS averaged over M,H,T. gsamp: (M,H,T,S)."""
    y = real[..., None]                       # (M,H,T,1)
    term1 = np.mean(np.abs(gsamp - y), axis=-1)                      # E|X-y|
    # E|X-X'| via sorted-sample estimator (O(S log S))
    xs = np.sort(gsamp, axis=-1)
    S = xs.shape[-1]
    w = (2 * np.arange(1, S + 1) - S - 1)                            # weights
    term2 = 2.0 * np.sum(w * xs, axis=-1) / (S * S)
    crps = term1 - 0.5 * term2
    return float(np.mean(crps))


def _avg_std(gsamp):
    return float(np.mean(np.std(gsamp, axis=-1)))


def _psd(real, gmean):
    """Mean abs diff of average log power spectral density along the horizon."""
    def logpsd(x):                            # x: (M,H,T)
        f = np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1)
        p = (np.abs(f) ** 2).mean(axis=0)     # (F,T) averaged over windows
        return np.log(p + 1e-8)
    return float(np.mean(np.abs(logpsd(real) - logpsd(gmean))))


def _acf(real, gmean, nlags=None):
    """Mean abs diff of the average autocorrelation function over lags."""
    H = real.shape[1]
    nlags = nlags or (H - 1)

    def acf(x):                               # x: (M,H,T) -> (nlags+1, T)
        xc = x - x.mean(axis=1, keepdims=True)
        var = (xc ** 2).mean(axis=1) + 1e-8   # (M,T)
        out = []
        for k in range(nlags + 1):
            if k == 0:
                c = np.ones_like(var)
            else:
                c = (xc[:, k:, :] * xc[:, :H - k, :]).mean(axis=1) / var
            out.append(c.mean(axis=0))        # avg over windows -> (T,)
        return np.stack(out, axis=0)          # (nlags+1, T)
    return float(np.mean(np.abs(acf(real) - acf(gmean))))


def _dtw_pair(a, b):
    """Classic DTW distance between two 1-D sequences."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m]


def _dtw(real, gmean, max_windows=1500, seed=0):
    """Mean DTW over a (sub)sample of windows and all targets."""
    M, H, T = real.shape
    idx = np.arange(M)
    if M > max_windows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(M, size=max_windows, replace=False)
    tot, cnt = 0.0, 0
    for i in idx:
        for t in range(T):
            tot += _dtw_pair(real[i, :, t], gmean[i, :, t])
            cnt += 1
    return float(tot / max(cnt, 1))


def _corr_matrix(x):
    """TxT Pearson correlation across all (M*H) points. x: (M,H,T)."""
    flat = x.reshape(-1, x.shape[-1])
    return np.corrcoef(flat, rowvar=False)


def _apd(real, gmean):
    cr, cg = _corr_matrix(real), _corr_matrix(gmean)
    T = cr.shape[0]
    off = ~np.eye(T, dtype=bool)
    return float(np.mean(np.abs(cr[off] - cg[off])))


def compute_metrics(real, gmean, gsamp):
    return {
        "CRPS":    _crps(real, gsamp),
        "PSD":     _psd(real, gmean),
        "APD":     _apd(real, gmean),
        "Avg Std": _avg_std(gsamp),
        "DTW":     _dtw(real, gmean),
        "MAE":     _mae(real, gmean),
        "RMSE":    _rmse(real, gmean),
        "ACF":     _acf(real, gmean),
    }


# ----------------------------------------------------------------------------- figures
def fig3_profiles(real, gmean, start_hours, out_path):
    """Fig.3: one representative daily (H-step) profile, real vs generated."""
    M, H, T = real.shape
    # prefer a window that starts at midnight so the x-axis is a clean day
    cand = np.where(start_hours == 0)[0] if start_hours is not None else np.arange(M)
    if len(cand) == 0:
        cand = np.arange(M)
    # pick the most "typical" day: closest to the median daily mean profile
    day_mean = real[cand].mean(axis=1).mean(axis=1)                 # (len(cand),)
    pick = cand[np.argsort(np.abs(day_mean - np.median(day_mean)))[0]]

    x = np.arange(H)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for t, ax in enumerate(axes.ravel()):
        ax.plot(x, real[pick, :, t],  color="#d1622b", lw=1.8, label="Real sample")
        ax.plot(x, gmean[pick, :, t], color="#4a7fb0", lw=1.8, label="Generated sample")
        ax.set_title(FIG_TITLES[t]); ax.set_xlabel("Time (Hours)")
        ax.set_ylabel("Value (p.u.)"); ax.set_ylim(0, 1.0); ax.legend(fontsize=8)
    fig.suptitle("Fig. 3  Comparison of the real and generated data")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def fig4_corr(real, gmean, out_path):
    """Fig.4: Pearson correlation matrices, generated vs real."""
    cg, cr = _corr_matrix(gmean), _corr_matrix(real)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, mat, name in zip(axes, [cg, cr],
                             ["Generated Data", "Real Data"]):
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(TARGET_NAMES); ax.set_yticklabels(TARGET_NAMES)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if abs(mat[i, j]) > 0.5 else "black", fontsize=9)
        ax.set_title(f"Variable Correlation Matrix ({name})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Fig. 4  Pearson correlation matrices of real and generated sample")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def _kde(values, grid):
    try:
        from scipy.stats import gaussian_kde
        return gaussian_kde(values)(grid)
    except Exception:
        h, edges = np.histogram(values, bins=60, density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        return np.interp(grid, centers, h)


def fig5_pdf(real, gmean, out_path):
    """Fig.5: probability density functions, real vs generated."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    titles = ["Electrical Load", "Cooling Load", "Heating Load", "PV power generation"]
    xlabels = ["Electrical load (Pu)", "Cooling Load (Pu)",
               "Heating load (Pu)", "PV power generation (Pu)"]
    for t, ax in enumerate(axes.ravel()):
        rv, gv = real[..., t].ravel(), gmean[..., t].ravel()
        lo = min(rv.min(), gv.min()); hi = max(rv.max(), gv.max())
        pad = 0.05 * (hi - lo + 1e-6)
        grid = np.linspace(lo - pad, hi + pad, 300)
        ax.fill_between(grid, _kde(rv, grid), color="red",  alpha=0.5, label="Real sample")
        ax.fill_between(grid, _kde(gv, grid), color="gray", alpha=0.5, label="Generated sample")
        ax.set_title(titles[t]); ax.set_xlabel(xlabels[t]); ax.set_ylabel("PDF")
        ax.legend(fontsize=8)
    fig.suptitle("Fig. 5  Probability Density Functions of real and generated sample")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def make_all(real, gmean, gsamp, start_hours, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig3_profiles(real, gmean, start_hours, os.path.join(out_dir, "fig3_profiles.png"))
    fig4_corr(real, gmean, os.path.join(out_dir, "fig4_correlation.png"))
    fig5_pdf(real, gmean, os.path.join(out_dir, "fig5_pdf.png"))
    metrics = compute_metrics(real, gmean, gsamp)

    order = ["CRPS", "PSD", "APD", "Avg Std", "DTW", "MAE", "RMSE", "ACF"]
    header = "".join(f"{k:>10}" for k in order)
    values = "".join(f"{metrics[k]:>10.4f}" for k in order)
    table = header + "\n" + values + "\n"
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(table)
    import json
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n===== IES generation metrics =====")
    print(table)
    print(f"figures + metrics saved under: {out_dir}")
    return metrics
