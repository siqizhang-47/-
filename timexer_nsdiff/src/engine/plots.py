"""The three result figures.

Figure 1  real vs generated 24-hour trajectories, one panel per energy variable
Figure 2  Pearson correlation matrices of the four variables, generated vs real
Figure 3  probability density functions of real and generated samples

All three take the scenario tensor ``pred [W, S, H, K]`` and the ground truth
``true [W, H, K]`` in PHYSICAL units, and plot in per-unit (p.u.) values, i.e.
divided by a per-variable base.  The default base is the maximum of the real
test series for that variable, so every curve lands in [0, 1].

Panel order and labels follow the reference figures: Electrical, Cooling,
Heating, PV -- note this is NOT the internal target order
(Electricity, PV, Cooling, Heat), so the plotting code reindexes explicitly.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..metrics.prob_metrics import TARGET_NAMES

# reference-figure display order and labels
PLOT_ORDER = ["Electricity", "Cooling", "Heat", "PV"]
DISPLAY_NAME = {"Electricity": "Electrical", "Cooling": "Cooling", "Heat": "Heating", "PV": "PV"}
PANEL_TITLE = {"Electricity": "Electrical Load", "Cooling": "Cooling Load",
               "Heat": "Heating Load", "PV": "PV Power"}
PDF_TITLE = {"Electricity": "Electrical Load", "Cooling": "Cooling Load",
             "Heat": "Heating Load", "PV": "PV power generation"}
PDF_XLABEL = {"Electricity": "Electrical load (Pu)", "Cooling": "Cooling Load (Pu)",
              "Heat": "Heating load (Pu)", "PV": "PV power generation (Pu)"}

REAL_COLOR = "#d1440a"     # figure 1 real curve
GEN_COLOR = "#3b6ea5"      # figure 1 generated curve
PDF_REAL = "#f03b20"       # figure 3 real density
PDF_GEN = "#9e9e9e"        # figure 3 generated density


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.titlesize": 10,
        "figure.dpi": 160,
    })
    return plt


def _as_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def per_unit_base(true, mode: str = "max"):
    """Per-variable normalisation base. true [W, H, K] in physical units."""
    t = _as_numpy(true).reshape(-1, true.shape[-1])
    if mode == "max":
        base = np.abs(t).max(axis=0)
    elif mode == "p99":
        base = np.percentile(np.abs(t), 99, axis=0)
    else:
        raise ValueError(f"unknown p.u. base mode '{mode}'")
    return np.where(base <= 0, 1.0, base).astype(np.float64)


def _idx(target_names=None):
    names = target_names or TARGET_NAMES
    return {n: i for i, n in enumerate(names)}


# =========================================================================== fig 1
def pick_window(true, group_info=None, target_names=None):
    """Pick a representative day: prefer a window starting at hour 0, then the
    one with the largest PV output (a clear sunny day, like the reference figure)."""
    idx = _idx(target_names)
    t = _as_numpy(true)
    candidates = np.arange(t.shape[0])
    if group_info is not None:
        g = _as_numpy(group_info)
        midnight = np.nonzero(g[:, 0, 0] == 0)[0]
        if midnight.size > 0:
            candidates = midnight
    pv_energy = t[candidates, :, idx["PV"]].sum(axis=1)
    return int(candidates[int(np.argmax(pv_energy))])


def plot_sample_trajectories(pred, true, out_path, window=None, group_info=None,
                             base_mode="max", scenario="mean", band=None,
                             target_names=None, seed=0):
    """Figure 1: real vs generated 24-hour curves, one panel per variable.

    scenario='mean'  -> the generated curve is the mean over the S scenarios
    scenario='single'-> one randomly drawn scenario (closest to the reference figure)
    band=0.9         -> additionally shade the central 90% scenario interval
    """
    plt = _mpl()
    idx = _idx(target_names)
    p = _as_numpy(pred)
    t = _as_numpy(true)
    base = per_unit_base(t, base_mode)

    w = pick_window(t, group_info, target_names) if window is None else int(window)
    hours = np.arange(t.shape[1])

    if scenario == "single":
        rng = np.random.default_rng(seed)
        gen = p[w, rng.integers(p.shape[1])]
    else:
        gen = p[w].mean(axis=0)
    real = t[w]

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.4))
    for ax, name in zip(axes.ravel(), PLOT_ORDER):
        k = idx[name]
        ax.plot(hours, real[:, k] / base[k], color=REAL_COLOR, lw=1.8, label="Real sample")
        ax.plot(hours, gen[:, k] / base[k], color=GEN_COLOR, lw=1.3, label="Generated sample")
        if band is not None:
            lo = np.quantile(p[w, :, :, k], (1 - band) / 2, axis=0) / base[k]
            hi = np.quantile(p[w, :, :, k], 1 - (1 - band) / 2, axis=0) / base[k]
            ax.fill_between(hours, lo, hi, color=GEN_COLOR, alpha=0.18, lw=0,
                            label=f"{int(band * 100)}% interval")
        ax.set_title(PANEL_TITLE[name])
        ax.set_xlabel("Time (Hours)")
        ax.set_ylabel("Value (p.u.)")
        ax.set_xlim(hours[0], hours[-1])
        ax.set_ylim(0.0, 1.0)
        # "best" keeps the legend off the PV bell, which peaks mid-panel
        ax.legend(fontsize=7.5, loc="best", framealpha=0.85, edgecolor="none")
    fig.tight_layout()
    _save(fig, out_path, plt)
    return {"window": w, "scenario": scenario, "pu_base": base.tolist(),
            "pu_base_order": target_names or TARGET_NAMES}


# =========================================================================== fig 2
def _flatten_for_correlation(pred, true, source="samples", max_rows=400_000, seed=0):
    """Return (generated [N, K], real [M, K]) hour-level matrices."""
    p = _as_numpy(pred)
    t = _as_numpy(true)
    real = t.reshape(-1, t.shape[-1])
    if source == "mean":
        gen = p.mean(axis=1).reshape(-1, p.shape[-1])
    else:  # pool every scenario -> reflects the generative distribution
        gen = p.reshape(-1, p.shape[-1])
    rng = np.random.default_rng(seed)
    if gen.shape[0] > max_rows:
        gen = gen[rng.choice(gen.shape[0], max_rows, replace=False)]
    if real.shape[0] > max_rows:
        real = real[rng.choice(real.shape[0], max_rows, replace=False)]
    return gen, real


def plot_correlation_matrices(pred, true, out_path, source="samples", target_names=None,
                              seed=0):
    """Figure 2: Pearson correlation matrices, generated (left) vs real (right)."""
    plt = _mpl()
    idx = _idx(target_names)
    order = [idx[n] for n in PLOT_ORDER]
    labels = [DISPLAY_NAME[n] for n in PLOT_ORDER]

    gen, real = _flatten_for_correlation(pred, true, source, seed=seed)
    c_gen = np.corrcoef(gen[:, order], rowvar=False)
    c_real = np.corrcoef(real[:, order], rowvar=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for ax, mat, title in zip(axes, [c_gen, c_real],
                              ["Variable Correlation Matrix (Generated Data)",
                               "Variable Correlation Matrix (Real Data)"]):
        im = ax.imshow(mat, cmap="RdBu", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels, rotation=90, va="center")
        ax.set_title(title, fontsize=10)
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 0.55 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, ticks=np.arange(-1.0, 1.01, 0.25))
    fig.tight_layout()
    _save(fig, out_path, plt)
    return {"labels": labels, "generated": c_gen.tolist(), "real": c_real.tolist(),
            "source": source}


# =========================================================================== fig 3
def gaussian_kde_1d(x, grid, bw=None, max_points=40_000, seed=0, chunk=64):
    """Gaussian KDE with Silverman's rule -- numpy only, no scipy dependency."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    if x.size > max_points:
        x = rng.choice(x, max_points, replace=False)
    n = x.size
    if n < 2:
        return np.zeros_like(grid)
    std = x.std(ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    sigma = min(std, iqr / 1.349) if iqr > 0 else std
    if bw is None:
        bw = 0.9 * sigma * n ** (-0.2)
    if not np.isfinite(bw) or bw <= 0:
        bw = max(1e-6, std if std > 0 else 1e-6)

    out = np.empty_like(grid, dtype=np.float64)
    for i in range(0, grid.size, chunk * 8):
        g = grid[i:i + chunk * 8]
        z = (g[:, None] - x[None, :]) / bw
        out[i:i + g.size] = np.exp(-0.5 * z * z).sum(axis=1)
    return out / (n * bw * np.sqrt(2.0 * np.pi))


def plot_pdf_comparison(pred, true, out_path, base_mode="max", n_grid=512,
                        target_names=None, seed=0, source="samples"):
    """Figure 3: KDE of the real and the generated sample distributions."""
    plt = _mpl()
    idx = _idx(target_names)
    p = _as_numpy(pred)
    t = _as_numpy(true)
    base = per_unit_base(t, base_mode)

    gen_all = (p.mean(axis=1) if source == "mean" else p).reshape(-1, p.shape[-1])
    real_all = t.reshape(-1, t.shape[-1])

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.8))
    summary = {}
    for ax, name in zip(axes.ravel(), PLOT_ORDER):
        k = idx[name]
        real = real_all[:, k] / base[k]
        gen = gen_all[:, k] / base[k]
        lo = min(real.min(), np.percentile(gen, 0.1))
        hi = max(real.max(), np.percentile(gen, 99.9))
        pad = 0.08 * (hi - lo + 1e-9)
        grid = np.linspace(lo - pad, hi + pad, n_grid)

        d_real = gaussian_kde_1d(real, grid, seed=seed)
        d_gen = gaussian_kde_1d(gen, grid, seed=seed)

        ax.fill_between(grid, d_real, color=PDF_REAL, alpha=0.65, lw=0)
        ax.plot(grid, d_real, color=PDF_REAL, lw=1.0, label="Real sample")
        ax.fill_between(grid, d_gen, color=PDF_GEN, alpha=0.65, lw=0)
        ax.plot(grid, d_gen, color="#6b6b6b", lw=1.0, label="Generated sample")

        ax.set_title(PDF_TITLE[name])
        ax.set_xlabel(PDF_XLABEL[name])
        ax.set_ylabel("PDF")
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(bottom=0.0)
        ax.legend(frameon=True, fontsize=7.5, loc="upper right")

        # L1 distance between the two densities: a compact goodness number
        summary[name] = {
            "kde_L1_distance": float(np.trapezoid(np.abs(d_real - d_gen), grid))
            if hasattr(np, "trapezoid") else float(np.trapz(np.abs(d_real - d_gen), grid)),
            "real_mean_pu": float(real.mean()), "gen_mean_pu": float(gen.mean()),
            "real_std_pu": float(real.std()), "gen_std_pu": float(gen.std()),
        }
    fig.tight_layout()
    _save(fig, out_path, plt)
    return summary


# =========================================================================== driver
def _save(fig, out_path, plt):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path} (+ .pdf)")


def make_all_figures(pred, true, out_dir, group_info=None, base_mode="max",
                     scenario="mean", band=None, corr_source="samples",
                     target_names=None, seed=0, window=None):
    """Draw figures 1-3 into `out_dir/figures/` and write figures.json alongside."""
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    info = {}
    info["fig1_trajectories"] = plot_sample_trajectories(
        pred, true, os.path.join(fig_dir, "fig1_real_vs_generated.png"),
        window=window, group_info=group_info, base_mode=base_mode, scenario=scenario,
        band=band, target_names=target_names, seed=seed)
    info["fig2_correlation"] = plot_correlation_matrices(
        pred, true, os.path.join(fig_dir, "fig2_correlation_matrix.png"),
        source=corr_source, target_names=target_names, seed=seed)
    info["fig3_pdf"] = plot_pdf_comparison(
        pred, true, os.path.join(fig_dir, "fig3_pdf_comparison.png"),
        base_mode=base_mode, target_names=target_names, seed=seed, source=corr_source)
    with open(os.path.join(fig_dir, "figures.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info
