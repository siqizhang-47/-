"""Per-model diagnostic figures (real vs generated), following the paper's
example style:

  Fig A  daily curves        real sample vs generated scenarios, 2x2 carriers
  Fig B  Pearson correlation generated vs real, side-by-side heatmaps
  Fig C  PDF (KDE) overlay   real sample vs generated sample, 2x2 carriers
  plus   comparison_daily_curves: all requested models on one day

Scenario sources per model name:
  frozen         -> the raw cache (data/scenarios/wnorm_on by default)
  anything else  -> results/<name>/adapted/*.npz, produced by
                    `run_adapter.py --method <m> --save-adapted`

Usage:
  python scripts/plot_model_figures.py --models frozen proposed cosa \
      --day 2021-06-15 --start 2021-06-01 --end 2021-06-30
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import load_config, repo_root
from utils.seed import set_seed

# display order/titles follow the example figures; our array order is
# [PV, Electricity, Cooling, Heat] -> plotted as E/C/H/PV
PANELS = [("Electrical Load", 1), ("Cooling Load", 2), ("Heating Load", 3), ("PV Power", 0)]
CORR_LABELS = ["Electrical", "Cooling", "Heating", "PV"]
CORR_IDX = [1, 2, 3, 0]

REAL_COLOR = "#D55E00"       # daily curves: real sample
GEN_COLOR = "#4878A8"        # daily curves: generated
PDF_REAL = "#E64545"         # KDE: real
PDF_GEN = "#8C8C8C"          # KDE: generated
MODEL_COLORS = ["#4878A8", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
                "#56B4E9", "#8C8C8C"]  # fixed categorical order for comparison plot

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


class NpzSource:
    """Loads (scenarios, y_true) npz days from a directory."""

    def __init__(self, directory):
        self.dir = Path(directory)
        files = sorted(self.dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(
                f"no npz files in {self.dir} -- for adapter models rerun "
                "run_adapter.py with --save-adapted")
        self.dates = [pd.Timestamp(f.stem) for f in files]

    def load(self, date):
        with np.load(self.dir / f"{pd.Timestamp(date).date()}.npz") as z:
            return z["scenarios"].astype(float), z["y_true"].astype(float)

    def in_range(self, start, end):
        return [d for d in self.dates if start <= d <= end]


def resolve_source(model, cache_dir, results_dir):
    """model: 'frozen' | results/<name> shorthand | explicit dir path.
    Returns (label, NpzSource)."""
    if model == "frozen":
        return "frozen", NpzSource(cache_dir)
    p = Path(model)
    if p.is_dir():  # explicit path to a results dir or npz dir
        d = p / "adapted" if (p / "adapted").is_dir() else p
        return p.name, NpzSource(d)
    return model, NpzSource(Path(results_dir) / model / "adapted")


def load_period(src, dates, desc):
    scen_all, y_all = [], []
    for d in tqdm(dates, desc=desc, leave=False):
        scen, y = src.load(d)
        scen_all.append(scen)
        y_all.append(y)
    return scen_all, y_all


# ---------------- Fig A: daily curves ----------------
def plot_daily_curves(model, scen, y, pu, day, out_dir, band=True):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    hours = np.arange(24)
    for ax, (title, c) in zip(axes.ravel(), PANELS):
        yr = y[:, c] / pu[c]
        mu = scen[:, :, c].mean(axis=0) / pu[c]
        if band:
            lo = np.quantile(scen[:, :, c], 0.05, axis=0) / pu[c]
            hi = np.quantile(scen[:, :, c], 0.95, axis=0) / pu[c]
            ax.fill_between(hours, lo, hi, color=GEN_COLOR, alpha=0.18,
                            linewidth=0, label="Generated 90% band")
        ax.plot(hours, yr, color=REAL_COLOR, lw=2, label="Real sample")
        ax.plot(hours, mu, color=GEN_COLOR, lw=2, label="Generated mean")
        ax.set_title(title)
        ax.set_xlabel("Time (Hours)")
        ax.set_ylabel("Value (p.u.)")
        ax.set_xlim(0, 23)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"{model}: real vs generated, {day.date()}", y=0.995)
    fig.tight_layout()
    path = out_dir / f"{model}_daily_curves_{day.date()}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------- Fig B: correlation heatmaps ----------------
def _corr_panel(ax, r, title):
    im = ax.imshow(r, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(4), CORR_LABELS)
    ax.set_yticks(range(4), CORR_LABELS)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{r[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(r[i, j]) > 0.55 else "black", fontsize=9)
    return im


def plot_corr_matrices(model, scen_all, y_all, out_dir, tag):
    # generated: pool scenarios x hours; real: pool hours -- over the period
    gen = np.concatenate([s.reshape(-1, 4) for s in scen_all])[:, CORR_IDX]
    real = np.concatenate(y_all)[:, CORR_IDX]
    r_gen = np.corrcoef(gen.T)
    r_real = np.corrcoef(real.T)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    _corr_panel(axes[0], r_gen, "Variable Correlation Matrix (Generated Data)")
    im = _corr_panel(axes[1], r_real, "Variable Correlation Matrix (Real Data)")
    for ax in axes:
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.03)
    fig.suptitle(f"{model}: Pearson correlation, {tag}", y=0.98)
    path = out_dir / f"{model}_corr_matrix_{tag}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------- Fig C: PDFs ----------------
def plot_pdfs(model, scen_all, y_all, pu, out_dir, tag, max_gen=50000):
    gen = np.concatenate([s.reshape(-1, 4) for s in scen_all])
    real = np.concatenate(y_all)
    if len(gen) > max_gen:  # subsample for KDE tractability (global seed governs)
        gen = gen[np.random.choice(len(gen), max_gen, replace=False)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, (title, c) in zip(axes.ravel(), PANELS):
        r, g = real[:, c] / pu[c], gen[:, c] / pu[c]
        lo = min(r.min(), g.min())
        hi = max(r.max(), g.max())
        pad = 0.05 * (hi - lo + 1e-9)
        xs = np.linspace(lo - pad, hi + pad, 400)
        for vals, color, label in ((r, PDF_REAL, "Real sample"),
                                   (g, PDF_GEN, "Generated sample")):
            if vals.std() < 1e-9:
                continue
            dens = gaussian_kde(vals)(xs)
            ax.fill_between(xs, dens, color=color, alpha=0.45, linewidth=0)
            ax.plot(xs, dens, color=color, lw=1.5, label=label)
        ax.set_title(title)
        ax.set_xlabel(f"{title} (p.u.)")
        ax.set_ylabel("PDF")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"{model}: distribution of real vs generated, {tag}", y=0.995)
    fig.tight_layout()
    path = out_dir / f"{model}_pdf_{tag}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------- comparison across models ----------------
def plot_comparison(sources, pu, day, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, (title, c) in zip(axes.ravel(), PANELS):
        first = True
        for k, (model, src) in enumerate(sources.items()):
            scen, y = src.load(day)
            if first:
                ax.plot(np.arange(24), y[:, c] / pu[c], color="black", lw=2.2,
                        label="Real sample")
                first = False
            # distinct linestyles keep near-identical models visible
            style = ["-", "--", "-.", ":"][k % 4]
            ax.plot(np.arange(24), scen[:, :, c].mean(axis=0) / pu[c],
                    color=MODEL_COLORS[k % len(MODEL_COLORS)], lw=1.8,
                    linestyle=style, label=model)
        ax.set_title(title)
        ax.set_xlabel("Time (Hours)")
        ax.set_ylabel("Value (p.u.)")
        ax.set_xlim(0, 23)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Model comparison (scenario means), {day.date()}", y=0.995)
    fig.tight_layout()
    path = out_dir / f"comparison_daily_curves_{day.date()}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = repo_root()
    parser.add_argument("--base-config", default=str(root / "configs/base.yaml"))
    parser.add_argument("--models", nargs="+", default=["frozen", "proposed"],
                        help="'frozen' and/or results/<name> dirs with adapted/")
    parser.add_argument("--cache", default=None, help="frozen scenario cache dir")
    parser.add_argument("--day", default=None, help="date for the daily-curve figure")
    parser.add_argument("--start", default=None, help="period start for corr/pdf")
    parser.add_argument("--end", default=None, help="period end for corr/pdf")
    parser.add_argument("--out", default=None, help="output dir (default results/figures)")
    parser.add_argument("--no-band", action="store_true", help="hide the 90% band")
    args = parser.parse_args()

    cfg = load_config(args.base_config)
    set_seed(cfg.seed)
    cache_dir = Path(args.cache or root / cfg.scenario_dir / "wnorm_on")
    results_dir = root / cfg.results_dir
    out_dir = Path(args.out or results_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = dict(resolve_source(m, cache_dir, results_dir) for m in args.models)
    ref = next(iter(sources.values()))
    deploy_dates = [d for d in ref.dates if d >= pd.Timestamp("2020-01-01")] or ref.dates
    day = pd.Timestamp(args.day) if args.day else deploy_dates[len(deploy_dates) // 2]
    start = pd.Timestamp(args.start) if args.start else day.replace(day=1)
    end = pd.Timestamp(args.end) if args.end else (start + pd.offsets.MonthEnd(1))
    tag = f"{start.date()}_{end.date()}"

    # p.u. base: per-carrier max |real| over the period (frozen/ref source)
    _, y_ref = load_period(ref, ref.in_range(start, end), "p.u. base")
    pu = np.abs(np.concatenate(y_ref)).max(axis=0)
    pu[pu < 1e-9] = 1.0

    written = []
    for model, src in sources.items():
        dates = src.in_range(start, end)
        assert dates, f"{model}: no cached days in {start.date()}..{end.date()}"
        assert day in src.dates, f"{model}: {day.date()} not in its npz days"
        scen_day, y_day = src.load(day)
        written.append(plot_daily_curves(model, scen_day, y_day, pu, day,
                                         out_dir, band=not args.no_band))
        scen_all, y_all = load_period(src, dates, f"{model} period")
        written.append(plot_corr_matrices(model, scen_all, y_all, out_dir, tag))
        written.append(plot_pdfs(model, scen_all, y_all, pu, out_dir, tag))
    if len(sources) > 1:
        written.append(plot_comparison(sources, pu, day, out_dir))

    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
