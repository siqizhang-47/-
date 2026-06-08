"""Plotting / visualization for SC-CDiff scenarios (and any baseline, by tag).

Reads scenarios_<tag>_<split>_k<k>.npz and writes PNGs under <artifacts>/plots/.
Figures:
  1. fan charts  : per representative day, 5 channels, truth vs scenario
                   median + 10-90% / 25-75% bands + a few sample trajectories
  2. rank histogram : ensemble calibration per channel (flat = well calibrated)
  3. marginals      : generated vs real value histograms per channel
  4. reliability    : nominal vs empirical coverage curve per channel

Usage:
  python -m sc_cdiff.eval.plots --split test --k 0 --tag sccdiff
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..artifacts import scen_path  # noqa: E402
from ..data.build_dataset import _load_cfg  # noqa: E402

BASE_TS = pd.Timestamp("2001-06-01 00:00:00")


def _load(cfg, split, k):
    d = np.load(scen_path(cfg, split, k))
    return d["scenarios"], d["truth"], d["irr"], d["start"]


def _months(start):
    return np.array([(BASE_TS + pd.Timedelta(hours=int(s))).month for s in start])


def _dates(start):
    return [(BASE_TS + pd.Timedelta(hours=int(s))).strftime("%Y-%m-%d") for s in start]


def _pick_days(months, n_per_season=1):
    """Pick representative days: one each from Jan/Apr/Jul/Oct if available."""
    picks = []
    for m in (1, 4, 7, 10):
        idx = np.where(months == m)[0]
        if len(idx):
            picks.extend(idx[:: max(1, len(idx) // n_per_season)][:n_per_season].tolist())
    return picks or [0]


def fan_charts(cfg, scen, truth, start, split, k, outdir, n_traj=8):
    chans = cfg["channels"]
    months = _months(start)
    dates = _dates(start)
    for di in _pick_days(months):
        S = scen[di]                      # [M,5,24]
        Y = truth[di]                     # [5,24]
        hours = np.arange(24)
        fig, axes = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
        for c in range(5):
            ax = axes[c]
            sc = S[:, c, :]               # [M,24]
            p10, p25, p50, p75, p90 = np.percentile(sc, [10, 25, 50, 75, 90], axis=0)
            ax.fill_between(hours, p10, p90, alpha=0.20, color="C0", label="10-90%")
            ax.fill_between(hours, p25, p75, alpha=0.30, color="C0", label="25-75%")
            for j in range(min(n_traj, sc.shape[0])):
                ax.plot(hours, sc[j], color="C0", lw=0.4, alpha=0.35)
            ax.plot(hours, p50, color="C0", lw=1.6, label="median")
            ax.plot(hours, Y[c], color="k", lw=2.0, label="truth")
            ax.set_ylabel(chans[c])
            if c == 0:
                ax.legend(loc="upper right", fontsize=7, ncol=2)
        axes[-1].set_xlabel("hour")
        fig.suptitle(f"{cfg.get('method_tag','sccdiff')}  {dates[di]}  (k={k})")
        fig.tight_layout()
        f = os.path.join(outdir, f"fan_{cfg.get('method_tag','sccdiff')}_{split}_k{k}_{dates[di]}.png")
        fig.savefig(f, dpi=120); plt.close(fig)
        print(f"[plots] wrote {f}")


def rank_histogram(cfg, scen, truth, split, k, outdir):
    chans = cfg["channels"]
    N, M, C, T = scen.shape
    fig, axes = plt.subplots(1, 5, figsize=(16, 3))
    for c in range(C):
        ranks = (scen[:, :, c, :] < truth[:, None, c, :]).sum(axis=1).reshape(-1)
        axes[c].hist(ranks, bins=np.arange(M + 2) - 0.5, color="C0", density=True)
        axes[c].axhline(1.0 / (M + 1), color="r", ls="--", lw=1)
        axes[c].set_title(chans[c]); axes[c].set_xlabel("rank")
    fig.suptitle(f"Rank histogram  {cfg.get('method_tag','sccdiff')}  k={k} (flat=calibrated)")
    fig.tight_layout()
    f = os.path.join(outdir, f"rankhist_{cfg.get('method_tag','sccdiff')}_{split}_k{k}.png")
    fig.savefig(f, dpi=120); plt.close(fig)
    print(f"[plots] wrote {f}")


def marginals(cfg, scen, truth, split, k, outdir):
    chans = cfg["channels"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3))
    for c in range(5):
        g = scen[:, :, c, :].reshape(-1)
        r = truth[:, c, :].reshape(-1)
        lo, hi = np.percentile(np.concatenate([g, r]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        axes[c].hist(r, bins=bins, density=True, alpha=0.5, label="real", color="k")
        axes[c].hist(g, bins=bins, density=True, alpha=0.5, label="gen", color="C0")
        axes[c].set_title(chans[c])
        if c == 0:
            axes[c].legend(fontsize=8)
    fig.suptitle(f"Marginal distributions  {cfg.get('method_tag','sccdiff')}  k={k}")
    fig.tight_layout()
    f = os.path.join(outdir, f"marginals_{cfg.get('method_tag','sccdiff')}_{split}_k{k}.png")
    fig.savefig(f, dpi=120); plt.close(fig)
    print(f"[plots] wrote {f}")


def reliability(cfg, scen, truth, split, k, outdir):
    chans = cfg["channels"]
    levels = np.arange(0.1, 1.0, 0.1)
    fig, ax = plt.subplots(figsize=(5, 5))
    for c in range(5):
        emp = []
        for a in levels:
            lo = np.percentile(scen[:, :, c, :], 50 - 50 * a, axis=1)
            hi = np.percentile(scen[:, :, c, :], 50 + 50 * a, axis=1)
            emp.append(np.mean((truth[:, c, :] >= lo) & (truth[:, c, :] <= hi)))
        ax.plot(levels, emp, marker="o", label=chans[c])
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.set_title(f"Reliability  {cfg.get('method_tag','sccdiff')}  k={k}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f = os.path.join(outdir, f"reliability_{cfg.get('method_tag','sccdiff')}_{split}_k{k}.png")
    fig.savefig(f, dpi=120); plt.close(fig)
    print(f"[plots] wrote {f}")


def make_all(cfg, split, k):
    outdir = os.path.join(cfg["paths"]["artifacts"], "plots")
    os.makedirs(outdir, exist_ok=True)
    scen, truth, irr, start = _load(cfg, split, k)
    fan_charts(cfg, scen, truth, start, split, k, outdir)
    rank_histogram(cfg, scen, truth, split, k, outdir)
    marginals(cfg, scen, truth, split, k, outdir)
    reliability(cfg, scen, truth, split, k, outdir)
    print(f"[plots] all figures in {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--tag", default="sccdiff")
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts
    cfg["method_tag"] = args.tag
    make_all(cfg, args.split, args.k)


if __name__ == "__main__":
    main()
