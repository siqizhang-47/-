#!/usr/bin/env python3
"""Re-draw the three publication figures from a saved ``figure_arrays.npz``.

The experiment now writes ``figure_arrays.npz`` (raw example window, truth
pool, sample pool and the p.u. base) next to ``metrics.json``. This script
re-renders fig1/fig2/fig3 from those arrays so titles, labels, dpi or the p.u.
base can be changed without re-running the model.

Example:
  python3 scripts/Energy4Exog/replot_figures.py \
      --npz energy4_exog_outputs/exomvd_v2_calibrated/seed_1/figure_arrays.npz \
      --out figures/seed_1 --dpi 300
  # optional: override the p.u. base per variable
  python3 scripts/Energy4Exog/replot_figures.py --npz ... --base Electricity=40000 PV=5000
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.experiments.energy_figures import render_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="path to figure_arrays.npz")
    ap.add_argument("--out", required=True, help="output directory for the three PNGs")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--base", nargs="*", default=[], help="override p.u. base, e.g. Electricity=40000")
    ap.add_argument("--fig1-title", default=None)
    ap.add_argument("--fig2-title", default=None)
    ap.add_argument("--fig3-title", default=None)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    names = [str(n) for n in z["feature_names"]]
    base = np.array(z["pu_base"], dtype=np.float64)
    for item in args.base:
        k, v = item.split("=")
        base[names.index(k)] = float(v)

    example = None
    if "example_truth" in z:
        example = {
            "history": z["example_history"],
            "truth": z["example_truth"],
            "samples": z["example_samples"],
            "selection_policy": str(z["example_selection_policy"]) if "example_selection_policy" in z else "unspecified",
        }
    truth_pool = z["truth_pool"] if "truth_pool" in z else None
    sample_pool = z["sample_pool"] if "sample_pool" in z else None

    render_all(args.out, example, truth_pool, sample_pool, base, names, dpi=args.dpi, seed=args.seed,
               fig1_title=args.fig1_title, fig2_title=args.fig2_title, fig3_title=args.fig3_title)
    print("p.u. base:", dict(zip(names, base.round(3))))
    print("Figures written to", args.out)


if __name__ == "__main__":
    main()
