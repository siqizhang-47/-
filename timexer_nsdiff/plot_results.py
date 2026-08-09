#!/usr/bin/env python
"""Redraw the three result figures for an already-trained run.

    # from a run directory (reuses samples_test.npz if it exists,
    # otherwise reloads final_model.pth and regenerates scenarios)
    python plot_results.py --run_dir results/A7/seed1 --gpu 2

    # a single random scenario instead of the scenario mean, plus a 90% band
    python plot_results.py --run_dir results/A7/seed1 --fig_scenario single --fig_band 0.9

    # cheap redraw: regenerate only every 24th test window
    python plot_results.py --run_dir results/A7/seed1 --stride 24

Figures land in <run_dir>/figures/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from src.config import ExpConfig
from src.data_provider.energy_weather_dataset import HEEWData, SplitYears
from src.engine.evaluate import generate_scenarios, inverse_transform
from src.engine.plots import make_all_figures
from src.models.nsdiff_conditioned import TimeXerNsDiff
from src.utils.reproduce import reproducible, resolve_device


def load_run_config(run_dir):
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        sys.exit(f"[fatal] no config.json in {run_dir}")
    with open(path) as f:
        raw = json.load(f)
    cfg = ExpConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Redraw figures 1-3 for a finished run")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None,
                    help="regenerate every Nth test window (default: the run's test_stride)")
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--fig_scenario", choices=["mean", "single"], default=None)
    ap.add_argument("--fig_band", type=float, default=None)
    ap.add_argument("--fig_pu_base", choices=["max", "p99"], default=None)
    ap.add_argument("--fig_corr_source", choices=["samples", "mean"], default=None)
    ap.add_argument("--fig_window", type=int, default=None,
                    help="index of the day drawn in figure 1 (default: auto-pick)")
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--force_regenerate", action="store_true")
    args = ap.parse_args()

    cfg = load_run_config(args.run_dir)
    for name in ("gpu", "num_samples", "fig_scenario", "fig_band", "fig_pu_base",
                 "fig_corr_source", "fig_window", "data_path"):
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, name, v)
    if args.stride is not None:
        cfg.test_stride = args.stride

    npz = os.path.join(args.run_dir, "samples_test.npz")
    if os.path.exists(npz) and not args.force_regenerate:
        print(f"[plot] reusing cached scenarios from {npz}")
        blob = np.load(npz)
        pred = torch.from_numpy(blob["pred"])
        true = torch.from_numpy(blob["true"])
        groups = torch.from_numpy(blob["group_info"]) if "group_info" in blob else None
    else:
        print("[plot] regenerating scenarios from final_model.pth")
        ckpt = os.path.join(args.run_dir, "final_model.pth")
        if not os.path.exists(ckpt):
            sys.exit(f"[fatal] neither {npz} nor {ckpt} exists")
        reproducible(cfg.seed, cfg.deterministic)
        device = resolve_device(cfg.gpu)
        data = HEEWData(cfg.data_path, window=cfg.window, horizon=cfg.horizon,
                        splits=SplitYears(tuple(cfg.train_years), tuple(cfg.val_years),
                                          tuple(cfg.test_years)),
                        cache_dir=cfg.cache_dir)
        model = TimeXerNsDiff(cfg).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        pred, true, groups = generate_scenarios(model, data.dataset("test"), cfg, device,
                                                stride=cfg.test_stride, desc="figures")
        pred, true = inverse_transform(pred, true, data.target_scaler)

    info = make_all_figures(pred, true, args.run_dir, group_info=groups,
                            base_mode=cfg.fig_pu_base, scenario=cfg.fig_scenario,
                            band=(cfg.fig_band if cfg.fig_band > 0 else None),
                            corr_source=cfg.fig_corr_source, seed=cfg.seed,
                            window=(None if cfg.fig_window < 0 else cfg.fig_window))
    print(f"\n[plot] figure 1 drew test window #{info['fig1_trajectories']['window']}")
    print("[plot] KDE L1 distance real vs generated:")
    for name, d in info["fig3_pdf"].items():
        print(f"    {name:<12} {d['kde_L1_distance']:.4f}")


if __name__ == "__main__":
    main()
