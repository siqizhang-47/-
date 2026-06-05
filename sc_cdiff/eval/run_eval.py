"""Step 13: evaluation suite. Loads sampled scenarios, computes all metric
groups, prints a summary table. Filled days are already excluded at sampling
(AlignedDays drops them)."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from ..data.build_dataset import _load_cfg
from .joint import (mean_energy_score, mean_variogram_score, seasonal_corr_error)
from .marginal import marginal_metrics
from .physical import physical_metrics
from .reliability import reliability_metrics
from .temporal import temporal_metrics
from .decision import decision_metrics


def _months_seasons(start, n_hours_start="2001-06-01 00:00:00"):
    base = pd.Timestamp(n_hours_start)
    months = np.array([(base + pd.Timedelta(hours=int(s))).month for s in start])
    season = ((months % 12) // 3)        # 0 winter..3 autumn
    return months, season


def evaluate(cfg, split="test", k=0, do_decision=True):
    paths = cfg["paths"]
    f = os.path.join(paths["artifacts"], f"scenarios_{split}_k{k}.npz")
    d = np.load(f)
    scen, truth, irr, start = d["scenarios"], d["truth"], d["irr"], d["start"]
    months, season = _months_seasons(start)
    print(f"[eval] {split} k={k}: scenarios {scen.shape}")

    report = {
        "energy_score": mean_energy_score(scen, truth),
        "variogram_score": mean_variogram_score(scen, truth),
        "seasonal_corr_err": seasonal_corr_error(scen, truth, season),
        "marginal": marginal_metrics(scen, truth),
        "temporal": temporal_metrics(scen, truth),
        "physical": physical_metrics(scen, truth, irr, pv_cap=cfg["pv_cap"], months=months),
        "reliability": reliability_metrics(scen, truth),
    }
    if do_decision and k == 0:
        report["decision"] = decision_metrics(scen, truth)

    out = os.path.join(paths["artifacts"], f"eval_{split}_k{k}.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)

    # console summary
    print(f"  Energy Score      : {report['energy_score']:.4f}")
    print(f"  Variogram Score   : {report['variogram_score']:.4f}")
    print(f"  Seasonal corr err : {report['seasonal_corr_err']:.4f}")
    print(f"  PV night viol     : {report['physical']['night_pv_viol_rate']:.4f}")
    print(f"  Neg value rate    : {report['physical']['neg_rate']:.4f}")
    chans = cfg["channels"]
    print("  PICP / PINAW per channel:")
    for ch, nm in enumerate(chans):
        r = report["reliability"][ch]
        print(f"    {nm:3s}: PICP={r['picp']:.3f} PINAW={r['pinaw']:.3f} RMSE={r['rmse']:.1f}")
    if "decision" in report:
        print(f"  Decision: {report['decision']}")
    print(f"[eval] wrote {out}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=None, help="evaluate a single k; default all configured")
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts
    ks = [args.k] if args.k is not None else cfg["sample"]["ks"]
    for k in ks:
        evaluate(cfg, args.split, k)


if __name__ == "__main__":
    main()
