"""figure1: Load prediction curve.

Per-variable subplots (a)(b)(c): truth (black), each model's predicted mean
curve, and the 95% interval of --interval_model (default nsdiff), over the
first PLOT_HOURS consecutive hours of the test stream (fixed in advance,
never cherry-picked).

python -m src.visualization.plot_intervals --prediction_root artifacts/predictions
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baselines.prediction_contract import iter_shards
from src.data.low_carbon_schema import TARGET_NAMES

MODEL_LABEL = {"deepvar": "DeepVAR", "d3u": "D3U", "wavestitch": "WaveStitch",
               "nsdiff": "NsDiff"}
MODEL_ORDER = ["deepvar", "d3u", "wavestitch", "nsdiff"]
VAR_LABEL = {"electricity": "Electricity", "cooling": "Cooling", "heat": "Heat"}
SUB = "abc"
PLOT_HOURS = 168


def stitch_first_window(pred_dir):
    """Concatenate non-overlapping consecutive forecasts (stride H) until the
    first PLOT_HOURS hours of the test stream are covered."""
    rows = {"ts": [], "truth": [], "lower": [], "upper": [], "mean": []}
    next_fsi = None
    hours = 0
    for shard in iter_shards(pred_dir):
        fsi = shard["forecast_start_index"]
        order = np.argsort(fsi)
        for i in order:
            if next_fsi is None:
                next_fsi = int(fsi[i])
            if int(fsi[i]) != next_fsi:
                continue
            s = shard["samples"][i]          # [H,D,S]
            rows["ts"].append(shard["timestamps"][i])
            rows["truth"].append(shard["truth"][i])
            rows["lower"].append(np.quantile(s, 0.025, axis=-1))
            rows["upper"].append(np.quantile(s, 0.975, axis=-1))
            rows["mean"].append(s.mean(axis=-1))
            H = s.shape[0]
            next_fsi += H
            hours += H
            if hours >= PLOT_HOURS:
                return {k: np.concatenate(v, axis=0)[:PLOT_HOURS] for k, v in rows.items()}
    if hours == 0:
        raise RuntimeError(f"no usable windows in {pred_dir}")
    return {k: np.concatenate(v, axis=0) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction_root", default="artifacts/predictions")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--interval_model", default="nsdiff",
                    help="model whose 95%% interval is shaded")
    ap.add_argument("--output", default="figures/figure1_load_prediction_curve.png")
    args = ap.parse_args()

    model_dirs = {os.path.basename(os.path.dirname(d)): d
                  for d in sorted(glob.glob(os.path.join(
                      args.prediction_root, "*", f"seed_{args.seed}")))}
    if not model_dirs:
        raise SystemExit("no predictions found")
    models = [m for m in MODEL_ORDER if m in model_dirs]
    models += [m for m in model_dirs if m not in models]
    windows = {m: stitch_first_window(model_dirs[m]) for m in models}
    interval_model = args.interval_model if args.interval_model in windows else models[-1]

    t = pd.to_datetime(windows[models[0]]["ts"], unit="s")
    fig, axes = plt.subplots(len(TARGET_NAMES), 1,
                             figsize=(11, 2.8 * len(TARGET_NAMES)), sharex=True)
    for d, (name, ax) in enumerate(zip(TARGET_NAMES, axes)):
        w_int = windows[interval_model]
        ax.fill_between(t, w_int["lower"][:, d], w_int["upper"][:, d], alpha=0.25,
                        color="tab:blue",
                        label=f"95% interval ({MODEL_LABEL.get(interval_model, interval_model)})")
        for m in models:
            ax.plot(t, windows[m]["mean"][:, d], lw=1.2, label=MODEL_LABEL.get(m, m))
        ax.plot(t, w_int["truth"][:, d], lw=1.4, color="k", label="Actual")
        ax.set_ylabel("kW")
        ax.set_title(f"({SUB[d]}) {VAR_LABEL[name]}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[0].legend(loc="upper right", ncol=3, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
