"""Truth / predicted mean / 95% interval over a FIXED test window (spec 18.3).

The plotting window is the first full 168 consecutive hours of the test
stream — fixed before results are known, never cherry-picked.

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

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "Microsoft YaHei",
                                   "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from src.baselines.prediction_contract import iter_shards
from src.data.low_carbon_schema import TARGET_NAMES

MODEL_LABEL = {"nsdiff": "NsDiff", "d3u": "D3U", "wavestitch": "WaveStitch",
               "zg_nsdiff": "ZG-NsDiff"}
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
            s = shard["samples"][i]          # [H,4,S]
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


VAR_LABEL = {"electricity": "电负荷", "cooling": "冷负荷", "heating": "热负荷", "pv": "光伏出力"}
SUB = "abcd"


def plot_combined(windows, model, output_dir):
    """Template figure: 4 stacked subplots (a)-(d) for one model —
    truth / predicted mean / 95% interval."""
    w = windows[model]
    t = pd.to_datetime(w["ts"], unit="s")
    fig, axes = plt.subplots(len(TARGET_NAMES), 1, figsize=(11, 2.6 * len(TARGET_NAMES)),
                             sharex=True)
    for d, (name, ax) in enumerate(zip(TARGET_NAMES, axes)):
        ax.fill_between(t, w["lower"][:, d], w["upper"][:, d], alpha=0.3, label="95%预测区间")
        ax.plot(t, w["mean"][:, d], lw=1.3, label="预测均值")
        ax.plot(t, w["truth"][:, d], lw=1.3, color="k", label="真实值")
        ax.set_ylabel("kW")
        ax.set_title(f"({SUB[d]}) {VAR_LABEL[name]}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes[0].legend(loc="upper right", ncol=3)
    fig.suptitle(f"真实值、预测均值与95%预测区间（{MODEL_LABEL.get(model, model)}）")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = os.path.join(output_dir, "interval_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction_root", default="artifacts/predictions")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default=None,
                    help="model for the combined template figure (default: zg_nsdiff if present)")
    ap.add_argument("--output_dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_dirs = {os.path.basename(os.path.dirname(d)): d
                  for d in sorted(glob.glob(os.path.join(args.prediction_root, "*", f"seed_{args.seed}")))}
    windows = {m: stitch_first_window(p) for m, p in model_dirs.items()}

    combined_model = args.model or ("zg_nsdiff" if "zg_nsdiff" in windows
                                    else next(iter(windows)))
    plot_combined(windows, combined_model, args.output_dir)

    for d, name in enumerate(TARGET_NAMES):
        fig, axes = plt.subplots(len(windows), 1, figsize=(11, 2.6 * len(windows)),
                                 sharex=True, squeeze=False)
        for ax, (m, w) in zip(axes[:, 0], windows.items()):
            t = pd.to_datetime(w["ts"], unit="s")
            ax.fill_between(t, w["lower"][:, d], w["upper"][:, d], alpha=0.3,
                            label="95% interval")
            ax.plot(t, w["mean"][:, d], label="pred mean", lw=1.2)
            ax.plot(t, w["truth"][:, d], label="truth", lw=1.2, color="k")
            ax.set_ylabel("kW")
            ax.set_title(f"{name} — {MODEL_LABEL.get(m, m)} (oracle future weather)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        axes[0, 0].legend(loc="upper right")
        fig.autofmt_xdate()
        fig.tight_layout()
        path = os.path.join(args.output_dir, f"interval_{name}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
