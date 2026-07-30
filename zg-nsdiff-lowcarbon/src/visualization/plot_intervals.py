"""figure1: Load prediction curve.

Per-variable subplots (a)(b)(c): truth (black), each model's predicted mean
curve, and the 95% interval of --interval_model.

The plotted segment is chosen AUTOMATICALLY as the smoothest consecutive
PLOT_HOURS stretch of the ACTUAL load curve over the whole test stream:
segments are scored by normalized first-difference roughness
(mean|Δy|/σ_d plus a large-jump penalty, summed over variables) and the
minimum-score segment is used. The rule is deterministic and data-driven —
rerunning always selects the same segment.

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

MODEL_LABEL = {"deepvar": "DeepVAR", "d3u": "WCRD", "wavestitch": "WaveStitch",
               "nsdiff": "NsDiff"}
MODEL_ORDER = ["deepvar", "wavestitch", "nsdiff", "d3u"]
VAR_LABEL = {"electricity": "Electricity", "cooling": "Cooling", "heat": "Heat"}
SUB = "abc"
PLOT_HOURS = 168
JUMP_PENALTY = 3.0  # weight of the max single-step jump in the roughness score


def load_forecasts(pred_dir):
    """All forecasts of one model, sorted by forecast_start_index."""
    fsi, ts, truth, lower, upper, mean = [], [], [], [], [], []
    for shard in iter_shards(pred_dir):
        fsi.append(shard["forecast_start_index"])
        ts.append(shard["timestamps"])
        truth.append(shard["truth"])
        s = shard["samples"]
        lower.append(np.quantile(s, 0.025, axis=-1))
        upper.append(np.quantile(s, 0.975, axis=-1))
        mean.append(s.mean(axis=-1))
    fsi = np.concatenate(fsi)
    order = np.argsort(fsi)
    return {
        "fsi": fsi[order],
        "ts": np.concatenate(ts)[order],
        "truth": np.concatenate(truth)[order],
        "lower": np.concatenate(lower)[order],
        "upper": np.concatenate(upper)[order],
        "mean": np.concatenate(mean)[order],
    }


def consecutive_runs(fsi, H, need):
    """Start positions of runs of `need` forecasts with stride exactly H."""
    starts = []
    i = 0
    n = len(fsi)
    while i + need <= n:
        ok = all(fsi[i + k + 1] - fsi[i + k] == H for k in range(need - 1))
        if ok:
            starts.append(i)
            i += 1
        else:
            i += 1
    return starts


def roughness(truth_seg, scale):
    """Normalized roughness of a [T,D] truth segment; lower = smoother."""
    d = np.abs(np.diff(truth_seg, axis=0))         # [T-1, D]
    per_var = d.mean(axis=0) + JUMP_PENALTY * d.max(axis=0)
    return float((per_var / scale).sum())


def select_smoothest_segment(fc, H):
    need = int(np.ceil(PLOT_HOURS / H))
    runs = consecutive_runs(fc["fsi"], H, need)
    if not runs:
        raise RuntimeError("no consecutive forecast run long enough to plot")
    scale = fc["truth"].reshape(-1, fc["truth"].shape[-1]).std(axis=0) + 1e-9
    best, best_score = None, np.inf
    for r in runs:
        seg = fc["truth"][r: r + need].reshape(-1, fc["truth"].shape[-1])[:PLOT_HOURS]
        score = roughness(seg, scale)
        if score < best_score:
            best, best_score = r, score
    print(f"selected smoothest segment: run starting at forecast_start_index="
          f"{fc['fsi'][best]} (roughness {best_score:.3f}, {len(runs)} candidates)")
    return best, need


def extract_segment(fc, run_start, need):
    out = {}
    for k in ["ts", "truth", "lower", "upper", "mean"]:
        out[k] = fc[k][run_start: run_start + need].reshape(
            -1, *fc[k].shape[2:])[:PLOT_HOURS]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction_root", default="artifacts/predictions")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--interval_model", default=None,
                    help="model whose 95%% interval is shaded "
                         "(default: d3u/WCRD if present, else last available)")
    ap.add_argument("--start_index", type=int, default=None,
                    help="override the automatic segment choice with an explicit "
                         "forecast_start_index")
    ap.add_argument("--output", default="figures/figure1_load_prediction_curve.png")
    args = ap.parse_args()

    model_dirs = {os.path.basename(os.path.dirname(d)): d
                  for d in sorted(glob.glob(os.path.join(
                      args.prediction_root, "*", f"seed_{args.seed}")))}
    if not model_dirs:
        raise SystemExit("no predictions found")
    models = [m for m in MODEL_ORDER if m in model_dirs]
    models += [m for m in model_dirs if m not in models]
    forecasts = {m: load_forecasts(model_dirs[m]) for m in models}

    interval_model = args.interval_model or ("d3u" if "d3u" in forecasts else models[-1])
    if interval_model not in forecasts:
        raise SystemExit(f"interval model '{interval_model}' not found; "
                         f"available: {list(forecasts)}")

    ref = forecasts[interval_model]
    H = ref["truth"].shape[1]
    if args.start_index is not None:
        pos = np.where(ref["fsi"] == args.start_index)[0]
        if len(pos) == 0:
            raise SystemExit(f"forecast_start_index {args.start_index} not found")
        run_start, need = int(pos[0]), int(np.ceil(PLOT_HOURS / H))
    else:
        run_start, need = select_smoothest_segment(ref, H)

    segments = {m: extract_segment(forecasts[m], run_start, need) for m in models}
    t = pd.to_datetime(segments[interval_model]["ts"], unit="s")

    fig, axes = plt.subplots(len(TARGET_NAMES), 1,
                             figsize=(11, 2.8 * len(TARGET_NAMES)), sharex=True)
    for d, (name, ax) in enumerate(zip(TARGET_NAMES, axes)):
        w_int = segments[interval_model]
        ax.fill_between(t, w_int["lower"][:, d], w_int["upper"][:, d], alpha=0.25,
                        color="tab:blue",
                        label=f"95% interval ({MODEL_LABEL.get(interval_model, interval_model)})")
        for m in models:
            ax.plot(t, segments[m]["mean"][:, d], lw=1.2, label=MODEL_LABEL.get(m, m))
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
