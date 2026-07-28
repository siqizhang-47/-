"""Build Table I: PERFORMANCE COMPARISON ON THE REAL-WORLD HEEW DATASET.

Best results in **bold**, second-best <u>underlined</u>.
MAPE/AW: smaller is better. PICP: closest to 0.95 is better.

python -m src.evaluation.build_tables --input results/summary_metrics.csv --output_dir results
"""
import argparse
import os

import numpy as np
import pandas as pd

from src.data.low_carbon_schema import TARGET_NAMES

MODEL_ORDER = ["deepvar", "d3u", "wavestitch", "nsdiff"]
MODEL_LABEL = {"deepvar": "DeepVAR", "d3u": "D3U", "wavestitch": "WaveStitch",
               "nsdiff": "NsDiff"}
VAR_LABEL = {"electricity": "Electricity", "cooling": "Cool", "heat": "Heat"}


def rank_order(values, mode):
    v = np.asarray(values, dtype=np.float64)
    key = np.abs(v - 0.95) if mode == "picp" else v
    return np.argsort(key)  # ascending: best first


def fmt(value, rank, digits=4):
    s = f"{value:.{digits}f}"
    if rank == 0:
        return f"**{s}**"
    if rank == 1:
        return f"<u>{s}</u>"
    return s


def build_table1(df: pd.DataFrame) -> str:
    models = [m for m in MODEL_ORDER if m in df["model"].values]
    models += [m for m in df["model"].values if m not in models]
    columns = []
    for name in TARGET_NAMES:
        v = VAR_LABEL[name]
        columns += [(f"{name}_mape", f"{v}-MAPE", "min"),
                    (f"{name}_aw", f"{v}-AW", "min"),
                    (f"{name}_picp", f"{v}-PICP", "picp")]

    ranks = {}
    for key, _, mode in columns:
        vals = [df.loc[df.model == m, f"{key}_mean"].iloc[0] for m in models]
        order = rank_order(vals, mode)
        r = np.empty(len(models), dtype=int)
        r[order] = np.arange(len(models))
        ranks[key] = r

    lines = [
        "### Table I  PERFORMANCE COMPARISON ON THE REAL-WORLD HEEW DATASET. "
        "THE BEST RESULTS ARE SHOWN IN BOLD AND THE SECOND-BEST RESULTS ARE UNDERLINED.",
        "",
        "| Model | " + " | ".join(h for _, h, _ in columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for i, m in enumerate(models):
        row = df.loc[df.model == m].iloc[0]
        cells = [fmt(row[f"{k}_mean"], ranks[k][i]) for k, _, _ in columns]
        lines.append(f"| {MODEL_LABEL.get(m, m)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/summary_metrics.csv")
    ap.add_argument("--output_dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input)
    table = build_table1(df)
    out = os.path.join(args.output_dir, "table1.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
