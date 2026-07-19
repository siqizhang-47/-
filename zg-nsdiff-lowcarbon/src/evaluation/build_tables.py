"""Build the paper result tables from summary_metrics.csv (spec section 16).

Bolding rules: MAPE/PINAW/NCRPS smaller is better; PICP closest to 0.95;
Active F1 larger is better.

python -m src.evaluation.build_tables --input results/summary_metrics.csv --output_dir results/tables
"""
import argparse
import os

import numpy as np
import pandas as pd

from src.data.low_carbon_schema import TARGET_NAMES, ZERO_TARGET_NAMES

MODEL_ORDER = ["nsdiff", "d3u", "wavestitch", "zg_nsdiff"]
MODEL_LABEL = {"nsdiff": "NsDiff", "d3u": "D3U", "wavestitch": "WaveStitch",
               "zg_nsdiff": "ZG-NsDiff"}


def fmt(mean, std, bold=False, digits=4):
    s = f"{mean:.{digits}f} ± {std:.{digits}f}"
    return f"**{s}**" if bold else s


def best_index(values, mode):
    v = np.asarray(values, dtype=np.float64)
    if mode == "min":
        return int(np.nanargmin(v))
    if mode == "max":
        return int(np.nanargmax(v))
    if mode == "picp":
        return int(np.nanargmin(np.abs(v - 0.95)))
    if mode == "none":
        return -1
    raise ValueError(mode)


def table(df, columns, out_path, title):
    """columns: list of (metric_key, header, mode)."""
    models = [m for m in MODEL_ORDER if m in df["model"].values]
    models += [m for m in df["model"].values if m not in models]
    lines = [f"### {title}", "",
             "| 模型 | " + " | ".join(h for _, h, _ in columns) + " |",
             "|---|" + "---:|" * len(columns)]
    best = {key: best_index(
        [df.loc[df.model == m, f"{key}_mean"].iloc[0] for m in models], mode)
        for key, _, mode in columns}
    for i, m in enumerate(models):
        row = df.loc[df.model == m].iloc[0]
        cells = [fmt(row[f"{k}_mean"], row[f"{k}_std"], bold=(best[k] == i))
                 for k, _, _ in columns]
        lines.append(f"| {MODEL_LABEL.get(m, m)} | " + " | ".join(cells) + " |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/summary_metrics.csv")
    ap.add_argument("--output_dir", default="results/tables")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input)

    # main 4-variable table (16.1)
    cols = []
    for name in TARGET_NAMES:
        if name == "electricity":
            cols.append(("electricity_mape", "Electricity MAPE ↓", "min"))
        else:
            cols.append((f"{name}_mape_pos", f"{name.capitalize()} MAPE+ ↓", "min"))
        cols.append((f"{name}_pinaw", f"{name.capitalize()} PINAW ↓", "min"))
        cols.append((f"{name}_picp", f"{name.capitalize()} PICP", "picp"))
    table(df, cols, os.path.join(args.output_dir, "main_table.md"), "四变量主表")

    # probabilistic & joint table (16.2)
    table(df, [
        ("ncrps_mean", "NCRPS ↓", "min"),
        ("energy_score", "Energy Score ↓", "min"),
        ("variogram_score", "Variogram Score ↓", "min"),
        ("corr_error", "Corr. Error ↓", "min"),
    ], os.path.join(args.output_dir, "joint_table.md"), "概率与联合指标")

    # zero-state tables (16.3)
    for name in ZERO_TARGET_NAMES:
        table(df, [
            (f"{name}_active_f1", "Active F1 ↑", "max"),
            (f"{name}_brier", "Brier ↓", "min"),
            (f"{name}_ece", "ECE ↓", "min"),
            (f"{name}_true_zero_rate", "True zero rate", "none"),
            (f"{name}_pred_zero_rate", "Pred. zero rate", "none"),
            (f"{name}_zero_rate_abs_error", "Rate error ↓", "min"),
        ], os.path.join(args.output_dir, f"zero_table_{name}.md"),
              f"零状态结果 — {name}")


if __name__ == "__main__":
    main()
