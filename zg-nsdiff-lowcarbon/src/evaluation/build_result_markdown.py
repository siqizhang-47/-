"""Assemble results/experiment_results.md — exactly the paper's outputs:

  Table I  (per-variable MAPE / AW / PICP, best bold, second underlined)
  figure1  Load prediction curve
  figure2  Pearson correlation matrices of the real and generated sample
  figure3  Actual vs. predicted load distribution

python -m src.evaluation.build_result_markdown
"""
import argparse
import os

import pandas as pd

from src.evaluation.build_tables import build_table1


def fig(path, caption):
    if os.path.exists(path):
        return f"![{caption}]({path})\n\n{caption}"
    return f"_（{caption} 尚未生成：{path}，先运行 bash scripts/build_results.sh）_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/summary_metrics.csv")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--output", default="results/experiment_results.md")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    parts = [
        "# Experiments — HEEW",
        "",
        build_table1(df),
        "",
        fig(os.path.join(args.figures, "figure1_load_prediction_curve.png"),
            "figure1 Load prediction curve."),
        "",
        fig(os.path.join(args.figures, "figure2_correlation_matrices.png"),
            "figure2 The Pearson correlation matrices of the real and generated sample."),
        "",
        fig(os.path.join(args.figures, "figure3_load_distribution.png"),
            "figure3 Actual vs. predicted load distribution."),
        "",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
