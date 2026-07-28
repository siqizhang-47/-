#!/usr/bin/env bash
# Table I + figure1/2/3 + experiment_results.md
set -e
source "$(dirname "$0")/_env.sh"
python -m src.evaluation.evaluate_predictions --prediction_root artifacts/predictions --output results/per_seed_metrics.csv
python -m src.evaluation.aggregate_seeds --input results/per_seed_metrics.csv --output results/summary_metrics.csv
python -m src.evaluation.build_tables --input results/summary_metrics.csv --output_dir results
python -m src.visualization.plot_intervals --prediction_root artifacts/predictions --output figures/figure1_load_prediction_curve.png
python -m src.visualization.plot_correlation_comparison --prediction_root artifacts/predictions --output figures/figure2_correlation_matrices.png
python -m src.visualization.plot_distributions --prediction_root artifacts/predictions --output figures/figure3_load_distribution.png
python -m src.evaluation.build_result_markdown --metrics results/summary_metrics.csv --figures figures --output results/experiment_results.md
