#!/usr/bin/env bash
# Unified evaluation + tables + figures + result markdown
set -e
source "$(dirname "$0")/_env.sh"
python -m src.evaluation.evaluate_predictions --prediction_root artifacts/predictions --output results/per_seed_metrics.csv
python -m src.evaluation.aggregate_seeds --input results/per_seed_metrics.csv --output results/summary_metrics.csv
python -m src.evaluation.build_tables --input results/summary_metrics.csv --output_dir results/tables
python -m src.visualization.plot_correlations --output_dir figures
python -m src.visualization.plot_distributions --prediction_root artifacts/predictions --output_dir figures
python -m src.visualization.plot_intervals --prediction_root artifacts/predictions --output_dir figures
python -m src.visualization.plot_calibration --per_seed results/per_seed_metrics.csv --output_dir figures
python -m src.evaluation.build_result_markdown --metrics results/summary_metrics.csv --figures figures --output results/experiment_results.md
