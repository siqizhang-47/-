#!/usr/bin/env bash
set -e
source "$(dirname "$0")/_env.sh"
python -m src.data.low_carbon_preprocess --config configs/low_carbon_common.yaml
python -m src.visualization.plot_correlations --output_dir figures
