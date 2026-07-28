#!/usr/bin/env bash
set -e
source "$(dirname "$0")/_env.sh"
python -m src.data.low_carbon_preprocess --config configs/heew_common.yaml
