#!/usr/bin/env bash
set -e
source "$(dirname "$0")/_env.sh"
python -m src.baselines.deepvar_adapter --config configs/deepvar_heew.yaml --seeds $SEEDS
