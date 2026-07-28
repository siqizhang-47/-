#!/usr/bin/env bash
set -e
source "$(dirname "$0")/_env.sh"
python -m src.baselines.wavestitch_adapter --config configs/wavestitch_heew.yaml --seeds $SEEDS
