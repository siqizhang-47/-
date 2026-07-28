#!/usr/bin/env bash
set -e
source "$(dirname "$0")/_env.sh"
python -m src.baselines.d3u_adapter --config configs/d3u_heew.yaml --seeds $SEEDS
