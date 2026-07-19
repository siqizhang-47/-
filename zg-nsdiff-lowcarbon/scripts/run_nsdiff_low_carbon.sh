#!/usr/bin/env bash
set -e
source "$(dirname "$0")/_env.sh"
python -m src.experiments.NsDiffLowCarbon --config configs/nsdiff_low_carbon.yaml --load_pretrain --seeds $SEEDS
