#!/usr/bin/env bash
set -e
source "$(dirname "$0")/_env.sh"
python -m src.experiments.ZGNsDiffLowCarbon --config configs/zg_nsdiff_low_carbon.yaml --pretrain --seeds $SEEDS
