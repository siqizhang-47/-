#!/usr/bin/env bash
set -e
source "$(dirname "$0")/_env.sh"
python -m src.experiments.pretrain_f_low_carbon --config configs/nsdiff_low_carbon.yaml --seeds $SEEDS
python -m src.experiments.pretrain_g_low_carbon --config configs/nsdiff_low_carbon.yaml --seeds $SEEDS
