#!/usr/bin/env bash
# NsDiff: F/G pretraining -> joint training -> test export
set -e
source "$(dirname "$0")/_env.sh"
python -m src.experiments.pretrain_f_low_carbon --config configs/nsdiff_heew.yaml --seeds $SEEDS
python -m src.experiments.pretrain_g_low_carbon --config configs/nsdiff_heew.yaml --seeds $SEEDS
python -m src.experiments.NsDiffLowCarbon --config configs/nsdiff_heew.yaml --load_pretrain --seeds $SEEDS
