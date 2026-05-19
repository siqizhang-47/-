#!/usr/bin/env bash
# Main comparison: all three baselines with the full feature set, H=96.
# Logs go to logs/, JSON results go to results/.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data/sxq_data/EWELD_labeled_output}"
GPU="${GPU:-2}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-64}"

mkdir -p logs results

for MODEL in iTransformer TimeMixer TimeXer; do
    echo "==> $MODEL (full features, H=96)"
    CUDA_VISIBLE_DEVICES=$GPU python -m experiment.run \
        --data_root "$DATA_ROOT" \
        --model $MODEL \
        --feature_set load_weather_event \
        --pred_len 96 \
        --seq_len 96 \
        --epochs $EPOCHS \
        --batch_size $BATCH \
        --gpu $GPU \
        --tag main \
        2>&1 | tee "logs/${MODEL}_main.log"
done

python -m experiment.aggregate results
