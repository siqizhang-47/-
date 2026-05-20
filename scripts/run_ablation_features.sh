#!/usr/bin/env bash
# Feature ablation: load / load+weather / load+weather+event, H=96.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data/sxq_data/EWELD_labeled_output}"
GPU="${GPU:-2}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-64}"
MAX_USERS="${MAX_USERS:-100}"
EVAL_STRIDE="${EVAL_STRIDE:-4}"

mkdir -p logs results

for MODEL in iTransformer TimeMixer TimeXer; do
    for FS in load load_weather load_weather_event; do
        echo "==> $MODEL feature_set=$FS (users=$MAX_USERS, eval_stride=$EVAL_STRIDE)"
        CUDA_VISIBLE_DEVICES=$GPU python -m experiment.run \
            --data_root "$DATA_ROOT" \
            --model $MODEL \
            --feature_set $FS \
            --pred_len 96 \
            --seq_len 96 \
            --epochs $EPOCHS \
            --batch_size $BATCH \
            --gpu $GPU \
            --max_users $MAX_USERS \
            --eval_stride $EVAL_STRIDE \
            --tag ablate_features \
            2>&1 | tee "logs/${MODEL}_${FS}.log"
    done
done

python -m experiment.aggregate results
