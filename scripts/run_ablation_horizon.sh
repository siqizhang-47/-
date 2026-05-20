#!/usr/bin/env bash
# Horizon ablation: H = 24 / 48 / 96 (6h / 12h / 24h), full feature set.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/data/sxq_data/EWELD_labeled_output}"
GPU="${GPU:-2}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-64}"
MAX_USERS="${MAX_USERS:-100}"
EVAL_STRIDE="${EVAL_STRIDE:-4}"

mkdir -p logs results

for MODEL in iTransformer TimeMixer TimeXer; do
    for H in 24 48 96; do
        echo "==> $MODEL H=$H (users=$MAX_USERS, eval_stride=$EVAL_STRIDE)"
        CUDA_VISIBLE_DEVICES=$GPU python -m experiment.run \
            --data_root "$DATA_ROOT" \
            --model $MODEL \
            --feature_set load_weather_event \
            --pred_len $H \
            --seq_len 96 \
            --epochs $EPOCHS \
            --batch_size $BATCH \
            --gpu $GPU \
            --max_users $MAX_USERS \
            --eval_stride $EVAL_STRIDE \
            --tag ablate_H${H} \
            2>&1 | tee "logs/${MODEL}_H${H}.log"
    done
done

python -m experiment.aggregate results
