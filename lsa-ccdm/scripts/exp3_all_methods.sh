#!/usr/bin/env bash
# Exp 3: all 8 drift-handling methods on the shared wnorm_on cache + DM tests.
# (Baselines 2/3 need `run_retrain_baselines.py` to have produced their caches.)
set -e
cd "$(dirname "$0")/.."

for METHOD in frozen tafas cosa shift_only scale_only proposed independent; do
    python experiments/run_adapter.py --method "$METHOD"
done

# periodic retraining / fine-tuning caches (skipped if not sampled yet)
for MODE in retrain finetune; do
    for YEAR in 2020 2021 2022; do
        CACHE="data/scenarios/${MODE}_${YEAR}"
        if [ -d "$CACHE" ]; then
            python experiments/run_adapter.py --method frozen --cache "$CACHE" --name "${MODE}_${YEAR}"
        fi
    done
done

python experiments/make_summary.py
