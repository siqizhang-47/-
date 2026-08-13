#!/usr/bin/env bash
# Ablation D (scale sharing), H (Delta structure), I (online objective).
set -e
cd "$(dirname "$0")/.."

python experiments/run_adapter.py --method proposed --override scale_mode=shared --name abl_D_shared_scale

for MODE in lowrank fullrank gated_only; do
    python experiments/run_adapter.py --method proposed --override delta_mode=$MODE --name "abl_H_${MODE}"
done

# ablation I: the s_c trajectory is logged per day in daily_metrics.parquet (s_* columns)
for LOSS in crps nll mse; do
    python experiments/run_adapter.py --method proposed --override loss=$LOSS --name "abl_I_${LOSS}"
done
