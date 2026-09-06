#!/usr/bin/env bash
set -euo pipefail

# Publication-style 3-seed run for ExoMV-D (NsDiff + TimeXer-Exog mean + Variance V2).
# Default seeds match the NsDiff paper: {1,2,3}.
# Example: PHYSICAL_GPU=0 bash scripts/Energy4Exog/run_exomvd_3seeds.sh

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU:-0}"
export PYTHONPATH=./

OUT_ROOT="${OUT_ROOT:-./energy4_exog_outputs/exomvd_v2_calibrated}"
SEEDS="${SEEDS:-1 2 3}"
EPOCHS="${EPOCHS:-10}"

for SEED in ${SEEDS}; do
  echo "============================================================"
  echo "Running ExoMV-D seed=${SEED} on physical GPU ${PHYSICAL_GPU:-0}"
  echo "============================================================"
  python3 -u ./src/experiments/NsDiff_energy_exog.py \
    --dataset_type=Energy4Exog \
    --model_type=ExoMV-D \
    --estimator_variant=mean_and_var_v2 \
    --data_path=./dataset \
    --device=cuda:0 \
    --windows=168 \
    --pred_len=24 \
    --horizon=1 \
    --batch_size=32 \
    --train_ratio=0.7 \
    --test_ratio=0.2 \
    --num_worker=4 \
    --epochs="${EPOCHS}" \
    --patience=5 \
    --rolling_length=96 \
    --timexer_patch_len=24 \
    --timexer_d_model=512 \
    --timexer_n_heads=8 \
    --timexer_e_layers=2 \
    --timexer_d_ff=1024 \
    --var_v2_patch_len=12 \
    --var_v2_d_model=96 \
    --var_v2_n_heads=4 \
    --var_v2_d_ff=192 \
    --var_v2_base_hidden=512 \
    --var_v2_max_log_correction=0.60 \
    --var_v2_weather_regime_len=24 \
    --enable_posthoc_calibration=True \
    --calibration_max_windows=512 \
    --calibration_bias_strength=1.0 \
    --calibration_spread_min=0.45 \
    --calibration_spread_max=1.15 \
    --calibration_spread_steps=15 \
    --enforce_nonnegative_outputs=True \
    --plot_example_policy=best \
    --plot_max_windows=512 \
    --load_pretrain=False \
    --analysis_dir="${OUT_ROOT}" \
    run "${SEED}"
done

python3 scripts/Energy4Exog/aggregate_exomvd_3seeds.py \
  --root "${OUT_ROOT}" \
  --seeds ${SEEDS} \
  --model-name "ExoMV-D"

echo "Done. Summary: ${OUT_ROOT}/summary"
