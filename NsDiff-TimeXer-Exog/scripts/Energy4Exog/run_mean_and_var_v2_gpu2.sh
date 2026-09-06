#!/usr/bin/env bash
set -euo pipefail

# Override with e.g. PHYSICAL_GPU=0 bash scripts/Energy4Exog/run_mean_and_var_v2_gpu2.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU:-2}"
export PYTHONPATH=./

python3 -u ./src/experiments/NsDiff_energy_exog.py \
  --dataset_type=Energy4Exog \
  --model_type=NsDiffMeanVarV2Exog \
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
  --epochs=10 \
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
  --var_v2_max_log_correction=1.0 \
  --var_v2_weather_regime_len=24 \
  --load_pretrain=False \
  --analysis_dir=./energy4_exog_outputs/mean_and_var_v2 \
  --plot_max_windows=512 \
  run 42
