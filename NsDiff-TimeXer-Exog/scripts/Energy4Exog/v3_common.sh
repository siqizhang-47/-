#!/usr/bin/env bash
# Shared argument sets for the V3 experiments (sourced by the run scripts).
# Every array below corresponds to one column of the ablation table in
# docs/V3_MODIFICATIONS.md, so a variant is simply the union of the enabled sets.

COMMON_ARGS=(
  --dataset_type=Energy4Exog
  --data_path=./dataset
  --device="${DEVICE:-cuda:0}"
  --windows=168
  --pred_len=24
  --horizon=1
  --batch_size=32
  --train_ratio=0.7
  --test_ratio=0.2
  --num_worker="${NUM_WORKER:-4}"
  --rolling_length=96
  --timexer_patch_len=24
  --enable_posthoc_calibration="${CALIBRATION:-True}"
  --calibration_max_windows=512
  --enforce_nonnegative_outputs=True
  --plot_example_policy=best
  --plot_max_windows=512
  --sample_minibatch="${SAMPLE_MINIBATCH:-25}"
  --load_pretrain=False
)

# V0: the current ExoMV-D model (TimeXer-Exog mean + Variance V2, joint training, CRPS checkpoint).
V0_ARGS=(
  --estimator_variant=mean_and_var_v2
  --timexer_d_model=512 --timexer_n_heads=8 --timexer_e_layers=2 --timexer_d_ff=1024
  --var_v2_patch_len=12 --var_v2_d_model=96 --var_v2_n_heads=4 --var_v2_d_ff=192
  --var_v2_base_hidden=512 --var_v2_max_log_correction=0.60 --var_v2_weather_regime_len=24
  --detach_mean_for_diffusion=False
  --training_schedule=joint
  --epochs="${EPOCHS:-10}" --patience="${PATIENCE:-5}"
)

# V3 mean backbone (plan section 21 starting point) + staged training (sections 8/9).
V3_MEAN_ARGS=(
  --estimator_variant=v3
  --v3_d_model="${V3_D_MODEL:-256}" --v3_n_heads=8 --v3_e_layers=2 --v3_d_ff="${V3_D_FF:-512}"
  --v3_dec_layers=2 --v3_tcn_hidden=128 --v3_tcn_kernel=3 --v3_tcn_dilations=1,2,4,8,16
  --v3_seasonal_gate="${SEASONAL_GATE:-fixed}" --v3_seasonal_daily_weight=0.7
  --detach_mean_for_diffusion=True
  --training_schedule=staged
  --mean_epochs="${MEAN_EPOCHS:-60}" --mean_patience="${MEAN_PATIENCE:-10}"
  --epochs="${EPOCHS:-10}" --patience="${PATIENCE:-5}"
  --joint_epochs="${JOINT_EPOCHS:-0}" --joint_mean_lr_scale=0.1
)

# Variance V2 (rolling-variance target) kept for the ablation rungs before Phase 3.
VAR_V2_ARGS=(
  --variance_estimator=residual_v2 --variance_loss=rolling
  --var_v2_patch_len=12 --var_v2_d_model=96 --var_v2_n_heads=4 --var_v2_d_ff=192
  --var_v2_base_hidden=512 --var_v2_max_log_correction=0.60 --var_v2_weather_regime_len=24
)
# Variance V3 (forecast-residual heteroscedastic NLL).
VAR_V3_ARGS=(
  --variance_estimator=residual_v3 --variance_loss=residual_nll
  --var_v3_d_model=128 --var_v3_n_heads=4 --var_v3_d_ff=256 --var_v3_patch_len=12 --var_v3_recent_len=48
)
SLOPE_ARGS=(--lambda_slope=0.3 --lambda_curve=0.05)
NO_SLOPE_ARGS=(--lambda_slope=0.0 --lambda_curve=0.0)
RESIDUAL_DIFF_ARGS=(--diffusion_space=residual)
ABSOLUTE_DIFF_ARGS=(--diffusion_space=absolute)
PV_ARGS=(--pv_transform=log1p --pv_daylight_gate="${PV_GATE:-False}")
NO_PV_ARGS=(--pv_transform=none --pv_daylight_gate=False)
