#!/usr/bin/env bash
set -euo pipefail
# Ablation ladder of docs/V3_MODIFICATIONS.md (plan section 19).
#   V0 current model | V1 +seasonal | V2 +horizon exog | V3 +slope loss | V4 +local TCN
#   V5 +residual variance | V6 +residual diffusion | V7 +PV constraint (full)
# Usage: VARIANTS="V0 V4 V7" SEEDS="1 2 3" bash scripts/Energy4Exog/run_v3_ablation.sh
cd "$(dirname "$0")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU:-0}"
export PYTHONPATH=./
source scripts/Energy4Exog/v3_common.sh

OUT_ROOT="${OUT_ROOT:-./energy4_exog_outputs/v3_ablation}"
SEEDS="${SEEDS:-1}"
VARIANTS="${VARIANTS:-V0 V1 V2 V3 V4 V5 V6 V7}"

variant_args() {
  case "$1" in
    V0) echo "${V0_ARGS[@]}" "${NO_SLOPE_ARGS[@]}" "${ABSOLUTE_DIFF_ARGS[@]}" "${NO_PV_ARGS[@]}" ;;
    V1) echo "${V3_MEAN_ARGS[@]}" --v3_use_seasonal=True --v3_use_horizon_query=False --v3_use_local_tcn=False "${VAR_V2_ARGS[@]}" "${NO_SLOPE_ARGS[@]}" "${ABSOLUTE_DIFF_ARGS[@]}" "${NO_PV_ARGS[@]}" ;;
    V2) echo "${V3_MEAN_ARGS[@]}" --v3_use_seasonal=True --v3_use_horizon_query=True  --v3_use_local_tcn=False "${VAR_V2_ARGS[@]}" "${NO_SLOPE_ARGS[@]}" "${ABSOLUTE_DIFF_ARGS[@]}" "${NO_PV_ARGS[@]}" ;;
    V3) echo "${V3_MEAN_ARGS[@]}" --v3_use_seasonal=True --v3_use_horizon_query=True  --v3_use_local_tcn=False "${VAR_V2_ARGS[@]}" "${SLOPE_ARGS[@]}"    "${ABSOLUTE_DIFF_ARGS[@]}" "${NO_PV_ARGS[@]}" ;;
    V4) echo "${V3_MEAN_ARGS[@]}" --v3_use_seasonal=True --v3_use_horizon_query=True  --v3_use_local_tcn=True  "${VAR_V2_ARGS[@]}" "${SLOPE_ARGS[@]}"    "${ABSOLUTE_DIFF_ARGS[@]}" "${NO_PV_ARGS[@]}" ;;
    V5) echo "${V3_MEAN_ARGS[@]}" --v3_use_seasonal=True --v3_use_horizon_query=True  --v3_use_local_tcn=True  "${VAR_V3_ARGS[@]}" "${SLOPE_ARGS[@]}"    "${ABSOLUTE_DIFF_ARGS[@]}" "${NO_PV_ARGS[@]}" ;;
    V6) echo "${V3_MEAN_ARGS[@]}" --v3_use_seasonal=True --v3_use_horizon_query=True  --v3_use_local_tcn=True  "${VAR_V3_ARGS[@]}" "${SLOPE_ARGS[@]}"    "${RESIDUAL_DIFF_ARGS[@]}" "${NO_PV_ARGS[@]}" ;;
    V7) echo "${V3_MEAN_ARGS[@]}" --v3_use_seasonal=True --v3_use_horizon_query=True  --v3_use_local_tcn=True  "${VAR_V3_ARGS[@]}" "${SLOPE_ARGS[@]}"    "${RESIDUAL_DIFF_ARGS[@]}" "${PV_ARGS[@]}" ;;
    *) echo "unknown variant $1" >&2; exit 1 ;;
  esac
}

for V in ${VARIANTS}; do
  for SEED in ${SEEDS}; do
    echo "============================================================"
    echo "Ablation ${V} seed=${SEED}"
    echo "============================================================"
    # shellcheck disable=SC2046
    python3 -u ./src/experiments/NsDiff_energy_exog.py \
      "${COMMON_ARGS[@]}" $(variant_args "${V}") \
      --model_type="ExoMV-D-${V}" --model_display_name="ExoMV-D ${V}" \
      --analysis_dir="${OUT_ROOT}/${V}" run "${SEED}"
  done
done
python3 scripts/Energy4Exog/compare_v3_ablation.py --root "${OUT_ROOT}" --seeds ${SEEDS} --variants ${VARIANTS}
