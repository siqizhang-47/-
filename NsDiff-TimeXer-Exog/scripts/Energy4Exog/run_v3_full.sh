#!/usr/bin/env bash
set -euo pipefail
# Full V3 model (V7 in the ablation table):
#   seasonal baseline + horizon-aligned exogenous queries + local TCN + slope/curvature loss
#   + residual-NLL variance + standardized-residual diffusion + log1p PV.
# Example: PHYSICAL_GPU=0 bash scripts/Energy4Exog/run_v3_full.sh
#          DEVICE=cpu NUM_WORKER=0 bash scripts/Energy4Exog/run_v3_full.sh
cd "$(dirname "$0")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU:-0}"
export PYTHONPATH=./
source scripts/Energy4Exog/v3_common.sh

OUT_ROOT="${OUT_ROOT:-./energy4_exog_outputs/v3_full}"
SEEDS="${SEEDS:-1}"
for SEED in ${SEEDS}; do
  python3 -u ./src/experiments/NsDiff_energy_exog.py \
    "${COMMON_ARGS[@]}" "${V3_MEAN_ARGS[@]}" "${VAR_V3_ARGS[@]}" "${SLOPE_ARGS[@]}" \
    "${RESIDUAL_DIFF_ARGS[@]}" "${PV_ARGS[@]}" \
    --model_type=ExoMV-D-V3 --model_display_name="ExoMV-D V3" \
    --analysis_dir="${OUT_ROOT}" run "${SEED}"
done
echo "Done. Outputs: ${OUT_ROOT}/seed_*/{metrics.json,diagnostics.json,fig1_prediction_intervals.png,fig2_marginal_pdfs.png,fig3_pearson_correlation.png}"
