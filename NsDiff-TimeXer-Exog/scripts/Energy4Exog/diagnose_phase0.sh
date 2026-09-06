#!/usr/bin/env bash
set -euo pipefail
# Phase 0 (plan section 10/18): diagnose the CURRENT model before changing the mean estimator.
# Trains the current ExoMV-D (V0) and reports direct-mean vs Monte-Carlo-mean MAE/RMSE, CRPS,
# PICP/MPIW 50/80/95, slope / peak / ramp errors and PV nighttime statistics
# (energy4_exog_outputs/phase0/seed_*/diagnostics.json) plus Figure 1 with the direct mean overlaid.
cd "$(dirname "$0")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU:-0}"
export PYTHONPATH=./
source scripts/Energy4Exog/v3_common.sh
OUT_ROOT="${OUT_ROOT:-./energy4_exog_outputs/phase0}"
python3 -u ./src/experiments/NsDiff_energy_exog.py \
  "${COMMON_ARGS[@]}" "${V0_ARGS[@]}" "${NO_SLOPE_ARGS[@]}" "${ABSOLUTE_DIFF_ARGS[@]}" "${NO_PV_ARGS[@]}" \
  --model_type=ExoMV-D-Phase0 --model_display_name="ExoMV-D (current)" \
  --analysis_dir="${OUT_ROOT}" run "${SEED:-1}"
python3 - <<PY
import json; d=json.load(open("${OUT_ROOT}/seed_${SEED:-1}/diagnostics.json"))
print("Direct mean MAE :", d["direct_mean_mae"], " RMSE:", d["direct_mean_rmse"])
print("MC mean MAE     :", d["mc_mean_mae"], " RMSE:", d["mc_mean_rmse"])
for l in ("50","80","95"): print(f"PICP-{l}: {d['picp_'+l]:.4f}  MPIW-{l}: {d['mpiw_'+l]:.4f}")
print(d["phase0_verdict"])
PY
