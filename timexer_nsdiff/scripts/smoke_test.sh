#!/usr/bin/env bash
# Two-minute sanity check: 1 epoch per stage, 8 scenarios, 1/400 of the test set.
# Verifies the data fixes, all four training stages, sampling and the metrics.
set -euo pipefail

cd "$(dirname "$0")/.."
python run.py --ablation A7 --gpu "${GPU:-2}" --seeds 1 \
  --epochs_mean 1 --epochs_scale 1 --epochs_joint 1 --epochs_diffusion 1 \
  --num_samples 8 --sample_chunk 8 --test_stride 400 \
  --output_dir results_smoke
