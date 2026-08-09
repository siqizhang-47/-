#!/usr/bin/env bash
# Main model only (A7 = full TimeXer condition -> f_phi + g_psi + denoiser),
# three seeds, GPU 2.
set -euo pipefail

cd "$(dirname "$0")/.."
python run.py --ablation A7 --gpu "${GPU:-2}"
