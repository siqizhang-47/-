#!/usr/bin/env bash
# Full ablation protocol from section 20 of the design document.
# Six models x 3 seeds, all on GPU 2.
set -euo pipefail

cd "$(dirname "$0")/.."

GPU=${GPU:-2}
CONFIG=${CONFIG:-configs/energy_timexer_nsdiff.yaml}
OUT=${OUT:-results}

for AB in A0 A3 A4 A5 A6 A7; do
  echo ""
  echo "================= ablation ${AB} ================="
  python run.py --config "${CONFIG}" --ablation "${AB}" --gpu "${GPU}" --output_dir "${OUT}"
done

python aggregate_ablations.py --results_dir "${OUT}"
