#!/usr/bin/env bash
# Ablation E: context window length K, and F: adaptation frequency.
set -e
cd "$(dirname "$0")/.."

for K in 1 3 7 14 30; do
    python experiments/run_adapter.py --method proposed --override K=$K --name "abl_E_K${K}"
done

for N in 1 3 7 14 30; do
    python experiments/run_adapter.py --method proposed --override adapt_every=$N --name "abl_F_every${N}"
done
