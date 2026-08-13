#!/usr/bin/env bash
# Exp 5: window-normalization x adapter 2x2 ablation.
# Requires both caches: data/scenarios/wnorm_on and data/scenarios/wnorm_off.
set -e
cd "$(dirname "$0")/.."

python experiments/run_adapter.py --method frozen   --cache data/scenarios/wnorm_on  --name exp5_wnorm_on_adapter_off
python experiments/run_adapter.py --method proposed --cache data/scenarios/wnorm_on  --name exp5_wnorm_on_adapter_on
python experiments/run_adapter.py --method frozen   --cache data/scenarios/wnorm_off --name exp5_wnorm_off_adapter_off
python experiments/run_adapter.py --method proposed --cache data/scenarios/wnorm_off --name exp5_wnorm_off_adapter_on
