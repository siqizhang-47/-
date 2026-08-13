#!/usr/bin/env bash
# End-to-end smoke test: tiny backbone training + Jan-2020 deployment (M=20)
# + all adapter methods on the mini cache. Run before any full experiment.
set -e
cd "$(dirname "$0")/.."

pytest tests/ -x -q

python scripts/prepare_data.py
python experiments/run_backbone_train.py --extra-config configs/exp/smoke.yaml
python experiments/run_deploy.py --extra-config configs/exp/smoke.yaml --out data/scenarios/smoke
for METHOD in frozen tafas cosa shift_only scale_only proposed independent; do
    python experiments/run_adapter.py --method "$METHOD" \
        --extra-config configs/exp/smoke.yaml \
        --cache data/scenarios/smoke --name "smoke_${METHOD}"
done
echo "SMOKE TEST PASSED"
