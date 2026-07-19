#!/usr/bin/env bash
# Ablations (spec section 17): gate-only / mask-only / deterministic gate.
# Each variant reuses the zg config with overrides written to configs/ablation/.
set -e
source "$(dirname "$0")/_env.sh"
mkdir -p configs/ablation

make_cfg () {  # name use_gate use_mask bernoulli
python - "$1" "$2" "$3" "$4" <<'EOF'
import sys, yaml
name, use_gate, use_mask, bern = sys.argv[1:5]
cfg = yaml.safe_load(open("configs/zg_nsdiff_low_carbon.yaml"))
cfg["model_name"] = name
cfg["zg"] = {"use_gate": use_gate == "1", "use_mask": use_mask == "1"}
cfg.setdefault("sampling", {})["bernoulli_gate"] = bern == "1"
yaml.safe_dump(cfg, open(f"configs/ablation/{name}.yaml", "w"), allow_unicode=True)
EOF
}

make_cfg zg_gate_only 1 0 1
make_cfg zg_mask_only 0 1 0
make_cfg zg_det_gate  1 1 0

for name in zg_gate_only zg_mask_only zg_det_gate; do
  python -m src.experiments.ZGNsDiffLowCarbon --config configs/ablation/$name.yaml --pretrain --seeds $SEEDS
done
