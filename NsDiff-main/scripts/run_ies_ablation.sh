#!/usr/bin/env bash
# §13 消融实验:依次跑 完整 / w-o天气 / w-o LSNM / w-o UANS / w-o Hurdle,均用物理 GPU 4。
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID

for cfg in ies_oracle ies_wo_weather ies_wo_lsnm ies_wo_uans ies_wo_hurdle; do
  echo "==================== running: $cfg ===================="
  python3 -m ies.run --config "configs/${cfg}.yaml" --gpu 0
done
