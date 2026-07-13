#!/usr/bin/env bash
# 快速 sanity check(1 epoch、少量 batch、num_samples=5),先跑这个确认不报形状错。
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python3 -m ies.run --config configs/ies_oracle_smoke.yaml --gpu 0
