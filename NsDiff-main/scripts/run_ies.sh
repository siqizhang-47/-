#!/usr/bin/env bash
# NsDiff 四变量联合概率预测(Oracle 天气版)—— 主训练/评测
# 使用物理 GPU 4:CUDA_VISIBLE_DEVICES=4 把 4 号卡映射为进程内的 cuda:0。
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python3 -m ies.run --config configs/ies_oracle.yaml --gpu 0
