#!/usr/bin/env bash
# Shared environment for all run scripts.
# GPU index defaults to 0; override with:  GPU=1 bash scripts/xxx.sh
export GPU="${GPU:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH=.
SEEDS="${SEEDS:-1}"
echo "[env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  SEEDS=$SEEDS"
