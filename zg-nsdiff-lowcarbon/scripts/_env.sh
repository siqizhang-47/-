#!/usr/bin/env bash
# Shared environment for all run scripts.
# GPU index defaults to 6; override with:  GPU=0 bash scripts/xxx.sh
export GPU="${GPU:-6}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH=.
SEEDS="${SEEDS:-1}"
echo "[env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  SEEDS=$SEEDS"
