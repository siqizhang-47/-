#!/usr/bin/env bash
# TimeXer-NsDiff: train on GPU 4, then evaluate (one-line metrics + 2 figures).
set -e
cd "$(dirname "$0")"
export PYTHONPATH=./

CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 main.py train_eval \
    --device=cuda:4 \
    --batch_size=64 \
    --epochs=40 \
    --lr=1e-3 \
    --patience=8 \
    --num_workers=4 \
    --n_samples=100
