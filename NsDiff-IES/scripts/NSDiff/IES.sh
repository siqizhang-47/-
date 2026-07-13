#!/usr/bin/env bash
# Train NsDiff on the merged IES dataset and produce Fig.3/4/5 + metrics.
# Runs on GPU 4 (cuda:4). No pretraining required: f(x) and g(x) are trained
# jointly with the diffusion model.
set -e
export PYTHONPATH=./

CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/NsDiff_IES.py \
    --dataset_type=IES \
    --device="cuda:4" \
    --batch_size=32 \
    --windows=168 \
    --horizon=1 \
    --pred_len=24 \
    --epochs=30 \
    --patience=8 \
    --lr=0.001 \
    --num_worker=8 \
    train_eval --seed=1
