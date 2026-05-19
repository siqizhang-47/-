"""End-to-end experiment runner.

Usage::

    python -m experiment.run \
        --data_root D:/EWELD_labeled_output \
        --model iTransformer \
        --feature_set load_weather_event \
        --pred_len 96 \
        --epochs 10 \
        --gpu 2

Outputs a JSON results file under ``results/`` with overall / normal / event
/ event-family / high-risk metrics and the configuration used.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

# Allow running as both ``python -m experiment.run`` and ``python experiment/run.py``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.data.features import EVENT_LABELS_USED, LOAD_COL, WEATHER_COLS
from experiment.data.filter import FilterConfig, filter_high_quality_users, time_splits
from experiment.data.loader import load_all_users
from experiment.data.normalize import LoadNormalizer, WeatherNormalizer
from experiment.data.window import WindowConfig, build_pooled_dataset, feature_columns
from experiment.models import MODEL_NAMES, build_model
from experiment.trainer import TrainConfig, fit_and_evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_normalizers(user_frames, user_city, splits) -> tuple[LoadNormalizer, WeatherNormalizer]:
    load_norm = LoadNormalizer()
    weather_norm = WeatherNormalizer()
    city_blocks: Dict[str, list] = {}
    for uid, df in user_frames.items():
        tr_slice, _, _ = splits[uid]
        load_train = df[LOAD_COL].to_numpy(dtype=np.float32)[tr_slice]
        load_norm.fit(uid, load_train)
        weather_train = df[WEATHER_COLS].to_numpy(dtype=np.float32)[tr_slice]
        city_blocks.setdefault(user_city[uid], []).append(weather_train)
    for city, blocks in city_blocks.items():
        weather_norm.fit(city, blocks)
    return load_norm, weather_norm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="Path to EWELD_labeled_output")
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--feature_set", default="load_weather_event",
                   choices=("load", "load_weather", "load_weather_event"))
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--train_stride", type=int, default=4)
    p.add_argument("--eval_stride", type=int, default=1)
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--test_ratio", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=2, help="CUDA device index")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max_users", type=int, default=0, help="0 = no cap, otherwise debug subset")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--tag", default="", help="extra suffix on results filename")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.gpu is not None and args.gpu >= 0:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES','-')}", flush=True)

    print(f"loading users from {args.data_root} ...", flush=True)
    user_frames, user_city = load_all_users(args.data_root)
    print(f"raw users: {len(user_frames)}", flush=True)

    fcfg = FilterConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    kept = filter_high_quality_users(user_frames, fcfg)
    print(f"high-quality users: {len(kept)}", flush=True)
    if args.max_users > 0:
        kept = dict(list(kept.items())[: args.max_users])
        print(f"capped to {len(kept)} users for debug", flush=True)
    if len(kept) == 0:
        raise SystemExit("no users passed the high-quality filter")

    user_idx_map = {uid: i for i, uid in enumerate(sorted(kept.keys()))}
    idx_to_user = {i: uid for uid, i in user_idx_map.items()}

    splits = {uid: time_splits(len(df), fcfg) for uid, df in kept.items()}
    load_norm, weather_norm = fit_normalizers(kept, user_city, splits)

    wcfg = WindowConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_stride=args.train_stride,
        eval_stride=args.eval_stride,
        feature_set=args.feature_set,
    )
    pooled, cols = build_pooled_dataset(
        kept, splits, load_norm, weather_norm, user_city, user_idx_map, wcfg,
    )
    print(
        f"channels={len(cols)}, train={len(pooled['train'])}, "
        f"val={len(pooled['val'])}, test={len(pooled['test'])}",
        flush=True,
    )

    enc_in = len(cols)
    model = build_model(args.model, seq_len=args.seq_len, pred_len=args.pred_len, enc_in=enc_in)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} params={n_params/1e6:.2f}M enc_in={enc_in}", flush=True)

    tcfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        num_workers=args.num_workers,
        device=device,
    )
    result = fit_and_evaluate(model, pooled, tcfg, idx_to_user, load_norm)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    name = f"{args.model}_{args.feature_set}_H{args.pred_len}{tag}.json"
    out_path = out_dir / name

    payload = {
        "args": vars(args),
        "n_users": len(kept),
        "channels": cols,
        **result,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}", flush=True)
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
