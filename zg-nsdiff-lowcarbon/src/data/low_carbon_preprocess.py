"""One-shot preprocessing: HEEW Excel -> numpy artifacts.

Usage:
    python -m src.data.low_carbon_preprocess --config configs/heew_common.yaml
"""
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data import low_carbon_schema as S
from src.data.target_transform import TargetTransform, WeatherTransform
from src.data.window_index import build_split_indices
from src.utils.config import load_yaml, save_json, sha256_file


def build_calendar(timestamps: pd.DatetimeIndex) -> np.ndarray:
    hour = timestamps.hour.values.astype(np.float64)
    dow = timestamps.dayofweek.values.astype(np.float64)
    doy = timestamps.dayofyear.values.astype(np.float64)
    cal = np.stack(
        [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            np.sin(2 * np.pi * doy / 366.0),
            np.cos(2 * np.pi * doy / 366.0),
            (dow >= 5).astype(np.float64),
        ],
        axis=1,
    )
    return cal.astype(np.float32)


def run(config_path: str, output_root: str = "artifacts"):
    cfg = load_yaml(config_path)
    dcfg = cfg["data"]
    xlsx = dcfg["input_xlsx"]
    out_dir = os.path.join(output_root, "data", "heew")
    os.makedirs(out_dir, exist_ok=True)

    steps = tqdm(total=7, desc="preprocess", ncols=100)

    # 1. read excel ------------------------------------------------------
    steps.set_postfix_str("reading excel")
    df = pd.read_excel(xlsx, sheet_name=dcfg.get("sheet_name", S.SHEET_NAME))
    steps.update(1)

    # 2. schema audit ----------------------------------------------------
    steps.set_postfix_str("schema audit")
    for col in S.TIME_COLUMNS + S.TARGET_COLUMNS + S.WEATHER_COLUMNS:
        assert col in df.columns, f"missing column: {col}"
    ts = pd.DatetimeIndex(pd.to_datetime(
        df[S.TIME_COLUMNS].rename(columns=str.lower)))
    epoch_s = ((ts - pd.Timestamp(0)) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
    diffs = np.diff(epoch_s)
    assert (diffs == S.EXPECTED_FREQ_SECONDS).all(), "timestamps are not hourly-continuous"
    n = len(df)
    steps.update(1)

    # 3. raw arrays ------------------------------------------------------
    steps.set_postfix_str("building arrays")
    target_raw = df[S.TARGET_COLUMNS].to_numpy(dtype=np.float64)
    weather_raw = df[S.WEATHER_COLUMNS].to_numpy(dtype=np.float64)
    observed = np.isfinite(target_raw)
    weather_observed = np.isfinite(weather_raw)
    calendar = build_calendar(ts)
    timestamps = epoch_s
    steps.update(1)

    # 4. splits + windows -------------------------------------------------
    steps.set_postfix_str("split + window indices")
    L = int(dcfg.get("context_length", S.CONTEXT_LENGTH))
    H = int(dcfg.get("prediction_length", S.PREDICTION_LENGTH))
    idx = build_split_indices(
        n,
        float(dcfg.get("train_ratio", S.TRAIN_RATIO)),
        float(dcfg.get("val_ratio", S.VAL_RATIO)),
        L,
        H,
        observed.all(axis=1),
        weather_observed.all(axis=1),
    )
    train_end, val_end = idx["train_end"], idx["val_end"]
    steps.update(1)

    # 5. fit transforms on train only -------------------------------------
    steps.set_postfix_str("fitting transforms (train only)")
    ttf = TargetTransform.fit(target_raw, observed, train_end)
    wtf = WeatherTransform.fit(weather_raw, weather_observed, train_end)
    stats = dict(ttf.stats)
    stats.update(wtf.to_stats())
    stats["context_length"] = L
    stats["prediction_length"] = H
    stats["train_end"] = int(train_end)
    stats["val_end"] = int(val_end)
    stats["n_rows"] = int(n)
    steps.update(1)

    # 6. save arrays -------------------------------------------------------
    steps.set_postfix_str("saving arrays")
    np.save(os.path.join(out_dir, "target_raw.npy"), target_raw.astype(np.float32))
    np.save(os.path.join(out_dir, "weather_raw.npy"), weather_raw.astype(np.float32))
    np.save(os.path.join(out_dir, "calendar.npy"), calendar)
    np.save(os.path.join(out_dir, "timestamps.npy"), timestamps)
    np.save(os.path.join(out_dir, "observed_mask.npy"), observed)
    np.save(os.path.join(out_dir, "weather_observed_mask.npy"), weather_observed)
    np.savez(
        os.path.join(out_dir, "split_indices.npz"),
        train_starts=idx["train_starts"],
        val_starts=idx["val_starts"],
        test_starts=idx["test_starts"],
        train_end=train_end,
        val_end=val_end,
        context_length=L,
        prediction_length=H,
    )
    save_json(stats, os.path.join(out_dir, "preprocess_stats.json"))
    steps.update(1)

    # 7. audit + manifests --------------------------------------------------
    steps.set_postfix_str("audit + manifests")
    audit = {
        "n_rows": int(n),
        "time_start": str(ts[0]),
        "time_end": str(ts[-1]),
        "missing_per_target": {
            name: int((~observed[:, d]).sum()) for d, name in enumerate(S.TARGET_NAMES)
        },
        "windows": {
            split: {
                "before_missing_filter": int(len(idx[f"{split}_starts_all"])),
                "after_missing_filter": int(len(idx[f"{split}_starts"])),
            }
            for split in ["train", "val", "test"]
        },
        "future_weather_mode": dcfg.get("future_weather_mode", S.FUTURE_WEATHER_MODE),
    }
    save_json(audit, os.path.join(out_dir, "data_audit.json"))
    save_json(
        {
            "train": [0, int(train_end)],
            "val": [int(train_end), int(val_end)],
            "test": [int(val_end), int(n)],
            "context_length": L,
            "prediction_length": H,
            "windows": audit["windows"],
        },
        os.path.join(out_dir, "split_manifest.json"),
    )
    repro_dir = os.path.join(output_root, "reproducibility")
    save_json(
        {"input_xlsx": os.path.abspath(xlsx), "input_sha256": sha256_file(xlsx)},
        os.path.join(repro_dir, "source_manifest.json"),
    )
    steps.update(1)
    steps.close()

    print("windows:", audit["windows"])
    print(f"artifacts written to {out_dir}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_root", default="artifacts")
    args = ap.parse_args()
    run(args.config, args.output_root)
