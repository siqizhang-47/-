"""Tiny synthetic smoke test that exercises the full data → train → eval path.

Builds a fake EWELD_labeled_output tree under /tmp, runs one epoch on each
model with a tiny model + tiny batches, and prints the resulting metrics.
This is intentionally CPU/GPU agnostic and finishes in well under a minute.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Make ``experiment`` importable when running as a plain script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_user_csv(path: Path, days: int = 400, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = days * 96
    times = pd.date_range("2022-01-01", periods=n, freq="15min")
    slot = np.arange(n) % 96
    base = 5 + 2 * np.sin(2 * np.pi * slot / 96)
    load = base + rng.normal(0, 0.2, size=n)
    temp = 20 + 8 * np.sin(2 * np.pi * np.arange(n) / (96 * 365)) + rng.normal(0, 0.5, n)
    dew = temp - 5 + rng.normal(0, 0.3, n)
    hum = 60 + rng.normal(0, 5, n)
    ws = np.clip(rng.normal(4, 1, n), 0, None)
    wg = ws + rng.normal(0, 0.5, n)
    pres = 1013 + rng.normal(0, 1, n)
    wind = ["NE"] * n
    cond = ["Clear"] * n
    events = {f"{i:02d}Label_x": np.zeros(n, dtype=np.int8) for i in range(1, 21)}
    # sprinkle a handful of category-1 events so the test split has event windows
    for i in range(0, n, 96 * 3):
        events["01Label_x"][i : i + 4] = 1
    for i in range(0, n, 96 * 5):
        events["06Label_x"][i : i + 4] = 1
    for i in range(0, n, 96 * 7):
        events["09Label_x"][i : i + 4] = 1
    df = pd.DataFrame({
        "Time": times,
        "Load": load,
        "User": "U1",
        "Temperature": temp,
        "Dew Point": dew,
        "Humidity": hum,
        "Wind": wind,
        "Wind Speed": ws,
        "Wind Gust": wg,
        "Pressure": pres,
        "Condition": cond,
        **events,
    })
    df.to_csv(path, index=False)


def build_fake_dataset(root: Path) -> None:
    for ct in ("CT1", "CT2"):
        for industry in ("C10 Manufacture of food products",):
            d = root / ct / industry
            d.mkdir(parents=True, exist_ok=True)
            for ui, seed in enumerate(range(2)):
                make_user_csv(d / f"U{ui+1}.csv", days=400, seed=seed)


def run(model_name: str, root: Path):
    cmd = [
        sys.executable,
        "-m",
        "experiment.run",
        "--data_root",
        str(root),
        "--model",
        model_name,
        "--feature_set",
        "load_weather_event",
        "--seq_len",
        "32",
        "--pred_len",
        "16",
        "--train_stride",
        "8",
        "--eval_stride",
        "4",
        "--epochs",
        "1",
        "--batch_size",
        "16",
        "--num_workers",
        "0",
        "--gpu",
        "-1",
        "--tag",
        "smoke",
    ]
    import subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["iTransformer", "TimeMixer", "TimeXer"])
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "EWELD_labeled_output"
        build_fake_dataset(root)
        print(f"built fake dataset under {root}", flush=True)
        for m in args.models:
            print(f"\n========== {m} ==========")
            run(m, root)


if __name__ == "__main__":
    main()
