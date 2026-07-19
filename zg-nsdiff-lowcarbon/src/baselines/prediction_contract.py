"""Unified prediction file contract (spec section 14).

Every model / seed writes:
    artifacts/predictions/<model>/seed_<k>/manifest.json
    artifacts/predictions/<model>/seed_<k>/predictions_000.npz ...

Shard content:
    samples float32 [n,H,4,S]  raw kW, >= 0
    truth   float32 [n,H,4]
    timestamps int64 [n,H]
    forecast_start_index int64 [n]
    gate_prob float32 [n,H,3]  (optional, ZG only)
"""
import glob
import json
import os

import numpy as np

from src.data.low_carbon_schema import TARGET_NAMES


class PredictionShardWriter:
    def __init__(self, out_dir: str, manifest: dict):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.manifest = dict(manifest)
        self.manifest.setdefault("target_order", TARGET_NAMES)
        self.manifest.setdefault("sample_shape", "[N,H,D,S]")
        self.shard_idx = 0
        self.total = 0

    def write(self, samples, truth, timestamps, forecast_start_index, gate_prob=None):
        samples = np.asarray(samples, dtype=np.float32)
        assert samples.ndim == 4, f"samples must be [n,H,D,S], got {samples.shape}"
        assert np.isfinite(samples).all(), "samples contain NaN/Inf"
        assert (samples >= 0).all(), "raw samples must be non-negative"
        payload = {
            "samples": samples,
            "truth": np.asarray(truth, dtype=np.float32),
            "timestamps": np.asarray(timestamps, dtype=np.int64),
            "forecast_start_index": np.asarray(forecast_start_index, dtype=np.int64),
        }
        if gate_prob is not None:
            payload["gate_prob"] = np.asarray(gate_prob, dtype=np.float32)
        path = os.path.join(self.out_dir, f"predictions_{self.shard_idx:03d}.npz")
        np.savez_compressed(path, **payload)
        self.shard_idx += 1
        self.total += samples.shape[0]

    def close(self):
        self.manifest["num_windows"] = int(self.total)
        self.manifest["num_shards"] = int(self.shard_idx)
        with open(os.path.join(self.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False, default=str)


def shard_paths(pred_dir: str):
    return sorted(glob.glob(os.path.join(pred_dir, "predictions_*.npz")))


def iter_shards(pred_dir: str):
    for p in shard_paths(pred_dir):
        with np.load(p) as z:
            yield {k: z[k] for k in z.files}


def load_manifest(pred_dir: str):
    with open(os.path.join(pred_dir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def assert_contract(pred_dir: str, H: int = 24, num_samples: int = None):
    man = load_manifest(pred_dir)
    n_total = 0
    for shard in iter_shards(pred_dir):
        s = shard["samples"]
        assert s.ndim == 4
        assert s.shape[1] == H and s.shape[2] == len(TARGET_NAMES)
        if num_samples is not None:
            assert s.shape[3] == num_samples
        assert np.isfinite(s).all()
        assert (s >= 0).all()
        assert shard["truth"].shape == s.shape[:3]
        n_total += s.shape[0]
    assert n_total == man["num_windows"]
    return man


def assert_alignment(pred_dirs: list):
    """All models must share identical forecast_start_index / timestamps / truth."""
    ref = None
    for d in pred_dirs:
        fsi, ts, tr = [], [], []
        for shard in iter_shards(d):
            fsi.append(shard["forecast_start_index"])
            ts.append(shard["timestamps"])
            tr.append(shard["truth"])
        cur = (np.concatenate(fsi), np.concatenate(ts), np.concatenate(tr))
        if ref is None:
            ref = cur
            ref_dir = d
        else:
            assert np.array_equal(ref[0], cur[0]), f"forecast_start_index mismatch: {ref_dir} vs {d}"
            assert np.array_equal(ref[1], cur[1]), f"timestamps mismatch: {ref_dir} vs {d}"
            assert np.allclose(ref[2], cur[2]), f"truth mismatch: {ref_dir} vs {d}"
