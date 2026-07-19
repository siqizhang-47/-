import numpy as np
import pytest

from src.baselines.prediction_contract import (
    PredictionShardWriter,
    assert_alignment,
    assert_contract,
    iter_shards,
)


def make_preds(tmp_path, model, n=6, H=24, D=4, S=10, seed=0, shard_size=4):
    rng = np.random.default_rng(seed)
    d = tmp_path / model / "seed_1"
    w = PredictionShardWriter(str(d), {"model": model, "seed": 1})
    rng_truth = np.random.default_rng(123)  # identical truth across models
    truth = np.abs(rng_truth.normal(size=(n, H, D))).astype(np.float32)
    ts = np.arange(n * H).reshape(n, H).astype(np.int64)
    fsi = np.arange(n).astype(np.int64)
    samples = np.abs(rng.normal(size=(n, H, D, S))).astype(np.float32)
    for s in range(0, n, shard_size):
        e = min(s + shard_size, n)
        w.write(samples[s:e], truth[s:e], ts[s:e], fsi[s:e])
    w.close()
    return str(d)


def test_contract_roundtrip(tmp_path):
    d = make_preds(tmp_path, "m1")
    man = assert_contract(d, H=24, num_samples=10)
    assert man["num_windows"] == 6
    total = sum(s["samples"].shape[0] for s in iter_shards(d))
    assert total == 6


def test_negative_samples_rejected(tmp_path):
    w = PredictionShardWriter(str(tmp_path / "bad" / "seed_1"), {"model": "bad", "seed": 1})
    with pytest.raises(AssertionError):
        w.write(-np.ones((1, 24, 4, 5), dtype=np.float32),
                np.zeros((1, 24, 4)), np.zeros((1, 24), dtype=np.int64),
                np.zeros(1, dtype=np.int64))


def test_alignment_check(tmp_path):
    d1 = make_preds(tmp_path, "m1", seed=0)
    d2 = make_preds(tmp_path, "m2", seed=99)  # different samples, same truth/ts
    assert_alignment([d1, d2])


def test_alignment_mismatch_detected(tmp_path):
    d1 = make_preds(tmp_path, "m1", seed=0)
    d3 = tmp_path / "m3" / "seed_1"
    w = PredictionShardWriter(str(d3), {"model": "m3", "seed": 1})
    rng = np.random.default_rng(5)
    w.write(np.abs(rng.normal(size=(6, 24, 4, 10))).astype(np.float32),
            np.abs(rng.normal(size=(6, 24, 4))).astype(np.float32),  # different truth
            np.arange(6 * 24).reshape(6, 24).astype(np.int64),
            np.arange(6).astype(np.int64))
    w.close()
    with pytest.raises(AssertionError):
        assert_alignment([d1, str(d3)])
