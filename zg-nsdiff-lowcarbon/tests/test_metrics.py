import numpy as np
import torch

from src.evaluation.empirical_crps import crps_samples
from src.evaluation.interval_metrics import StreamingIntervalMetrics, interval_bounds
from src.evaluation.point_metrics import StreamingPointMetrics
from src.evaluation.zero_metrics import StreamingZeroMetrics, active_prob_from_samples


def test_picp_all_hit_is_one():
    m = StreamingIntervalMetrics(levels=[0.95])
    samples = np.linspace(-10, 10, 100)[None, :].repeat(5, axis=0)  # wide
    truth = np.zeros(5)
    m.update(samples, truth)
    out = m.compute(truth_range=1.0)
    assert out["picp_95"] == 1.0


def test_picp_all_miss_is_zero():
    m = StreamingIntervalMetrics(levels=[0.95])
    samples = np.linspace(1, 2, 100)[None, :].repeat(5, axis=0)
    truth = np.full(5, 100.0)
    m.update(samples, truth)
    assert m.compute(truth_range=1.0)["picp_95"] == 0.0


def test_95_interval_uses_0025_0975():
    samples = np.arange(1000, dtype=np.float64)[None, :]
    lo, up = interval_bounds(samples, 0.025, 0.975)
    assert abs(lo[0] - np.quantile(samples[0], 0.025)) < 1e-9
    assert abs(up[0] - np.quantile(samples[0], 0.975)) < 1e-9


def test_picp_accumulation_is_count_based():
    # regression for the original bug: batch means must NOT be averaged again
    m = StreamingIntervalMetrics(levels=[0.95])
    wide = np.linspace(-100, 100, 50)
    narrow = np.linspace(0.4, 0.6, 50)
    m.update(wide[None, :], np.array([0.0]))                       # covered
    m.update(np.tile(narrow, (3, 1)), np.array([10.0, 10.0, 10.0]))  # 3 misses
    assert abs(m.compute(truth_range=1.0)["picp_95"] - 0.25) < 1e-9


def test_crps_matches_bruteforce():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(4, 50))
    truth = rng.normal(size=4)
    fast = crps_samples(samples, truth)
    for i in range(4):
        x, y = samples[i], truth[i]
        t1 = np.abs(x - y).mean()
        t2 = np.abs(x[:, None] - x[None, :]).sum() / (2 * len(x) ** 2)
        assert abs(fast[i] - (t1 - t2)) < 1e-8


def test_crps_perfect_forecast_near_zero():
    samples = np.full((2, 100), 5.0)
    truth = np.full(2, 5.0)
    assert np.allclose(crps_samples(samples, truth), 0.0, atol=1e-10)


def test_mape_variants():
    m = StreamingPointMetrics(positive_scale=10.0)
    m.update(np.array([110.0, 0.0]), np.array([100.0, 0.0]))
    out = m.compute()
    assert abs(out["mape_pos"] - 10.0) < 1e-9      # only y>0 coordinate counted
    assert out["mape_eps"] > 0                     # stabilised version is finite


def test_zero_metrics_and_brier():
    m = StreamingZeroMetrics(n_bins=2)
    m.update(np.array([0.9, 0.1, 0.8, 0.2]), np.array([True, False, True, False]))
    out = m.compute()
    assert out["active_accuracy"] == 1.0
    assert abs(out["brier"] - np.mean([0.01, 0.01, 0.04, 0.04])) < 1e-9
    assert out["true_zero_rate"] == 0.5


def test_active_prob_from_samples():
    samples = np.array([[0.0, 0.0, 1.0, 2.0]])  # 2 of 4 positive
    assert active_prob_from_samples(samples)[0] == 0.5
