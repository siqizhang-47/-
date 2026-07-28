import numpy as np

from src.evaluation.empirical_crps import crps_samples
from src.evaluation.interval_metrics import StreamingIntervalMetrics, interval_bounds
from src.evaluation.point_metrics import StreamingPointMetrics


def test_picp_all_hit_is_one():
    m = StreamingIntervalMetrics(levels=[0.95])
    samples = np.linspace(-10, 10, 100)[None, :].repeat(5, axis=0)
    m.update(samples, np.zeros(5))
    assert m.compute(truth_range=1.0)["picp_95"] == 1.0


def test_picp_all_miss_is_zero():
    m = StreamingIntervalMetrics(levels=[0.95])
    samples = np.linspace(1, 2, 100)[None, :].repeat(5, axis=0)
    m.update(samples, np.full(5, 100.0))
    assert m.compute(truth_range=1.0)["picp_95"] == 0.0


def test_95_interval_uses_0025_0975():
    samples = np.arange(1000, dtype=np.float64)[None, :]
    lo, up = interval_bounds(samples, 0.025, 0.975)
    assert abs(lo[0] - np.quantile(samples[0], 0.025)) < 1e-9
    assert abs(up[0] - np.quantile(samples[0], 0.975)) < 1e-9


def test_picp_accumulation_is_count_based():
    m = StreamingIntervalMetrics(levels=[0.95])
    wide = np.linspace(-100, 100, 50)
    narrow = np.linspace(0.4, 0.6, 50)
    m.update(wide[None, :], np.array([0.0]))
    m.update(np.tile(narrow, (3, 1)), np.array([10.0, 10.0, 10.0]))
    assert abs(m.compute(truth_range=1.0)["picp_95"] - 0.25) < 1e-9


def test_aw_is_average_width():
    m = StreamingIntervalMetrics(levels=[0.95])
    samples = np.linspace(0, 10, 1001)[None, :]  # 2.5%..97.5% width = 9.5
    m.update(samples, np.array([5.0]))
    assert abs(m.compute(truth_range=10.0)["aw_95"] - 9.5) < 0.05


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


def test_mape():
    m = StreamingPointMetrics()
    m.update(np.array([110.0, 90.0]), np.array([100.0, 100.0]))
    assert abs(m.compute()["mape_pos"] - 10.0) < 1e-9
