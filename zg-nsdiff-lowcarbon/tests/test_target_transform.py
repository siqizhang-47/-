import numpy as np
import torch

from src.data.target_transform import TargetTransform


def make_transform(tiny_stats):
    return TargetTransform(tiny_stats)


def test_roundtrip_positive(tiny_stats):
    tf = make_transform(tiny_stats)
    y = np.array([[500.0, 80.0, 60.0, 40.0], [600.0, 10.0, 5.0, 3.0]])
    u = tf.transform(y)
    back = tf.inverse(u)
    assert np.allclose(back, y, rtol=1e-4, atol=1e-3)


def test_positive_inverse_zero_is_zero(tiny_stats):
    tf = make_transform(tiny_stats)
    for d in [1, 2, 3]:
        assert tf.positive_inverse(0.0, d) == 0.0
        assert tf.positive_inverse(torch.zeros(3), d).eq(0).all()


def test_nan_passthrough(tiny_stats):
    tf = make_transform(tiny_stats)
    y = np.array([[np.nan, 1.0, 2.0, 3.0]])
    u = tf.transform(y)
    assert np.isnan(u[0, 0])
    assert np.isfinite(u[0, 1:]).all()


def test_inverse_torch_matches_numpy(tiny_stats):
    tf = make_transform(tiny_stats)
    u = np.random.RandomState(0).randn(5, 4).astype(np.float32)
    a = tf.inverse(u)
    b = tf.inverse_torch(torch.from_numpy(u), target_dim=-1).numpy()
    assert np.allclose(a, b, rtol=1e-4, atol=1e-4)


def test_fit_uses_train_only():
    rng = np.random.RandomState(0)
    N = 200
    y = np.abs(rng.randn(N, 4)) * 10 + 1
    y[150:, :] *= 100  # extreme val/test values must not affect the scaler
    obs = np.ones((N, 4), dtype=bool)
    tf1 = TargetTransform.fit(y, obs, train_end=140)
    y2 = y.copy()
    y2[150:, :] *= 7.0
    tf2 = TargetTransform.fit(y2, obs, train_end=140)
    assert tf1.stats == tf2.stats
