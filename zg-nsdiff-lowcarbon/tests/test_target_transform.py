import numpy as np
import torch

from src.data.target_transform import TargetTransform


def test_roundtrip(tiny_stats):
    tf = TargetTransform(tiny_stats)
    y = np.array([[500.0, 80.0, 6.0], [600.0, 10.0, 3.0]])
    u = tf.transform(y)
    back = tf.inverse(u)
    assert np.allclose(back, y, rtol=1e-4, atol=1e-3)


def test_inverse_clamps_nonnegative(tiny_stats):
    tf = TargetTransform(tiny_stats)
    u = np.full((2, 3), -100.0)  # extreme negative model-space values
    assert (tf.inverse(u) >= 0).all()


def test_inverse_torch_matches_numpy(tiny_stats):
    tf = TargetTransform(tiny_stats)
    u = np.random.RandomState(0).randn(5, 3).astype(np.float32)
    a = tf.inverse(u)
    b = tf.inverse_torch(torch.from_numpy(u), target_dim=-1).numpy()
    assert np.allclose(a, b, rtol=1e-4, atol=1e-4)


def test_fit_uses_train_only():
    rng = np.random.RandomState(0)
    N = 200
    y = np.abs(rng.randn(N, 3)) * 10 + 1
    obs = np.ones((N, 3), dtype=bool)
    tf1 = TargetTransform.fit(y, obs, train_end=140)
    y2 = y.copy()
    y2[150:, :] *= 7.0  # extreme val/test values must not affect the scaler
    tf2 = TargetTransform.fit(y2, obs, train_end=140)
    assert tf1.stats == tf2.stats
