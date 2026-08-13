"""Phase 5 acceptance: metric correctness checks."""
import numpy as np
import pytest

from evaluation import metrics as M
from evaluation.dm_test import dm_test


def test_crps_matches_properscoring():
    ps = pytest.importorskip("properscoring")
    scen = np.random.randn(80, 24, 4)
    y = np.random.randn(24, 4)
    ours = M.ensemble_crps(scen, y)
    ref = np.empty((24, 4))
    for h in range(24):
        for c in range(4):
            ref[h, c] = ps.crps_ensemble(y[h, c], scen[:, h, c])
    assert np.abs(ours - ref).max() < 1e-6


def test_picp_on_standard_normal():
    scen = np.random.randn(2000, 24, 4)
    y = np.random.randn(24 * 40, 4).reshape(-1, 4)
    # aggregate coverage over many truth draws
    covs = [M.picp(scen, y[i * 24:(i + 1) * 24], 0.90) for i in range(40)]
    assert abs(np.mean(covs) - 0.90) < 0.02


def test_energy_score_zero_for_perfect_point_mass():
    y = np.random.randn(24, 4)
    scen = np.repeat(y[None], 50, axis=0)
    assert M.energy_score(scen, y) < 1e-10
    # and positive for an informative but dispersed ensemble
    scen2 = y[None] + np.random.randn(50, 24, 4)
    assert M.energy_score(scen2, y) > 0


def test_variogram_score_zero_for_perfect():
    y = np.random.randn(24, 4)
    scen = np.repeat(y[None], 50, axis=0)
    assert M.variogram_score(scen, y) < 1e-10


def test_winkler_penalizes_misses():
    y = np.zeros((24, 4))
    inside = np.random.randn(500, 24, 4)
    outside = inside + 10.0  # interval no longer covers 0
    assert M.winkler(outside, y) > M.winkler(inside, y)


def test_correlation_helpers():
    L = np.linalg.cholesky(np.array([[1, .7, 0, 0], [.7, 1, 0, 0],
                                     [0, 0, 1, -.4], [0, 0, -.4, 1.0]]))
    scen = np.random.randn(4000, 24, 4) @ L.T
    r = M.scenario_corr(scen, "pearson")
    assert abs(r[0, 1] - 0.7) < 0.05
    assert abs(r[2, 3] + 0.4) < 0.05
    assert M.cme(r, r) == 0.0


def test_dm_self_comparison():
    loss = np.random.rand(300) + 1
    stat, p = dm_test(loss, loss.copy())
    assert stat == 0.0 and p == 1.0


def test_dm_detects_difference():
    a = np.random.rand(400)
    b = a + 0.5
    stat, p = dm_test(a, b)
    assert stat < 0 and p < 1e-4
