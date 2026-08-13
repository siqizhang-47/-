"""Diebold-Mariano test with HAC (Newey-West) variance for daily loss series."""
import numpy as np
from scipy import stats


def newey_west_variance(d, lag=7):
    """HAC long-run variance of the mean of d with Bartlett kernel."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    d_center = d - d.mean()
    gamma0 = (d_center ** 2).mean()
    var = gamma0
    for j in range(1, min(lag, n - 1) + 1):
        gamma_j = (d_center[j:] * d_center[:-j]).mean()
        var += 2.0 * (1.0 - j / (lag + 1)) * gamma_j
    return var / n


def dm_test(loss_a, loss_b, lag=7):
    """DM test on two aligned daily loss series. H0: equal predictive accuracy.

    Returns (dm_stat, p_value). Negative stat -> method A better (lower loss).
    """
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    assert loss_a.shape == loss_b.shape
    d = loss_a - loss_b
    if np.allclose(d, 0):
        return 0.0, 1.0
    var = newey_west_variance(d, lag=lag)
    if var <= 0:
        return 0.0, 1.0
    dm_stat = d.mean() / np.sqrt(var)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def dm_matrix(losses_by_method: dict, lag=7):
    """Pairwise p-value (and stat) matrices for {method: daily_loss_series}."""
    methods = list(losses_by_method)
    n = len(methods)
    p_mat = np.ones((n, n))
    stat_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            stat, p = dm_test(losses_by_method[methods[i]], losses_by_method[methods[j]], lag=lag)
            stat_mat[i, j] = stat
            p_mat[i, j] = p
    return methods, stat_mat, p_mat
