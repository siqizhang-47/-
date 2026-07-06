"""Missing-mask generators for the IES benchmark.

Mask convention (matching MIRI's ``rectified_impute``):
    M = 1  -> observed
    M = 0  -> missing
"""

from __future__ import annotations

import numpy as np


def make_mcar_mask(X: np.ndarray, missing_rate: float, seed: int) -> np.ndarray:
    """Missing Completely At Random: each entry dropped independently."""
    rng = np.random.default_rng(seed)
    M = (rng.random(X.shape) > missing_rate).astype(np.float32)

    # Avoid all-missing rows (keep at least one observed entry per row).
    all_missing = M.sum(axis=1) == 0
    if all_missing.any():
        keep_col = rng.integers(0, X.shape[1], size=int(all_missing.sum()))
        M[np.where(all_missing)[0], keep_col] = 1.0

    return M


def make_mar_mask(
    X: np.ndarray,
    missing_rate: float,
    cond_idx: list[int],
    target_idx: list[int],
    seed: int,
) -> np.ndarray:
    """Missing At Random.

    Condition variables (weather) remain fully observed.  Each target variable
    (energy) is masked according to a linear score of the condition variables;
    the exact per-target missing rate is enforced by a quantile threshold.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    M = np.ones((n, d), dtype=np.float32)

    Xc = X[:, cond_idx]
    for j in target_idx:
        beta = rng.normal(size=(len(cond_idx),))
        score = Xc @ beta + 0.1 * rng.normal(size=n)
        threshold = np.quantile(score, 1.0 - missing_rate)
        miss = score >= threshold
        M[miss, j] = 0.0

    return M


def make_mnar_mask(X: np.ndarray, missing_rate: float, seed: int) -> np.ndarray:
    """Missing Not At Random.

    Missingness of each variable depends on its own value.  For every variable a
    random direction (high-value-missing or low-value-missing) is drawn, and the
    ``missing_rate`` fraction with the largest score is dropped.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    M = np.ones((n, d), dtype=np.float32)

    for j in range(d):
        direction = rng.choice([-1.0, 1.0])
        score = direction * X[:, j] + 0.05 * rng.normal(size=n)
        threshold = np.quantile(score, 1.0 - missing_rate)
        miss = score >= threshold
        M[miss, j] = 0.0

    return M


def make_test_mask(
    X_test: np.ndarray,
    mechanism: str,
    missing_rate: float,
    seed: int,
    feature_cols: list[str],
    config: dict,
) -> np.ndarray:
    """Dispatch to the requested mechanism and return a test-set mask."""
    mechanism = mechanism.lower()
    if mechanism == "mcar":
        return make_mcar_mask(X_test, missing_rate, seed)
    if mechanism == "mar":
        cond_idx = [feature_cols.index(c) for c in config["missing"]["mar_condition_cols"]]
        target_idx = [feature_cols.index(c) for c in config["missing"]["mar_target_cols"]]
        return make_mar_mask(X_test, missing_rate, cond_idx, target_idx, seed)
    if mechanism == "mnar":
        return make_mnar_mask(X_test, missing_rate, seed)
    raise ValueError(f"Unknown mechanism: {mechanism}")
