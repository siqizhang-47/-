"""Imputation methods for the IES benchmark.

Every ``impute_*`` function takes ``(X_init, M, ...)`` where ``X_init`` already
has the missing entries filled with an initial guess and ``M`` is the mask
(1 = observed, 0 = missing).  Classical baselines re-insert NaNs at missing
positions before calling their sklearn / hyperimpute backend; MIRI consumes the
initialized matrix directly as its rectified-flow starting point.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Missing-value initialization
# --------------------------------------------------------------------------- #
def initialize_missing(
    X_std: np.ndarray,
    M: np.ndarray,
    seed: int,
    strategy: str = "train_mean_noise",
    noise_std: float = 0.01,
) -> np.ndarray:
    """Fill missing entries of a standardized matrix with an initial guess.

    After train-standardization the (training) mean is approximately zero, so the
    mean-fill strategies simply fill with 0 (+ optional small Gaussian noise).
    """
    rng = np.random.default_rng(seed)
    X0 = X_std.copy()
    rows, cols = np.where(M == 0)

    if strategy == "zero":
        X0[rows, cols] = 0.0

    elif strategy == "train_mean_noise":
        X0[rows, cols] = 0.0 + noise_std * rng.normal(size=len(rows))

    elif strategy == "gaussian":
        X0[rows, cols] = rng.normal(size=len(rows))

    else:
        raise ValueError(strategy)

    return X0.astype(np.float32)


# --------------------------------------------------------------------------- #
# Classical baselines
# --------------------------------------------------------------------------- #
def impute_mean(X_init, M):
    """Column-mean imputation (mean over observed entries per column)."""
    X = X_init.copy()
    for j in range(X.shape[1]):
        obs = M[:, j] == 1
        fill = X[obs, j].mean() if obs.any() else 0.0
        X[M[:, j] == 0, j] = fill
    return X.astype(np.float32)


def impute_knn(X_init, M, n_neighbors=5):
    from sklearn.impute import KNNImputer
    X_nan = X_init.copy()
    X_nan[M == 0] = np.nan
    return KNNImputer(n_neighbors=n_neighbors).fit_transform(X_nan).astype(np.float32)


def impute_mice(X_init, M, seed=0):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    X_nan = X_init.copy()
    X_nan[M == 0] = np.nan
    imp = IterativeImputer(max_iter=100, random_state=seed, sample_posterior=False)
    return imp.fit_transform(X_nan).astype(np.float32)


def impute_hyper_plugin(X_init, M, plugin_name: str):
    """Impute via a HyperImpute plugin (missforest / gain / hyperimpute)."""
    X_nan = X_init.copy()
    X_nan[M == 0] = np.nan

    from hyperimpute.plugins.imputers import Imputers
    imputer = Imputers().get(plugin_name)
    df = imputer.fit_transform(X_nan.astype(np.float64))

    if hasattr(df, "to_numpy"):
        return df.to_numpy().astype(np.float32)
    return np.asarray(df, dtype=np.float32)


# --------------------------------------------------------------------------- #
# MIRI
# --------------------------------------------------------------------------- #
def impute_miri(
    X_init,
    M,
    X_true,
    max_rounds,
    batch_size,
    max_epochs,
    ode_steps,
    lr,
    hidden_dims,
    estimate_mi,
    checkpoint_path=None,
):
    """Run the MIRI rectified-flow imputer with the IES-specific MLP."""
    import torch
    from src.imputer import rectified_impute
    from experiments.ies.models import MLPIES

    hidden = tuple(hidden_dims)

    class ModelFactory(MLPIES):
        def __init__(self, d):
            super().__init__(d=d, hidden_dims=hidden)

    X0_t = torch.tensor(X_init, dtype=torch.float32)
    M_t = torch.tensor(M, dtype=torch.float32)
    Xstar_t = torch.tensor(X_true, dtype=torch.float32)

    X_imp_t, mmd_list, mi_list, _ = rectified_impute(
        X0_t,
        M_t,
        Xstar_t,
        ModelFactory,
        max_rounds=max_rounds,
        clamp=False,
        verbose=False,
        batchsize=batch_size,
        maxepochs=max_epochs,
        odesteps=ode_steps,
        lr=lr,
        estimate_mi=estimate_mi,
        checkpoint_path=checkpoint_path,
    )

    X_imp = X_imp_t.detach().cpu().numpy().astype(np.float32)
    # rectified_impute returns torch tensors in mmd_list; cast to plain floats.
    mmd_list = [float(x) for x in mmd_list]
    mi_list = [float(x) for x in mi_list]
    return X_imp, mmd_list, mi_list


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def run_selected_method(method, X_init, M, X_true, config, seed):
    """Run one imputation method and return (X_imp, extra_info dict).

    Raises on failure; the caller records ``status="failed"``.
    """
    method = method.lower()
    extra = {}

    if method == "mean":
        X_imp = impute_mean(X_init, M)
    elif method == "knn":
        X_imp = impute_knn(X_init, M)
    elif method == "mice":
        X_imp = impute_mice(X_init, M, seed=seed)
    elif method in ("missforest", "gain", "hyperimpute"):
        X_imp = impute_hyper_plugin(X_init, M, plugin_name=method)
    elif method == "miri":
        mcfg = config["miri"]
        ckpt = None
        if config.get("output", {}).get("save_checkpoints", False):
            ckpt = config.get("_checkpoint_path")
        X_imp, mmd_list, mi_list = impute_miri(
            X_init,
            M,
            X_true,
            max_rounds=mcfg["max_rounds"],
            batch_size=mcfg["batch_size"],
            max_epochs=mcfg["max_epochs"],
            ode_steps=mcfg["ode_steps"],
            lr=mcfg["lr"],
            hidden_dims=mcfg["hidden_dims"],
            estimate_mi=mcfg["estimate_mi"],
            checkpoint_path=ckpt,
        )
        extra["mmd_list"] = mmd_list
        extra["mi_list"] = mi_list
    else:
        raise ValueError(f"Unknown method: {method}")

    return X_imp.astype(np.float32), extra
