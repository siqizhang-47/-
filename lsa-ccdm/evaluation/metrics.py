"""All evaluation metrics (v4 §15), operating on scenario ensembles.

Inputs: scen (M, H, C) numpy, y (H, C) numpy, in the ORIGINAL physical units.
Every function returns floats / arrays; ``daily_metrics`` bundles everything
into one flat dict per deployment day.
"""
import numpy as np
from scipy.stats import rankdata

CARRIERS = ["PV", "Electricity", "Cooling", "Heat"]
PICP_LEVELS = (0.80, 0.90, 0.95)
RELIABILITY_LEVELS = np.linspace(0.05, 0.95, 10)


# ---------------- point metrics ----------------
def point_metrics(scen, y):
    mu = scen.mean(axis=0)
    return {
        "mae": float(np.abs(mu - y).mean()),
        "rmse": float(np.sqrt(((mu - y) ** 2).mean())),
        "bias": float((mu - y).mean()),
    }


# ---------------- marginal probabilistic metrics ----------------
def ensemble_crps(scen, y):
    """Fair (unbiased) sample CRPS per (h, c): (H, C).

    Pair term uses 1/(2M(M-1)) instead of the naive 1/(2M^2).
    """
    M = scen.shape[0]
    t1 = np.abs(scen - y[None]).mean(axis=0)
    t2 = np.abs(scen[None] - scen[:, None]).mean(axis=(0, 1))
    fair = M / (M - 1) if M > 1 else 1.0
    return t1 - 0.5 * fair * t2


def crps_sum(scen, y):
    """CRPS of the cross-carrier sum (joint effect)."""
    return float(ensemble_crps(scen.sum(axis=2, keepdims=True), y.sum(axis=1, keepdims=True)).mean())


# ---------------- calibration / sharpness ----------------
def interval_bounds(scen, level):
    alpha = 1.0 - level
    lo = np.quantile(scen, alpha / 2, axis=0)
    hi = np.quantile(scen, 1 - alpha / 2, axis=0)
    return lo, hi


def picp(scen, y, level):
    lo, hi = interval_bounds(scen, level)
    return float(((y >= lo) & (y <= hi)).mean())


def mpiw(scen, level):
    lo, hi = interval_bounds(scen, level)
    return float((hi - lo).mean())


def pinaw(scen, y, level):
    lo, hi = interval_bounds(scen, level)
    rng = y.max(axis=0) - y.min(axis=0) + 1e-8
    return float(((hi - lo) / rng[None, :]).mean())


def winkler(scen, y, level=0.90):
    alpha = 1.0 - level
    lo, hi = interval_bounds(scen, level)
    width = hi - lo
    below = (y < lo)
    above = (y > hi)
    score = width + below * (2.0 / alpha) * (lo - y) + above * (2.0 / alpha) * (y - hi)
    return float(score.mean())


def reliability_points(scen, y, levels=RELIABILITY_LEVELS):
    """Nominal central-interval coverage vs empirical coverage (for reliability diagrams)."""
    return {f"cov@{lvl:.2f}": picp(scen, y, lvl) for lvl in levels}


def pit_values(scen, y):
    """Probability integral transform values, flattened over (h, c)."""
    M = scen.shape[0]
    below = (scen < y[None]).sum(axis=0)
    equal = (scen == y[None]).sum(axis=0)
    u = (below + np.random.uniform(size=equal.shape) * (equal + 1)) / (M + 1)
    return u.reshape(-1)


# ---------------- joint metrics ----------------
def energy_score(scen, y):
    """ES = mean ||X_m - y|| - 0.5 mean ||X_m - X_m'||, vectors over (H*C)."""
    flat = scen.reshape(scen.shape[0], -1)
    yv = y.reshape(-1)
    t1 = np.linalg.norm(flat - yv[None], axis=1).mean()
    diff = flat[None] - flat[:, None]
    t2 = np.linalg.norm(diff, axis=2).mean()
    return float(t1 - 0.5 * t2)


def variogram_score(scen, y, p=0.5):
    """Variogram score across the C carriers, averaged over horizon steps."""
    M, H, C = scen.shape
    score = 0.0
    for h in range(H):
        yd = np.abs(y[h][:, None] - y[h][None, :]) ** p          # (C, C)
        sd = (np.abs(scen[:, h, :, None] - scen[:, h, None, :]) ** p).mean(axis=0)
        score += ((yd - sd) ** 2).sum()
    return float(score / H)


# ---------------- dependence ----------------
def scenario_corr(scen, method="pearson"):
    """Cross-carrier correlation of the generated ensemble.

    For each horizon step, correlate carriers across the M scenarios, then
    average the C x C matrices over the horizon.
    """
    M, H, C = scen.shape
    mats = []
    for h in range(H):
        x = scen[:, h, :]
        if method == "spearman":
            x = np.apply_along_axis(rankdata, 0, x)
        sd = x.std(axis=0)
        ok = sd > 1e-10
        cm = np.eye(C)
        if ok.sum() >= 2:
            sub = np.corrcoef(x[:, ok].T)
            cm[np.ix_(ok, ok)] = sub
        mats.append(cm)
    return np.mean(mats, axis=0)


def residual_corr(residuals, method="pearson"):
    """Correlation of true residuals pooled over a rolling window.

    residuals: (N, C) array (days*hours flattened).
    """
    x = residuals
    if method == "spearman":
        x = np.apply_along_axis(rankdata, 0, x)
    return np.corrcoef(x.T)


def cme(r_real, r_gen):
    """Correlation matrix error: Frobenius norm of the difference."""
    return float(np.linalg.norm(r_real - r_gen, ord="fro"))


# ---------------- naive baselines (Phase 1 acceptance) ----------------
def persistence_forecast(history, H=24):
    """Repeat the last H hours of history: (H, C)."""
    return history[-H:].copy()


def climatology_forecast(clim_table, month, hours):
    """Month-hour climatological mean forecast. clim_table[(month, hour)] -> (C,)."""
    return np.stack([clim_table[(month, h)] for h in hours])


def build_climatology(df, target_cols):
    """Month-hour mean table from a training dataframe with a datetime index."""
    table = {}
    grouped = df.groupby([df.index.month, df.index.hour])[target_cols].mean()
    for (m, h), row in grouped.iterrows():
        table[(m, h)] = row.values
    return table


# ---------------- aggregation ----------------
def daily_metrics(scen, y, prefix=""):
    """Full flat metric dict for one deployment day."""
    out = {}
    out.update({f"{prefix}{k}": v for k, v in point_metrics(scen, y).items()})
    crps_hc = ensemble_crps(scen, y)
    out[f"{prefix}crps"] = float(crps_hc.mean())
    for c, name in enumerate(CARRIERS):
        out[f"{prefix}crps_{name}"] = float(crps_hc[:, c].mean())
        out[f"{prefix}mae_{name}"] = float(np.abs(scen.mean(0)[:, c] - y[:, c]).mean())
        out[f"{prefix}bias_{name}"] = float((scen.mean(0)[:, c] - y[:, c]).mean())
    out[f"{prefix}crps_sum"] = crps_sum(scen, y)
    for lvl in PICP_LEVELS:
        tag = int(lvl * 100)
        out[f"{prefix}picp{tag}"] = picp(scen, y, lvl)
        out[f"{prefix}ce{tag}"] = abs(out[f"{prefix}picp{tag}"] - lvl)
        out[f"{prefix}mpiw{tag}"] = mpiw(scen, lvl)
    out[f"{prefix}pinaw90"] = pinaw(scen, y, 0.90)
    out[f"{prefix}winkler90"] = winkler(scen, y, 0.90)
    out[f"{prefix}energy_score"] = energy_score(scen, y)
    out[f"{prefix}variogram_score"] = variogram_score(scen, y)
    out.update({f"{prefix}{k}": v for k, v in reliability_points(scen, y).items()})
    return out
