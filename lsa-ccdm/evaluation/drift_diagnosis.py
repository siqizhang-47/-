"""Exp 1: long-term drift diagnosis (v4 §14).

Produces per-year location / dispersion / distribution / dependence drift
statistics from the cleaned csv. Outputs csv tables under results/drift/.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wasserstein_distance
from tqdm import tqdm

TARGETS = ["PV", "Electricity", "Cooling", "Heat"]
TRAIN_YEARS = range(2014, 2019)


def yearly_moments(df):
    rows = []
    for year, g in df.groupby(df.index.year):
        for c in TARGETS:
            mu, sd = g[c].mean(), g[c].std()
            rows.append({"year": year, "carrier": c, "mean": mu, "std": sd,
                         "cv": sd / (abs(mu) + 1e-8)})
    return pd.DataFrame(rows)


def deseasonalized_residuals(df):
    """Remove the month-hour climatology fitted on the TRAINING years only."""
    train = df[df.index.year.isin(list(TRAIN_YEARS))]
    clim = train.groupby([train.index.month, train.index.hour])[TARGETS].mean()
    keys = list(zip(df.index.month, df.index.hour))
    base = clim.loc[keys].values
    return pd.DataFrame(df[TARGETS].values - base, index=df.index, columns=TARGETS)


def conditional_residual_mean(df, temp_range=(70, 80), hour_range=(10, 16)):
    """Mean load under controlled weather (reproduces the §2.1 concept-drift check)."""
    mask = (df["Temperature"].between(*temp_range)
            & (df.index.hour >= hour_range[0]) & (df.index.hour <= hour_range[1]))
    sub = df[mask]
    rows = []
    for year, g in sub.groupby(sub.index.year):
        row = {"year": year, "n_hours": len(g)}
        row.update({c: g[c].mean() for c in TARGETS})
        rows.append(row)
    return pd.DataFrame(rows)


def rbf_mmd(x, y, subsample=5000, sigma=None):
    """MMD^2 with an RBF kernel (median heuristic), subsampled for tractability."""
    # uses the legacy global numpy RNG so the single global seed governs it
    if len(x) > subsample:
        x = x[np.random.choice(len(x), subsample, replace=False)]
    if len(y) > subsample:
        y = y[np.random.choice(len(y), subsample, replace=False)]

    def sqdist(a, b):
        return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)

    dxy = sqdist(x, y)
    if sigma is None:
        sigma = np.sqrt(np.median(dxy) / 2 + 1e-12)
    k = lambda d: np.exp(-d / (2 * sigma ** 2))
    m, n = len(x), len(y)
    kxx = (k(sqdist(x, x)).sum() - m) / (m * (m - 1))
    kyy = (k(sqdist(y, y)).sum() - n) / (n * (n - 1))
    kxy = k(dxy).mean()
    return float(kxx + kyy - 2 * kxy)


def distribution_drift(df):
    """Per-year Wasserstein (per carrier, standardized) and joint MMD vs train."""
    train = df[df.index.year.isin(list(TRAIN_YEARS))][TARGETS]
    mu, sd = train.mean(), train.std()
    train_z = ((train - mu) / sd).values
    rows = []
    for year, g in tqdm(df.groupby(df.index.year), desc="distribution drift"):
        gz = ((g[TARGETS] - mu) / sd).values
        row = {"year": year, "mmd": rbf_mmd(train_z, gz)}
        for i, c in enumerate(TARGETS):
            row[f"wasserstein_{c}"] = wasserstein_distance(train_z[:, i], gz[:, i])
        rows.append(row)
    return pd.DataFrame(rows)


def _corr(x, method):
    if method == "spearman":
        x = np.apply_along_axis(rankdata, 0, x)
    return np.corrcoef(x.T)


def dependence_drift(df, n_boot=1000):
    """Per-year residual-space correlation matrices + bootstrap CIs."""
    resid = deseasonalized_residuals(df)
    rows = []
    for year, g in tqdm(resid.groupby(resid.index.year), desc="dependence drift"):
        x = g[TARGETS].values
        for method in ("pearson", "spearman"):
            r = _corr(x, method)
            boots = np.empty((n_boot, len(TARGETS), len(TARGETS)))
            n = len(x)
            for b in range(n_boot):
                idx = np.random.randint(0, n, n)
                boots[b] = _corr(x[idx], method)
            lo = np.percentile(boots, 2.5, axis=0)
            hi = np.percentile(boots, 97.5, axis=0)
            for i in range(len(TARGETS)):
                for j in range(i + 1, len(TARGETS)):
                    rows.append({"year": year, "method": method,
                                 "pair": f"{TARGETS[i]}-{TARGETS[j]}",
                                 "corr": r[i, j], "ci_lo": lo[i, j], "ci_hi": hi[i, j]})
    return pd.DataFrame(rows)


def run_all(csv_path, out_dir, n_boot=1000):
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")

    yearly_moments(df).to_csv(out_dir / "yearly_moments.csv", index=False)
    resid = deseasonalized_residuals(df)
    resid_std = resid.groupby(resid.index.year).std()
    resid_std.to_csv(out_dir / "deseasonalized_residual_std.csv")
    conditional_residual_mean(df).to_csv(out_dir / "conditional_load_mean.csv", index=False)
    distribution_drift(df).to_csv(out_dir / "distribution_drift.csv", index=False)
    dependence_drift(df, n_boot=n_boot).to_csv(out_dir / "dependence_drift.csv", index=False)
    # PV conversion-efficiency decay (PV/GHI on GHI>200 hours)
    mask = df["GHI"] > 200
    pv_eff = (df.loc[mask, "PV"] / df.loc[mask, "GHI"]).groupby(df.index[mask].year).mean()
    pv_eff.rename("pv_ghi_ratio").to_csv(out_dir / "pv_efficiency_decay.csv")
    print(f"Drift diagnosis written to {out_dir}")
