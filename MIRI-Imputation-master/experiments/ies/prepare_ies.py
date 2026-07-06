"""Data loading, splitting and standardization for the IES imputation benchmark.

The pipeline:
    1. read the ``Merged`` sheet of ``processed_data.xlsx``;
    2. parse / sort by ``Date``;
    3. drop the 2011-03 real missing segment (a real sensor outage);
    4. keep only rows whose 8 feature values are all non-null;
    5. build a stratified train/val/test split by (month, hour);
    6. standardize using training statistics only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_ies_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean the IES data.

    Returns:
        df_meta: contains Date, month, hour, original index.
        df_x:    feature dataframe with the 8 IES variables only.
    """
    data_cfg = config["data"]
    df = pd.read_excel(data_cfg["excel_path"], sheet_name=data_cfg["sheet_name"])

    date_col = data_cfg["date_col"]
    feature_cols = data_cfg["feature_cols"]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    drop_start = pd.to_datetime(data_cfg["drop_start"])
    drop_end = pd.to_datetime(data_cfg["drop_end"])
    keep = ~((df[date_col] >= drop_start) & (df[date_col] <= drop_end))
    df = df.loc[keep].reset_index(drop=True)

    df_x = df[feature_cols].copy()
    valid = df_x.notna().all(axis=1)
    df = df.loc[valid].reset_index(drop=True)
    df_x = df_x.loc[valid].reset_index(drop=True)

    df_meta = pd.DataFrame({
        "Date": df[date_col],
        "month": df[date_col].dt.month,
        "hour": df[date_col].dt.hour,
        "orig_index": df.index,
    })

    return df_meta, df_x


def stratified_month_hour_split(
    df_meta: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratify by a month-hour label using two successive train_test_splits.

        label = month + "_" + hour

    Some (month, hour) strata may contain few samples; if sklearn refuses to
    stratify (a class with < 2 members in a split), we fall back to an
    unstratified random split so the pipeline never crashes.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    idx = np.arange(len(df_meta))
    labels = df_meta["month"].astype(str) + "_" + df_meta["hour"].astype(str)

    def _split(indices, train_size, strat):
        try:
            return train_test_split(
                indices,
                train_size=train_size,
                random_state=seed,
                stratify=strat,
            )
        except ValueError:
            return train_test_split(
                indices,
                train_size=train_size,
                random_state=seed,
                stratify=None,
            )

    train_idx, temp_idx = _split(idx, train_ratio, labels)

    temp_labels = labels.iloc[temp_idx]
    val_fraction_in_temp = val_ratio / (val_ratio + test_ratio)

    val_idx, test_idx = _split(temp_idx, val_fraction_in_temp, temp_labels)

    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def standardize_by_train(
    X: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize columns using training rows only.

    Returns:
        X_std, mean, std  (all float32)
    """
    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    X_std = (X - mean) / std
    return X_std.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)
