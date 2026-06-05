"""Step 1+2: build a continuous hourly array from processed_data.xlsx and
compute normalization statistics.

This reads the `Merged` sheet ONCE and produces a single continuous tape of
hourly readings (no slicing into days here -- step 5 does sliding windows or
aligned natural days on top of this tape).

Outputs (under <artifacts>/):
  dataset.npz     : Yfull, Wfull, CALfull, ERAfull, IRRfull, FILLEDfull, bounds
  norm_stats.json : per-channel mean/std (PV per-era, C/H activated-only)

The 2011-03 earthquake gap is still NaN in this processed_data (PV, E, temp,
wind direction). We impute those holes with (month, hour) climatology so the
continuous tape is numerically usable, but mark them in FILLEDfull so they are
(a) skippable during training and (b) excluded from evaluation (plan C8).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def _load_cfg(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _hour_index_of(dates: pd.Series, ts: str) -> int:
    """Return number of rows with Date <= ts (i.e. the exclusive end index)."""
    return int((dates <= pd.Timestamp(ts)).sum())


def _era_of(dates: pd.Series, era0_end: str, era1_end: str) -> np.ndarray:
    era = np.full(len(dates), 2, dtype=np.int64)
    era[dates <= pd.Timestamp(era1_end)] = 1
    era[dates <= pd.Timestamp(era0_end)] = 0
    return era


def _climatology_impute(values: np.ndarray, month: np.ndarray, hour: np.ndarray,
                        valid_train: np.ndarray) -> np.ndarray:
    """Fill NaNs in `values` with the (month, hour) mean computed over rows that
    are both non-NaN and inside the training period (avoids leakage)."""
    out = values.copy()
    nan_mask = np.isnan(out)
    if not nan_mask.any():
        return out
    # build (month,hour) -> mean table from valid training rows
    key_m = month * 100 + hour
    fit_mask = valid_train & ~np.isnan(values)
    table = {}
    for k in np.unique(key_m[fit_mask]):
        table[int(k)] = float(np.mean(values[fit_mask & (key_m == k)]))
    global_mean = float(np.nanmean(values[valid_train])) if valid_train.any() else float(np.nanmean(values))
    fill = np.array([table.get(int(k), global_mean) for k in key_m[nan_mask]])
    out[nan_mask] = fill
    return out


def build(cfg: dict) -> dict:
    paths = cfg["paths"]
    os.makedirs(paths["artifacts"], exist_ok=True)

    df = pd.read_excel(paths["raw_xlsx"], sheet_name="Merged")
    dates = pd.to_datetime(df["Date"])
    n = len(df)
    print(f"[build] read Merged: {n} rows, {dates.min()} -> {dates.max()}")

    # ---- boundaries (hour indices) ----
    train_end_hour = _hour_index_of(dates, cfg["split"]["train_end"])
    val_end_hour = _hour_index_of(dates, cfg["split"]["val_end"])
    print(f"[build] train_end_hour={train_end_hour} val_end_hour={val_end_hour} test_hours={n - val_end_hour}")

    valid_train = np.arange(n) < train_end_hour

    # ---- raw fields ----
    Ycols = cfg["cols"]["Y"]
    Yraw = df.iloc[:, Ycols].to_numpy(dtype=np.float64)             # [n,5]
    irr = df.iloc[:, cfg["cols"]["irr"]].to_numpy(dtype=np.float64) # complete (0 NaN)
    temp = df.iloc[:, cfg["cols"]["temp"]].to_numpy(dtype=np.float64)
    humid = df.iloc[:, cfg["cols"]["humidity"]].to_numpy(dtype=np.float64)
    wspd = df.iloc[:, cfg["cols"]["windspeed"]].to_numpy(dtype=np.float64)
    wdir = df.iloc[:, cfg["cols"]["winddir"]].to_numpy(dtype=np.float64)

    month = dates.dt.month.to_numpy()
    hour = dates.dt.hour.to_numpy()
    weekday = dates.dt.weekday.to_numpy()

    # ---- FILLED marker = any original NaN among the imputed fields ----
    filled = (np.isnan(Yraw).any(axis=1) | np.isnan(temp) | np.isnan(wdir)).astype(np.int64)
    print(f"[build] FILLED hours={filled.sum()} (= {filled.sum()//24} days)")

    # ---- impute NaNs with (month,hour) climatology from training rows ----
    for j in range(Yraw.shape[1]):
        Yraw[:, j] = _climatology_impute(Yraw[:, j], month, hour, valid_train)
    temp = _climatology_impute(temp, month, hour, valid_train)
    wdir = _climatology_impute(wdir, month, hour, valid_train)
    # humidity/windspeed/irr have no NaN but guard anyway
    humid = _climatology_impute(humid, month, hour, valid_train)
    wspd = _climatology_impute(wspd, month, hour, valid_train)

    # ---- weather feature vector [n,6]: irr,temp,humidity,windspeed,wdir_sin,wdir_cos
    wdir_rad = np.deg2rad(wdir % 360.0)
    Wfull = np.stack([irr, temp, humid, wspd, np.sin(wdir_rad), np.cos(wdir_rad)], axis=1)

    # ---- calendar feature vector [n,8] ----
    hour_ang = 2 * np.pi * hour / 24.0
    month_ang = 2 * np.pi * (month - 1) / 12.0
    wday_ang = 2 * np.pi * weekday / 7.0
    is_weekend = (weekday >= 5).astype(np.float64)
    is_daytime = (irr > 0).astype(np.float64)
    CALfull = np.stack([
        np.sin(hour_ang), np.cos(hour_ang),
        np.sin(month_ang), np.cos(month_ang),
        np.sin(wday_ang), np.cos(wday_ang),
        is_weekend, is_daytime,
    ], axis=1)

    ERAfull = _era_of(dates, cfg["era_bounds"]["era0_end"], cfg["era_bounds"]["era1_end"])
    IRRfull = irr.copy()

    out_path = os.path.join(paths["artifacts"], paths["dataset_npz"])
    np.savez_compressed(
        out_path,
        Yfull=Yraw.astype(np.float32),
        Wfull=Wfull.astype(np.float32),
        CALfull=CALfull.astype(np.float32),
        ERAfull=ERAfull.astype(np.int64),
        IRRfull=IRRfull.astype(np.float32),
        FILLEDfull=filled.astype(np.int64),
        train_end_hour=np.int64(train_end_hour),
        val_end_hour=np.int64(val_end_hour),
        n_hours=np.int64(n),
    )
    print(f"[build] wrote {out_path}")

    # ---- sanity checks (compare against plan's verified numbers) ----
    days = n // 24
    print(f"[check] hours={n} days={days} "
          f"train_days={train_end_hour//24} val_days={(val_end_hour-train_end_hour)//24} "
          f"test_days={(n-val_end_hour)//24}")
    for cnt, label in [((ERAfull == 0).sum(), "era0"), ((ERAfull == 1).sum(), "era1"),
                       ((ERAfull == 2).sum(), "era2")]:
        print(f"[check] {label} days={cnt//24}")
    names = cfg["channels"]
    for j, nm in enumerate(names):
        zr = (Yraw[:, j] == 0).mean() * 100
        print(f"[check] {nm} zero%={zr:.1f}")

    # ---- normalization statistics (step 2) ----
    stats = _compute_norm_stats(cfg, Yraw, ERAfull, IRRfull, valid_train)
    # weather feature stats (per-feature, train rows) for conditioning input scaling
    w_mu = Wfull[valid_train].mean(axis=0)
    w_sd = Wfull[valid_train].std(axis=0) + 1e-6
    stats["W"] = [list(map(float, w_mu)), list(map(float, w_sd))]
    ns_path = os.path.join(paths["artifacts"], paths["norm_stats"])
    with open(ns_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[build] wrote {ns_path}")
    return stats


def _compute_norm_stats(cfg, Yraw, ERAfull, IRRfull, valid_train) -> dict:
    """PV: per-era daytime (irr>0). C/H: activated-only (val>0). E/HW: all hours.
    Only training rows are used."""
    pv, e, c, h, hw = (cfg["pv_idx"], cfg["e_idx"], cfg["c_idx"], cfg["h_idx"], cfg["hw_idx"])
    eps = 1e-6

    def ms(vals):
        return float(np.mean(vals)), float(np.std(vals) + eps)

    stats = {"per_era_pv": {}}
    day = IRRfull > 0
    for era in [0, 1, 2]:
        m = valid_train & (ERAfull == era) & day
        if m.any():
            mu, sd = ms(Yraw[m, pv])
        else:
            mu, sd = 0.0, 1.0
        stats["per_era_pv"][str(era)] = [mu, sd]

    # E: all training hours
    stats["E"] = list(ms(Yraw[valid_train, e]))
    # C / H: activated only
    mc = valid_train & (Yraw[:, c] > 0)
    mh = valid_train & (Yraw[:, h] > 0)
    stats["C"] = list(ms(Yraw[mc, c])) if mc.any() else [0.0, 1.0]
    stats["H"] = list(ms(Yraw[mh, h])) if mh.any() else [0.0, 1.0]
    # HW: all hours
    stats["HW"] = list(ms(Yraw[valid_train, hw]))

    print(f"[check] PV(era0 day) mean/std={stats['per_era_pv']['0'][0]:.2f}/{stats['per_era_pv']['0'][1]:.2f}")
    print(f"[check] E mean/std={stats['E'][0]:.2f}/{stats['E'][1]:.2f}")
    print(f"[check] C act mean/std={stats['C'][0]:.2f}/{stats['C'][1]:.2f}")
    print(f"[check] H act mean/std={stats['H'][0]:.2f}/{stats['H'][1]:.2f}")
    print(f"[check] HW mean/std={stats['HW'][0]:.2f}/{stats['HW'][1]:.2f}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--raw_xlsx", default=None, help="override raw xlsx path")
    ap.add_argument("--artifacts", default=None, help="override artifacts dir")
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    if args.raw_xlsx:
        cfg["paths"]["raw_xlsx"] = args.raw_xlsx
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts
    build(cfg)


if __name__ == "__main__":
    main()
