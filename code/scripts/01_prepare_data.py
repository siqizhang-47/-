"""
A1. Data loading and stage (Regime) partitioning for PS-BCD v4.1.

Produces:
    data/processed/train.npz
    data/processed/val.npz
    data/processed/test.npz
    data/processed/split_report.json

Each .npz contains:
    Y      : (N, 24, 14)  float32  output variables
    C      : (N, 24, 10)  float32  condition variables
    regime : (N,)         int8     1=A, 2=B, 3=C
    season : (N,)         int8     0=summer, 1=winter, 2=shoulder
    dates  : (N,)         datetime64[D]
    Y_names: list[str]
    C_names: list[str]

Y order  (14): E, C, H, W, P_PV, g_GE, P_GE, g_FC, P_FC, g_AC1, g_AC2, g_AC3, g_B, P_grid
C order  (10): T_out, I, RH, v, h, d, w, s, a_GE, a_FC
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path("/workspace/code")
SRC_XLSX = ROOT / "processed_data.xlsx"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Column mapping (raw -> internal)
# ----------------------------------------------------------------------------
COL_E = "[Electricity] Electricity load (kW)"
COL_C = "[Electricity] Cooling load (kW)"
COL_H = "[Electricity] Heating load (kW)"
COL_W = "[Electricity] Hot water load (kW)"
COL_GB = "[Gas] Boiler (m3)"
COL_GFC = "[Gas] Fuel cell (m3)"
COL_GGE = "[Gas] Gas engine (m3)"
COL_GAC1 = "[Gas] Absorption chiller 1 (m3)"
COL_GAC2 = "[Gas] Absorption chiller 2 (m3)"
COL_GAC3 = "[Gas] Absorption chiller 3 (m3)"
COL_PV = "[Power] Solar energy generation (kW)"
COL_GRID = "[Power] Electricity imported from grid (kW)"
COL_PGE = "[Power] Gas engine (kW)"
COL_PFC = "[Power] Fuel cell (kW)"
COL_I = "[Weather] Horizontal solar irradition (W)"
COL_T = "[Weather] Outdoor air temperature (℃)"
COL_RH = "[Weather] Outdoor air humidity (%)"
COL_WS = "[Weather] Wind speed (m/s)"

Y_COLS = [
    COL_E, COL_C, COL_H, COL_W,
    COL_PV,
    COL_GGE, COL_PGE,
    COL_GFC, COL_PFC,
    COL_GAC1, COL_GAC2, COL_GAC3,
    COL_GB,
    COL_GRID,
]
Y_NAMES = ["E", "C", "H", "W", "P_PV",
           "g_GE", "P_GE", "g_FC", "P_FC",
           "g_AC1", "g_AC2", "g_AC3", "g_B", "P_grid"]

C_NAMES = ["T_out", "I", "RH", "v", "h", "d", "w", "s", "a_GE", "a_FC"]

# ----------------------------------------------------------------------------
# Regime definition (Part II Table 1.1)
# ----------------------------------------------------------------------------
def regime_of(ts: pd.Timestamp) -> int:
    """1 = A (2001-06..2011-12), 2 = B (2012-01..2016-12), 3 = C (2017-01..)."""
    y = ts.year
    if y <= 2011:
        return 1
    if y <= 2016:
        return 2
    return 3

# Equipment masks (Step 1)
def a_GE_of(ts: pd.Timestamp) -> int:
    """GE active: year<=2016 AND 8<=hour<=22."""
    return int(ts.year <= 2016 and 8 <= ts.hour <= 22)

def a_FC_of(ts: pd.Timestamp) -> int:
    """FC active: year<=2011."""
    return int(ts.year <= 2011)

# Seasonal sets (Step 1)
M_C = {5, 6, 7, 8, 9, 10}   # cooling months
M_H = {1, 2, 3, 4, 11, 12}  # heating months
SHOULDER_MONTHS = {4, 5, 10, 11}  # boundary months for shoulder check


def main():
    print(f"[load] {SRC_XLSX}")
    df = pd.read_excel(SRC_XLSX, sheet_name="Merged")
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  raw rows: {len(df)}  range: {df['Date'].min()} -> {df['Date'].max()}")

    # ------------------------------------------------------------------
    # 1. Add day-key + drop days that contain any NaN in Y columns
    # ------------------------------------------------------------------
    df["day"] = df["Date"].dt.floor("D")
    nan_mask_y = df[Y_COLS].isna().any(axis=1)
    bad_days = df.loc[nan_mask_y, "day"].unique()
    n_bad = len(bad_days)
    print(f"  dropping {n_bad} days with any NaN in Y/weather columns")
    df = df[~df["day"].isin(bad_days)].reset_index(drop=True)

    # Sanity: every remaining day must have exactly 24 hours
    counts = df.groupby("day").size()
    incomplete = counts[counts != 24]
    if len(incomplete) > 0:
        raise RuntimeError(f"Found {len(incomplete)} days with != 24 hours; first: {incomplete.head()}")
    n_days = df["day"].nunique()
    print(f"  clean days: {n_days}  (={n_days * 24} rows)")

    # ------------------------------------------------------------------
    # 2. Build per-row condition features
    # ------------------------------------------------------------------
    ts = df["Date"]
    df["h"] = ts.dt.hour.astype(np.int16)
    df["d"] = ts.dt.dayofweek.astype(np.int16)        # 0=Mon
    df["w"] = (df["d"] >= 5).astype(np.int8)          # 1 if weekend
    month = ts.dt.month
    # season: 0=summer (M_C), 1=winter (M_H), 2=other (none, since the two sets cover all 12 months)
    season_hr = np.where(month.isin(M_C), 0, np.where(month.isin(M_H), 1, 2)).astype(np.int8)
    df["s"] = season_hr
    df["a_GE"] = ts.apply(a_GE_of).astype(np.int8)
    df["a_FC"] = ts.apply(a_FC_of).astype(np.int8)
    df["regime"] = ts.apply(regime_of).astype(np.int8)

    # ------------------------------------------------------------------
    # 3. Reshape to (N_days, 24, ...)
    # ------------------------------------------------------------------
    df = df.sort_values("Date").reset_index(drop=True)
    n = n_days
    Y = df[Y_COLS].to_numpy(np.float32).reshape(n, 24, len(Y_COLS))
    C_arr = np.stack([
        df["T_out"].to_numpy(np.float32) if "T_out" in df else df[COL_T].to_numpy(np.float32),
        df[COL_I].to_numpy(np.float32),
        df[COL_RH].to_numpy(np.float32),
        df[COL_WS].to_numpy(np.float32),
        df["h"].to_numpy(np.float32),
        df["d"].to_numpy(np.float32),
        df["w"].to_numpy(np.float32),
        df["s"].to_numpy(np.float32),
        df["a_GE"].to_numpy(np.float32),
        df["a_FC"].to_numpy(np.float32),
    ], axis=1).reshape(n, 24, 10)

    # Day-level metadata: take first hour of each day
    first_per_day = df.groupby("day").head(1).reset_index(drop=True)
    assert len(first_per_day) == n
    regime_day = first_per_day["regime"].to_numpy(np.int8)
    dates_day = first_per_day["day"].to_numpy("datetime64[D]")

    # Day-level season label.  A day is "summer" if its month is in M_C,
    # "winter" if in M_H.  (The two sets partition the year, so no "other".)
    month_day = pd.to_datetime(dates_day).month
    # 0 = summer, 1 = winter, 2 = shoulder (filled below if shoulder rate >=5%)
    season_day = np.where(np.isin(month_day, list(M_C)), 0,
                          np.where(np.isin(month_day, list(M_H)), 1, 2)).astype(np.int8)

    # ------------------------------------------------------------------
    # 4. Shoulder-month overlap check (Step 2 rule)
    # ------------------------------------------------------------------
    iC, iH = Y_NAMES.index("C"), Y_NAMES.index("H")
    shoulder_mask_hr = np.isin(month.to_numpy(), list(SHOULDER_MONTHS))
    Csh = df[COL_C].to_numpy()[shoulder_mask_hr]
    Hsh = df[COL_H].to_numpy()[shoulder_mask_hr]
    both_nonzero = ((Csh > 0) & (Hsh > 0)).mean() if len(Csh) > 0 else 0.0
    shoulder_overlap_rate = float(both_nonzero)
    needs_shoulder = shoulder_overlap_rate >= 0.05
    print(f"  shoulder months {sorted(SHOULDER_MONTHS)} C&H>0 rate = {shoulder_overlap_rate:.4f}"
          f"  -> shoulder model {'NEEDED' if needs_shoulder else 'NOT needed'}")

    # If we don't need shoulder, fold boundary months into nearest summer/winter
    # (Apr/May -> winter side? Apr is closer to winter, May to summer.  Use the
    # default split: months in M_C stay summer, months in M_H stay winter -- this
    # already exhausts all 12 months, so season_day has no '2' entries.)
    assert (season_day != 2).all(), "Season labels must cover every day"

    # ------------------------------------------------------------------
    # 5. Per-regime chronological 70/15/15 split
    # ------------------------------------------------------------------
    splits = {1: {}, 2: {}, 3: {}}
    train_idx, val_idx, test_idx = [], [], []
    all_idx = np.arange(n)
    for r in (1, 2, 3):
        idx_r = all_idx[regime_day == r]
        n_r = len(idx_r)
        n_train = int(round(0.70 * n_r))
        n_val = int(round(0.15 * n_r))
        n_test = n_r - n_train - n_val
        tr = idx_r[:n_train]
        va = idx_r[n_train:n_train + n_val]
        te = idx_r[n_train + n_val:]
        splits[r] = {
            "n_total": int(n_r),
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
            "train_range": [str(dates_day[tr[0]]), str(dates_day[tr[-1]])] if len(tr) else None,
            "val_range":   [str(dates_day[va[0]]), str(dates_day[va[-1]])] if len(va) else None,
            "test_range":  [str(dates_day[te[0]]), str(dates_day[te[-1]])] if len(te) else None,
        }
        train_idx.append(tr); val_idx.append(va); test_idx.append(te)
        print(f"  Regime {r}: total={n_r}  train={len(tr)}  val={len(va)}  test={len(te)}")

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)

    # ------------------------------------------------------------------
    # 6. Save .npz files
    # ------------------------------------------------------------------
    def _save(name: str, idx: np.ndarray):
        out = OUT_DIR / f"{name}.npz"
        np.savez_compressed(
            out,
            Y=Y[idx],
            C=C_arr[idx],
            regime=regime_day[idx],
            season=season_day[idx],
            dates=dates_day[idx],
            Y_names=np.array(Y_NAMES),
            C_names=np.array(C_NAMES),
        )
        print(f"  wrote {out}  Y={Y[idx].shape}  C={C_arr[idx].shape}")

    _save("train", train_idx)
    _save("val", val_idx)
    _save("test", test_idx)

    # ------------------------------------------------------------------
    # 7. Validation checks (per A1 spec)
    # ------------------------------------------------------------------
    print("\n=== A1 checks ===")
    checks = {}

    # (1) each regime: train+val+test == total
    for r in (1, 2, 3):
        s = splits[r]
        ok = s["n_train"] + s["n_val"] + s["n_test"] == s["n_total"]
        checks[f"regime{r}_sum_equals_total"] = ok
        print(f"  [1] regime{r}: train+val+test={s['n_train']+s['n_val']+s['n_test']} == total={s['n_total']} -> {ok}")

    # (2) chronological no-leak (per regime)
    for r in (1, 2, 3):
        s = splits[r]
        ok = (s["train_range"][1] < s["val_range"][0]) and (s["val_range"][1] < s["test_range"][0])
        checks[f"regime{r}_no_leak"] = ok
        print(f"  [2] regime{r}: train.max < val.min < test.max -> {ok}")

    # (3) Y has no NaN
    nan_y = bool(np.isnan(Y).any())
    checks["Y_no_nan"] = not nan_y
    print(f"  [3] Y no NaN: {not nan_y}")

    # (4) g_FC zero for regime>=2 (FC retired)
    iGFC = Y_NAMES.index("g_FC")
    fc_zero_in_BC = (Y[regime_day >= 2][:, :, iGFC] == 0).mean()
    checks["gFC_zero_in_regimeBC_rate"] = float(fc_zero_in_BC)
    print(f"  [4] g_FC==0 in Regime B+C: {fc_zero_in_BC:.6f}  (target: 1.0)")

    # (5) g_GE zero for regime==3 (GE retired)
    iGGE = Y_NAMES.index("g_GE")
    ge_zero_in_C = (Y[regime_day == 3][:, :, iGGE] == 0).mean()
    checks["gGE_zero_in_regimeC_rate"] = float(ge_zero_in_C)
    print(f"  [5] g_GE==0 in Regime C: {ge_zero_in_C:.6f}  (target: 1.0)")

    # (6) E min > 0
    iE = Y_NAMES.index("E")
    e_min = float(Y[:, :, iE].min())
    checks["E_min"] = e_min
    print(f"  [6] E.min = {e_min}  (target: > 0)")

    # (7) shoulder C&H overlap rate
    checks["shoulder_CH_overlap_rate"] = shoulder_overlap_rate
    print(f"  [7] shoulder CH overlap rate = {shoulder_overlap_rate:.4f}  (target: < 0.05)")

    # ------------------------------------------------------------------
    # 8. Write split_report.json
    # ------------------------------------------------------------------
    report = {
        "source": str(SRC_XLSX),
        "raw_hours": int(len(df) + n_bad * 24),
        "kept_days": int(n),
        "dropped_days_due_to_NaN": int(n_bad),
        "Y_names": Y_NAMES,
        "C_names": C_NAMES,
        "regimes": {
            "A": {"years": "2001-06..2011-12", **splits[1]},
            "B": {"years": "2012-01..2016-12", **splits[2]},
            "C": {"years": "2017-01..2022-10", **splits[3]},
        },
        "total_train": int(len(train_idx)),
        "total_val": int(len(val_idx)),
        "total_test": int(len(test_idx)),
        "shoulder_months": sorted(SHOULDER_MONTHS),
        "shoulder_CH_overlap_rate": shoulder_overlap_rate,
        "shoulder_model_needed": bool(needs_shoulder),
        "M_C": sorted(M_C),
        "M_H": sorted(M_H),
        "checks": checks,
    }
    with open(OUT_DIR / "split_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  wrote {OUT_DIR / 'split_report.json'}")
    print("\n[done]")


if __name__ == "__main__":
    main()
