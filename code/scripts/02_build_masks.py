"""
A2. Build deterministic masks for PS-BCD v4.1.

For each split in {train, val, test}, builds:
    a_GE (N,24) bool   GE active   = I(year<=2016) * I(8<=h<=22)
    a_FC (N,24) bool   FC active   = I(year<=2011)
    m_C  (N,24) bool   cooling active month
    m_H  (N,24) bool   heating active month
    m_PV (N,24) bool   PV daylight = I(I_t > delta_I)

Output: data/processed/masks_{train,val,test}.npz
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace/code")
PROC_DIR = ROOT / "data" / "processed"

# From plan Step 1
M_C = {5, 6, 7, 8, 9, 10}
M_H = {1, 2, 3, 4, 11, 12}
# Plan suggested delta_I in [5, 20] W/m^2 but inspection of this dataset shows
# PV is exactly zero iff I == 0 (the irradiation sensor reads down to ~0.02
# W/m^2 with PV already producing).  Use a tiny positive threshold so the
# mask captures actual nighttime without false-flagging cloudy daylight hours.
DELTA_I = 0.0  # W/m^2  -> equivalent to I > 0


def build_masks_for_split(split: str):
    in_path = PROC_DIR / f"{split}.npz"
    out_path = PROC_DIR / f"masks_{split}.npz"
    d = np.load(in_path, allow_pickle=False)
    Y = d["Y"]                          # (N,24,14) float32
    C = d["C"]                          # (N,24,10) float32
    dates = d["dates"]                  # (N,)  datetime64[D]
    Y_names = list(d["Y_names"])
    C_names = list(d["C_names"])

    N = Y.shape[0]
    hours = np.arange(24, dtype=np.int16)[None, :]            # (1,24)
    months = pd.to_datetime(dates).month.to_numpy()[:, None]  # (N,1)
    years = pd.to_datetime(dates).year.to_numpy()[:, None]    # (N,1)

    # ----- deterministic masks -----
    a_GE = (years <= 2016) & (hours >= 8) & (hours <= 22)     # (N,24)
    a_FC = np.broadcast_to(years <= 2011, (N, 24)).copy()
    m_C = np.isin(months, list(M_C)) & np.ones((1, 24), bool) # (N,24)
    m_H = np.isin(months, list(M_H)) & np.ones((1, 24), bool)

    # PV mask from irradiation column in C
    iI = C_names.index("I")
    I_arr = C[:, :, iI]                                        # (N,24)
    m_PV = I_arr > DELTA_I

    a_GE = a_GE.astype(bool)
    a_FC = a_FC.astype(bool)
    m_C = m_C.astype(bool)
    m_H = m_H.astype(bool)
    m_PV = m_PV.astype(bool)

    np.savez_compressed(
        out_path,
        a_GE=a_GE, a_FC=a_FC, m_C=m_C, m_H=m_H, m_PV=m_PV,
        delta_I=np.float32(DELTA_I),
    )
    print(f"  wrote {out_path}  a_GE={a_GE.shape}  a_FC={a_FC.shape} "
          f"m_C={m_C.shape}  m_H={m_H.shape}  m_PV={m_PV.shape}")
    return d, {"a_GE": a_GE, "a_FC": a_FC, "m_C": m_C, "m_H": m_H, "m_PV": m_PV}


def main():
    report = {"delta_I": DELTA_I, "M_C": sorted(M_C), "M_H": sorted(M_H), "splits": {}}
    print("[A2] Building masks ...")
    for split in ("train", "val", "test"):
        data, masks = build_masks_for_split(split)
        Y = data["Y"]
        C = data["C"]
        regime = data["regime"]
        Y_names = list(data["Y_names"])
        iE = Y_names.index("E")
        iC_ = Y_names.index("C")
        iH = Y_names.index("H")
        iPGE = Y_names.index("P_GE")
        iPFC = Y_names.index("P_FC")
        iPPV = Y_names.index("P_PV")

        a_GE = masks["a_GE"]; a_FC = masks["a_FC"]
        m_C = masks["m_C"];   m_H = masks["m_H"];   m_PV = masks["m_PV"]

        # ---------- checks ----------
        # (1) a_GE all False where regime == 3 (Regime C)
        idx_C_regime = (regime == 3)
        chk1 = (not a_GE[idx_C_regime].any()) if idx_C_regime.any() else True

        # (2) a_FC all False where regime in {2,3}  (FC retired in B and C)
        idx_BC = (regime >= 2)
        chk2 = (not a_FC[idx_BC].any()) if idx_BC.any() else True

        # (3) real P_GE > 0 but a_GE == 0  -> rate < 2%
        PGE = Y[:, :, iPGE]
        denom_pge = (PGE > 0).sum()
        rate_pge = float(((PGE > 0) & (~a_GE)).sum() / denom_pge) if denom_pge > 0 else 0.0

        # (3b) real P_FC > 0 but a_FC == 0  (informational)
        PFC = Y[:, :, iPFC]
        denom_pfc = (PFC > 0).sum()
        rate_pfc = float(((PFC > 0) & (~a_FC)).sum() / denom_pfc) if denom_pfc > 0 else 0.0

        # (4) real C > 0 but m_C == 0  -> rate < 1%
        Cv = Y[:, :, iC_]
        denom_c = (Cv > 0).sum()
        rate_c = float(((Cv > 0) & (~m_C)).sum() / denom_c) if denom_c > 0 else 0.0

        # (4b) real H > 0 but m_H == 0 (informational)
        Hv = Y[:, :, iH]
        denom_h = (Hv > 0).sum()
        rate_h = float(((Hv > 0) & (~m_H)).sum() / denom_h) if denom_h > 0 else 0.0

        # (5) real P_PV > 0 but m_PV == 0  -> rate < 0.5%
        PPV = Y[:, :, iPPV]
        denom_pv = (PPV > 0).sum()
        rate_pv = float(((PPV > 0) & (~m_PV)).sum() / denom_pv) if denom_pv > 0 else 0.0

        info = {
            "N": int(Y.shape[0]),
            "check1_aGE_falseInRegimeC": bool(chk1),
            "check2_aFC_falseInRegimeBC": bool(chk2),
            "check3_PGE_pos_but_aGE0_rate": rate_pge,   # target <0.02
            "check4_C_pos_but_mC0_rate": rate_c,         # target <0.01
            "check5_PPV_pos_but_mPV0_rate": rate_pv,     # target <0.005
            "info_PFC_pos_but_aFC0_rate": rate_pfc,
            "info_H_pos_but_mH0_rate": rate_h,
            "active_fractions": {
                "a_GE": float(a_GE.mean()),
                "a_FC": float(a_FC.mean()),
                "m_C": float(m_C.mean()),
                "m_H": float(m_H.mean()),
                "m_PV": float(m_PV.mean()),
            },
        }
        report["splits"][split] = info

        print(f"\n  === A2 checks for {split} ===")
        print(f"  [1] a_GE all False in Regime C            : {chk1}  (target True)")
        print(f"  [2] a_FC all False in Regime B+C          : {chk2}  (target True)")
        print(f"  [3] P_GE>0 & a_GE==0 rate                 : {rate_pge:.6f}  (target <0.02)")
        print(f"  [4] C>0    & m_C==0  rate                 : {rate_c:.6f}  (target <0.01)")
        print(f"  [5] P_PV>0 & m_PV==0 rate                 : {rate_pv:.6f}  (target <0.005)")
        print(f"  (info) P_FC>0 & a_FC==0 rate              : {rate_pfc:.6f}")
        print(f"  (info) H>0    & m_H==0  rate              : {rate_h:.6f}")

    with open(PROC_DIR / "masks_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  wrote {PROC_DIR / 'masks_report.json'}")
    print("\n[done]")


if __name__ == "__main__":
    main()
