"""Physical-feasibility / structural-zero metrics (plan §8.5)."""
from __future__ import annotations

import numpy as np


def physical_metrics(scenarios, truth, irr, pv_cap=160.0,
                     pv_idx=0, c_idx=2, h_idx=3, summer=(6, 7, 8), winter=(12, 1, 2),
                     months=None):
    """scenarios:[N,M,5,24], irr:[N,24] true irradiance, months:[N] calendar month.
    Off-season uses cooling-in-winter / heating-in-summer as the violation proxy."""
    N, M, C, T = scenarios.shape
    eps = 1e-6
    night = (irr <= 0)[:, None, :]                                   # [N,1,24]
    night_pv_viol = (scenarios[:, :, pv_idx, :][:, :, :] > eps) & night
    night_rate = float(night_pv_viol.sum() / max(night.sum() * M, 1))

    neg_rate = float((scenarios < -eps).mean())
    cap_rate = float((scenarios[:, :, pv_idx, :] > pv_cap + eps).mean())

    res = {"night_pv_viol_rate": night_rate, "neg_rate": neg_rate,
           "pv_cap_viol_rate": cap_rate}

    if months is not None:
        is_summer = np.isin(months, summer)
        is_winter = np.isin(months, winter)
        # heating active in summer
        if is_summer.any():
            heat_summer = (scenarios[is_summer, :, h_idx, :] > eps).mean()
            res["heat_offseason_rate"] = float(heat_summer)
        # cooling active in winter
        if is_winter.any():
            cool_winter = (scenarios[is_winter, :, c_idx, :] > eps).mean()
            res["cool_offseason_rate"] = float(cool_winter)
    return res
