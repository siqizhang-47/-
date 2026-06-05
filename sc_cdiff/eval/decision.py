"""Optional decision-relevant evaluation, consistent with the real system
(no storage, no fabricated dispatch) -- plan §8.6 / step 15.

Decision object = procurement quantity / peak reservation, NOT unit dispatch.
In test era (era2) the grid-buy residual electricity load ~= electricity load.
Gas demand is mapped from cooling/heating/hot-water via documented efficiencies.
"""
from __future__ import annotations

import numpy as np

# documented efficiencies (plan §1.1)
COP_COOL = 1.1     # absorption chiller cooling
COP_HEAT = 0.93    # absorption chiller heating
# day-ahead procurement nominal quantile (risk-averse reservation)
RESERVE_Q = 0.9


def electricity_procurement(scenarios, truth, e_idx=1, energy_price=1.0,
                            demand_price=15.0):
    """scenarios:[N,M,5,24], truth:[N,5,24]. Returns cost regret vs perfect
    foresight and peak-coverage reliability for day-ahead electricity buy."""
    S = scenarios[:, :, e_idx, :]            # [N,M,24]
    y = truth[:, e_idx, :]                   # [N,24]
    # energy: reserve = ensemble mean total; cost regret = |buy - actual| * price
    buy_energy = S.sum(axis=2).mean(axis=1)  # [N]
    true_energy = y.sum(axis=1)
    energy_regret = float(np.abs(buy_energy - true_energy).mean() * energy_price)
    # peak demand: reserve = high quantile of scenario daily peak
    scen_peak = S.max(axis=2)                # [N,M]
    reserve_peak = np.quantile(scen_peak, RESERVE_Q, axis=1)  # [N]
    true_peak = y.max(axis=1)
    peak_cover = float((reserve_peak >= true_peak).mean())
    # demand-charge regret: pay for max(reserve, actual) shortfall penalty
    peak_regret = float((np.abs(reserve_peak - true_peak) * demand_price).mean())
    return {"energy_regret": energy_regret, "peak_coverage": peak_cover,
            "peak_regret": peak_regret}


def gas_procurement(scenarios, truth, c_idx=2, h_idx=3, hw_idx=4):
    """Map cooling/heating/hot-water to gas demand and assess day-ahead gas buy
    deviation (in load-equivalent units)."""
    def gas_of(arr):
        # arr:[...,5,24] -> daily gas-equivalent demand
        cool = arr[..., c_idx, :] / COP_COOL
        heat = arr[..., h_idx, :] / COP_HEAT
        hw = arr[..., hw_idx, :]
        return (cool + heat + hw).sum(axis=-1)
    gas_scen = gas_of(scenarios)             # [N,M]
    gas_true = gas_of(truth)                 # [N]
    buy = gas_scen.mean(axis=1)
    dev = float(np.abs(buy - gas_true).mean())
    return {"gas_procure_dev": dev}


def decision_metrics(scenarios, truth):
    out = {}
    out.update(electricity_procurement(scenarios, truth))
    out.update(gas_procurement(scenarios, truth))
    return out
