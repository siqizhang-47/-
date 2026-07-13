"""
Merge the raw Integrated-Energy-System (IES) energy + weather CSVs into a single
multivariate hourly time-series file usable by the NsDiff framework.

Inputs  (place under ./data/IES/):
    - Total_energy.csv   : Year,Month,Day,Hour,Weekday,Electricity,PV,Cooling,Heat,Emission,Total Energy
    - Total_weather.csv  : Year,Month,Day,Hour,DOW,Temperature,Dew Point,Humidity,Wind Speed,Wind Gust,Pressure,Precip

Output (./data/IES/IES.csv):
    date, Electricity, Cooling, Heat, PV, Temperature, Dew Point, Humidity,
    Wind Speed, Wind Gust, Pressure, Precip

The first four columns are the four IES targets evaluated in the paper
(Electrical / Cooling / Heating loads and PV power); the remaining seven weather
channels are kept as exogenous covariates so the model sees the full context.
A per-feature base (the max over the whole series) is also written to
IES_base.csv so results can be expressed in per-unit (p.u.) for plotting.
"""
import os
import argparse
import numpy as np
import pandas as pd

# Column order of the merged file. The first TARGETS are what the paper plots.
TARGET_COLS = ["Electricity", "Cooling", "Heat", "PV"]          # -> Electrical, Cooling, Heating, PV
WEATHER_COLS = ["Temperature", "Dew Point", "Humidity",
                "Wind Speed", "Wind Gust", "Pressure", "Precip"]
FEATURE_COLS = TARGET_COLS + WEATHER_COLS
KEYS = ["Year", "Month", "Day", "Hour"]


def merge(energy_path, weather_path, out_path):
    energy = pd.read_csv(energy_path)
    weather = pd.read_csv(weather_path)

    # Align strictly on the calendar keys (inner join keeps only shared timestamps).
    merged = pd.merge(energy, weather, on=KEYS, how="inner", suffixes=("", "_w"))
    merged = merged.sort_values(KEYS).reset_index(drop=True)

    # Build a proper hourly datetime index.
    date = pd.to_datetime(merged[KEYS].rename(
        columns={"Year": "year", "Month": "month", "Day": "day", "Hour": "hour"}))

    out = pd.DataFrame({"date": date})
    for c in FEATURE_COLS:
        if c not in merged.columns:
            raise KeyError(f"expected column '{c}' not found in merged data")
        out[c] = merged[c].astype("float32").values

    # Basic sanity: no NaNs, strictly increasing hourly timestamps.
    assert out[FEATURE_COLS].isna().sum().sum() == 0, "NaNs after merge"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)

    # Per-feature base value for per-unit conversion (used only for plotting).
    base = out[FEATURE_COLS].abs().max()
    base.to_csv(os.path.join(os.path.dirname(out_path), "IES_base.csv"),
                header=["base"])

    print(f"[merge] wrote {out_path}  shape={out.shape}  "
          f"range={out.date.min()} .. {out.date.max()}")
    print(f"[merge] features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./data/IES",
                    help="folder containing Total_energy.csv and Total_weather.csv")
    ap.add_argument("--energy", default=None)
    ap.add_argument("--weather", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    energy = a.energy or os.path.join(a.dir, "Total_energy.csv")
    weather = a.weather or os.path.join(a.dir, "Total_weather.csv")
    out = a.out or os.path.join(a.dir, "IES.csv")
    merge(energy, weather, out)
