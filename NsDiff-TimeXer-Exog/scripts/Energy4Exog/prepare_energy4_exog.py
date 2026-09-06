#!/usr/bin/env python3
"""Rebuild the 2018-2019 Energy4Exog CSV from the supplied workbook."""
from pathlib import Path
import argparse
import pandas as pd

TARGETS = ["Electricity", "PV", "Cooling", "Heat"]
WEATHER = ["Temperature", "Dew Point", "Humidity", "GHI"]

p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="Path to aligned_energy_weather...xlsx")
p.add_argument("--output", default="dataset/Energy4Exog/energy4_exog.csv")
a = p.parse_args()

df = pd.read_excel(a.input, sheet_name="Aligned_Data", engine="openpyxl")
needed = ["Year", "Month", "Day", "Hour"] + TARGETS + WEATHER
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

date = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day, hour=df.Hour))
out = pd.DataFrame({"date": date})
for c in TARGETS + WEATHER:
    out[c] = pd.to_numeric(df[c], errors="raise")
out = out[(out.date >= "2018-01-01") & (out.date < "2020-01-01")].reset_index(drop=True)
if len(out) != 17520:
    raise ValueError(f"Expected 17,520 rows, got {len(out)}")
if not (out.date.diff().dropna() == pd.Timedelta(hours=1)).all():
    raise ValueError("Timestamps are not continuous hourly data")
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(a.output, index=False)
print(f"Saved {len(out):,} rows to {a.output}")
