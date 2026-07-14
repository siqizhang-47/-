"""
IES dataset for the TimeXer-NsDiff experiment (plan §2-5).

Source: aligned_energy_weather_summary.xlsx, sheet 'Aligned_Data' (energy +
weather already merged on the hourly calendar; 78 888 rows, 2014-2022, no gaps).

- Targets Y (4): Electricity, PV, Cooling, Heat
- Exogenous: calendar raw [Year,Month,Day,Hour,Weekday] (encoded in-model) +
  weather (7): Temperature, Dew Point, Humidity, Wind Speed, Wind Gust,
  Pressure, Clearsky GHI.
- Window: L=168 history, H=24 horizon.
- Split by YEAR of the forecast window: train 2014-2020, val 2021, test 2022.
- **Min-max normalisation to [0,1]** for BOTH targets and weather, fit on TRAIN
  rows only (separate scalers). Everything the model sees — and every metric /
  figure produced downstream — is therefore in the normalised [0,1] range.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

TARGET_COLS = ["Electricity", "PV", "Cooling", "Heat"]
WEATHER_COLS = ["Temperature", "Dew Point", "Humidity", "Wind Speed",
                "Wind Gust", "Pressure", "Clearsky GHI"]
KEYS = ["Year", "Month", "Day", "Hour"]
SHEET = "Aligned_Data"


def _load(path):
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=SHEET)
    else:
        df = pd.read_csv(path)
    df = df.sort_values(KEYS).reset_index(drop=True)
    ts = pd.to_datetime(df[KEYS].rename(columns={"Year": "year", "Month": "month",
                                                 "Day": "day", "Hour": "hour"}))
    energy = df[TARGET_COLS].to_numpy(np.float32)                       # [T,4]
    weather = df[WEATHER_COLS].to_numpy(np.float32)                     # [T,7]
    calendar = np.stack([ts.dt.year, ts.dt.month, ts.dt.day,
                         ts.dt.hour, ts.dt.dayofweek], axis=-1).astype(np.float32)
    years = ts.dt.year.to_numpy()
    return energy, weather, calendar, years


class MinMaxScaler:
    """Scale each column to [0,1] using train-set min/max."""
    def __init__(self, dmin, dmax):
        self.min = dmin.astype(np.float32)
        rng = (dmax - dmin).astype(np.float32)
        self.range = np.where(rng == 0, 1.0, rng)

    def transform(self, x):  return (x - self.min) / self.range
    def inverse(self, x):    return x * self.range + self.min


class IESWindows(Dataset):
    def __init__(self, energy_n, weather_n, calendar, starts, L, H):
        self.en, self.wn, self.cal = energy_n, weather_n, calendar
        self.starts, self.L, self.H = starts, L, H

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t = self.starts[i]
        L, H = self.L, self.H
        return {
            "history_energy": torch.from_numpy(self.en[t - L:t]),        # [L,4] in [0,1]
            "future_energy":  torch.from_numpy(self.en[t:t + H]),         # [H,4] in [0,1]
            "future_calendar": torch.from_numpy(self.cal[t:t + H]),       # [H,5] raw
            "future_weather":  torch.from_numpy(self.wn[t:t + H]),        # [H,7] in [0,1]
            "start_hour": int(self.cal[t, 3]),
        }


def build_dataloaders(root="./data/IES/aligned_energy_weather_summary.xlsx",
                      L=168, H=24, batch_size=64, num_workers=4,
                      train_years=(2014, 2020), val_year=2021, test_year=2022):
    if os.path.isdir(root):     # accept a folder too
        root = os.path.join(root, "aligned_energy_weather_summary.xlsx")
    energy, weather, calendar, years = _load(root)
    T = len(energy)

    starts = np.arange(L, T - H + 1)
    ys = years[starts]                                   # forecast-start year
    train_starts = starts[(ys >= train_years[0]) & (ys <= train_years[1])]
    val_starts = starts[ys == val_year]
    test_starts = starts[ys == test_year]

    row_train = (years >= train_years[0]) & (years <= train_years[1])
    tgt_scaler = MinMaxScaler(energy[row_train].min(0), energy[row_train].max(0))
    wth_scaler = MinMaxScaler(weather[row_train].min(0), weather[row_train].max(0))

    energy_n = tgt_scaler.transform(energy)
    weather_n = wth_scaler.transform(weather)

    def mk(s):
        return IESWindows(energy_n, weather_n, calendar, s, L, H)

    loaders = {
        "train": DataLoader(mk(train_starts), batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True),
        "val": DataLoader(mk(val_starts), batch_size=batch_size, shuffle=False,
                          num_workers=num_workers),
        "test": DataLoader(mk(test_starts), batch_size=batch_size, shuffle=False,
                           num_workers=num_workers),
    }
    meta = {"target_scaler": tgt_scaler, "weather_scaler": wth_scaler,
            "n_train": len(train_starts), "n_val": len(val_starts), "n_test": len(test_starts),
            "target_names": TARGET_COLS}
    print(f"[data] windows -> train {meta['n_train']}  val {meta['n_val']}  test {meta['n_test']}  (min-max [0,1])")
    return loaders, meta
