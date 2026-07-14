"""
IES dataset for the TimeXer-NsDiff experiment (plan §2-5).

- Targets Y (4): Electricity, PV, Cooling, Heat
- Exogenous: calendar raw [Year,Month,Day,Hour,Weekday] (encoded in-model) +
  weather (7): Temperature, Dew Point, Humidity, Wind Speed, Wind Gust,
  Pressure, Precip  (Precip stands in for 'Clearsky GHI', absent from the data).
- Window: L=168 history, H=24 horizon.
- Split by YEAR of the forecast window: train 2014-2020, val 2021, test 2022
  (history may reach back into the previous year; labels stay in the split year).
- Separate standardisation for targets and weather, fit on TRAIN rows only.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

TARGET_COLS = ["Electricity", "PV", "Cooling", "Heat"]
WEATHER_COLS = ["Temperature", "Dew Point", "Humidity", "Wind Speed",
                "Wind Gust", "Pressure", "Precip"]
KEYS = ["Year", "Month", "Day", "Hour"]


def _load_merged(root):
    e = pd.read_csv(os.path.join(root, "Total_energy.csv"))
    w = pd.read_csv(os.path.join(root, "Total_weather.csv"))
    m = pd.merge(e, w, on=KEYS, how="inner", suffixes=("", "_w")).sort_values(KEYS).reset_index(drop=True)
    ts = pd.to_datetime(m[KEYS].rename(columns={"Year": "year", "Month": "month",
                                                "Day": "day", "Hour": "hour"}))
    energy = m[TARGET_COLS].to_numpy(np.float32)                 # [T,4]
    weather = m[WEATHER_COLS].to_numpy(np.float32)               # [T,7]
    calendar = np.stack([ts.dt.year, ts.dt.month, ts.dt.day,
                         ts.dt.hour, ts.dt.dayofweek], axis=-1).astype(np.float32)  # [T,5]
    years = ts.dt.year.to_numpy()
    return energy, weather, calendar, years


class Scaler:
    def __init__(self, mean, std):
        self.mean = mean.astype(np.float32)
        self.std = np.where(std == 0, 1.0, std).astype(np.float32)

    def transform(self, x):      return (x - self.mean) / self.std
    def inverse(self, x):        return x * self.std + self.mean


class IESWindows(Dataset):
    def __init__(self, energy_s, weather_s, calendar, energy_raw, starts, L, H):
        self.es, self.ws, self.cal, self.er = energy_s, weather_s, calendar, energy_raw
        self.starts, self.L, self.H = starts, L, H

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t = self.starts[i]                       # first FUTURE step index
        L, H = self.L, self.H
        return {
            "history_energy": torch.from_numpy(self.es[t - L:t]),         # [L,4] scaled
            "future_energy":  torch.from_numpy(self.es[t:t + H]),          # [H,4] scaled
            "future_calendar": torch.from_numpy(self.cal[t:t + H]),        # [H,5] raw
            "future_weather":  torch.from_numpy(self.ws[t:t + H]),         # [H,7] scaled
            "future_energy_raw": torch.from_numpy(self.er[t:t + H]),       # [H,4] raw (metrics)
            "start_hour": int(self.cal[t, 3]),
        }


def build_dataloaders(root="./data/IES", L=168, H=24, batch_size=64, num_workers=4,
                      train_years=(2014, 2020), val_year=2021, test_year=2022):
    energy, weather, calendar, years = _load_merged(root)
    T = len(energy)

    def year_of_start(t):
        return years[t]                          # year of the first forecast step

    starts = np.arange(L, T - H + 1)
    ys = years[starts]                           # forecast-start year
    train_starts = starts[(ys >= train_years[0]) & (ys <= train_years[1])]
    val_starts = starts[ys == val_year]
    test_starts = starts[ys == test_year]

    # scalers fit on TRAIN ROWS only (rows whose timestamp year is in train range)
    row_train = (years >= train_years[0]) & (years <= train_years[1])
    tgt_scaler = Scaler(energy[row_train].mean(0), energy[row_train].std(0))
    wth_scaler = Scaler(weather[row_train].mean(0), weather[row_train].std(0))

    energy_s = tgt_scaler.transform(energy)
    weather_s = wth_scaler.transform(weather)

    def mk(starts_):
        return IESWindows(energy_s, weather_s, calendar, energy, starts_, L, H)

    loaders = {}
    loaders["train"] = DataLoader(mk(train_starts), batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
    loaders["val"] = DataLoader(mk(val_starts), batch_size=batch_size, shuffle=False,
                                num_workers=num_workers)
    loaders["test"] = DataLoader(mk(test_starts), batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)
    meta = {"target_scaler": tgt_scaler, "weather_scaler": wth_scaler,
            "n_train": len(train_starts), "n_val": len(val_starts), "n_test": len(test_starts),
            "target_names": TARGET_COLS}
    print(f"[data] windows -> train {meta['n_train']}  val {meta['n_val']}  test {meta['n_test']}")
    return loaders, meta
