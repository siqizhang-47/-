import os
import numpy as np
import pandas as pd
from typing import Optional
from torch_timeseries.core.dataset.dataset import TimeSeriesDataset


class IES(TimeSeriesDataset):
    """Integrated Energy System (IES) hourly dataset.

    Built by merging ``Total_energy.csv`` and ``Total_weather.csv`` on the
    calendar keys (Year, Month, Day, Hour). 78 888 hourly steps from
    2014-01-01 to 2022-12-31.

    Feature order (11 channels), first four are the paper targets::

        0 Electricity  (Electrical load)
        1 Cooling      (Cooling load)
        2 Heat         (Heating load)
        3 PV           (PV power)
        4 Temperature  5 Dew Point   6 Humidity   7 Wind Speed
        8 Wind Gust    9 Pressure   10 Precip     (weather covariates)
    """

    name: str = "IES"
    num_features: int = 11
    freq: str = "h"           # hourly data
    length: int = 78888

    def download(self) -> None:
        # If the merged file is already present, nothing to do. Otherwise build
        # it from the two raw CSVs that must be placed in the dataset folder.
        merged = os.path.join(self.dir, "IES.csv")
        if os.path.exists(merged):
            return
        energy = os.path.join(self.dir, "Total_energy.csv")
        weather = os.path.join(self.dir, "Total_weather.csv")
        if not (os.path.exists(energy) and os.path.exists(weather)):
            raise FileNotFoundError(
                f"Place Total_energy.csv and Total_weather.csv under '{self.dir}', "
                f"or run data_merge.py to create IES.csv first.")
        # Lazy import to avoid a hard dependency at package import time.
        import importlib.util
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        spec = importlib.util.spec_from_file_location(
            "data_merge", os.path.join(here, "data_merge.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.merge(energy, weather, merged)

    def _load(self) -> np.ndarray:
        self.file_name = os.path.join(self.dir, "IES.csv")
        self.df = pd.read_csv(self.file_name, parse_dates=["date"])
        self.dates = pd.DataFrame({"date": self.df.date})
        self.data = self.df.drop("date", axis=1).values.astype("float32")
        return self.data
