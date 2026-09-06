import os
import numpy as np
import pandas as pd
from torch_timeseries.core import TimeSeriesDataset


class Energy4(TimeSeriesDataset):
    """Hourly four-channel energy dataset used for the custom NsDiff experiment.

    Feature order is fixed to:
        Electricity, PV, Cooling, Heat

    The repository ships ``dataset/Energy4/energy4.csv``.  With the command
    ``--data_path=./dataset`` the inherited TimeSeriesDataset creates/uses
    ``./dataset/Energy4`` as ``self.dir``.
    """

    name: str = "Energy4"
    num_features: int = 4
    freq: str = "h"
    length: int = 17520
    feature_names = ["Electricity", "PV", "Cooling", "Heat"]

    def download(self) -> None:
        # Local-only dataset: never download or silently replace user data.
        path = os.path.join(self.dir, "energy4.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/Energy4/prepare_energy4.py first "
                "or copy energy4.csv into dataset/Energy4/."
            )

    def _load(self) -> np.ndarray:
        self.file_path = os.path.join(self.dir, "energy4.csv")
        self.df = pd.read_csv(self.file_path, parse_dates=["date"])
        required = ["date"] + self.feature_names
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Energy4 CSV is missing columns: {missing}")

        self.df = self.df[required].copy()

        # This experiment intentionally uses only calendar years 2018-2019.
        start = pd.Timestamp("2018-01-01 00:00:00")
        end_exclusive = pd.Timestamp("2020-01-01 00:00:00")
        self.df = self.df[(self.df["date"] >= start) & (self.df["date"] < end_exclusive)].reset_index(drop=True)
        if self.df.empty:
            raise ValueError("Energy4 has no observations in the required 2018-2019 date range.")

        if self.df[self.feature_names].isna().any().any():
            raise ValueError("Energy4 contains missing values in the four target channels.")
        if not self.df["date"].is_monotonic_increasing:
            raise ValueError("Energy4 timestamps must be monotonically increasing.")

        delta = self.df["date"].diff().dropna()
        if len(delta) and not (delta == pd.Timedelta(hours=1)).all():
            raise ValueError("Energy4 must be strictly hourly with no timestamp gaps.")

        self.dates = pd.DataFrame({"date": self.df["date"]})
        self.data = self.df[self.feature_names].to_numpy(dtype=np.float32)
        self.num_features = self.data.shape[1]
        self.length = self.data.shape[0]
        return self.data
