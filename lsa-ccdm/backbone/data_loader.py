"""Dataset for the energy MW task, adapted from CCDM data_producer/data_loader.py.

MW ("multivariate with weather") returns five tensors per window:
    seq_x      (cont_len, 8)   history: 4 targets + 4 weather covariates
    seq_y      (pred_len, 4)   future targets (modeled by the diffusion)
    seq_w      (pred_len, 4)   future weather (condition, perfect-forecast assumption)
    seq_x_mark (cont_len, n_time_feat)
    seq_y_mark (pred_len, n_time_feat)

Both scalers are fit on the training rows ONLY (no leakage).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils.time_features import time_features

# train/val/deploy row boundaries:
# 2014-2018 train (43,824 rows incl. 2016 leap) / 2019 val (8,760) / 2020-2022 deploy (26,304)
data_intervals = {
    "energy": [43824, 52584, 78888],
}

TARGET_COLS = ["PV", "Electricity", "Cooling", "Heat"]
COV_COLS = ["Temperature", "DewPoint", "Humidity", "GHI"]


class Dataset_MTS(Dataset):
    def __init__(self, data_name, data_path, cont_len, pred_len, status="train",
                 task="MW", scale=True, freq="h", intervals=None):
        assert status in ["train", "val", "test"]
        assert task == "MW", "this repo only implements the MW task"
        self.cont_len = cont_len
        self.pred_len = pred_len
        self.set_type = {"train": 0, "val": 1, "test": 2}[status]
        self.scale = scale
        self.freq = freq
        self.intervals = list(intervals) if intervals is not None else data_intervals[data_name]
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.data_path)
        assert list(df_raw.columns) == ["date"] + TARGET_COLS + COV_COLS, \
            f"unexpected column order in {self.data_path}: {list(df_raw.columns)}"

        start_indices = [0, self.intervals[0] - self.cont_len, self.intervals[1] - self.cont_len]
        end_indices = [self.intervals[0], self.intervals[1], self.intervals[2]]
        start_index = start_indices[self.set_type]
        end_index = end_indices[self.set_type]

        target = df_raw[TARGET_COLS].values.astype(np.float64)
        cov = df_raw[COV_COLS].values.astype(np.float64)

        self.scaler_target = StandardScaler()
        self.scaler_cov = StandardScaler()
        if self.scale:
            train_end = end_indices[0]
            self.scaler_target.fit(target[:train_end])
            self.scaler_cov.fit(cov[:train_end])
            target = self.scaler_target.transform(target)
            cov = self.scaler_cov.transform(cov)

        dates = pd.to_datetime(df_raw["date"].values[start_index:end_index])
        data_stamp = time_features(dates, freq=self.freq).transpose(1, 0)

        self.dates = dates
        self.data_target = target[start_index:end_index].astype(np.float32)
        self.data_cov = cov[start_index:end_index].astype(np.float32)
        self.data_stamp = data_stamp.astype(np.float32)

    def __getitem__(self, index):
        x_begin = index
        x_end = x_begin + self.cont_len
        y_begin = x_end
        y_end = y_begin + self.pred_len

        seq_x = np.concatenate(
            [self.data_target[x_begin:x_end], self.data_cov[x_begin:x_end]], axis=-1)
        seq_y = self.data_target[y_begin:y_end]
        seq_w = self.data_cov[y_begin:y_end]
        seq_x_mark = self.data_stamp[x_begin:x_end]
        seq_y_mark = self.data_stamp[y_begin:y_end]
        return seq_x, seq_y, seq_w, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_target) - self.cont_len - self.pred_len + 1

    def inverse_transform_target(self, data):
        return self.scaler_target.inverse_transform(data)


def create_mts_loader(cfg, status, batch_size=None, intervals=None, shuffle=None):
    dataset = Dataset_MTS(
        data_name=cfg.data_name, data_path=cfg.data_csv,
        cont_len=cfg.cont_len, pred_len=cfg.pred_len,
        status=status, task=cfg.task, freq=cfg.freq, intervals=intervals)
    is_train = status == "train"
    loader = DataLoader(
        dataset,
        batch_size=batch_size or cfg.train_batch_size,
        shuffle=is_train if shuffle is None else shuffle,
        num_workers=getattr(cfg, "num_workers", 0),
        drop_last=is_train)
    print(f"Sample number in {status} set: {len(dataset)}")
    return loader
