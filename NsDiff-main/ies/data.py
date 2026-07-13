from __future__ import annotations
import numpy as np, pandas as pd, torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

TARGETS = ['[Electricity] Electricity load (kW)',
           '[Electricity] Cooling load (kW)',
           '[Electricity] Heating load (kW)',
           '[Power] Solar energy generation (kW)']
WEATHER = ['[Weather] Horizontal solar irradition (W)',
           '[Weather] Outdoor air temperature (℃)',
           '[Weather] Outdoor air humidity (%)',
           '[Weather] Wind speed (m/s)']
PV_IDX = 3                      # Solar 在 TARGETS 中的下标


def calendar_feats(dt: pd.DatetimeIndex) -> np.ndarray:
    """4 维日历特征,归一化到 [-0.5, 0.5]。"""
    return np.stack([dt.month / 12 - .5, dt.day / 31 - .5,
                     dt.weekday / 6 - .5, dt.hour / 23 - .5], axis=-1).astype(np.float32)


class IESJointDataset(Dataset):
    """滑窗数据集。每个样本:
       x_y (L,4) 目标历史;  y_y (O,4) 目标未来(标准化)
       x_w (L,W) 天气历史;  y_w (O,W) 未来天气实测(Oracle,标准化)
       x_mark (L,4+W)=[日历‖天气]历史;  y_mark (O,4+W)=[日历‖Oracle天气]未来
       y_hour (O,) 未来整点小时(夜间 PV 约束用)
       y_y_raw (O,4) 目标未来原始 kW(打分反标准化对照用)
    """
    def __init__(self, df, starts, L, O, sy, sw):
        self.starts, self.L, self.O = np.asarray(starts), L, O
        self.Yz = sy.transform(df[TARGETS].values).astype(np.float32)
        self.Wz = sw.transform(df[WEATHER].values).astype(np.float32)
        self.Yraw = df[TARGETS].values.astype(np.float32)
        dt = pd.DatetimeIndex(pd.to_datetime(df['Date']))
        self.cal = calendar_feats(dt)
        self.hour = dt.hour.values.astype(np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s, L, O = int(self.starts[i]), self.L, self.O
        x_cal, y_cal = self.cal[s - L:s], self.cal[s:s + O]
        x_w, y_w = self.Wz[s - L:s], self.Wz[s:s + O]
        return {
            'x_y':    torch.tensor(self.Yz[s - L:s]),
            'y_y':    torch.tensor(self.Yz[s:s + O]),
            'x_w':    torch.tensor(x_w),
            'y_w':    torch.tensor(y_w),                                   # Oracle 未来天气
            'x_mark': torch.tensor(np.concatenate([x_cal, x_w], -1)),
            'y_mark': torch.tensor(np.concatenate([y_cal, y_w], -1)),      # 含 Oracle 天气
            'y_hour': torch.tensor(self.hour[s:s + O]),
            'y_y_raw': torch.tensor(self.Yraw[s:s + O]),
        }


def _valid_starts(n, L, O, lo, hi, stride):
    lo = max(lo, L); hi = min(hi, n - O + 1)
    return np.arange(lo, hi, stride, dtype=np.int64)


def build_dataloaders(cfg):
    dc = cfg['data']
    df = pd.read_excel(dc['path'])
    df['Date'] = pd.to_datetime(df['Date']); df = df.sort_values('Date').reset_index(drop=True)
    for c in TARGETS + WEATHER:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df[TARGETS + WEATHER] = df[TARGETS + WEATHER].interpolate(limit_direction='both').ffill().bfill()

    n = len(df); L, O = dc['input_len'], dc['pred_len']
    tr = int(n * dc['train_ratio']); va = int(n * (dc['train_ratio'] + dc['val_ratio']))

    sy = StandardScaler().fit(df[TARGETS].values[:tr])       # 仅训练段 fit
    sw = StandardScaler().fit(df[WEATHER].values[:tr])

    st_tr = _valid_starts(n, L, O, L,  tr - O + 1, dc.get('stride', 1))
    st_va = _valid_starts(n, L, O, tr, va - O + 1, dc.get('stride', 1))
    st_te = _valid_starts(n, L, O, va, n - O + 1, dc.get('test_stride', 1))

    def mk(starts, shuf):
        ds = IESJointDataset(df, starts, L, O, sy, sw)
        return DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=shuf,
                          num_workers=dc.get('num_workers', 0), drop_last=False)

    return {'train': mk(st_tr, True), 'val': mk(st_va, False), 'test': mk(st_te, False),
            'scaler_y': sy, 'scaler_w': sw, 'n_weather': len(WEATHER)}


def inverse_y(arr_std, sy):
    """arr_std: (...,4) 标准化 → 原始 kW。"""
    shp = arr_std.shape
    return sy.inverse_transform(arr_std.reshape(-1, shp[-1])).reshape(shp)
