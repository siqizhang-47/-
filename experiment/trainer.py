"""Unified training / evaluation loop for the three baselines."""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data.event import classify_windows
from .data.normalize import LoadNormalizer
from .metrics import evaluate_all


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 3
    grad_clip: float = 5.0
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    log_every: int = 200


def _forward(model: nn.Module, batch, device: str):
    x_enc, x_mark, y, y_mark, ev_pred, user_idx = batch
    x_enc = x_enc.to(device, non_blocking=True)
    x_mark = x_mark.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    y_mark = y_mark.to(device, non_blocking=True)
    label_len = x_enc.shape[1] // 2
    dec_inp = torch.zeros_like(y[:, :, :]).to(device)
    dec_inp = torch.cat([x_enc[:, -label_len:, :], dec_inp], dim=1)
    dec_mark = torch.cat([x_mark[:, -label_len:, :], y_mark], dim=1)
    out = model(x_enc, x_mark, dec_inp, dec_mark)
    if isinstance(out, tuple):
        out = out[0]
    # MS convention -> target is the last channel; horizon = last pred_len steps
    pred_len = y.shape[1]
    pred = out[:, -pred_len:, -1:]
    target = y[:, :, -1:]
    return pred, target, ev_pred, user_idx


def train_one_epoch(model, loader, optimizer, device, grad_clip):
    model.train()
    crit = nn.MSELoss()
    total = 0.0
    n = 0
    for batch in loader:
        optimizer.zero_grad()
        pred, target, _, _ = _forward(model, batch, device)
        loss = crit(pred, target)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = pred.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    crit = nn.MSELoss()
    total = 0.0
    n = 0
    for batch in loader:
        pred, target, _, _ = _forward(model, batch, device)
        bs = pred.size(0)
        total += crit(pred, target).item() * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def collect_predictions(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, yhats, evs, uids = [], [], [], []
    for batch in loader:
        pred, target, ev_pred, user_idx = _forward(model, batch, device)
        ys.append(target.squeeze(-1).cpu().numpy())
        yhats.append(pred.squeeze(-1).cpu().numpy())
        evs.append(ev_pred.cpu().numpy())
        uids.append(user_idx.cpu().numpy())
    return (
        np.concatenate(ys, axis=0),
        np.concatenate(yhats, axis=0),
        np.concatenate(evs, axis=0),
        np.concatenate(uids, axis=0),
    )


def inverse_normalize(values_norm: np.ndarray, user_ids: np.ndarray,
                       idx_to_user: Dict[int, str], load_norm: LoadNormalizer) -> np.ndarray:
    """Per-window inverse z-score using each window's user statistics."""
    out = np.empty_like(values_norm)
    for i, uidx in enumerate(user_ids):
        uid = idx_to_user[int(uidx)]
        out[i] = load_norm.inverse(uid, values_norm[i])
    return out


def fit_and_evaluate(
    model: nn.Module,
    datasets,
    cfg: TrainConfig,
    idx_to_user: Dict[int, str],
    load_norm: LoadNormalizer,
) -> Dict:
    device = cfg.device if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    common_dl = dict(num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    train_loader = DataLoader(datasets["train"], batch_size=cfg.batch_size, shuffle=True, drop_last=True, **common_dl)
    val_loader = DataLoader(datasets["val"], batch_size=cfg.batch_size, shuffle=False, **common_dl)
    test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size, shuffle=False, **common_dl)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = math.inf
    best_state = None
    patience = cfg.patience
    bad = 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg.grad_clip)
        val_loss = eval_loss(model, val_loader, device)
        dt = time.time() - t0
        print(f"[epoch {epoch:02d}] train={train_loss:.4f} val={val_loss:.4f} ({dt:.1f}s)", flush=True)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"early-stopping at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_norm, yhat_norm, events_pred, user_ids = collect_predictions(model, test_loader, device)
    y_real = inverse_normalize(y_norm, user_ids, idx_to_user, load_norm)
    yhat_real = inverse_normalize(yhat_norm, user_ids, idx_to_user, load_norm)
    masks = classify_windows(events_pred)
    metrics = evaluate_all(y_real, yhat_real, masks)

    n_windows = y_real.shape[0]
    counts = {
        "n_total_windows": int(n_windows),
        "n_event_windows": int(masks["event"].sum()),
        "n_normal_windows": int(masks["normal"].sum()),
        "n_temp_windows": int(masks["temp"].sum()),
        "n_wind_windows": int(masks["wind"].sum()),
        "n_typhoon_windows": int(masks["typhoon"].sum()),
    }
    return {"metrics": metrics, "counts": counts, "best_val_loss": best_val}
