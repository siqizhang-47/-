"""Classify each prediction window as normal / event, and assign family flags.

Computed once per (user, window) on the *raw* binary labels of the prediction
horizon ``[t, t+H-1]``. Only labels 1-12 participate; labels 13-20 are ignored.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .features import EVENT_FAMILIES


def classify_windows(events_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """events_pred: array [num_windows, pred_len, 12] of 0/1 labels.

    Returns a dict of boolean masks indexed by category.
    """
    assert events_pred.ndim == 3 and events_pred.shape[-1] == 12, events_pred.shape
    any_event = (events_pred.sum(axis=(1, 2)) > 0)
    masks = {
        "event": any_event,
        "normal": ~any_event,
    }
    for family, cols in EVENT_FAMILIES.items():
        idx = [int(c[1:]) - 1 for c in cols]  # e01 -> 0, e04 -> 3, ...
        block = events_pred[:, :, idx]
        masks[family] = (block.sum(axis=(1, 2)) > 0)
    return masks
