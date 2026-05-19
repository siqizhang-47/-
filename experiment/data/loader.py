"""Walk the EWELD_labeled_output tree and build per-user DataFrames."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterator, Tuple

import pandas as pd

from .features import build_user_dataframe


CITY_RE = re.compile(r"^CT\d+$")
INDUSTRY_RE = re.compile(r"^C\d+")
USER_RE = re.compile(r"^U[\w-]+\.csv$", re.IGNORECASE)


def iter_user_csvs(root: str | Path) -> Iterator[Tuple[str, str, str, Path]]:
    """Yield ``(city, industry, user_id, csv_path)`` over a EWELD_labeled_output tree."""
    root = Path(root)
    for city in sorted(os.listdir(root)):
        if not CITY_RE.match(city):
            continue
        city_dir = root / city
        if not city_dir.is_dir():
            continue
        for industry in sorted(os.listdir(city_dir)):
            ind_dir = city_dir / industry
            if not ind_dir.is_dir():
                continue
            for fname in sorted(os.listdir(ind_dir)):
                if not fname.lower().endswith(".csv"):
                    continue
                if not USER_RE.match(fname):
                    continue
                user_id = f"{city}__{industry}__{fname[:-4]}"
                yield city, industry, user_id, ind_dir / fname


def load_all_users(root: str | Path) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Return ``(user_frames, user_city)``.

    Files that fail to parse are silently skipped.
    """
    frames: Dict[str, pd.DataFrame] = {}
    user_city: Dict[str, str] = {}
    for city, _industry, uid, path in iter_user_csvs(root):
        df = build_user_dataframe(path)
        if df is None or df.empty:
            continue
        frames[uid] = df
        user_city[uid] = city
    return frames, user_city
