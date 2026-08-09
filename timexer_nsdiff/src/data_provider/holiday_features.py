"""IsWeekend / IsHoliday flags (design document, 1.1 and 5.3).

Holidays come from ``holidays.US(subdiv='AZ')``.  A self-contained fallback
implementation of the US federal calendar is provided so the pipeline still runs
if the ``holidays`` package is not installed -- it is used only as a last
resort and prints a warning, because the design document fixes the Arizona
calendar as the reference.

``IsBreak`` (University of Arizona academic calendar) is left as an explicit
extension hook: the design document lists it as the highest-priority future
feature but excludes it from the first version.
"""
from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd


class HolidayFeatureBuilder:
    def __init__(self, subdiv: str = "AZ", country: str = "US", use_break: bool = False,
                 break_ranges=None, verbose: bool = True):
        self.subdiv = subdiv
        self.country = country
        self.use_break = use_break
        self.break_ranges = break_ranges or []
        self.verbose = verbose
        self._backend = None
        self._cal = None

    # ------------------------------------------------------------------ holidays
    def _build_calendar(self, years):
        try:
            import holidays  # noqa: F401
            self._cal = holidays.country_holidays(self.country, subdiv=self.subdiv, years=list(years))
            self._backend = "holidays"
        except Exception as exc:  # pragma: no cover - environment dependent
            if self.verbose:
                print(f"[holiday] `holidays` package unavailable ({exc}); "
                      f"falling back to the built-in US federal calendar")
            self._cal = _federal_us_holidays(years)
            self._backend = "builtin"

    def is_holiday(self, timestamps: pd.Series) -> np.ndarray:
        years = sorted(timestamps.dt.year.unique().tolist())
        if self._cal is None:
            self._build_calendar(years)
        dates = timestamps.dt.date
        flags = np.fromiter((d in self._cal for d in dates), dtype=np.int8, count=len(dates))
        if self.verbose:
            n_days = len(set(d for d in dates if d in self._cal))
            print(f"[holiday] backend={self._backend} | {n_days} holiday days "
                  f"over {years[0]}-{years[-1]} | {int(flags.sum())} flagged hours")
        return flags.astype(np.float32)

    # ------------------------------------------------------------------ weekend
    @staticmethod
    def is_weekend(weekday: np.ndarray) -> np.ndarray:
        """Weekday follows the Monday=0 convention (verified: 2014-01-01 -> 2 = Wednesday)."""
        return np.isin(weekday, [5, 6]).astype(np.float32)

    # ------------------------------------------------------------------ IsBreak hook
    def is_break(self, timestamps: pd.Series) -> np.ndarray:
        """UA academic-break flag. Disabled by default (see design document 1.1)."""
        flags = np.zeros(len(timestamps), dtype=np.float32)
        if not self.use_break:
            return flags
        dates = timestamps.dt.date
        for start, end in self.break_ranges:
            start = pd.to_datetime(start).date()
            end = pd.to_datetime(end).date()
            flags[np.array([(start <= d <= end) for d in dates])] = 1.0
        return flags


# ---------------------------------------------------------------------- fallback
def _nth_weekday(year, month, weekday, n):
    d = dt.date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year, month, weekday):
    if month == 12:
        d = dt.date(year + 1, 1, 1) - dt.timedelta(days=1)
    else:
        d = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d):
    """Federal observation rule: Saturday -> Friday, Sunday -> Monday."""
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def _federal_us_holidays(years):
    """Minimal stand-in for holidays.US(subdiv='AZ'), 2014-2022."""
    out = set()
    for y in years:
        out.add(_observed(dt.date(y, 1, 1)))                  # New Year's Day
        out.add(_nth_weekday(y, 1, 0, 3))                     # MLK Day
        out.add(_nth_weekday(y, 2, 0, 3))                     # Washington's Birthday
        out.add(_last_weekday(y, 5, 0))                       # Memorial Day
        if y >= 2021:
            out.add(_observed(dt.date(y, 6, 19)))             # Juneteenth
        out.add(_observed(dt.date(y, 7, 4)))                  # Independence Day
        out.add(_nth_weekday(y, 9, 0, 1))                     # Labor Day
        out.add(_nth_weekday(y, 10, 0, 2))                    # Columbus Day
        out.add(_observed(dt.date(y, 11, 11)))                # Veterans Day
        out.add(_nth_weekday(y, 11, 3, 4))                    # Thanksgiving
        out.add(_observed(dt.date(y, 12, 25)))                # Christmas
    return out
