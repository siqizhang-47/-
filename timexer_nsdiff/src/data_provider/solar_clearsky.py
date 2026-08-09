"""Deterministic clear-sky GHI for Tucson, AZ (design document, 1.2 / 2).

ClearskyGHI is a *solar-geometry* exogenous variable: it is exactly known for
any future timestamp, unlike the measured GHI which would need a forecast.  It
gives the model a physical envelope for PV and lets the attention map separate
"where the sun is" from "how cloudy it is".

Primary backend is pvlib (Ineichen/Perez with the Linke turbidity climatology).
If pvlib is unavailable, a Haurwitz clear-sky model is used instead -- it needs
only the solar zenith angle, which is computed here from the NOAA solar
position algorithm, so the fallback has no third-party dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# University of Arizona main campus, Tucson
TUCSON_LAT = 32.2319
TUCSON_LON = -110.9501
TUCSON_ALT = 728.0  # metres
TUCSON_TZ = "Etc/GMT+7"  # MST, Arizona does not observe DST


def clearsky_ghi(timestamps: pd.Series, backend: str = "auto", verbose: bool = True) -> np.ndarray:
    """Clear-sky GHI in W/m^2 aligned with `timestamps` (naive local MST)."""
    if backend in ("auto", "pvlib"):
        try:
            values = _clearsky_pvlib(timestamps)
            if verbose:
                print(f"[clearsky] backend=pvlib | max={values.max():.1f} W/m^2")
            return values
        except Exception as exc:  # pragma: no cover - environment dependent
            if backend == "pvlib":
                raise
            if verbose:
                print(f"[clearsky] pvlib unavailable ({exc}); using the Haurwitz fallback")
    values = _clearsky_haurwitz(timestamps)
    if verbose:
        print(f"[clearsky] backend=haurwitz | max={values.max():.1f} W/m^2")
    return values


def _clearsky_pvlib(timestamps: pd.Series) -> np.ndarray:
    import pvlib

    idx = pd.DatetimeIndex(timestamps).tz_localize(TUCSON_TZ)
    location = pvlib.location.Location(TUCSON_LAT, TUCSON_LON, tz=TUCSON_TZ, altitude=TUCSON_ALT)
    # Hourly irradiance is an interval average: evaluate at the interval centre.
    cs = location.get_clearsky(idx + pd.Timedelta(minutes=30), model="ineichen")
    return cs["ghi"].to_numpy(dtype=np.float32)


def _solar_zenith_cos(timestamps: pd.Series) -> np.ndarray:
    """cos(solar zenith) via the NOAA solar-position equations (MST, +30 min centring)."""
    ts = pd.DatetimeIndex(timestamps) + pd.Timedelta(minutes=30)
    doy = ts.dayofyear.to_numpy(dtype=np.float64)
    hour = ts.hour.to_numpy(dtype=np.float64) + ts.minute.to_numpy(dtype=np.float64) / 60.0

    gamma = 2.0 * np.pi / 365.0 * (doy - 1.0 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))

    # MST is UTC-7, i.e. a standard meridian of -105 degrees.
    time_offset = eqtime + 4.0 * (TUCSON_LON - (-105.0))
    tst = hour * 60.0 + time_offset
    ha = np.deg2rad(tst / 4.0 - 180.0)

    lat = np.deg2rad(TUCSON_LAT)
    cos_zen = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    return np.clip(cos_zen, 0.0, 1.0)


def _clearsky_haurwitz(timestamps: pd.Series) -> np.ndarray:
    cos_zen = _solar_zenith_cos(timestamps)
    ghi = np.zeros_like(cos_zen)
    day = cos_zen > 0.0
    ghi[day] = 1098.0 * cos_zen[day] * np.exp(-0.059 / cos_zen[day])
    return ghi.astype(np.float32)
