"""Phase 0: raw xlsx -> cleaned data/processed/energy.csv + cleaning report.

Column order contract (used by the WHOLE repo, targets are indices 0-3 after
`date`, covariates 4-7):
    [date, PV, Electricity, Cooling, Heat, Temperature, DewPoint, Humidity, GHI]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import load_config, repo_root
from utils.seed import set_seed

TARGETS = ["PV", "Electricity", "Cooling", "Heat"]
COVARIATES = ["Temperature", "DewPoint", "Humidity", "GHI"]
COLUMNS = ["date"] + TARGETS + COVARIATES
# columns whose exact-zero values are missing-value markers (GHI zeros are real: night)
ZERO_AS_NAN = ["DewPoint", "Humidity", "Temperature"]
ZERO_MARKER_THRESHOLD = 5.0  # zero counts as a missing marker only if the
                             # neighbor-interpolated estimate is > this far from 0
MAX_GAP_HOURS = 3       # plain time interpolation up to this gap length
MAX_LONG_GAP_HOURS = 12  # longer gaps (manually checked: brief 2014/2015 sensor
                         # outages, longest 8h on 2015-08-31) are interpolated too
                         # but itemized in the report; beyond this -> error


def clean(df: pd.DataFrame):
    report = []
    df = df.rename(columns={"Dew Point": "DewPoint"})
    df["date"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour"]])

    # hourly continuity check
    full_range = pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="h")
    if len(full_range) != len(df) or not (df["date"].values == full_range.values).all():
        raise ValueError("Raw data is not a continuous hourly series")
    report.append(f"- rows: {len(df)}, span: {df['date'].iloc[0]} .. {df['date'].iloc[-1]} (continuous hourly)")

    df = df.set_index("date")
    for col in tqdm(ZERO_AS_NAN, desc="cleaning zero-markers"):
        n_zero = int((df[col] == 0).sum())
        # a zero is a MISSING MARKER only when it is contextually anomalous:
        # the neighbor-interpolated estimate is far from 0. Genuine zero
        # crossings (e.g. dew point ... -2, 0, 2 ... in a cold snap) are kept.
        est = df[col].replace(0, np.nan).interpolate(method="time", limit_area="inside")
        est = est.bfill().ffill()
        marker = (df[col] == 0) & (est.abs() > ZERO_MARKER_THRESHOLD)
        n_genuine = int(((df[col] == 0) & ~marker).sum())
        df.loc[marker, col] = np.nan
        isna = df[col].isna()
        n_nan_before = int(isna.sum())

        # itemize gaps longer than MAX_GAP_HOURS (manual-check record)
        grp = (isna != isna.shift()).cumsum()
        long_gaps = []
        for g, run in df[col][isna].groupby(grp[isna]):
            if len(run) > MAX_LONG_GAP_HOURS:
                raise ValueError(
                    f"{col}: gap of {len(run)}h starting {run.index[0]} exceeds "
                    f"{MAX_LONG_GAP_HOURS}h -> manual check required")
            if len(run) > MAX_GAP_HOURS:
                long_gaps.append(f"{run.index[0]} .. {run.index[-1]} ({len(run)}h)")

        df[col] = df[col].interpolate(method="time", limit_area="inside")
        # series-boundary values cannot be interpolated -> nearest value
        df[col] = df[col].bfill().ffill()
        assert not df[col].isna().any()
        report.append(
            f"- {col}: {n_zero} zeros found, {n_nan_before} classified as missing "
            f"markers -> time-interpolated (boundary values back/forward-filled), "
            f"{n_genuine} kept as genuine zero crossings, 0 NaN remaining"
        )
        if long_gaps:
            report.append(f"  - gaps > {MAX_GAP_HOURS}h (checked, brief sensor "
                          f"outages, interpolated): " + "; ".join(long_gaps))

    out = df.reset_index()[COLUMNS]
    assert not out.drop(columns=["date"]).isna().any().any(), "NaN left after cleaning"
    return out, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(repo_root() / "configs/base.yaml"))
    parser.add_argument("--raw", default=None, help="override raw xlsx path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    root = repo_root()
    raw_path = Path(args.raw) if args.raw else root / cfg.data_raw
    out_path = root / cfg.data_csv
    report_path = root / cfg.results_dir / "data_cleaning_report.md"

    print(f"Reading {raw_path} ...")
    df = pd.read_excel(raw_path)
    out, report = clean(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Data cleaning report\n\n"
        + "\n".join(report)
        + "\n\nColumn order: " + ", ".join(COLUMNS) + "\n"
    )
    print(f"Wrote {out_path} ({len(out)} rows) and {report_path}")


if __name__ == "__main__":
    main()
