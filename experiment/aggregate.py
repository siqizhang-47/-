"""Collect ``results/*.json`` runs and emit the spec tables as CSV."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List


MAIN_COLUMNS = [
    "Model", "Overall MAE", "Overall RMSE", "Normal MAE",
    "Event MAE", "Event RMSE", "Event sMAPE",
    "P95AE (event)", "Peak MAE (event)",
]

FAMILY_COLUMNS = [
    "Model",
    "Temp/Humidity MAE", "Wind/Gust MAE", "Typhoon MAE",
    "Temp/Humidity RMSE", "Wind/Gust RMSE", "Typhoon RMSE",
]


def load_runs(results_dir: Path) -> List[dict]:
    runs = []
    for f in sorted(results_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            runs.append({"path": f.name, **json.load(fh)})
    return runs


def write_csv(rows, header, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(h, "") for h in header])


def main(results_dir: str = "results") -> None:
    rdir = Path(results_dir)
    runs = load_runs(rdir)
    if not runs:
        print(f"no runs in {rdir}", file=sys.stderr)
        return

    main_rows = []
    family_rows = []
    for run in runs:
        args = run["args"]
        m = run["metrics"]
        tag = f"{args['model']} ({args['feature_set']}, H={args['pred_len']})"
        main_rows.append({
            "Model": tag,
            "Overall MAE": f"{m['overall_MAE']:.4f}",
            "Overall RMSE": f"{m['overall_RMSE']:.4f}",
            "Normal MAE": f"{m['normal_MAE']:.4f}",
            "Event MAE": f"{m['event_MAE']:.4f}",
            "Event RMSE": f"{m['event_RMSE']:.4f}",
            "Event sMAPE": f"{m['event_sMAPE']:.2f}",
            "P95AE (event)": f"{m['event_P95AE']:.4f}",
            "Peak MAE (event)": f"{m['event_MAE_peak']:.4f}",
        })
        family_rows.append({
            "Model": tag,
            "Temp/Humidity MAE": f"{m['temp_MAE']:.4f}",
            "Wind/Gust MAE": f"{m['wind_MAE']:.4f}",
            "Typhoon MAE": f"{m['typhoon_MAE']:.4f}",
            "Temp/Humidity RMSE": f"{m['temp_RMSE']:.4f}",
            "Wind/Gust RMSE": f"{m['wind_RMSE']:.4f}",
            "Typhoon RMSE": f"{m['typhoon_RMSE']:.4f}",
        })

    write_csv(main_rows, MAIN_COLUMNS, rdir / "summary_main.csv")
    write_csv(family_rows, FAMILY_COLUMNS, rdir / "summary_family.csv")
    print(f"wrote {rdir/'summary_main.csv'} and {rdir/'summary_family.csv'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
