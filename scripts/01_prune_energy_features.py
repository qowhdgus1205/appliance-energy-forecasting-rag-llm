#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

KEEP_SUFFIXES = ("_lag3", "_rmean3", "_rstd3", "_delta3")
DROP_EXACT = {"hour", "minute", "day", "month", "weekday"}
DROP_SUFFIXES = (
    "_lag1",
    "_lag5",
    "_delta1",
    "_log1p",
    "_rmean6",
    "_rmean12",
    "_rstd6",
    "_rstd12",
)


def is_feature_column(col: str) -> bool:
    if col in {"server_time", "inst_heat"}:
        return True
    if col in DROP_EXACT:
        return False
    if any(col.endswith(suffix) for suffix in KEEP_SUFFIXES):
        return True
    if any(col.endswith(suffix) for suffix in DROP_SUFFIXES):
        return False
    return True


def prune_table(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    df = pd.read_csv(path, parse_dates=["server_time"])
    keep_cols = [c for c in df.columns if is_feature_column(c)]

    calendar_cols = [c for c in df.columns if c in DROP_EXACT]
    drop_cols = [c for c in df.columns if c not in keep_cols]

    pruned = df[keep_cols].copy()
    stats = {
        "input_cols": df.shape[1],
        "output_cols": pruned.shape[1],
        "dropped_cols": len(drop_cols),
        "kept_lag3": sum(c.endswith("_lag3") for c in keep_cols),
        "kept_rmean3": sum(c.endswith("_rmean3") for c in keep_cols),
        "kept_rstd3": sum(c.endswith("_rstd3") for c in keep_cols),
        "kept_delta3": sum(c.endswith("_delta3") for c in keep_cols),
        "dropped_calendar": len(calendar_cols),
    }
    return pruned, stats


def write_report(rows: Iterable[dict[str, object]], output_path: Path) -> None:
    rows = list(rows)
    lines = [
        "# Industrial Energy feature pruning summary",
        "",
        "## Rule",
        "",
        "- keep `server_time` and `inst_heat`",
        "- keep only engineered features with suffixes `_lag3`, `_rmean3`, `_rstd3`, `_delta3` if they exist",
        "- drop calendar features such as `hour`, `minute`, `day`, `month`, `weekday`",
        "- drop other lag/rolling/log variants such as `_lag1`, `_lag5`, `_delta1`, `_log1p`, `_rmean6`, `_rmean12`, `_rstd6`, `_rstd12`",
        "",
        "## Files",
        "",
        "| file | input_cols | output_cols | dropped_cols | kept_lag3 | kept_rmean3 | kept_rstd3 | kept_delta3 | dropped_calendar |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['file_name']} | {row['input_cols']} | {row['output_cols']} | {row['dropped_cols']} | "
            f"{row['kept_lag3']} | {row['kept_rmean3']} | {row['kept_rstd3']} | {row['kept_delta3']} | {row['dropped_calendar']} |"
        )
    lines += ["", "## Notes", "", "- `delta3` columns are not present in the current CSVs, so that count is 0.", ""]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune Industrial Energy engineered features.")
    parser.add_argument(
        "--files",
        nargs="*",
        default=["facility_b.csv", "facility_c.csv", "facility_a.csv", "facility_d.csv"],
        help="CSV files under data/raw to prune.",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_rows = []
    for file_name in args.files:
        in_path = RAW_DIR / file_name
        out_path = PROCESSED_DIR / file_name
        if not in_path.exists():
            raise FileNotFoundError(f"Missing file: {in_path}")

        pruned, stats = prune_table(in_path)
        pruned.to_csv(out_path, index=False)
        report_rows.append({"file_name": file_name, **stats})
        print(f"{file_name}: {stats['input_cols']} -> {stats['output_cols']} columns")

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(REPORT_DIR / "lg_feature_pruning.csv", index=False)
    write_report(report_rows, REPORT_DIR / "lg_feature_pruning.md")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
