#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs" / "reports"


@dataclass
class FileSummary:
    file_name: str
    rows: int
    cols: int
    start_time: str
    end_time: str
    null_ratio: float
    numeric_cols: int
    object_cols: int


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["server_time"])


def summarize_file(path: Path) -> FileSummary:
    df = load_table(path)
    numeric_cols = df.select_dtypes(include="number").shape[1]
    object_cols = df.select_dtypes(exclude="number").shape[1]
    return FileSummary(
        file_name=path.name,
        rows=len(df),
        cols=df.shape[1],
        start_time=str(df["server_time"].min()),
        end_time=str(df["server_time"].max()),
        null_ratio=float(df.isna().mean().mean()),
        numeric_cols=int(numeric_cols),
        object_cols=int(object_cols),
    )


def write_markdown_report(summaries: Iterable[FileSummary], output_path: Path) -> None:
    rows = list(summaries)
    total_rows = sum(r.rows for r in rows)
    total_cols = rows[0].cols if rows else 0
    lines = [
        "# Industrial Energy data scan summary",
        "",
        "## Files",
        "",
        "| file | rows | cols | start_time | end_time | null_ratio | numeric_cols | object_cols |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r.file_name} | {r.rows} | {r.cols} | {r.start_time} | {r.end_time} | "
            f"{r.null_ratio:.4f} | {r.numeric_cols} | {r.object_cols} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        f"- total rows: {total_rows}",
        f"- column count: {total_cols}",
        "- the four tables already contain engineered lag / rolling / calendar features",
        "- first modeling step should focus on target selection and time-based split",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Industrial Energy site tables.")
    parser.add_argument(
        "--files",
        nargs="*",
        default=["facility_b.csv", "facility_c.csv", "facility_a.csv", "facility_d.csv"],
        help="CSV files under data/raw to scan.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for file_name in args.files:
        path = RAW_DIR / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        summaries.append(summarize_file(path))

    report_df = pd.DataFrame([s.__dict__ for s in summaries])
    report_df.to_csv(OUTPUT_DIR / "lg_data_scan.csv", index=False)
    write_markdown_report(summaries, OUTPUT_DIR / "lg_data_scan.md")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
