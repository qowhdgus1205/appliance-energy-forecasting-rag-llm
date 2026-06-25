#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
FIG_DIR = BASE_DIR / "outputs" / "figures"

os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["server_time"])


def summarize_file(path: Path) -> dict[str, object]:
    df = load_table(path)
    y = df["inst_heat"]
    return {
        "file_name": path.name,
        "rows": len(df),
        "start_time": df["server_time"].min(),
        "end_time": df["server_time"].max(),
        "mean": y.mean(),
        "std": y.std(),
        "min": y.min(),
        "max": y.max(),
        "p05": y.quantile(0.05),
        "p50": y.quantile(0.50),
        "p95": y.quantile(0.95),
        "zero_ratio": (y == 0).mean(),
        "missing_ratio": y.isna().mean(),
    }


def make_plots(df: pd.DataFrame, file_name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    y = df["inst_heat"]
    t = df["server_time"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(t, y, color="#1f77b4", linewidth=1)
    axes[0].set_title(f"{file_name} - inst_heat over time")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("inst_heat")

    axes[1].hist(y, bins=50, color="#4C78A8", edgecolor="white")
    axes[1].set_title("inst_heat distribution")
    axes[1].set_xlabel("inst_heat")
    axes[1].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{Path(file_name).stem}_inst_heat_overview.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile inst_heat across Industrial Energy tables.")
    parser.add_argument(
        "--files",
        nargs="*",
        default=["facility_b.csv", "facility_c.csv", "facility_a.csv", "facility_d.csv"],
        help="CSV files under data/processed to profile.",
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for file_name in args.files:
        path = PROCESSED_DIR / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        df = load_table(path)
        rows.append(summarize_file(path))
        make_plots(df, file_name)

    report_df = pd.DataFrame(rows)
    report_df.to_csv(REPORT_DIR / "lg_inst_heat_profile.csv", index=False)

    md_lines = [
        "# Industrial Energy inst_heat profile",
        "",
        "| file | rows | start_time | end_time | mean | std | min | max | p05 | p50 | p95 | zero_ratio | missing_ratio |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            "| {file_name} | {rows} | {start_time} | {end_time} | {mean:.2f} | {std:.2f} | {min:.2f} | {max:.2f} | {p05:.2f} | {p50:.2f} | {p95:.2f} | {zero_ratio:.4f} | {missing_ratio:.4f} |".format(
                **row
            )
        )
    md_lines += [
        "",
        "## Notes",
        "",
        "- `inst_heat` is heavily zero-inflated in some files, especially `facility_c.csv` and the site tables.",
        "- the first baseline should probably treat long zero stretches carefully.",
        "- plots are saved under `outputs/figures/`.",
        "",
    ]
    (REPORT_DIR / "lg_inst_heat_profile.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
