#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
FIG_DIR = BASE_DIR / "outputs" / "figures"


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["server_time"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile facility_a operating modes.")
    parser.add_argument("--file", default="facility_a.csv", help="Processed facility_a CSV.")
    args = parser.parse_args()

    path = PROCESSED_DIR / args.file if not Path(args.file).is_absolute() else Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = load_table(path)
    if "opermode" not in df.columns:
        raise ValueError("Expected opermode column in facility_a.csv")

    df["heat_mode"] = df["opermode"].map({1: "gas_like", 0: "heating_like"}).fillna("unknown")
    zero = df["inst_heat"] == 0

    summary = (
        df.groupby("heat_mode")
        .agg(
            rows=("inst_heat", "size"),
            zero_ratio=("inst_heat", lambda s: (s == 0).mean()),
            mean_inst_heat=("inst_heat", "mean"),
            median_inst_heat=("inst_heat", "median"),
            opermode_mean=("opermode", "mean"),
        )
        .reset_index()
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(REPORT_DIR / "facility_a_mode_profile.csv", index=False)

    md_lines = [
        "# Facility A mode profile",
        "",
        "## Interpretation",
        "",
        "- `opermode == 1` is used as the first gas-like proxy.",
        "- `opermode == 0` is used as the first heating-like proxy.",
        "- This mapping is inferred from the observed zero-heat concentration.",
        "",
        "## Summary",
        "",
        "| heat_mode | rows | zero_ratio | mean_inst_heat | median_inst_heat | opermode_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        md_lines.append(
            f"| {row.heat_mode} | {row.rows} | {row.zero_ratio:.4f} | {row.mean_inst_heat:.2f} | {row.median_inst_heat:.2f} | {row.opermode_mean:.4f} |"
        )
    md_lines += ["", "## Notes", "", "- zero heat is concentrated in gas-like mode; mode-specific thresholds are required.", ""]
    (REPORT_DIR / "facility_a_mode_profile.md").write_text("\n".join(md_lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary["heat_mode"], summary["zero_ratio"], color=["#E45756", "#4C78A8"])
    ax.set_title("facility_a zero heat ratio by mode proxy")
    ax.set_xlabel("heat_mode")
    ax.set_ylabel("zero_ratio")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "facility_a_mode_zero_ratio.png", dpi=160)
    plt.close(fig)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

