#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "outputs" / "reports"
FIG_DIR = BASE_DIR / "outputs" / "figures"


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["server_time"])
    required = {"server_time", "inst_heat", "prediction", "residual", "abs_error", "pred_anomaly"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def extract_episodes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    episode_id = 0
    current = []

    for _, row in df.iterrows():
        if int(row["pred_anomaly"]) == 1:
            current.append(row)
        else:
            if current:
                episode_id += 1
                block = pd.DataFrame(current)
                rows.append(
                    {
                        "episode_id": episode_id,
                        "start_time": block["server_time"].iloc[0],
                        "end_time": block["server_time"].iloc[-1],
                        "rows": len(block),
                        "mean_actual": block["inst_heat"].mean(),
                        "mean_prediction": block["prediction"].mean(),
                        "mean_residual": block["residual"].mean(),
                        "mean_abs_error": block["abs_error"].mean(),
                        "max_abs_error": block["abs_error"].max(),
                    }
                )
                current = []

    if current:
        episode_id += 1
        block = pd.DataFrame(current)
        rows.append(
            {
                "episode_id": episode_id,
                "start_time": block["server_time"].iloc[0],
                "end_time": block["server_time"].iloc[-1],
                "rows": len(block),
                "mean_actual": block["inst_heat"].mean(),
                "mean_prediction": block["prediction"].mean(),
                "mean_residual": block["residual"].mean(),
                "mean_abs_error": block["abs_error"].mean(),
                "max_abs_error": block["abs_error"].max(),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_abs_error", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize inst_heat residual episodes.")
    parser.add_argument(
        "--predictions",
        default="outputs/reports/facility_a_inst_heat_baseline_predictions.csv",
        help="Prediction CSV from the baseline model.",
    )
    args = parser.parse_args()

    pred_path = BASE_DIR / args.predictions if not Path(args.predictions).is_absolute() else Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    df = load_predictions(pred_path)
    episodes = extract_episodes(df)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORT_DIR / f"{pred_path.stem}_episodes.csv"
    out_md = REPORT_DIR / f"{pred_path.stem}_episodes.md"

    episodes.to_csv(out_csv, index=False)

    lines = [
        "# Industrial Energy inst_heat residual episode summary",
        "",
        f"- source: `{pred_path.name}`",
        f"- pred_anomaly ratio: {df['pred_anomaly'].mean():.4f}",
        f"- residual threshold: {df['abs_error'].quantile(0.95):.4f}",
        "",
        "| episode_id | start_time | end_time | rows | mean_actual | mean_prediction | mean_residual | mean_abs_error | max_abs_error |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in episodes.head(20).iterrows():
        lines.append(
            f"| {int(row['episode_id'])} | {row['start_time']} | {row['end_time']} | {int(row['rows'])} | "
            f"{row['mean_actual']:.2f} | {row['mean_prediction']:.2f} | {row['mean_residual']:.2f} | "
            f"{row['mean_abs_error']:.2f} | {row['max_abs_error']:.2f} |"
        )
    lines += ["", "## Notes", "", "- this is a first residual-based episode grouping without domain labels.", ""]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(episodes.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
