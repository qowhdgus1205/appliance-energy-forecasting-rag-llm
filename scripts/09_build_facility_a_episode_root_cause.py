#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["server_time"])


def extract_episodes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    episode_id = 0
    current = []
    start_idx = None

    for idx, row in df.iterrows():
        if int(row["pred_anomaly"]) == 1:
            if start_idx is None:
                start_idx = idx
            current.append(row)
        else:
            if current:
                episode_id += 1
                block = pd.DataFrame(current)
                rows.append(
                    {
                        "episode_id": episode_id,
                        "start_idx": start_idx,
                        "end_idx": idx - 1,
                        "start_time": block["server_time"].iloc[0],
                        "end_time": block["server_time"].iloc[-1],
                        "rows": len(block),
                        "mean_actual": block["inst_heat"].mean(),
                        "mean_prediction": block["prediction"].mean(),
                        "mean_target_future": block["target_future"].mean(),
                        "mean_residual": block["residual"].mean(),
                        "mean_abs_error": block["abs_error"].mean(),
                        "max_abs_error": block["abs_error"].max(),
                    }
                )
                current = []
                start_idx = None

    if current:
        episode_id += 1
        block = pd.DataFrame(current)
        rows.append(
            {
                "episode_id": episode_id,
                "start_idx": start_idx,
                "end_idx": len(df) - 1,
                "start_time": block["server_time"].iloc[0],
                "end_time": block["server_time"].iloc[-1],
                "rows": len(block),
                "mean_actual": block["inst_heat"].mean(),
                "mean_prediction": block["prediction"].mean(),
                "mean_target_future": block["target_future"].mean(),
                "mean_residual": block["residual"].mean(),
                "mean_abs_error": block["abs_error"].mean(),
                "max_abs_error": block["abs_error"].max(),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_abs_error", ascending=False).reset_index(drop=True)


def compare_windows(df: pd.DataFrame, start_idx: int, end_idx: int, window: int | None = None) -> pd.DataFrame:
    episode = df.iloc[start_idx : end_idx + 1]
    if window is None:
        window = len(episode)
    prev_end = start_idx
    prev_start = max(0, prev_end - window)
    baseline = df.iloc[prev_start:prev_end]
    if baseline.empty:
        return pd.DataFrame()

    numeric_cols = [
        c
        for c in df.columns
        if c not in {"server_time", "target_future", "prediction", "residual", "abs_error", "pred_anomaly"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    rows = []
    for col in numeric_cols:
        ep_mean = episode[col].mean()
        base_mean = baseline[col].mean()
        delta = ep_mean - base_mean
        base_std = baseline[col].std()
        z_delta = delta / base_std if pd.notna(base_std) and base_std > 0 else 0.0
        rows.append(
            {
                "feature": col,
                "episode_mean": ep_mean,
                "baseline_mean": base_mean,
                "delta": delta,
                "z_delta": z_delta,
            }
        )
    return pd.DataFrame(rows).sort_values("z_delta", key=lambda s: s.abs(), ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build root-cause summaries for facility_a forecast episodes.")
    parser.add_argument(
        "--predictions",
        default="outputs/reports/facility_a_inst_heat_exogenous_predictions.csv",
        help="Prediction CSV from forecasting model.",
    )
    parser.add_argument(
        "--processed",
        default="data/processed/facility_a.csv",
        help="Processed site table used to recover all features.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of episodes to summarize.")
    args = parser.parse_args()

    pred_path = BASE_DIR / args.predictions if not Path(args.predictions).is_absolute() else Path(args.predictions)
    proc_path = BASE_DIR / args.processed if not Path(args.processed).is_absolute() else Path(args.processed)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")
    if not proc_path.exists():
        raise FileNotFoundError(f"Missing processed file: {proc_path}")

    pred = load_table(pred_path)
    proc = load_table(proc_path).sort_values("server_time").reset_index(drop=True)
    merged = pred.merge(proc, on="server_time", how="left", suffixes=("", "_proc"))
    episodes = extract_episodes(merged)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORT_DIR / f"{pred_path.stem}_root_cause.csv"
    out_md = REPORT_DIR / f"{pred_path.stem}_root_cause.md"
    out_json = REPORT_DIR / f"{pred_path.stem}_root_cause.json"

    episode_payload = []
    md_lines = [
        "# Facility A inst_heat forecast episode root-cause summary",
        "",
        f"- source: `{pred_path.name}`",
        f"- top-k episodes: {args.top_k}",
        f"- residual threshold: {pred['abs_error'].quantile(0.95):.4f}",
        "",
    ]

    for _, ep in episodes.head(args.top_k).iterrows():
        shifts = compare_windows(merged, int(ep["start_idx"]), int(ep["end_idx"]), int(ep["rows"]))
        if shifts.empty:
            continue
        shifts = shifts.head(8).copy()
        episode_payload.append(
            {
                "episode": ep.to_dict(),
                "top_feature_shifts": shifts.to_dict(orient="records"),
            }
        )
        md_lines += [
            f"## Episode {int(ep['episode_id'])}",
            "",
            f"- time: {ep['start_time']} -> {ep['end_time']}",
            f"- rows: {int(ep['rows'])}",
            f"- mean actual future inst_heat: {ep['mean_target_future']:.2f}",
            f"- mean prediction: {ep['mean_prediction']:.2f}",
            f"- mean residual: {ep['mean_residual']:.2f}",
            f"- mean abs error: {ep['mean_abs_error']:.2f}",
            "",
            "| feature | episode_mean | baseline_mean | delta | z_delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for _, row in shifts.iterrows():
            md_lines.append(
                f"| {row['feature']} | {row['episode_mean']:.4f} | {row['baseline_mean']:.4f} | {row['delta']:.4f} | {row['z_delta']:.4f} |"
            )
        md_lines.append("")

    flat_rows = []
    for item in episode_payload:
        ep = item["episode"]
        for shift in item["top_feature_shifts"]:
            flat_rows.append(
                {
                    "episode_id": ep["episode_id"],
                    "start_time": ep["start_time"],
                    "end_time": ep["end_time"],
                    "rows": ep["rows"],
                    "mean_target_future": ep["mean_target_future"],
                    "mean_prediction": ep["mean_prediction"],
                    "mean_residual": ep["mean_residual"],
                    "mean_abs_error": ep["mean_abs_error"],
                    "feature": shift["feature"],
                    "episode_mean": shift["episode_mean"],
                    "baseline_mean": shift["baseline_mean"],
                    "delta": shift["delta"],
                    "z_delta": shift["z_delta"],
                }
            )

    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    out_json.write_text(json.dumps(episode_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"saved: {out_md}")


if __name__ == "__main__":
    main()
