#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "outputs" / "reports"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["server_time"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact incident record for facility_a.")
    parser.add_argument(
        "--predictions",
        default="outputs/reports/facility_a_inst_heat_baseline_predictions.csv",
        help="Prediction CSV from baseline model.",
    )
    parser.add_argument(
        "--episodes",
        default="outputs/reports/facility_a_inst_heat_baseline_predictions_episodes.csv",
        help="Episode summary CSV.",
    )
    parser.add_argument(
        "--importance",
        default="outputs/reports/facility_a_inst_heat_feature_importance.csv",
        help="Feature importance CSV.",
    )
    args = parser.parse_args()

    pred_path = BASE_DIR / args.predictions if not Path(args.predictions).is_absolute() else Path(args.predictions)
    ep_path = BASE_DIR / args.episodes if not Path(args.episodes).is_absolute() else Path(args.episodes)
    imp_path = BASE_DIR / args.importance if not Path(args.importance).is_absolute() else Path(args.importance)

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")
    if not ep_path.exists():
        raise FileNotFoundError(f"Missing episode file: {ep_path}")
    if not imp_path.exists():
        raise FileNotFoundError(f"Missing importance file: {imp_path}")

    pred = load_csv(pred_path)
    episodes = pd.read_csv(ep_path)
    importance = pd.read_csv(imp_path)
    prefix = pred_path.stem.replace("_predictions", "")

    top_episode = episodes.iloc[0].to_dict() if len(episodes) else {}
    top_features = importance.head(10)[["feature", "importance_mean"]].to_dict(orient="records")

    record = {
        "site": "facility_a",
        "target": "inst_heat",
        "prediction_file": pred_path.name,
        "episode_file": ep_path.name,
        "importance_file": imp_path.name,
        "data_window": {
            "start_time": str(pred["server_time"].min()),
            "end_time": str(pred["server_time"].max()),
            "rows": int(len(pred)),
        },
        "residual_threshold": float(pred["abs_error"].quantile(0.95)),
        "pred_anomaly_ratio": float(pred["pred_anomaly"].mean()),
        "top_episode": top_episode,
        "top_features": top_features,
        "analysis_note": (
            "This is a first baseline residual incident record for facility_a. "
            "It is intended for later RAG and LLM summarization."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / f"{prefix}_incident.json"
    md_path = REPORT_DIR / f"{prefix}_incident.md"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Facility A inst_heat incident record",
        "",
        f"- data window: {record['data_window']['start_time']} -> {record['data_window']['end_time']}",
        f"- rows: {record['data_window']['rows']}",
        f"- residual threshold: {record['residual_threshold']:.4f}",
        f"- predicted anomaly ratio: {record['pred_anomaly_ratio']:.4f}",
        "",
        "## Top episode",
        "",
        "| episode_id | start_time | end_time | rows | mean_actual | mean_prediction | mean_residual | mean_abs_error | max_abs_error |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if top_episode:
        lines.append(
            "| {episode_id} | {start_time} | {end_time} | {rows} | {mean_actual:.2f} | {mean_prediction:.2f} | {mean_residual:.2f} | {mean_abs_error:.2f} | {max_abs_error:.2f} |".format(
                **top_episode
            )
        )

    lines += [
        "",
        "## Top features",
        "",
        "| rank | feature | importance_mean |",
        "| --- | --- | ---: |",
    ]
    for i, row in enumerate(top_features, start=1):
        lines.append(f"| {i} | {row['feature']} | {row['importance_mean']:.6f} |")

    lines += [
        "",
        "## Note",
        "",
        "This record is compact on purpose. It is meant to be a stable incident summary for later RAG/LLM use.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
