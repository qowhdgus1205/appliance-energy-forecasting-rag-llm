#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "outputs" / "reports"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["server_time"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact RAG context for facility_a.")
    parser.add_argument(
        "--incident",
        default="outputs/reports/facility_a_inst_heat_exogenous_incident.json",
        help="Compact incident JSON.",
    )
    parser.add_argument(
        "--root-cause",
        default="outputs/reports/facility_a_inst_heat_exogenous_predictions_root_cause.json",
        help="Root cause JSON.",
    )
    parser.add_argument(
        "--predictions",
        default="outputs/reports/facility_a_inst_heat_exogenous_predictions.csv",
        help="Forecast predictions CSV.",
    )
    args = parser.parse_args()

    incident_path = BASE_DIR / args.incident if not Path(args.incident).is_absolute() else Path(args.incident)
    root_path = BASE_DIR / args.root_cause if not Path(args.root_cause).is_absolute() else Path(args.root_cause)
    pred_path = BASE_DIR / args.predictions if not Path(args.predictions).is_absolute() else Path(args.predictions)
    if not incident_path.exists():
        raise FileNotFoundError(f"Missing incident file: {incident_path}")
    if not root_path.exists():
        raise FileNotFoundError(f"Missing root cause file: {root_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    incident = load_json(incident_path)
    root = load_json(root_path)
    pred = load_csv(pred_path)

    top_root = root[0] if root else {}
    context = {
        "site": incident["site"],
        "target": incident["target"],
        "horizon": "t+1",
        "data_window": incident["data_window"],
        "forecast_metrics_proxy": {
            "residual_threshold": incident["residual_threshold"],
            "pred_anomaly_ratio": incident["pred_anomaly_ratio"],
        },
        "top_episode": incident["top_episode"],
        "top_features": incident["top_features"],
        "root_cause_top_episode": top_root.get("episode", {}),
        "root_cause_top_feature_shifts": top_root.get("top_feature_shifts", [])[:8],
        "visual_assets": [
            "outputs/figures/facility_a_eda_overview.png",
            "outputs/figures/facility_a_eda_residuals.png",
            "outputs/figures/facility_a_eda_top_episode_zoom.png",
        ],
        "notes": [
            "This context is compact and intended for later RAG / LangChain prompts.",
            "The forecasting target is t+1 inst_heat.",
            "Residual episode summaries are derived from absolute forecast error on the test split.",
        ],
        "test_window_rows": int(len(pred)),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORT_DIR / "facility_a_inst_heat_rag_context.json"
    out_md = REPORT_DIR / "facility_a_inst_heat_rag_context.md"
    out_json.write_text(json.dumps(context, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Facility A inst_heat RAG context",
        "",
        f"- target: `{context['target']}`",
        f"- horizon: `{context['horizon']}`",
        f"- residual threshold: {context['forecast_metrics_proxy']['residual_threshold']:.4f}",
        f"- predicted anomaly ratio: {context['forecast_metrics_proxy']['pred_anomaly_ratio']:.4f}",
        "",
        "## Top episode",
        "",
        f"- time: {context['top_episode']['start_time']} -> {context['top_episode']['end_time']}",
        f"- mean actual future inst_heat: {context['top_episode']['mean_actual']:.2f}",
        f"- mean prediction: {context['top_episode']['mean_prediction']:.2f}",
        f"- mean residual: {context['top_episode']['mean_residual']:.2f}",
        "",
        "## Visual assets",
        "",
    ]
    for item in context["visual_assets"]:
        lines.append(f"- `{item}`")
    lines += [
        "",
        "## Root-cause hints",
        "",
    ]
    for row in context["root_cause_top_feature_shifts"]:
        lines.append(
            f"- {row['feature']}: delta={float(row['delta']):.4f}, z_delta={float(row['z_delta']):.4f}"
        )
    lines += ["", "## Notes", ""]
    for note in context["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(context, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
