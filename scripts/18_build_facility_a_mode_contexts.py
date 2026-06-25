#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"


def load_csv(path: Path) -> pd.DataFrame:
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
                        "mean_actual_sum_15": block["actual_sum_15"].mean(),
                        "mean_prediction_sum_15": block["pred_sum_15"].mean(),
                        "mean_residual_sum_15": block["residual_sum_15"].mean(),
                        "mean_abs_error_sum_15": block["abs_error_sum_15"].mean(),
                        "max_abs_error_sum_15": block["abs_error_sum_15"].max(),
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
                "mean_actual_sum_15": block["actual_sum_15"].mean(),
                "mean_prediction_sum_15": block["pred_sum_15"].mean(),
                "mean_residual_sum_15": block["residual_sum_15"].mean(),
                "mean_abs_error_sum_15": block["abs_error_sum_15"].mean(),
                "max_abs_error_sum_15": block["abs_error_sum_15"].max(),
            }
        )
    episodes = pd.DataFrame(rows)
    if episodes.empty:
        return episodes
    return episodes.sort_values("mean_abs_error_sum_15", ascending=False).reset_index(drop=True)


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
        if c not in {"server_time", "pred_anomaly", "actual_sum_15", "pred_sum_15", "residual_sum_15", "abs_error_sum_15"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    rows = []
    for col in numeric_cols:
        ep_mean = episode[col].mean()
        base_mean = baseline[col].mean()
        delta = ep_mean - base_mean
        base_std = baseline[col].std()
        z_delta = delta / base_std if pd.notna(base_std) and base_std > 0 else 0.0
        rows.append({"feature": col, "episode_mean": ep_mean, "baseline_mean": base_mean, "delta": delta, "z_delta": z_delta})
    return pd.DataFrame(rows).sort_values("z_delta", key=lambda s: s.abs(), ascending=False)


def build_mode_context(mode_label: str, mode_value: int, pred: pd.DataFrame, proc: pd.DataFrame, feature_importance: pd.DataFrame, suite_summary: pd.DataFrame, cost_inputs: dict) -> dict:
    merged = pred.merge(proc, on="server_time", how="left", suffixes=("", "_proc"))
    mode_df = merged[merged["opermode"] == mode_value].reset_index(drop=True)
    if mode_df.empty:
        raise ValueError(f"No rows found for mode {mode_label}")

    threshold = float(mode_df["abs_error_sum_15"].quantile(0.95))
    mode_df = mode_df.copy()
    mode_df["pred_anomaly"] = (mode_df["abs_error_sum_15"] > threshold).astype(int)
    episodes = extract_episodes(mode_df)
    top_episode = episodes.iloc[0].to_dict() if len(episodes) else {}
    root_shifts = compare_windows(mode_df, int(top_episode["start_idx"]), int(top_episode["end_idx"])) if top_episode else pd.DataFrame()
    top_features = feature_importance.head(8)[["feature", "importance"]].to_dict(orient="records") if not feature_importance.empty else []

    context = {
        "site": "facility_a",
        "mode_label": mode_label,
        "mode_proxy": "opermode",
        "mode_value": mode_value,
        "mode_interpretation": "gas_like" if mode_value == 1 else "heating_like",
        "target": "inst_heat",
        "forecast_window": "t+1..t+15",
        "representative_horizon": "t+15",
        "model_scope": "shared_multioutput_forecast",
        "data_window": {
            "start_time": str(mode_df["server_time"].min()),
            "end_time": str(mode_df["server_time"].max()),
            "rows": int(len(mode_df)),
        },
        "forecast_metrics_proxy": {
            "residual_threshold": threshold,
            "pred_anomaly_ratio": float(mode_df["pred_anomaly"].mean()),
        },
        "top_episode": top_episode,
        "top_features": top_features,
        "root_cause_top_feature_shifts": root_shifts.head(8).to_dict(orient="records") if not root_shifts.empty else [],
        "mode_notes": [
            "The forecasting model predicts the next 15 inst_heat values in one shot.",
            "Use opermode as the first gas/electric proxy when interpreting residuals.",
            "A zero inst_heat value is not automatically abnormal in gas-like mode.",
            "Residual thresholds are mode-aware and should be compared separately by operating mode.",
        ],
        "multioutput_suite_summary": suite_summary.to_dict(orient="records"),
        "horizon_metrics": suite_summary.to_dict(orient="records"),
        "cost_inputs": cost_inputs,
    }
    return context


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mode-aware Industrial Energy facility_a contexts from the multi-output forecast.")
    parser.add_argument(
        "--predictions",
        default="outputs/reports/facility_a_inst_heat_multioutput_predictions.csv",
        help="Multi-output forecast predictions CSV.",
    )
    parser.add_argument(
        "--processed",
        default="data/processed/facility_a.csv",
        help="Processed facility_a CSV.",
    )
    parser.add_argument(
        "--importance",
        default="outputs/reports/facility_a_inst_heat_multioutput_feature_importance.csv",
        help="Feature importance CSV.",
    )
    parser.add_argument(
        "--suite-summary",
        default="outputs/reports/facility_a_inst_heat_multioutput_summary.json",
        help="Multi-output suite summary JSON.",
    )
    args = parser.parse_args()

    pred_path = BASE_DIR / args.predictions if not Path(args.predictions).is_absolute() else Path(args.predictions)
    proc_path = BASE_DIR / args.processed if not Path(args.processed).is_absolute() else Path(args.processed)
    imp_path = BASE_DIR / args.importance if not Path(args.importance).is_absolute() else Path(args.importance)
    suite_summary_path = BASE_DIR / args.suite_summary if not Path(args.suite_summary).is_absolute() else Path(args.suite_summary)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not proc_path.exists():
        raise FileNotFoundError(proc_path)
    if not imp_path.exists():
        raise FileNotFoundError(imp_path)
    if not suite_summary_path.exists():
        raise FileNotFoundError(suite_summary_path)

    pred = load_csv(pred_path)
    proc = load_csv(proc_path).sort_values("server_time").reset_index(drop=True)
    importance = pd.read_csv(imp_path)
    suite_summary = json.loads(suite_summary_path.read_text(encoding="utf-8"))
    suite_rows = pd.DataFrame(suite_summary.get("horizon_metrics", []))
    cost_inputs = suite_summary.get("cost_inputs")
    if cost_inputs is None:
        cost_inputs = {
            "electric_rate": suite_summary.get("electric_rate"),
            "gas_rate": suite_summary.get("gas_rate"),
        }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for mode_label, mode_value in [("gas_like", 1), ("heating_like", 0)]:
        context = build_mode_context(mode_label, mode_value, pred, proc, importance, suite_rows, cost_inputs)
        json_path = REPORT_DIR / f"facility_a_{mode_label}_rag_context.json"
        md_path = REPORT_DIR / f"facility_a_{mode_label}_rag_context.md"
        json_path.write_text(json.dumps(context, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        lines = [
            f"# Facility A {mode_label} RAG context",
            "",
            f"- mode_proxy: `{context['mode_proxy']}`",
            f"- mode_value: `{mode_value}`",
            f"- interpretation: `{context['mode_interpretation']}`",
            f"- model_scope: `{context['model_scope']}`",
            f"- forecast_window: `{context['forecast_window']}`",
            f"- representative_horizon: `{context['representative_horizon']}`",
            f"- residual_threshold: {context['forecast_metrics_proxy']['residual_threshold']:.4f}",
            f"- predicted_anomaly_ratio: {context['forecast_metrics_proxy']['pred_anomaly_ratio']:.4f}",
            "",
            "## Top episode",
            "",
            f"- time: {context['top_episode'].get('start_time')} -> {context['top_episode'].get('end_time')}",
            f"- mean actual sum 15: {float(context['top_episode'].get('mean_actual_sum_15', 0)):.2f}",
            f"- mean prediction sum 15: {float(context['top_episode'].get('mean_prediction_sum_15', 0)):.2f}",
            f"- mean residual sum 15: {float(context['top_episode'].get('mean_residual_sum_15', 0)):.2f}",
            "",
            "## Multi-output suite",
            "",
            "| horizon | test_mae | test_rmse | test_r2 | test_mape | mean_abs_residual |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in suite_rows.itertuples(index=False):
            lines.append(
                f"| t+{int(row.horizon)} | {float(row.test_mae):.2f} | {float(row.test_rmse):.2f} | {float(row.test_r2):.4f} | {float(row.test_mape):.4f} | {float(row.mean_abs_residual):.2f} |"
            )
        lines += ["", "## Notes", ""]
        for note in context["mode_notes"]:
            lines.append(f"- {note}")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
