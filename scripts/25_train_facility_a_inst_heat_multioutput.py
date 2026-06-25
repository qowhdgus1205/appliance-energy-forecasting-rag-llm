#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
MODEL_DIR = BASE_DIR / "outputs" / "models"


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["server_time"])


def add_time_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15) -> pd.Series:
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    split = pd.Series(index=df.index, dtype="object")
    split.iloc[:train_end] = "train"
    split.iloc[train_end:val_end] = "val"
    split.iloc[val_end:] = "test"
    return split


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    nonzero_mask = np.abs(y_true) > 1e-8
    mape = (
        float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])))
        if nonzero_mask.any()
        else float("nan")
    )
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "r2": r2_score(y_true, y_pred),
        "mape": mape,
        "mape_n": int(nonzero_mask.sum()),
    }


def build_supervised_frame(df: pd.DataFrame, target: str, horizon: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.sort_values("server_time").reset_index(drop=True).copy()
    future_cols = []
    for step in range(1, horizon + 1):
        col = f"{target}_future_{step}"
        work[col] = work[target].shift(-step)
        future_cols.append(col)
    work = work.dropna(subset=future_cols).reset_index(drop=True)
    feature_cols = [c for c in work.columns if c not in {"server_time", *future_cols}]
    X = work[feature_cols].select_dtypes(include="number")
    y = work[future_cols]
    return work, X, y


def extract_episodes(df: pd.DataFrame, anomaly_col: str = "pred_anomaly") -> pd.DataFrame:
    rows = []
    episode_id = 0
    current = []
    start_idx = None

    for idx, row in df.iterrows():
        if int(row[anomaly_col]) == 1:
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
    parser = argparse.ArgumentParser(description="Train a shared multi-output forecast model for facility_a inst_heat t+1..t+15.")
    parser.add_argument("--file", default="facility_a.csv", help="Processed CSV to use.")
    parser.add_argument("--target", default="inst_heat", help="Target column.")
    parser.add_argument("--horizon", type=int, default=15, help="Forecast horizon length.")
    parser.add_argument("--electric-rate", type=float, default=0.18, help="Electric rate per heat unit in EUR/kWh-equivalent.")
    parser.add_argument("--gas-rate", type=float, default=0.07, help="Gas rate per heat unit in EUR/kWh-equivalent.")
    args = parser.parse_args()

    path = PROCESSED_DIR / args.file
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = load_table(path).sort_values("server_time").reset_index(drop=True)
    if args.target not in df.columns:
        raise ValueError(f"Target not found: {args.target}")

    work, X, y = build_supervised_frame(df, target=args.target, horizon=args.horizon)
    split = add_time_split(work)
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    model = ExtraTreesRegressor(
        n_estimators=120,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X.loc[train_mask], y.loc[train_mask])

    pred_train = model.predict(X.loc[train_mask])
    pred_val = model.predict(X.loc[val_mask])
    pred_test = model.predict(X.loc[test_mask])

    train_m = metrics(y.loc[train_mask].to_numpy(), pred_train)
    val_m = metrics(y.loc[val_mask].to_numpy(), pred_val)
    test_m = metrics(y.loc[test_mask].to_numpy(), pred_test)

    pred_cols = [f"pred_t{step}" for step in range(1, args.horizon + 1)]
    actual_cols = [f"actual_t{step}" for step in range(1, args.horizon + 1)]
    residual_cols = [f"residual_t{step}" for step in range(1, args.horizon + 1)]

    result = work.loc[test_mask, ["server_time", "opermode", "mode", "mode_x", "mode_y", "oper", args.target]].copy()
    result["split"] = "test"
    for step, col in enumerate(pred_cols, start=1):
        result[col] = pred_test[:, step - 1]
    for step, col in enumerate(actual_cols, start=1):
        result[col] = y.loc[test_mask, f"{args.target}_future_{step}"].to_numpy()
    for step, col in enumerate(residual_cols, start=1):
        result[col] = result[actual_cols[step - 1]] - result[pred_cols[step - 1]]

    result["actual_sum_15"] = result[actual_cols].sum(axis=1)
    result["pred_sum_15"] = result[pred_cols].sum(axis=1)
    result["residual_sum_15"] = result["actual_sum_15"] - result["pred_sum_15"]
    result["abs_error_sum_15"] = result["residual_sum_15"].abs()
    threshold = float(result["abs_error_sum_15"].quantile(0.95))
    result["pred_anomaly"] = (result["abs_error_sum_15"] > threshold).astype(int)
    result["mode_label"] = result["opermode"].map({1: "gas_like", 0: "heating_like"}).fillna("unknown")

    result["electric_cost_15"] = np.nan if args.electric_rate is None else result["pred_sum_15"] * args.electric_rate
    result["gas_cost_15"] = np.nan if args.gas_rate is None else result["pred_sum_15"] * args.gas_rate
    result["pred_cost_delta_15"] = (
        np.nan
        if args.electric_rate is None or args.gas_rate is None
        else result["pred_sum_15"] * (args.electric_rate - args.gas_rate)
    )

    top_horizon_cols = [f"residual_t{step}" for step in range(1, args.horizon + 1)]
    abs_horizon = result[top_horizon_cols].abs().mean(axis=0).sort_values(ascending=False)
    top_horizon_summary = pd.DataFrame(
        {
            "horizon": [int(col.replace("residual_t", "")) for col in abs_horizon.index],
            "mean_abs_residual": abs_horizon.to_numpy(),
        }
    )

    importances = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": getattr(model, "feature_importances_", np.zeros(X.shape[1])),
        }
    ).sort_values("importance", ascending=False)

    horizon_metrics = []
    for step in range(1, args.horizon + 1):
        actual_vec = result[f"actual_t{step}"].to_numpy()
        pred_vec = result[f"pred_t{step}"].to_numpy()
        met = metrics(actual_vec, pred_vec)
        horizon_metrics.append(
            {
                "horizon": step,
                "test_mae": met["mae"],
                "test_rmse": met["rmse"],
                "test_r2": met["r2"],
                "test_mape": met["mape"],
                "mape_n": met["mape_n"],
                "mean_abs_residual": float(np.mean(np.abs(result[f"residual_t{step}"].to_numpy()))),
            }
        )

    episodes = extract_episodes(result)
    top_episode = episodes.iloc[0].to_dict() if len(episodes) else {}
    root_shifts = (
        compare_windows(result, int(top_episode["start_idx"]), int(top_episode["end_idx"]), int(top_episode["rows"]))
        if top_episode
        else pd.DataFrame()
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    pred_path = REPORT_DIR / f"{path.stem}_{args.target}_multioutput_predictions.csv"
    model_path = MODEL_DIR / f"{path.stem}_{args.target}_multioutput.joblib"
    imp_path = REPORT_DIR / f"{path.stem}_{args.target}_multioutput_feature_importance.csv"
    suite_md = REPORT_DIR / f"{path.stem}_{args.target}_multioutput_summary.md"
    suite_json = REPORT_DIR / f"{path.stem}_{args.target}_multioutput_summary.json"
    root_json = REPORT_DIR / f"{path.stem}_{args.target}_multioutput_root_cause.json"
    root_md = REPORT_DIR / f"{path.stem}_{args.target}_multioutput_root_cause.md"
    root_csv = REPORT_DIR / f"{path.stem}_{args.target}_multioutput_root_cause.csv"
    incident_json = REPORT_DIR / f"{path.stem}_{args.target}_multioutput_incident.json"
    incident_md = REPORT_DIR / f"{path.stem}_{args.target}_multioutput_incident.md"

    result.to_csv(pred_path, index=False)
    joblib.dump(model, model_path)
    importances.to_csv(imp_path, index=False)

    suite = {
        "file": args.file,
        "target": args.target,
        "horizon": args.horizon,
        "train_metrics": train_m,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "residual_threshold": threshold,
        "pred_anomaly_ratio": float(result["pred_anomaly"].mean()),
        "horizon_metrics": horizon_metrics,
        "top_horizon_summary": top_horizon_summary.head(15).to_dict(orient="records"),
        "electric_rate": args.electric_rate,
        "gas_rate": args.gas_rate,
        "cost_inputs": {"electric_rate": args.electric_rate, "gas_rate": args.gas_rate},
    }
    suite_json.write_text(json.dumps(suite, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    suite_lines = [
        "# Industrial Energy facility_a inst_heat multi-output forecast",
        "",
        f"- target: `{args.target}`",
        f"- forecast window: `t+1..t+{args.horizon}`",
        f"- model: `ExtraTreesRegressor` (single shared multi-output model)",
        f"- numeric features used: `{X.shape[1]}`",
        "",
        "## Metrics",
        "",
        "| split | MAE | RMSE | R2 | MAPE | n_nonzero |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| train | {train_m['mae']:.4f} | {train_m['rmse']:.4f} | {train_m['r2']:.4f} | {train_m['mape']:.4f} | {train_m['mape_n']} |",
        f"| val | {val_m['mae']:.4f} | {val_m['rmse']:.4f} | {val_m['r2']:.4f} | {val_m['mape']:.4f} | {val_m['mape_n']} |",
        f"| test | {test_m['mae']:.4f} | {test_m['rmse']:.4f} | {test_m['r2']:.4f} | {test_m['mape']:.4f} | {test_m['mape_n']} |",
        "",
        "## Window totals",
        "",
        f"- test abs_error(sum over t+1..t+15) 95th percentile: {threshold:.4f}",
        f"- predicted anomaly ratio on test: {result['pred_anomaly'].mean():.4f}",
        "",
        "## Optional cost inputs",
        "",
        f"- electric_rate: {args.electric_rate}",
        f"- gas_rate: {args.gas_rate}",
        "",
        "## Top horizons by mean absolute residual",
        "",
        "| horizon | test_mae | test_rmse | test_r2 | test_mape | mean_abs_residual |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in horizon_metrics:
        suite_lines.append(
            f"| t+{row['horizon']} | {row['test_mae']:.4f} | {row['test_rmse']:.4f} | {row['test_r2']:.4f} | {row['test_mape']:.4f} | {row['mean_abs_residual']:.4f} |"
        )
    suite_md.write_text("\n".join(suite_lines), encoding="utf-8")

    root_payload = []
    root_lines = [
        "# Industrial Energy facility_a inst_heat multi-output root-cause summary",
        "",
        f"- source: `{pred_path.name}`",
        f"- residual threshold: {threshold:.4f}",
        "",
    ]
    if not episodes.empty:
        for _, ep in episodes.head(10).iterrows():
            shifts = compare_windows(result, int(ep["start_idx"]), int(ep["end_idx"]), int(ep["rows"]))
            if shifts.empty:
                continue
            shifts = shifts.head(8).copy()
            root_payload.append(
                {
                    "episode": ep.to_dict(),
                    "top_feature_shifts": shifts.to_dict(orient="records"),
                }
            )
            root_lines += [
                f"## Episode {int(ep['episode_id'])}",
                "",
                f"- time: {ep['start_time']} -> {ep['end_time']}",
                f"- rows: {int(ep['rows'])}",
                f"- mean actual sum 15: {ep['mean_actual_sum_15']:.2f}",
                f"- mean prediction sum 15: {ep['mean_prediction_sum_15']:.2f}",
                f"- mean residual sum 15: {ep['mean_residual_sum_15']:.2f}",
                f"- mean abs error sum 15: {ep['mean_abs_error_sum_15']:.2f}",
                "",
                "| feature | episode_mean | baseline_mean | delta | z_delta |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
            for _, row in shifts.iterrows():
                root_lines.append(
                    f"| {row['feature']} | {row['episode_mean']:.4f} | {row['baseline_mean']:.4f} | {row['delta']:.4f} | {row['z_delta']:.4f} |"
                )
            root_lines.append("")

    flat_rows = []
    for item in root_payload:
        ep = item["episode"]
        for shift in item["top_feature_shifts"]:
            flat_rows.append(
                {
                    "episode_id": ep["episode_id"],
                    "start_time": ep["start_time"],
                    "end_time": ep["end_time"],
                    "rows": ep["rows"],
                    "mean_actual_sum_15": ep["mean_actual_sum_15"],
                    "mean_prediction_sum_15": ep["mean_prediction_sum_15"],
                    "mean_residual_sum_15": ep["mean_residual_sum_15"],
                    "mean_abs_error_sum_15": ep["mean_abs_error_sum_15"],
                    "feature": shift["feature"],
                    "episode_mean": shift["episode_mean"],
                    "baseline_mean": shift["baseline_mean"],
                    "delta": shift["delta"],
                    "z_delta": shift["z_delta"],
                }
            )

    pd.DataFrame(flat_rows).to_csv(root_csv, index=False)
    root_md.write_text("\n".join(root_lines), encoding="utf-8")
    root_json.write_text(json.dumps(root_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    top_episode_payload = {
        "site": "facility_a",
        "forecast_window": "t+1..t+15",
        "representative_horizon": "t+15",
        "mode_proxy": "opermode",
        "mode_values": {"gas_like": 1, "heating_like": 0},
        "residual_threshold": threshold,
        "pred_anomaly_ratio": float(result["pred_anomaly"].mean()),
        "top_episode": top_episode,
        "top_horizon_summary": top_horizon_summary.head(10).to_dict(orient="records"),
        "top_features": importances.head(10).to_dict(orient="records"),
        "root_cause_top_feature_shifts": root_shifts.head(8).to_dict(orient="records") if not root_shifts.empty else [],
        "cost_inputs": {"electric_rate": args.electric_rate, "gas_rate": args.gas_rate},
    }
    incident_json.write_text(json.dumps(top_episode_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    incident_lines = [
        "# Facility A inst_heat multi-output incident record",
        "",
        f"- forecast_window: `{top_episode_payload['forecast_window']}`",
        f"- representative_horizon: `{top_episode_payload['representative_horizon']}`",
        f"- residual_threshold: {threshold:.4f}",
        f"- predicted_anomaly_ratio: {float(result['pred_anomaly'].mean()):.4f}",
        "",
        "## Top episode",
        "",
        f"- time: {top_episode.get('start_time')} -> {top_episode.get('end_time')}",
        f"- mean actual sum 15: {float(top_episode.get('mean_actual_sum_15', 0)):.2f}",
        f"- mean prediction sum 15: {float(top_episode.get('mean_prediction_sum_15', 0)):.2f}",
        f"- mean residual sum 15: {float(top_episode.get('mean_residual_sum_15', 0)):.2f}",
        "",
        "## Top horizons by residual",
        "",
        "| horizon | mean_abs_residual |",
        "| --- | ---: |",
    ]
    for row in top_horizon_summary.head(10).itertuples(index=False):
        incident_lines.append(f"| t+{row.horizon} | {row.mean_abs_residual:.4f} |")
    incident_md.write_text("\n".join(incident_lines), encoding="utf-8")

    print(suite_lines[:40])
    print(suite)


if __name__ == "__main__":
    main()
