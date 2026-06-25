#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
MODEL_DIR = BASE_DIR / "outputs" / "models"

MODE_MAP = {1: "gas_like", 0: "heating_like"}
MODE_COLUMNS = {"opermode", "mode", "mode_x", "mode_y", "oper", "segment_id", "mode_label"}


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


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_arr = y_true.to_numpy()
    residual = y_arr - y_pred
    nonzero_mask = np.abs(y_arr) > 1e-8
    mape = (
        float(np.mean(np.abs((y_arr[nonzero_mask] - y_pred[nonzero_mask]) / y_arr[nonzero_mask])))
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


def build_mode_frame(df: pd.DataFrame, mode_value: int, target: str, horizon: int) -> pd.DataFrame:
    mode_label = MODE_MAP[mode_value]
    ordered = df.sort_values("server_time").reset_index(drop=True).copy()
    ordered["mode_label"] = ordered["opermode"].map(MODE_MAP)
    ordered["segment_id"] = (ordered["mode_label"] != ordered["mode_label"].shift()).cumsum()

    blocks: list[pd.DataFrame] = []
    for _, block in ordered.loc[ordered["opermode"] == mode_value].groupby("segment_id", sort=False):
        block = block.copy()
        block["target_future"] = block[target].shift(-horizon)
        block = block.dropna(subset=["target_future"]).reset_index(drop=True)
        if not block.empty:
            blocks.append(block)

    if not blocks:
        raise ValueError(f"No usable rows found for mode proxy {mode_label}")

    mode_df = pd.concat(blocks, ignore_index=True).sort_values("server_time").reset_index(drop=True)
    return mode_df


def build_feature_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    feature_cols = [
        c
        for c in df.columns
        if c not in {"server_time", target, "target_future"} and c not in MODE_COLUMNS and not c.startswith(f"{target}_")
    ]
    X = df[feature_cols].select_dtypes(include="number")
    return X


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
                        "mean_actual_future": block["target_future"].mean(),
                        "mean_prediction": block["prediction"].mean(),
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
                "mean_actual_future": block["target_future"].mean(),
                "mean_prediction": block["prediction"].mean(),
                "mean_residual": block["residual"].mean(),
                "mean_abs_error": block["abs_error"].mean(),
                "max_abs_error": block["abs_error"].max(),
            }
        )

    episodes = pd.DataFrame(rows)
    if episodes.empty:
        return episodes
    return episodes.sort_values("mean_abs_error", ascending=False).reset_index(drop=True)


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
        and c not in MODE_COLUMNS
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


def train_one_mode(
    full_df: pd.DataFrame,
    mode_value: int,
    target: str,
    horizon: int,
    mode_tag: str,
) -> dict[str, object]:
    mode_df = build_mode_frame(full_df, mode_value=mode_value, target=target, horizon=horizon)
    split = add_time_split(mode_df)
    X = build_feature_frame(mode_df, target=target)
    y = mode_df["target_future"]

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=42,
    )
    model.fit(X.loc[train_mask], y.loc[train_mask])

    pred_train = model.predict(X.loc[train_mask])
    pred_val = model.predict(X.loc[val_mask])
    pred_test = model.predict(X.loc[test_mask])

    train_m = metrics(y.loc[train_mask], pred_train)
    val_m = metrics(y.loc[val_mask], pred_val)
    test_m = metrics(y.loc[test_mask], pred_test)

    result = mode_df.loc[test_mask, ["server_time", "opermode", "mode_label", target, "target_future"]].copy()
    result["prediction"] = pred_test
    result["residual"] = result["target_future"] - result["prediction"]
    result["abs_error"] = result["residual"].abs()
    threshold = float(result["abs_error"].quantile(0.95))
    result["pred_anomaly"] = (result["abs_error"] > threshold).astype(int)

    feature_importance = permutation_importance(
        model,
        X.loc[test_mask],
        y.loc[test_mask],
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": feature_importance.importances_mean,
            "importance_std": feature_importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    episodes = extract_episodes(result)
    top_episode = episodes.iloc[0].to_dict() if len(episodes) else {}
    root_shifts = (
        compare_windows(result, int(top_episode["start_idx"]), int(top_episode["end_idx"]), int(top_episode["rows"]))
        if top_episode
        else pd.DataFrame()
    )

    return {
        "mode_df": mode_df,
        "X": X,
        "model": model,
        "train_metrics": train_m,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "result": result,
        "threshold": threshold,
        "importance": importance,
        "episodes": episodes,
        "top_episode": top_episode,
        "root_shifts": root_shifts,
        "mode_tag": mode_tag,
        "mode_value": mode_value,
    }


def write_mode_outputs(mode_result: dict[str, object], target: str, horizon: int) -> dict[str, Path]:
    mode_tag = str(mode_result["mode_tag"])
    result = mode_result["result"]
    importance = mode_result["importance"]
    episodes = mode_result["episodes"]
    root_shifts = mode_result["root_shifts"]
    top_episode = mode_result["top_episode"]
    train_m = mode_result["train_metrics"]
    val_m = mode_result["val_metrics"]
    test_m = mode_result["test_metrics"]
    threshold = float(mode_result["threshold"])
    mode_value = int(mode_result["mode_value"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    pred_path = REPORT_DIR / f"facility_a_{mode_tag}_{target}_mode_forecasting_predictions.csv"
    imp_path = REPORT_DIR / f"facility_a_{mode_tag}_{target}_mode_forecasting_feature_importance.csv"
    model_path = MODEL_DIR / f"facility_a_{mode_tag}_{target}_mode_forecasting_h{horizon}.joblib"
    summary_path = REPORT_DIR / f"facility_a_{mode_tag}_{target}_mode_forecasting.md"
    episodes_path = REPORT_DIR / f"facility_a_{mode_tag}_{target}_mode_forecasting_episodes.csv"
    episodes_md_path = REPORT_DIR / f"facility_a_{mode_tag}_{target}_mode_forecasting_episodes.md"
    root_json_path = REPORT_DIR / f"facility_a_{mode_tag}_{target}_mode_forecasting_root_cause.json"
    root_md_path = REPORT_DIR / f"facility_a_{mode_tag}_{target}_mode_forecasting_root_cause.md"
    root_csv_path = REPORT_DIR / f"facility_a_{mode_tag}_{target}_mode_forecasting_root_cause.csv"

    result.to_csv(pred_path, index=False)
    importance.to_csv(imp_path, index=False)
    episodes.to_csv(episodes_path, index=False)
    joblib.dump(mode_result["model"], model_path)

    md_lines = [
        f"# Industrial Energy facility_a {mode_tag} inst_heat mode forecasting",
        "",
        f"- target: `{target}`",
        f"- horizon: `t+{horizon}`",
        f"- mode_proxy: `opermode`",
        f"- mode_value: `{mode_value}`",
        f"- interpretation: `{mode_tag}`",
        f"- model: `HistGradientBoostingRegressor`",
        f"- rows used for this mode: `{len(mode_result['mode_df'])}`",
        f"- numeric features used: `{mode_result['X'].shape[1]}`",
        "",
        "## Metrics",
        "",
        "| split | MAE | RMSE | R2 | MAPE | n_nonzero |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| train | {train_m['mae']:.4f} | {train_m['rmse']:.4f} | {train_m['r2']:.4f} | {train_m['mape']:.4f} | {train_m['mape_n']} |",
        f"| val | {val_m['mae']:.4f} | {val_m['rmse']:.4f} | {val_m['r2']:.4f} | {val_m['mape']:.4f} | {val_m['mape_n']} |",
        f"| test | {test_m['mae']:.4f} | {test_m['rmse']:.4f} | {test_m['r2']:.4f} | {test_m['mape']:.4f} | {test_m['mape_n']} |",
        "",
        "## Residual threshold",
        "",
        f"- test abs_error 95th percentile: {threshold:.4f}",
        f"- predicted anomaly ratio on test: {result['pred_anomaly'].mean():.4f}",
        "",
        "## Top features",
        "",
        "| rank | feature | importance_mean | importance_std |",
        "| --- | --- | ---: | ---: |",
    ]
    for idx, row in enumerate(importance.head(20).itertuples(index=False), start=1):
        md_lines.append(f"| {idx} | {row.feature} | {row.importance_mean:.6f} | {row.importance_std:.6f} |")

    md_lines += [
        "",
        "## Top episode",
        "",
        f"- episode_id: {top_episode.get('episode_id')}",
        f"- time: {top_episode.get('start_time')} -> {top_episode.get('end_time')}",
        f"- rows: {top_episode.get('rows')}",
        f"- mean actual future inst_heat: {float(top_episode.get('mean_actual_future', 0)):.2f}",
        f"- mean prediction: {float(top_episode.get('mean_prediction', 0)):.2f}",
        f"- mean residual: {float(top_episode.get('mean_residual', 0)):.2f}",
        "",
        "## Notes",
        "",
        "- This model is trained after filtering the full timeline by operating mode proxy and creating t+1 labels within each contiguous same-mode segment.",
        "- MAPE is computed only on rows where the actual target is non-zero.",
        "",
    ]
    summary_path.write_text("\n".join(md_lines), encoding="utf-8")

    root_payload = []
    root_lines = [
        f"# Industrial Energy facility_a {mode_tag} inst_heat mode forecasting root-cause summary",
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
                f"- mean actual future inst_heat: {ep['mean_target_future']:.2f}",
                f"- mean prediction: {ep['mean_prediction']:.2f}",
                f"- mean residual: {ep['mean_residual']:.2f}",
                f"- mean abs error: {ep['mean_abs_error']:.2f}",
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

    pd.DataFrame(flat_rows).to_csv(root_csv_path, index=False)
    root_md_path.write_text("\n".join(root_lines), encoding="utf-8")
    root_json_path.write_text(json.dumps(root_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    return {
        "predictions": pred_path,
        "importance": imp_path,
        "model": model_path,
        "summary": summary_path,
        "episodes": episodes_path,
        "episodes_md": episodes_md_path,
        "root_json": root_json_path,
        "root_md": root_md_path,
        "root_csv": root_csv_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mode-specific inst_heat forecasting models for Industrial Energy facility_a.")
    parser.add_argument("--file", default="facility_a.csv", help="Processed CSV to use.")
    parser.add_argument("--target", default="inst_heat", help="Target column.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in rows.")
    args = parser.parse_args()

    path = PROCESSED_DIR / args.file
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = load_table(path)
    if args.target not in df.columns:
        raise ValueError(f"Target not found: {args.target}")
    if "opermode" not in df.columns:
        raise ValueError("Expected opermode column in facility_a.csv")

    overview_rows = []
    outputs = {}
    for mode_value, mode_tag in [(1, "gas_like"), (0, "heating_like")]:
        mode_result = train_one_mode(df, mode_value=mode_value, target=args.target, horizon=args.horizon, mode_tag=mode_tag)
        files = write_mode_outputs(mode_result, target=args.target, horizon=args.horizon)
        outputs[mode_tag] = files
        overview_rows.append(
            {
                "mode_tag": mode_tag,
                "mode_value": mode_value,
                "rows": len(mode_result["mode_df"]),
                "train_mae": mode_result["train_metrics"]["mae"],
                "val_mae": mode_result["val_metrics"]["mae"],
                "test_mae": mode_result["test_metrics"]["mae"],
                "train_rmse": mode_result["train_metrics"]["rmse"],
                "val_rmse": mode_result["val_metrics"]["rmse"],
                "test_rmse": mode_result["test_metrics"]["rmse"],
                "train_r2": mode_result["train_metrics"]["r2"],
                "val_r2": mode_result["val_metrics"]["r2"],
                "test_r2": mode_result["test_metrics"]["r2"],
                "train_mape": mode_result["train_metrics"]["mape"],
                "val_mape": mode_result["val_metrics"]["mape"],
                "test_mape": mode_result["test_metrics"]["mape"],
                "residual_threshold": mode_result["threshold"],
                "pred_anomaly_ratio": float(mode_result["result"]["pred_anomaly"].mean()),
            }
        )

    overview = pd.DataFrame(overview_rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    overview_csv = REPORT_DIR / f"{path.stem}_{args.target}_mode_forecasting_summary.csv"
    overview_md = REPORT_DIR / f"{path.stem}_{args.target}_mode_forecasting_summary.md"
    overview.to_csv(overview_csv, index=False)
    overview_lines = [
        "# Industrial Energy facility_a mode-specific inst_heat forecasting summary",
        "",
        f"- file: `{args.file}`",
        f"- target: `{args.target}`",
        f"- horizon: `t+{args.horizon}`",
        "- mode proxy: `opermode`",
        "- `opermode == 1` is treated as the first gas-like proxy.",
        "- `opermode == 0` is treated as the first heating-like proxy.",
        "",
        "| mode_tag | rows | test_MAE | test_RMSE | test_R2 | test_MAPE | residual_threshold | pred_anomaly_ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overview.itertuples(index=False):
        overview_lines.append(
            f"| {row.mode_tag} | {row.rows} | {row.test_mae:.4f} | {row.test_rmse:.4f} | {row.test_r2:.4f} | {row.test_mape:.4f} | {row.residual_threshold:.4f} | {row.pred_anomaly_ratio:.4f} |"
        )
    overview_lines += [
        "",
        "## Notes",
        "",
        "- Each model is trained on rows filtered by the operating-mode proxy and split by time within that mode-specific subset.",
        "- Root-cause summaries and RAG contexts are written per mode in `outputs/reports/`.",
        "",
    ]
    overview_md.write_text("\n".join(overview_lines), encoding="utf-8")

    print(overview.to_string(index=False))


if __name__ == "__main__":
    main()
