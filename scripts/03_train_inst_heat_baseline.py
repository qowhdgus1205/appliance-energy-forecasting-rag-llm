#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
MODEL_DIR = BASE_DIR / "outputs" / "models"

os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline model for inst_heat.")
    parser.add_argument("--file", default="facility_a.csv", help="Processed CSV to use.")
    parser.add_argument("--target", default="inst_heat", help="Target column.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in rows.")
    args = parser.parse_args()

    path = PROCESSED_DIR / args.file
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = load_table(path).sort_values("server_time").reset_index(drop=True)
    if args.target not in df.columns:
        raise ValueError(f"Target not found: {args.target}")

    df["target_future"] = df[args.target].shift(-args.horizon)
    df = df.dropna(subset=["target_future"]).reset_index(drop=True)
    split = add_time_split(df)
    feature_cols = [c for c in df.columns if c not in {"server_time", args.target, "target_future"}]
    X = df[feature_cols]
    y = df["target_future"]

    # Keep only numeric features for the first baseline.
    X = X.select_dtypes(include="number")

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

    def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        residual = y_true.to_numpy() - y_pred
        y_arr = y_true.to_numpy()
        nonzero_mask = np.abs(y_arr) > 1e-8
        mape = float(np.mean(np.abs((y_arr[nonzero_mask] - y_pred[nonzero_mask]) / y_arr[nonzero_mask]))) if nonzero_mask.any() else float("nan")
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "r2": r2_score(y_true, y_pred),
            "mape": mape,
            "mape_n": int(nonzero_mask.sum()),
        }

    train_m = metrics(y.loc[train_mask], pred_train)
    val_m = metrics(y.loc[val_mask], pred_val)
    test_m = metrics(y.loc[test_mask], pred_test)

    result = df.loc[test_mask, ["server_time", args.target, "target_future"]].copy()
    result["prediction"] = pred_test
    result["residual"] = result["target_future"] - result["prediction"]
    result["abs_error"] = result["residual"].abs()
    result["pred_anomaly"] = (result["abs_error"] > result["abs_error"].quantile(0.95)).astype(int)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / f"{path.stem}_{args.target}_h{args.horizon}_baseline.joblib")
    result.to_csv(REPORT_DIR / f"{path.stem}_{args.target}_baseline_predictions.csv", index=False)

    md_lines = [
        "# Industrial Energy inst_heat baseline",
        "",
        f"- file: `{args.file}`",
        f"- target: `{args.target}`",
        f"- horizon: `t+{args.horizon}`",
        f"- model: `HistGradientBoostingRegressor`",
        f"- features: numeric columns only, excluding `server_time`, current target, and future target label",
        "",
        "## Metrics",
        "",
        "| split | MAE | RMSE | R2 |",
        "| --- | ---: | ---: | ---: | ---: | n_nonzero |",
        f"| train | {train_m['mae']:.4f} | {train_m['rmse']:.4f} | {train_m['r2']:.4f} | {train_m['mape']:.4f} | {train_m['mape_n']} |",
        f"| val | {val_m['mae']:.4f} | {val_m['rmse']:.4f} | {val_m['r2']:.4f} | {val_m['mape']:.4f} | {val_m['mape_n']} |",
        f"| test | {test_m['mae']:.4f} | {test_m['rmse']:.4f} | {test_m['r2']:.4f} | {test_m['mape']:.4f} | {test_m['mape_n']} |",
        "",
        "## Residual threshold",
        "",
        f"- test abs_error 95th percentile: {result['abs_error'].quantile(0.95):.4f}",
        f"- predicted anomaly ratio on test: {result['pred_anomaly'].mean():.4f}",
        f"- MAPE computed only on rows where actual target is non-zero",
        "",
    ]
    (REPORT_DIR / f"{path.stem}_{args.target}_baseline.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
