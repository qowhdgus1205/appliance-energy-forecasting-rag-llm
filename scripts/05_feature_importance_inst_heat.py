#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
FIG_DIR = BASE_DIR / "outputs" / "figures"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute feature importance for inst_heat baseline.")
    parser.add_argument("--file", default="facility_a.csv", help="Processed CSV to use.")
    parser.add_argument("--target", default="inst_heat", help="Target column.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in rows.")
    parser.add_argument("--model-path", default=None, help="Optional explicit model path.")
    args = parser.parse_args()

    path = PROCESSED_DIR / args.file
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = load_table(path).sort_values("server_time").reset_index(drop=True)
    df["target_future"] = df[args.target].shift(-args.horizon)
    df = df.dropna(subset=["target_future"]).reset_index(drop=True)
    split = add_time_split(df)
    feature_cols = [c for c in df.columns if c not in {"server_time", args.target, "target_future"}]
    X = df[feature_cols].select_dtypes(include="number")
    y = df["target_future"]

    model_path = Path(args.model_path) if args.model_path else MODEL_DIR / f"{path.stem}_{args.target}_h{args.horizon}_baseline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    model = joblib.load(model_path)
    test_mask = split == "test"
    test_X = X.loc[test_mask]
    test_y = y.loc[test_mask]

    result = permutation_importance(
        model,
        test_X,
        test_y,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )

    importance = pd.DataFrame(
        {
            "feature": test_X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = REPORT_DIR / f"{path.stem}_{args.target}_feature_importance.csv"
    md_path = REPORT_DIR / f"{path.stem}_{args.target}_feature_importance.md"
    fig_path = FIG_DIR / f"{path.stem}_{args.target}_feature_importance.png"

    importance.to_csv(csv_path, index=False)

    top = importance.head(20)
    lines = [
        "# Industrial Energy inst_heat feature importance",
        "",
        f"- file: `{args.file}`",
        f"- target: `{args.target}`",
        f"- horizon: `t+{args.horizon}`",
        f"- metric: permutation importance on test split using negative MAE",
        "",
        "| rank | feature | importance_mean | importance_std |",
        "| --- | --- | ---: | ---: |",
    ]
    for i, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(
            f"| {i} | {row.feature} | {row.importance_mean:.6f} | {row.importance_std:.6f} |"
        )
    lines += ["", "## Notes", "", "- larger positive values mean the feature hurts performance more when permuted.", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    plot_df = top.sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(plot_df["feature"], plot_df["importance_mean"], xerr=plot_df["importance_std"], color="#4C78A8")
    ax.set_title(f"{path.stem} - {args.target} permutation importance")
    ax.set_xlabel("importance (mean decrease in neg MAE)")
    ax.set_ylabel("feature")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
