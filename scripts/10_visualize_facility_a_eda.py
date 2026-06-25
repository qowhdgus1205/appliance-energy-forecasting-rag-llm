#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
FIG_DIR = BASE_DIR / "outputs" / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["server_time"])


def top_corrs(df: pd.DataFrame, target: str = "inst_heat", k: int = 15) -> pd.Series:
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr(numeric_only=True)[target].drop(target).abs().sort_values(ascending=False)
    return corr.head(k)


def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["server_time"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create facility_a EDA visuals.")
    parser.add_argument("--site-file", default="facility_a.csv", help="Processed facility_a CSV.")
    parser.add_argument(
        "--predictions",
        default="outputs/reports/facility_a_inst_heat_exogenous_predictions.csv",
        help="Forecast predictions CSV.",
    )
    parser.add_argument(
        "--root-cause",
        default="outputs/reports/facility_a_inst_heat_exogenous_predictions_root_cause.json",
        help="Root-cause JSON from episode summarization.",
    )
    args = parser.parse_args()

    site_path = PROCESSED_DIR / args.site_file if not Path(args.site_file).is_absolute() else Path(args.site_file)
    pred_path = BASE_DIR / args.predictions if not Path(args.predictions).is_absolute() else Path(args.predictions)
    root_path = BASE_DIR / args.root_cause if not Path(args.root_cause).is_absolute() else Path(args.root_cause)
    if not site_path.exists():
        raise FileNotFoundError(f"Missing site file: {site_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")
    if not root_path.exists():
        raise FileNotFoundError(f"Missing root cause file: {root_path}")

    site = load_table(site_path).sort_values("server_time").reset_index(drop=True)
    pred = load_predictions(pred_path)
    with open(root_path, "r", encoding="utf-8") as f:
        root = json.load(f)
    corr = top_corrs(site)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Overview plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.ravel()

    axes[0].plot(site["server_time"], site["inst_heat"], color="#1f77b4", linewidth=0.8)
    axes[0].set_title("facility_a inst_heat over time")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("inst_heat")

    axes[1].hist(site["inst_heat"], bins=60, color="#4C78A8", edgecolor="white")
    axes[1].axvline(site["inst_heat"].median(), color="#E45756", linestyle="--", linewidth=1.2, label="median")
    axes[1].set_title("inst_heat distribution")
    axes[1].set_xlabel("inst_heat")
    axes[1].set_ylabel("count")
    axes[1].legend(frameon=False)

    axes[2].barh(list(reversed(corr.index.tolist())), list(reversed(corr.values.tolist())), color="#72B7B2")
    axes[2].set_title("top absolute correlations with inst_heat")
    axes[2].set_xlabel("|corr|")

    axes[3].plot(pred["server_time"], pred["target_future"], color="#1f77b4", linewidth=0.8, label="actual t+1")
    axes[3].plot(pred["server_time"], pred["prediction"], color="#E45756", linewidth=0.8, alpha=0.8, label="prediction")
    axes[3].set_title("forecasting result on test window")
    axes[3].set_xlabel("time")
    axes[3].set_ylabel("inst_heat t+1")
    axes[3].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "facility_a_eda_overview.png", dpi=160)
    plt.close(fig)

    # 2) Residual plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(pred["server_time"], pred["abs_error"], color="#F58518", linewidth=0.9)
    thr = float(pred["abs_error"].quantile(0.95))
    axes[0].axhline(thr, color="#E45756", linestyle="--", linewidth=1.2, label="95th threshold")
    axes[0].set_title("absolute forecasting error on test window")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("|residual|")
    axes[0].legend(frameon=False)

    axes[1].hist(pred["residual"], bins=60, color="#54A24B", edgecolor="white")
    axes[1].axvline(0, color="#2F2F2F", linestyle="--", linewidth=1)
    axes[1].set_title("residual distribution")
    axes[1].set_xlabel("residual")
    axes[1].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "facility_a_eda_residuals.png", dpi=160)
    plt.close(fig)

    # 3) Zoom around top episode
    top = root[0]
    ep = top["episode"]
    start_time = pd.to_datetime(ep["start_time"])
    end_time = pd.to_datetime(ep["end_time"])
    window = pred[(pred["server_time"] >= start_time - pd.Timedelta(hours=6)) & (pred["server_time"] <= end_time + pd.Timedelta(hours=6))]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(window["server_time"], window["target_future"], color="#1f77b4", linewidth=1.0, label="actual t+1")
    ax.plot(window["server_time"], window["prediction"], color="#E45756", linewidth=1.0, alpha=0.85, label="prediction")
    ax.fill_between(window["server_time"], window["target_future"], window["prediction"], color="#9D755D", alpha=0.15)
    ax.axvspan(start_time, end_time, color="#F58518", alpha=0.18, label="top episode")
    ax.set_title(f"top residual episode zoom: episode {ep['episode_id']}")
    ax.set_xlabel("time")
    ax.set_ylabel("inst_heat t+1")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "facility_a_eda_top_episode_zoom.png", dpi=160)
    plt.close(fig)

    md = [
        "# Facility A EDA visuals",
        "",
        "Generated figures:",
        "",
        "- `facility_a_eda_overview.png`",
        "- `facility_a_eda_residuals.png`",
        "- `facility_a_eda_top_episode_zoom.png`",
        "",
        "## Quick read",
        "",
        f"- inst_heat median: {site['inst_heat'].median():.2f}",
        f"- inst_heat zero ratio: {(site['inst_heat'] == 0).mean():.4f}",
        f"- top correlation with inst_heat: `{corr.index[0]}` ({corr.iloc[0]:.4f})",
        f"- test residual 95th percentile: {thr:.4f}",
        f"- top episode id: {ep['episode_id']}",
        f"- top episode time: {ep['start_time']} -> {ep['end_time']}",
        "",
    ]
    (REPORT_DIR / "facility_a_eda_visuals.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
