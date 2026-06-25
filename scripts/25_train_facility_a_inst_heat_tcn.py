#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
MODEL_DIR = BASE_DIR / "outputs" / "models"


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TCNForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        channels: int = 64,
        levels: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = n_features
        for level in range(levels):
            dilation = 2**level
            blocks.append(TemporalBlock(in_channels, channels, kernel_size, dilation, dropout))
            in_channels = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: batch, seq, features; conv expects batch, features, seq
        y = self.tcn(x.transpose(1, 2))
        return self.head(y[:, :, -1])


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["server_time"])


def add_time_split(n: int, train_frac: float = 0.7, val_frac: float = 0.15) -> np.ndarray:
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    split = np.full(n, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"
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
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape,
        "mape_n": int(nonzero_mask.sum()),
    }


def build_sequence_frame(
    df: pd.DataFrame,
    target: str,
    seq_len: int,
    horizon: int,
    use_engineered_features: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    work = df.sort_values("server_time").reset_index(drop=True).copy()
    future_cols = []
    for step in range(1, horizon + 1):
        col = f"{target}_future_{step}"
        work[col] = work[target].shift(-step)
        future_cols.append(col)
    work = work.dropna(subset=future_cols).reset_index(drop=True)
    feature_cols = [c for c in work.columns if c not in {"server_time", *future_cols}]
    feature_cols = list(work[feature_cols].select_dtypes(include="number").columns)
    if not use_engineered_features:
        engineered_markers = ("_lag", "_rmean", "_rstd")
        feature_cols = [c for c in feature_cols if not any(marker in c for marker in engineered_markers)]

    features = work[feature_cols].to_numpy(dtype=np.float32)
    targets = work[future_cols].to_numpy(dtype=np.float32)
    rows = []
    x_seq = []
    y_seq = []
    for end_idx in range(seq_len - 1, len(work)):
        start_idx = end_idx - seq_len + 1
        x_seq.append(features[start_idx : end_idx + 1])
        y_seq.append(targets[end_idx])
        rows.append(end_idx)
    return work.iloc[rows].reset_index(drop=True), np.stack(x_seq), np.stack(y_seq), feature_cols


def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x.astype(np.float32))), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            outs.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(outs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact TCN forecast model for facility_a inst_heat.")
    parser.add_argument("--file", default="facility_a.csv")
    parser.add_argument("--target", default="inst_heat")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--use-engineered-features", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--electric-rate", type=float, default=0.18)
    parser.add_argument("--gas-rate", type=float, default=0.07)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = PROCESSED_DIR / args.file
    df = load_table(path)
    work, x, y, feature_cols = build_sequence_frame(
        df,
        args.target,
        args.seq_len,
        args.horizon,
        args.use_engineered_features,
    )
    split = add_time_split(len(work))
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_flat = x[train_mask].reshape(-1, x.shape[-1])
    x_scaler.fit(x_train_flat)
    x_scaled = x_scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
    y_scaler.fit(y[train_mask])
    y_scaled = y_scaler.transform(y).astype(np.float32)

    train_idx = np.flatnonzero(train_mask)
    if args.max_train_samples and len(train_idx) > args.max_train_samples:
        train_idx = np.linspace(0, len(train_idx) - 1, args.max_train_samples).round().astype(int)
        train_idx = np.flatnonzero(train_mask)[train_idx]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_scaled[train_idx]), torch.from_numpy(y_scaled[train_idx])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = TCNForecaster(
        n_features=x.shape[-1],
        horizon=args.horizon,
        channels=args.channels,
        levels=args.levels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    history = []
    best_state = None
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_pred_scaled = predict(model, x_scaled[val_mask], args.batch_size, device)
        val_loss = float(np.mean(np.abs(val_pred_scaled - y_scaled[val_mask])))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_mae_scaled": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch} train_loss={np.mean(train_losses):.5f} val_mae_scaled={val_loss:.5f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_train = y_scaler.inverse_transform(predict(model, x_scaled[train_mask], args.batch_size, device))
    pred_val = y_scaler.inverse_transform(predict(model, x_scaled[val_mask], args.batch_size, device))
    pred_test = y_scaler.inverse_transform(predict(model, x_scaled[test_mask], args.batch_size, device))

    train_m = metrics(y[train_mask], pred_train)
    val_m = metrics(y[val_mask], pred_val)
    test_m = metrics(y[test_mask], pred_test)

    pred_cols = [f"pred_t{step}" for step in range(1, args.horizon + 1)]
    actual_cols = [f"actual_t{step}" for step in range(1, args.horizon + 1)]
    residual_cols = [f"residual_t{step}" for step in range(1, args.horizon + 1)]

    result = work.loc[test_mask, ["server_time", "opermode", "mode", "mode_x", "mode_y", "oper", args.target]].copy()
    result["split"] = "test"
    for step, col in enumerate(pred_cols, start=1):
        result[col] = pred_test[:, step - 1]
    for step, col in enumerate(actual_cols, start=1):
        result[col] = y[test_mask, step - 1]
    for step, col in enumerate(residual_cols, start=1):
        result[col] = result[actual_cols[step - 1]] - result[pred_cols[step - 1]]
    result["actual_sum_15"] = result[actual_cols].sum(axis=1)
    result["pred_sum_15"] = result[pred_cols].sum(axis=1)
    result["residual_sum_15"] = result["actual_sum_15"] - result["pred_sum_15"]
    result["abs_error_sum_15"] = result["residual_sum_15"].abs()
    threshold = float(result["abs_error_sum_15"].quantile(0.95))
    result["pred_anomaly"] = (result["abs_error_sum_15"] > threshold).astype(int)
    result["mode_label"] = result["opermode"].map({1: "gas_like", 0: "heating_like"}).fillna("unknown")
    result["electric_cost_15"] = result["pred_sum_15"] * args.electric_rate
    result["gas_cost_15"] = result["pred_sum_15"] * args.gas_rate
    result["pred_cost_delta_15"] = result["pred_sum_15"] * (args.electric_rate - args.gas_rate)

    horizon_metrics = []
    for step in range(1, args.horizon + 1):
        met = metrics(result[f"actual_t{step}"].to_numpy(), result[f"pred_t{step}"].to_numpy())
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

    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv1d) and module.weight.shape[1] == len(feature_cols):
            first_conv = module.weight.detach().cpu().abs().sum(dim=(0, 2)).numpy()
            break
    if first_conv is None:
        first_conv = np.zeros(len(feature_cols))
    importances = pd.DataFrame({"feature": feature_cols, "importance": first_conv})
    if importances["importance"].sum() > 0:
        importances["importance"] = importances["importance"] / importances["importance"].sum()
    importances = importances.sort_values("importance", ascending=False)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    pred_path = REPORT_DIR / f"{stem}_{args.target}_multioutput_predictions.csv"
    model_path = MODEL_DIR / f"{stem}_{args.target}_tcn.pt"
    scaler_path = MODEL_DIR / f"{stem}_{args.target}_tcn_scalers.joblib"
    imp_path = REPORT_DIR / f"{stem}_{args.target}_multioutput_feature_importance.csv"
    suite_md = REPORT_DIR / f"{stem}_{args.target}_multioutput_summary.md"
    suite_json = REPORT_DIR / f"{stem}_{args.target}_multioutput_summary.json"

    result.to_csv(pred_path, index=False)
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "feature_cols": feature_cols}, model_path)
    joblib.dump({"x_scaler": x_scaler, "y_scaler": y_scaler, "feature_cols": feature_cols}, scaler_path)
    importances.to_csv(imp_path, index=False)

    suite = {
        "file": args.file,
        "target": args.target,
        "model": "Temporal Convolutional Network (TCN)",
        "sequence_length": args.seq_len,
        "use_engineered_features": args.use_engineered_features,
        "max_train_samples": args.max_train_samples,
        "horizon": args.horizon,
        "train_metrics": train_m,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "training_history": history,
        "residual_threshold": threshold,
        "pred_anomaly_ratio": float(result["pred_anomaly"].mean()),
        "horizon_metrics": horizon_metrics,
        "electric_rate": args.electric_rate,
        "gas_rate": args.gas_rate,
        "cost_inputs": {"electric_rate": args.electric_rate, "gas_rate": args.gas_rate},
    }
    suite_json.write_text(json.dumps(suite, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Facility A inst_heat TCN forecast",
        "",
        f"- target: `{args.target}`",
        "- model: `Temporal Convolutional Network (TCN)`",
        f"- sequence_length: `{args.seq_len}`",
        f"- engineered lag/rolling features used: `{args.use_engineered_features}`",
        f"- numeric features used: `{len(feature_cols)}`",
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
        f"- test abs_error(sum over prediction window) 95th percentile: {threshold:.4f}",
        f"- predicted anomaly ratio on test: {result['pred_anomaly'].mean():.4f}",
        "",
        "## Optional cost inputs",
        "",
        f"- electric_rate: {args.electric_rate}",
        f"- gas_rate: {args.gas_rate}",
        "",
        "## Horizons by mean absolute residual",
        "",
        "| horizon | test_mae | test_rmse | test_r2 | test_mape | mean_abs_residual |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in horizon_metrics:
        lines.append(
            f"| {row['horizon']} | {row['test_mae']:.4f} | {row['test_rmse']:.4f} | {row['test_r2']:.4f} | {row['test_mape']:.4f} | {row['mean_abs_residual']:.4f} |"
        )
    suite_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"train": train_m, "val": val_m, "test": test_m}, indent=2))
    print(f"saved: {suite_md}")


if __name__ == "__main__":
    main()
