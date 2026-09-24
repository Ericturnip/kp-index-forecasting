"""
Experiment 17: true PyTorch transformer Seq2Seq candidate.

This experiment uses the same operational tensors as Experiment 12/16:
    8 past assimilation tokens: observed Kp + observed/known physics
    20 future tokens: forecast-valid physics only, no Kp
    20-step Kp target vector

It trains a small transformer encoder with future-token heads for mean Kp,
p90/p95 risk envelopes, and Kp>=5/6/7/8 probabilities. It is a research
candidate and does not replace Experiments 12-14.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - exercised only on machines without torch
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from temporal_model import _cadence_hours, _ensure_directory, _markdown_table, chronological_split, prepare_temporal_dataframe
from unified_refresh_pipeline import FUTURE_STEPS, PAST_STEPS, _lead_time_bin
from experiment16_transformer_seq2seq import (
    THRESHOLDS,
    TOKEN_FEATURES,
    build_transformer_tensor_dataset,
)


RANDOM_STATE = 42
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.15
BATCH_SIZE = 256
DEFAULT_EPOCHS = 10
LEARNING_RATE = 8e-4
POS_WEIGHTS = [2.0, 6.0, 12.0, 20.0]


def torch_install_instructions() -> str:
    return (
        "PyTorch is not installed. Install it for this environment with:\n"
        "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "Then rerun: python experiment17_pytorch_transformer.py"
    )


def _device() -> "torch.device":
    if torch is None:
        raise ImportError(torch_install_instructions())
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.isclose(np.nanstd(y_true), 0.0) or np.isclose(np.nanstd(y_pred), 0.0):
        return 0.0
    corr, _ = pearsonr(y_true, y_pred)
    return 0.0 if np.isnan(corr) else float(corr)


def _safe_average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, score))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = PAST_STEPS + FUTURE_STEPS):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return x + self.pe[:, : x.size(1), :]


class KpTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = D_MODEL, n_heads: int = N_HEADS, n_layers: int = N_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 7),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h = self.proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        future = h[:, PAST_STEPS:, :]
        return self.head(future)


def _standardize_tokens(train_tokens: np.ndarray, test_tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_tokens.reshape(-1, train_tokens.shape[-1]).mean(axis=0).astype(np.float32)
    std = train_tokens.reshape(-1, train_tokens.shape[-1]).std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ((train_tokens - mean) / std).astype(np.float32), ((test_tokens - mean) / std).astype(np.float32), mean, std


def quantile_loss(pred: "torch.Tensor", target: "torch.Tensor", q: float) -> "torch.Tensor":
    err = target - pred
    return torch.maximum((q - 1.0) * err, q * err).mean()


def focal_bce_with_logits(logits: "torch.Tensor", targets: "torch.Tensor", pos_weight: "torch.Tensor", gamma: float = 1.5) -> "torch.Tensor":
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = targets * prob + (1.0 - targets) * (1.0 - prob)
    focal = (1.0 - p_t).pow(gamma)
    return (focal * bce).mean()


def total_loss(outputs: "torch.Tensor", y: "torch.Tensor", pos_weight: "torch.Tensor") -> "torch.Tensor":
    mean = outputs[..., 0]
    p90 = outputs[..., 1]
    p95 = outputs[..., 2]
    logits = outputs[..., 3:]
    targets = torch.stack([(y >= threshold).float() for threshold in THRESHOLDS], dim=-1)
    huber = nn.functional.huber_loss(mean, y, delta=1.0)
    q90 = quantile_loss(p90, y, 0.90)
    q95 = quantile_loss(p95, y, 0.95)
    bce = focal_bce_with_logits(logits, targets, pos_weight=pos_weight)
    monotonic = torch.relu(mean - p90).mean() + torch.relu(p90 - p95).mean()
    return huber + 0.55 * q90 + 0.70 * q95 + 1.6 * bce + 0.15 * monotonic


def train_model(train_tokens: np.ndarray, train_y: np.ndarray, epochs: int = DEFAULT_EPOCHS) -> Tuple[KpTransformer, dict]:
    if torch is None:
        raise ImportError(torch_install_instructions())
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    device = _device()
    idx = np.arange(len(train_tokens))
    fit_idx, val_idx = train_test_split(idx, test_size=0.15, shuffle=False)
    x_fit = torch.tensor(train_tokens[fit_idx], dtype=torch.float32)
    y_fit = torch.tensor(train_y[fit_idx], dtype=torch.float32)
    x_val = torch.tensor(train_tokens[val_idx], dtype=torch.float32).to(device)
    y_val = torch.tensor(train_y[val_idx], dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(x_fit, y_fit), batch_size=BATCH_SIZE, shuffle=True)
    model = KpTransformer(input_dim=train_tokens.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    pos_weight = torch.tensor(POS_WEIGHTS, dtype=torch.float32, device=device)
    best_state = None
    best_val = float("inf")
    history = []
    patience = 4
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = total_loss(model(xb), yb, pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(total_loss(model(x_val), y_val, pos_weight).detach().cpu())
        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(f"Experiment 17 epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history, "device": str(device), "best_val_loss": best_val}


def predict_model(model: KpTransformer, test_tokens: np.ndarray) -> Dict[str, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    preds = []
    loader = DataLoader(torch.tensor(test_tokens, dtype=torch.float32), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for xb in loader:
            out = model(xb.to(device)).detach().cpu().numpy()
            preds.append(out)
    raw = np.concatenate(preds, axis=0)
    mean = np.clip(raw[..., 0], 0.0, 9.0)
    p90 = np.maximum(mean, np.clip(raw[..., 1], 0.0, 9.0))
    p95 = np.maximum(p90, np.clip(raw[..., 2], 0.0, 9.0))
    probs = expit(raw[..., 3:])
    out = {
        "kp_torch_transformer_mean": mean,
        "kp_torch_transformer_p90": p90,
        "kp_torch_transformer_p95": p95,
    }
    for idx, threshold in enumerate(THRESHOLDS):
        out[f"prob_torch_transformer_kp_ge_{int(threshold)}"] = probs[..., idx]
    return out


def flatten_predictions(feature_df: pd.DataFrame, dataset, prediction_arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for sample_idx, init_idx in enumerate(dataset.init_indices):
        init_time = feature_df["timestamp"].iloc[init_idx]
        for step in range(FUTURE_STEPS):
            target_idx = init_idx + step + 1
            valid_time = feature_df["timestamp"].iloc[target_idx]
            lead_hours = float((valid_time - init_time) / pd.Timedelta(hours=1))
            row = {
                "timestamp": valid_time,
                "t_init": init_time,
                "lead_step": step + 1,
                "lead_time_hours": lead_hours,
                "lead_time_bin": _lead_time_bin(lead_hours),
                "observed_kp": float(feature_df["kp"].iloc[target_idx]),
                "kp_history_source_max_time": init_time,
                "uses_future_observed_kp": False,
            }
            for name, values in prediction_arrays.items():
                row[name] = float(values[sample_idx, step])
            rows.append(row)
    return pd.DataFrame(rows)


def _metric_row(frame: pd.DataFrame, pred_col: str) -> dict:
    y = frame["observed_kp"].to_numpy(dtype=float)
    pred = frame[pred_col].to_numpy(dtype=float)
    quiet = y < 5.0
    row = {
        "n": int(len(frame)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "pearson_corr": _safe_corr(y, pred),
        "bias": float(np.mean(pred - y)),
        "quiet_false_storm_inflation": float(np.mean(pred[quiet] >= 5.0)) if quiet.any() else 0.0,
        "max_underprediction": float(np.max(np.maximum(y - pred, 0.0))) if len(y) else 0.0,
    }
    for threshold in THRESHOLDS:
        true_binary = y >= threshold
        pred_binary = pred >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(true_binary, pred_binary, average="binary", zero_division=0)
        row[f"precision_kp_ge_{int(threshold)}"] = float(precision)
        row[f"recall_kp_ge_{int(threshold)}"] = float(recall)
        row[f"f1_kp_ge_{int(threshold)}"] = float(f1)
        prob_col = f"prob_torch_transformer_kp_ge_{int(threshold)}"
        if prob_col in frame.columns:
            row[f"average_precision_kp_ge_{int(threshold)}"] = _safe_average_precision(true_binary, frame[prob_col].to_numpy(dtype=float))
    return row


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pred_col in ["kp_torch_transformer_mean", "kp_torch_transformer_p90", "kp_torch_transformer_p95"]:
        rows.append({"model": pred_col, "lead_time_bin": "all", "lead_step": 0, **_metric_row(predictions, pred_col)})
        for lead_step, group in predictions.groupby("lead_step", sort=True):
            rows.append({"model": pred_col, "lead_time_bin": group["lead_time_bin"].iloc[0], "lead_step": int(lead_step), **_metric_row(group, pred_col)})
    return pd.DataFrame(rows)


def _compare_frame(transformer_predictions: pd.DataFrame, path: str, model: str, pred_col: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path, parse_dates=["timestamp", "t_init"])
    common = transformer_predictions[["timestamp", "t_init", "observed_kp"]].merge(frame[["timestamp", "t_init", pred_col]], on=["timestamp", "t_init"], how="left")
    common = common.dropna(subset=[pred_col])
    if common.empty:
        return None
    return {"model": model, "status": "available", **_metric_row(common, pred_col)}


def comparison_metrics(predictions: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    rows = []
    specs = [
        ("experiment12_mean", "kp_seq_mean_forecast", "kp_experiment12_unified_rolling_predictions.csv"),
        ("experiment13_severe_overlay", "kp_final_production_forecast", "kp_experiment13_final_production_predictions.csv"),
        ("experiment14_extreme_tier", "kp_experiment14_extreme_production", "kp_experiment14_extreme_predictions.csv"),
        ("numpy_transformer_exp16_mean", "kp_transformer_mean", "kp_experiment16_transformer_predictions.csv"),
        ("numpy_transformer_exp16_p95", "kp_transformer_p95", "kp_experiment16_transformer_predictions.csv"),
    ]
    for model, pred_col, filename in specs:
        row = _compare_frame(predictions, os.path.join(output_dir, filename), model, pred_col)
        if row is not None:
            rows.append(row)
    exp15_row = None
    for filename, columns in [
        ("kp_experiment15_website_one_line_forecast.csv", ["kp_experiment15_one_line_forecast", "kp_website_one_line_forecast", "one_line_forecast"]),
        ("kp_experiment15_predictions.csv", ["kp_experiment15_one_line_forecast", "kp_website_one_line_forecast", "one_line_forecast"]),
    ]:
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            continue
        candidate = pd.read_csv(path, nrows=1)
        for col in columns:
            if col in candidate.columns:
                exp15_row = _compare_frame(predictions, path, "experiment15_website_one_line", col)
                break
        if exp15_row is not None:
            break
    rows.append(exp15_row if exp15_row is not None else {"model": "experiment15_website_one_line", "status": "not_found"})
    for pred_col in ["kp_torch_transformer_mean", "kp_torch_transformer_p90", "kp_torch_transformer_p95"]:
        rows.append({"model": pred_col, "status": "candidate", **_metric_row(predictions, pred_col)})
    return pd.DataFrame(rows)


def _plot_timeseries(predictions: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "kp_experiment17_pytorch_transformer_timeseries.png")
    work = predictions[predictions["lead_step"] == 1].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    storm_rows = work[work["observed_kp"] >= 6.0]
    if not storm_rows.empty:
        center = storm_rows.iloc[len(storm_rows) // 2]["timestamp"]
        plot_df = work[(work["timestamp"] >= center - pd.Timedelta(days=16)) & (work["timestamp"] <= center + pd.Timedelta(days=16))]
    else:
        plot_df = work.head(320)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(plot_df["timestamp"], plot_df["observed_kp"], color="black", linewidth=2, label="Observed Kp")
    axes[0].plot(plot_df["timestamp"], plot_df["kp_torch_transformer_mean"], color="tab:blue", linewidth=1.3, label="PyTorch transformer mean")
    axes[0].plot(plot_df["timestamp"], plot_df["kp_torch_transformer_p90"], color="tab:orange", linewidth=1.0, label="p90")
    axes[0].plot(plot_df["timestamp"], plot_df["kp_torch_transformer_p95"], color="tab:red", linewidth=1.0, label="p95")
    axes[0].axhline(5.0, color="gray", linestyle="--", linewidth=1)
    axes[0].axhline(6.0, color="gray", linestyle=":", linewidth=1)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("Kp")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(plot_df["timestamp"], plot_df["prob_torch_transformer_kp_ge_5"], color="tab:orange", label="P(Kp>=5)")
    axes[1].plot(plot_df["timestamp"], plot_df["prob_torch_transformer_kp_ge_6"], color="tab:red", label="P(Kp>=6)")
    axes[1].plot(plot_df["timestamp"], plot_df["prob_torch_transformer_kp_ge_7"], color="tab:purple", label="P(Kp>=7)")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(fontsize=8)
    axes[1].set_ylabel("Probability")
    axes[1].set_xlabel("Valid time")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_leadtime(metrics: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "kp_experiment17_pytorch_transformer_leadtime.png")
    work = metrics[(metrics["model"] == "kp_torch_transformer_mean") & (metrics["lead_step"] > 0)].copy()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(work["lead_step"], work["mae"], marker="o", color="tab:blue", label="MAE")
    ax1.set_xlabel("Future step")
    ax1.set_ylabel("MAE")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(work["lead_step"], work["pearson_corr"], marker="s", color="tab:green", label="Pearson r")
    ax2.set_ylabel("Pearson r")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# Research role: evaluate a PyTorch sequence model with past context and future
# forecast-driver tokens. Interpretation boundary: future observed Kp remains
# forbidden, and upper envelopes are diagnostic rather than direct alerts.
def run_experiment17_pytorch_transformer(raw_df: pd.DataFrame, output_dir: str = "testing", test_size: float = 0.2, epochs: Optional[int] = None) -> dict:
    _ensure_directory(output_dir)
    if torch is None:
        instructions = torch_install_instructions()
        paths = {"summary": os.path.join(output_dir, "kp_experiment17_pytorch_transformer_install.md")}
        with open(paths["summary"], "w", encoding="utf-8") as f:
            f.write("# Experiment 17 PyTorch Transformer\n\n" + instructions + "\n")
        return {"status": "torch_not_installed", "instructions": instructions, "paths": paths}
    feature_df, _ = prepare_temporal_dataframe(raw_df, return_metadata=True)
    train_df, _ = chronological_split(feature_df, test_size)
    split_index = len(train_df)
    cadence_hours = _cadence_hours(feature_df)
    train_data = build_transformer_tensor_dataset(feature_df, PAST_STEPS - 1, split_index - FUTURE_STEPS - 1)
    test_data = build_transformer_tensor_dataset(feature_df, split_index - 1, len(feature_df) - FUTURE_STEPS - 1)
    train_tokens, test_tokens, token_mean, token_std = _standardize_tokens(train_data.tokens, test_data.tokens)
    epochs = int(os.environ.get("KP_EXPERIMENT17_EPOCHS", epochs or DEFAULT_EPOCHS))
    model, train_debug = train_model(train_tokens, train_data.y, epochs=epochs)
    pred_arrays = predict_model(model, test_tokens)
    predictions = flatten_predictions(feature_df, test_data, pred_arrays)
    metrics = build_metrics(predictions)
    comparison = comparison_metrics(predictions, output_dir)

    paths = {
        "predictions": os.path.join(output_dir, "kp_experiment17_pytorch_transformer_predictions.csv"),
        "metrics": os.path.join(output_dir, "kp_experiment17_pytorch_transformer_metrics.csv"),
        "comparison": os.path.join(output_dir, "kp_experiment17_backbone_comparison.csv"),
        "debug": os.path.join(output_dir, "kp_experiment17_debug.json"),
        "summary": os.path.join(output_dir, "kp_experiment17_pi_summary.md"),
    }
    paths["timeseries"] = _plot_timeseries(predictions, output_dir)
    paths["leadtime"] = _plot_leadtime(metrics, output_dir)
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    comparison.to_csv(paths["comparison"], index=False)

    debug = {
        "objective": "True PyTorch transformer candidate backbone for 5-day Kp trajectory and severe-tail probabilities.",
        "torch_version": torch.__version__,
        "device": train_debug["device"],
        "cadence_hours": cadence_hours,
        "past_steps": PAST_STEPS,
        "future_steps": FUTURE_STEPS,
        "token_features": TOKEN_FEATURES,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "dropout": DROPOUT,
        "epochs_requested": epochs,
        "train_samples": int(len(train_data.tokens)),
        "test_samples": int(len(test_data.tokens)),
        "prediction_rows": int(len(predictions)),
        "uses_future_observed_kp": bool(predictions["uses_future_observed_kp"].astype(bool).any()),
        "training": train_debug,
        "token_mean": token_mean.tolist(),
        "token_std": token_std.tolist(),
        "paths": paths,
    }
    with open(paths["debug"], "w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2, default=str)
        f.write("\n")

    summary = [
        "# Experiment 17 PyTorch Transformer",
        "## Objective\nTrain a real PyTorch transformer candidate backbone using the same tensors as Experiment 12/16, with future-token heads for mean Kp, p90/p95, and Kp>=5/6/7/8 probabilities.",
        f"## Runtime\nPyTorch `{torch.__version__}` on `{train_debug['device']}`. Architecture: d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, dropout={DROPOUT}.",
        "## Loss\nHuber(mean) + p90/p95 quantile loss + focal weighted BCE for Kp>=5/6/7/8 with requested severe class weights.",
        "## Transformer Metrics\n" + _markdown_table(metrics[metrics["lead_time_bin"] == "all"]),
        "## Backbone Comparison\n" + _markdown_table(comparison),
        "## Artifacts\n" + "\n".join([f"- `{value}`" for value in paths.values()]),
    ]
    with open(paths["summary"], "w", encoding="utf-8") as f:
        f.write("\n\n".join(summary) + "\n")
    return {"status": "trained", "predictions": predictions, "metrics": metrics, "comparison": comparison, "debug": debug, "paths": paths}


def print_validation_matrix(result: dict) -> None:
    if result.get("status") != "trained":
        print(result.get("instructions", torch_install_instructions()))
        return
    comparison = result["comparison"]
    cols = [
        "model",
        "status",
        "mae",
        "rmse",
        "pearson_corr",
        "recall_kp_ge_5",
        "recall_kp_ge_6",
        "recall_kp_ge_7",
        "recall_kp_ge_8",
        "quiet_false_storm_inflation",
        "max_underprediction",
    ]
    print("\nEXPERIMENT 17 PYTORCH TRANSFORMER COMPARISON")
    print(comparison[[col for col in cols if col in comparison.columns]].to_string(index=False))


def main() -> None:
    data_path = os.environ.get("KP_EXPERIMENT17_DATA", "testing/2008_2025_df.txt")
    output_dir = os.environ.get("KP_EXPERIMENT17_OUTPUT_DIR", "testing")
    raw_df = pd.read_csv(data_path)
    result = run_experiment17_pytorch_transformer(raw_df, output_dir=output_dir)
    print_validation_matrix(result)
    print("\nExperiment 17 artifacts:")
    for key, value in result.get("paths", {}).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
