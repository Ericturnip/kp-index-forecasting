"""
Experiment 16: small attention/transformer candidate backbone.

The environment for this project currently has no PyTorch/TensorFlow runtime, so
this experiment implements a compact NumPy self-attention encoder with
supervised scikit-learn heads. It uses the same operational tensor contract as
Experiment 12:

    X_past   = 48h observed Kp + observed physics
    X_future = 120h forecast-valid physics only
    Y        = 20-step future Kp

The attention encoder is deliberately separate from Experiments 12-14 and is
evaluated only as a candidate backbone.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, precision_recall_fscore_support

from temporal_model import _cadence_hours, _ensure_directory, _markdown_table, chronological_split, prepare_temporal_dataframe
from unified_refresh_pipeline import FUTURE_STEPS, PAST_STEPS, _lead_time_bin


RANDOM_STATE = 42
MODEL_DIM = 32
ATTENTION_LAYERS = 2
THRESHOLDS = [5.0, 6.0, 7.0, 8.0]

TOKEN_FEATURES = [
    "bz",
    "bs",
    "velocity",
    "density",
    "pdyn",
    "ey",
    "newell_coupling",
    "coupling_simple",
    "compact_storm_score",
    "bz_southward_drop_6h",
    "bz_southward_drop_12h",
    "bz_southward_drop_24h",
    "bz_crossed_southward",
    "strong_southward_turning",
    "kp",
    "is_past",
    "is_future",
    "time_index_norm",
    "lead_hours_norm",
]


@dataclass
class TransformerTensorDataset:
    tokens: np.ndarray
    y: np.ndarray
    init_indices: np.ndarray
    init_times: pd.Series
    token_features: List[str]


@dataclass
class AttentionEncoder:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    weights: Dict[str, np.ndarray]
    token_features: List[str]


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.isclose(np.nanstd(y_true), 0.0) or np.isclose(np.nanstd(y_pred), 0.0):
        return 0.0
    corr, _ = pearsonr(y_true, y_pred)
    return 0.0 if np.isnan(corr) else float(corr)


def _safe_average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, score))


def _available(columns: Iterable[str], frame: pd.DataFrame) -> List[str]:
    return [column for column in columns if column in frame.columns]


def _token_value(row: pd.Series, feature: str, is_past: bool, rel_index: int) -> float:
    if feature == "kp":
        return float(row["kp"]) if is_past else 0.0
    if feature == "is_past":
        return 1.0 if is_past else 0.0
    if feature == "is_future":
        return 0.0 if is_past else 1.0
    if feature == "time_index_norm":
        return float(rel_index / max(PAST_STEPS + FUTURE_STEPS - 1, 1))
    if feature == "lead_hours_norm":
        return 0.0 if is_past else float((rel_index - PAST_STEPS + 1) / FUTURE_STEPS)
    return float(row[feature]) if feature in row.index else 0.0


def build_transformer_tensor_dataset(
    feature_df: pd.DataFrame,
    start_init_idx: int,
    end_init_idx: int,
) -> TransformerTensorDataset:
    token_features = TOKEN_FEATURES.copy()
    first = max(PAST_STEPS - 1, start_init_idx)
    last = min(end_init_idx, len(feature_df) - FUTURE_STEPS - 1)
    tokens = []
    targets = []
    init_indices = []
    for init_idx in range(first, last + 1):
        sample_tokens = []
        past = feature_df.iloc[init_idx - PAST_STEPS + 1 : init_idx + 1]
        future = feature_df.iloc[init_idx + 1 : init_idx + FUTURE_STEPS + 1]
        for step, (_, row) in enumerate(past.iterrows()):
            sample_tokens.append([_token_value(row, feature, True, step) for feature in token_features])
        for step, (_, row) in enumerate(future.iterrows()):
            sample_tokens.append([_token_value(row, feature, False, PAST_STEPS + step) for feature in token_features])
        tokens.append(sample_tokens)
        targets.append(future["kp"].to_numpy(dtype=float))
        init_indices.append(init_idx)
    return TransformerTensorDataset(
        tokens=np.asarray(tokens, dtype=np.float32),
        y=np.asarray(targets, dtype=np.float32),
        init_indices=np.asarray(init_indices, dtype=int),
        init_times=feature_df["timestamp"].iloc[init_indices].reset_index(drop=True),
        token_features=token_features,
    )


def fit_attention_encoder(train_tokens: np.ndarray, token_features: List[str]) -> AttentionEncoder:
    rng = np.random.default_rng(RANDOM_STATE)
    mean = train_tokens.reshape(-1, train_tokens.shape[-1]).mean(axis=0).astype(np.float32)
    std = train_tokens.reshape(-1, train_tokens.shape[-1]).std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    input_dim = train_tokens.shape[-1]
    weights = {"input": (rng.normal(0.0, 1.0 / np.sqrt(input_dim), size=(input_dim, MODEL_DIM))).astype(np.float32)}
    for layer in range(ATTENTION_LAYERS):
        scale = 1.0 / np.sqrt(MODEL_DIM)
        weights[f"q{layer}"] = rng.normal(0.0, scale, size=(MODEL_DIM, MODEL_DIM)).astype(np.float32)
        weights[f"k{layer}"] = rng.normal(0.0, scale, size=(MODEL_DIM, MODEL_DIM)).astype(np.float32)
        weights[f"v{layer}"] = rng.normal(0.0, scale, size=(MODEL_DIM, MODEL_DIM)).astype(np.float32)
        weights[f"o{layer}"] = rng.normal(0.0, scale, size=(MODEL_DIM, MODEL_DIM)).astype(np.float32)
    return AttentionEncoder(feature_mean=mean, feature_std=std, weights=weights, token_features=token_features)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.maximum(exp_scores.sum(axis=-1, keepdims=True), 1e-9)


def _layer_norm(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def transform_tokens(encoder: AttentionEncoder, tokens: np.ndarray, return_attention: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    x = ((tokens - encoder.feature_mean) / encoder.feature_std).astype(np.float32)
    h = np.tanh(np.einsum("ntf,fd->ntd", x, encoder.weights["input"]))
    final_attention = None
    for layer in range(ATTENTION_LAYERS):
        q = np.einsum("ntd,df->ntf", h, encoder.weights[f"q{layer}"])
        k = np.einsum("ntd,df->ntf", h, encoder.weights[f"k{layer}"])
        v = np.einsum("ntd,df->ntf", h, encoder.weights[f"v{layer}"])
        scores = np.einsum("ntd,nsd->nts", q, k) / np.sqrt(MODEL_DIM)
        attention = _softmax(scores).astype(np.float32)
        context = np.einsum("nts,nsd->ntd", attention, v)
        projected = np.einsum("ntd,df->ntf", context, encoder.weights[f"o{layer}"])
        h = _layer_norm(np.tanh(projected + h)).astype(np.float32)
        final_attention = attention
    if return_attention:
        return h, final_attention
    return h, None


def attention_step_features(embeddings: np.ndarray, tokens: np.ndarray) -> pd.DataFrame:
    future = embeddings[:, PAST_STEPS:, :]
    past_mean = embeddings[:, :PAST_STEPS, :].mean(axis=1)
    future_mean = future.mean(axis=1)
    future_max = future.max(axis=1)
    rows = []
    for step in range(FUTURE_STEPS):
        lead = np.full((len(embeddings), 1), (step + 1) / FUTURE_STEPS, dtype=np.float32)
        raw_token = tokens[:, PAST_STEPS + step, :]
        features = np.concatenate(
            [
                future[:, step, :],
                past_mean,
                future_mean,
                future_max,
                raw_token,
                lead,
            ],
            axis=1,
        )
        rows.append(features)
    matrix = np.vstack(rows)
    columns = (
        [f"future_token_h_{i}" for i in range(MODEL_DIM)]
        + [f"past_mean_h_{i}" for i in range(MODEL_DIM)]
        + [f"future_mean_h_{i}" for i in range(MODEL_DIM)]
        + [f"future_max_h_{i}" for i in range(MODEL_DIM)]
        + [f"raw_{feature}" for feature in TOKEN_FEATURES]
        + ["lead_fraction"]
    )
    return pd.DataFrame(matrix, columns=columns)


def _row_sample_weights(y_flat: np.ndarray) -> np.ndarray:
    weights = np.ones(len(y_flat), dtype=float)
    weights *= np.where(y_flat >= 5.0, 4.0, 1.0)
    weights *= np.where(y_flat >= 6.0, 3.0, 1.0)
    weights *= np.where(y_flat >= 7.0, 2.0, 1.0)
    return weights / max(float(weights.mean()), 1e-9)


def train_transformer_heads(train_features: pd.DataFrame, y_train: np.ndarray) -> Dict[str, object]:
    y_flat = y_train.T.reshape(-1)
    weights = _row_sample_weights(y_flat)
    heads: Dict[str, object] = {}
    mean_model = HistGradientBoostingRegressor(
        learning_rate=0.045,
        max_iter=180,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=RANDOM_STATE,
    )
    mean_model.fit(train_features, y_flat, sample_weight=weights)
    heads["mean"] = mean_model
    for label, quantile in [("p90", 0.90), ("p95", 0.95)]:
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            learning_rate=0.04,
            max_iter=170,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        )
        model.fit(train_features, y_flat, sample_weight=weights)
        heads[label] = model
    classifiers = {}
    for threshold in THRESHOLDS:
        y_binary = (y_flat >= threshold).astype(int)
        if int(y_binary.sum()) < 8 or len(np.unique(y_binary)) < 2:
            continue
        clf = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=160,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        )
        class_weights = np.where(y_binary == 1, max(2.0, len(y_binary) / max(int(y_binary.sum()), 1) / 2.0), 1.0)
        clf.fit(train_features, y_binary, sample_weight=class_weights)
        classifiers[str(int(threshold))] = clf
    heads["classifiers"] = classifiers
    return heads


def predict_transformer_heads(heads: Dict[str, object], features: pd.DataFrame, n_samples: int) -> Dict[str, np.ndarray]:
    mean = np.clip(heads["mean"].predict(features), 0.0, 9.0).reshape(FUTURE_STEPS, n_samples).T
    p90 = np.clip(heads["p90"].predict(features), 0.0, 9.0).reshape(FUTURE_STEPS, n_samples).T
    p95 = np.clip(heads["p95"].predict(features), 0.0, 9.0).reshape(FUTURE_STEPS, n_samples).T
    predictions = {
        "kp_transformer_mean": mean,
        "kp_transformer_p90": np.maximum(mean, p90),
        "kp_transformer_p95": np.maximum(mean, p95),
    }
    for threshold in THRESHOLDS:
        label = str(int(threshold))
        if label in heads["classifiers"]:
            prob = heads["classifiers"][label].predict_proba(features)[:, 1].reshape(FUTURE_STEPS, n_samples).T
        else:
            prob = np.zeros((n_samples, FUTURE_STEPS), dtype=float)
        predictions[f"prob_transformer_kp_ge_{label}"] = prob
    return predictions


def flatten_transformer_predictions(feature_df: pd.DataFrame, dataset: TransformerTensorDataset, prediction_arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for sample_idx, init_idx in enumerate(dataset.init_indices):
        init_time = feature_df["timestamp"].iloc[init_idx]
        for step in range(FUTURE_STEPS):
            target_idx = init_idx + step + 1
            valid_time = feature_df["timestamp"].iloc[target_idx]
            row = {
                "timestamp": valid_time,
                "t_init": init_time,
                "lead_step": step + 1,
                "lead_time_hours": float((valid_time - init_time) / pd.Timedelta(hours=1)),
                "lead_time_bin": _lead_time_bin(float((valid_time - init_time) / pd.Timedelta(hours=1))),
                "observed_kp": float(feature_df["kp"].iloc[target_idx]),
                "kp_history_source_max_time": init_time,
                "uses_future_observed_kp": False,
            }
            for name, values in prediction_arrays.items():
                row[name] = float(values[sample_idx, step])
            rows.append(row)
    return pd.DataFrame(rows)


def _prediction_metrics(frame: pd.DataFrame, pred_col: str) -> dict:
    y = frame["observed_kp"].to_numpy(dtype=float)
    pred = frame[pred_col].to_numpy(dtype=float)
    row = {
        "n": int(len(frame)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "pearson_corr": _safe_corr(y, pred),
        "bias": float(np.mean(pred - y)),
        "max_underprediction": float(np.max(np.maximum(y - pred, 0.0))) if len(y) else 0.0,
    }
    quiet = y < 5.0
    row["quiet_false_storm_inflation"] = float(np.mean(pred[quiet] >= 5.0)) if quiet.any() else 0.0
    for threshold in THRESHOLDS:
        pred_binary = pred >= threshold
        true_binary = y >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(true_binary, pred_binary, average="binary", zero_division=0)
        row[f"precision_kp_ge_{int(threshold)}"] = float(precision)
        row[f"recall_kp_ge_{int(threshold)}"] = float(recall)
        row[f"f1_kp_ge_{int(threshold)}"] = float(f1)
        prob_col = f"prob_transformer_kp_ge_{int(threshold)}"
        if prob_col in frame.columns:
            row[f"average_precision_kp_ge_{int(threshold)}"] = _safe_average_precision(true_binary, frame[prob_col].to_numpy(dtype=float))
    return row


def transformer_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pred_col in ["kp_transformer_mean", "kp_transformer_p90", "kp_transformer_p95"]:
        rows.append({"model": pred_col, "lead_time_bin": "all", "lead_step": 0, **_prediction_metrics(predictions, pred_col)})
        for lead_step, group in predictions.groupby("lead_step", sort=True):
            rows.append(
                {
                    "model": pred_col,
                    "lead_time_bin": str(group["lead_time_bin"].iloc[0]),
                    "lead_step": int(lead_step),
                    **_prediction_metrics(group, pred_col),
                }
            )
    return pd.DataFrame(rows)


def _load_baseline_predictions(output_dir: str) -> List[Tuple[str, str, pd.DataFrame]]:
    specs = [
        ("experiment12_mean", "kp_seq_mean_forecast", os.path.join(output_dir, "kp_experiment12_unified_rolling_predictions.csv")),
        ("experiment13_severe_overlay", "kp_final_production_forecast", os.path.join(output_dir, "kp_experiment13_final_production_predictions.csv")),
        ("experiment14_extreme_tier", "kp_experiment14_extreme_production", os.path.join(output_dir, "kp_experiment14_extreme_predictions.csv")),
    ]
    loaded = []
    for name, pred_col, path in specs:
        if os.path.exists(path):
            loaded.append((name, pred_col, pd.read_csv(path, parse_dates=["timestamp", "t_init"])))
    return loaded


def _load_experiment15(output_dir: str) -> Optional[Tuple[str, str, pd.DataFrame]]:
    candidates = [
        os.path.join(output_dir, "kp_experiment15_website_one_line_forecast.csv"),
        os.path.join(output_dir, "kp_experiment15_predictions.csv"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path, parse_dates=["timestamp", "t_init"])
        for col in ["kp_experiment15_one_line_forecast", "kp_website_one_line_forecast", "one_line_forecast"]:
            if col in frame.columns:
                return ("experiment15_website_one_line", col, frame)
    return None


def comparison_metrics(transformer_predictions: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    rows = []
    for model, pred_col, frame in _load_baseline_predictions(output_dir):
        common = transformer_predictions[["timestamp", "t_init"]].merge(frame, on=["timestamp", "t_init"], how="left")
        if pred_col not in common.columns:
            continue
        eval_frame = pd.DataFrame(
            {
                "observed_kp": transformer_predictions["observed_kp"].to_numpy(dtype=float),
                pred_col: common[pred_col].to_numpy(dtype=float),
            }
        ).dropna()
        if not eval_frame.empty:
            rows.append({"model": model, "status": "available", **_prediction_metrics(eval_frame, pred_col)})
    exp15 = _load_experiment15(output_dir)
    if exp15 is None:
        rows.append({"model": "experiment15_website_one_line", "status": "not_found"})
    else:
        model, pred_col, frame = exp15
        common = transformer_predictions[["timestamp", "t_init"]].merge(frame, on=["timestamp", "t_init"], how="left")
        eval_frame = pd.DataFrame({"observed_kp": transformer_predictions["observed_kp"], pred_col: common[pred_col]}).dropna()
        rows.append({"model": model, "status": "available", **_prediction_metrics(eval_frame, pred_col)})
    for pred_col in ["kp_transformer_mean", "kp_transformer_p90", "kp_transformer_p95"]:
        rows.append({"model": pred_col, "status": "candidate", **_prediction_metrics(transformer_predictions, pred_col)})
    return pd.DataFrame(rows)


def _plot_transformer_timeseries(predictions: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "kp_experiment16_transformer_timeseries.png")
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
    axes[0].plot(plot_df["timestamp"], plot_df["kp_transformer_mean"], color="tab:blue", linewidth=1.3, label="Transformer mean")
    axes[0].plot(plot_df["timestamp"], plot_df["kp_transformer_p90"], color="tab:orange", linewidth=1.0, label="Transformer p90")
    axes[0].plot(plot_df["timestamp"], plot_df["kp_transformer_p95"], color="tab:red", linewidth=1.0, label="Transformer p95")
    axes[0].axhline(5.0, color="gray", linestyle="--", linewidth=1)
    axes[0].axhline(6.0, color="gray", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Kp")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(plot_df["timestamp"], plot_df["prob_transformer_kp_ge_5"], color="tab:orange", label="P(Kp>=5)")
    axes[1].plot(plot_df["timestamp"], plot_df["prob_transformer_kp_ge_6"], color="tab:red", label="P(Kp>=6)")
    axes[1].plot(plot_df["timestamp"], plot_df["prob_transformer_kp_ge_7"], color="tab:purple", label="P(Kp>=7)")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("Probability")
    axes[1].set_xlabel("Valid time")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_attention_heatmap(attention: np.ndarray, output_dir: str) -> str:
    path = os.path.join(output_dir, "kp_experiment16_attention_heatmap.png")
    mean_attention = attention[:, PAST_STEPS:, :].mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mean_attention, aspect="auto", cmap="viridis")
    ax.set_xlabel("Attended token: past 1-8, future 1-20")
    ax.set_ylabel("Future output token")
    ax.set_title("Experiment 16 mean attention from future tokens")
    ax.axvline(PAST_STEPS - 0.5, color="white", linewidth=1)
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_leadtime(metrics: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "kp_experiment16_leadtime_metrics.png")
    work = metrics[(metrics["model"] == "kp_transformer_mean") & (metrics["lead_step"] > 0)].sort_values("lead_step")
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


# Research role: test a compact NumPy attention model and establish the sequence
# tensor shape. Interpretation boundary: p90/p95 risk envelopes trade accuracy
# for recall and must not be treated as the primary physical Kp calculation.
def run_experiment16_transformer_seq2seq(raw_df: pd.DataFrame, output_dir: str = "testing", test_size: float = 0.2) -> dict:
    _ensure_directory(output_dir)
    feature_df, _ = prepare_temporal_dataframe(raw_df, return_metadata=True)
    train_df, _ = chronological_split(feature_df, test_size)
    split_index = len(train_df)
    cadence_hours = _cadence_hours(feature_df)
    train_data = build_transformer_tensor_dataset(feature_df, PAST_STEPS - 1, split_index - FUTURE_STEPS - 1)
    test_data = build_transformer_tensor_dataset(feature_df, split_index - 1, len(feature_df) - FUTURE_STEPS - 1)

    encoder = fit_attention_encoder(train_data.tokens, train_data.token_features)
    train_embeddings, _ = transform_tokens(encoder, train_data.tokens, return_attention=False)
    test_embeddings, test_attention = transform_tokens(encoder, test_data.tokens, return_attention=True)
    train_features = attention_step_features(train_embeddings, train_data.tokens)
    test_features = attention_step_features(test_embeddings, test_data.tokens)
    heads = train_transformer_heads(train_features, train_data.y)
    prediction_arrays = predict_transformer_heads(heads, test_features, len(test_data.tokens))
    predictions = flatten_transformer_predictions(feature_df, test_data, prediction_arrays)
    metrics = transformer_metrics(predictions)
    comparison = comparison_metrics(predictions, output_dir)

    paths = {
        "predictions": os.path.join(output_dir, "kp_experiment16_transformer_predictions.csv"),
        "metrics": os.path.join(output_dir, "kp_experiment16_transformer_metrics.csv"),
        "comparison": os.path.join(output_dir, "kp_experiment16_backbone_comparison.csv"),
        "debug": os.path.join(output_dir, "kp_experiment16_debug.json"),
        "summary": os.path.join(output_dir, "kp_experiment16_pi_summary.md"),
    }
    paths["timeseries"] = _plot_transformer_timeseries(predictions, output_dir)
    paths["attention_heatmap"] = _plot_attention_heatmap(test_attention, output_dir) if test_attention is not None else ""
    paths["leadtime_plot"] = _plot_leadtime(metrics, output_dir)
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    comparison.to_csv(paths["comparison"], index=False)

    debug = {
        "objective": "Small transformer/attention Seq2Seq candidate backbone using Experiment 12 tensors.",
        "runtime_note": "No PyTorch/TensorFlow installed; implemented NumPy self-attention encoder with supervised sklearn heads.",
        "cadence_hours": cadence_hours,
        "past_steps": PAST_STEPS,
        "future_steps": FUTURE_STEPS,
        "token_features": TOKEN_FEATURES,
        "model_dim": MODEL_DIM,
        "attention_layers": ATTENTION_LAYERS,
        "train_samples": int(len(train_data.tokens)),
        "test_samples": int(len(test_data.tokens)),
        "prediction_rows": int(len(predictions)),
        "uses_future_observed_kp": bool(predictions["uses_future_observed_kp"].astype(bool).any()),
        "experiment15_status": "available" if _load_experiment15(output_dir) is not None else "not_found",
        "paths": paths,
    }
    with open(paths["debug"], "w", encoding="utf-8") as debug_file:
        json.dump(debug, debug_file, indent=2, default=str)
        debug_file.write("\n")

    summary = [
        "# Experiment 16 Transformer Seq2Seq Candidate",
        "## Objective\nTrain and evaluate a small attention-based sequence model using the same operational tensors as Experiment 12. This does not replace Experiments 12-14; it is a candidate backbone.",
        "## Architecture\nA 28-token sequence is built from 8 past assimilation tokens and 20 future forecast-physics tokens. Future tokens contain no Kp. A compact NumPy self-attention encoder produces contextual future-token embeddings; supervised heads emit mean Kp, p90/p95 envelopes, and Kp>=5/6/7/8 probabilities.",
        "## Runtime Note\nPyTorch/TensorFlow are not installed in this workspace, so this branch uses a dependency-light NumPy attention encoder plus scikit-learn heads.",
        "## Transformer Metrics\n" + _markdown_table(metrics[metrics["lead_time_bin"] == "all"]),
        "## Backbone Comparison\n" + _markdown_table(comparison),
        "## Artifacts\n" + "\n".join([f"- `{value}`" for value in paths.values()]),
    ]
    with open(paths["summary"], "w", encoding="utf-8") as summary_file:
        summary_file.write("\n\n".join(summary) + "\n")
    return {"predictions": predictions, "metrics": metrics, "comparison": comparison, "debug": debug, "paths": paths}


def print_validation_matrix(comparison: pd.DataFrame) -> None:
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
    print("\nEXPERIMENT 16 BACKBONE COMPARISON")
    print(comparison[[col for col in cols if col in comparison.columns]].to_string(index=False))


def main() -> None:
    data_path = os.environ.get("KP_EXPERIMENT16_DATA", "testing/2008_2025_df.txt")
    output_dir = os.environ.get("KP_EXPERIMENT16_OUTPUT_DIR", "testing")
    raw_df = pd.read_csv(data_path)
    result = run_experiment16_transformer_seq2seq(raw_df, output_dir=output_dir)
    print_validation_matrix(result["comparison"])
    print("\nExperiment 16 artifacts:")
    for key, value in result["paths"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
