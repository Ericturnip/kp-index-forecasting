"""
Experiment 12: unified rolling Seq2Seq-style Kp refresh pipeline.

This module builds leak-safe sequence tensors:
    past 48h observed physics + observed Kp
    future 120h forecast-valid physics
and trains a multi-output sequence wrapper to predict the full 20-step Kp
trajectory in one pass.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from temporal_model import (
    _cadence_hours,
    _ensure_directory,
    _markdown_table,
    chronological_split,
    prepare_temporal_dataframe,
)


PAST_STEPS = 8
FUTURE_STEPS = 20
FORECAST_HORIZON_HOURS = 120.0
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42

PAST_FEATURES = [
    "bz",
    "velocity",
    "density",
    "pdyn",
    "ey",
    "newell_coupling",
    "kp",
]
FUTURE_FEATURES = [
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
]


@dataclass
class TensorDataset:
    X: pd.DataFrame
    y: np.ndarray
    init_indices: np.ndarray
    init_times: pd.Series
    feature_columns: List[str]


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.isclose(np.nanstd(y_true), 0.0) or np.isclose(np.nanstd(y_pred), 0.0):
        return 0.0
    corr, _ = pearsonr(y_true, y_pred)
    return 0.0 if np.isnan(corr) else float(corr)


def _lead_time_bin(lead_hours: float) -> str:
    for low, high in [(0, 6), (6, 12), (12, 24), (24, 48), (48, 72), (72, 96), (96, 120)]:
        if low < lead_hours <= high or (low == 0 and 0 <= lead_hours <= high):
            return f"{low}-{high}h"
    return ">120h"


def _available(columns: Iterable[str], frame: pd.DataFrame) -> List[str]:
    return [column for column in columns if column in frame.columns]


def _sample_weight_from_target(y: np.ndarray) -> np.ndarray:
    max_kp = np.max(y, axis=1)
    weights = np.ones(len(y), dtype=float)
    weights *= np.where(max_kp >= 4.0, 1.8, 1.0)
    weights *= np.where(max_kp >= 5.0, 3.5, 1.0)
    weights *= np.where(max_kp >= 6.0, 2.5, 1.0)
    weights *= np.where(max_kp >= 7.0, 1.8, 1.0)
    return weights / max(float(np.mean(weights)), 1e-9)


def _sequence_feature_names(
    past_features: List[str],
    future_features: List[str],
    include_past_kp: bool = True,
) -> List[str]:
    names: List[str] = []
    for step in range(PAST_STEPS):
        lag = PAST_STEPS - step - 1
        for feature in past_features:
            if include_past_kp or feature != "kp":
                names.append(f"past_tminus_{lag}_{feature}")
    for step in range(1, FUTURE_STEPS + 1):
        for feature in future_features:
            names.append(f"future_tplus_{step}_{feature}")
    for feature in future_features:
        names.append(f"future_{feature}_mean_120h")
        names.append(f"future_{feature}_max_120h")
        names.append(f"future_{feature}_min_120h")
        names.append(f"future_{feature}_sum_120h")
    return names


# The past tensor may include Kp already observed at initialization. The future
# tensor contains upstream forecast drivers only, never future observed Kp.
def _build_one_sample(
    feature_df: pd.DataFrame,
    init_idx: int,
    past_features: List[str],
    future_features: List[str],
    include_past_kp: bool = True,
) -> Tuple[List[float], np.ndarray]:
    past = feature_df.iloc[init_idx - PAST_STEPS + 1 : init_idx + 1]
    future = feature_df.iloc[init_idx + 1 : init_idx + FUTURE_STEPS + 1]
    values: List[float] = []
    for _, row in past.iterrows():
        for feature in past_features:
            if include_past_kp or feature != "kp":
                values.append(float(row[feature]))
    for _, row in future.iterrows():
        for feature in future_features:
            values.append(float(row[feature]))
    for feature in future_features:
        series = future[feature].astype(float)
        values.extend([float(series.mean()), float(series.max()), float(series.min()), float(series.sum())])
    return values, future["kp"].to_numpy(dtype=float)


def build_sequence_tensor_dataset(
    feature_df: pd.DataFrame,
    start_init_idx: int,
    end_init_idx: int,
    include_past_kp: bool = True,
) -> TensorDataset:
    past_features = _available(PAST_FEATURES, feature_df)
    future_features = _available(FUTURE_FEATURES, feature_df)
    if "kp" not in past_features and include_past_kp:
        raise ValueError("Observed Kp is required in the assimilation lookback tensor.")
    rows = []
    targets = []
    init_indices = []
    first = max(PAST_STEPS - 1, start_init_idx)
    last = min(end_init_idx, len(feature_df) - FUTURE_STEPS - 1)
    for init_idx in range(first, last + 1):
        x_row, y_row = _build_one_sample(feature_df, init_idx, past_features, future_features, include_past_kp)
        rows.append(x_row)
        targets.append(y_row)
        init_indices.append(init_idx)
    feature_columns = _sequence_feature_names(past_features, future_features, include_past_kp=include_past_kp)
    X = pd.DataFrame(rows, columns=feature_columns)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True).fillna(0.0))
    return TensorDataset(
        X=X,
        y=np.asarray(targets, dtype=float),
        init_indices=np.asarray(init_indices, dtype=int),
        init_times=feature_df["timestamp"].iloc[init_indices].reset_index(drop=True),
        feature_columns=feature_columns,
    )


def _tree_quantile_predictions(model: RandomForestRegressor, X: pd.DataFrame, quantile: float) -> np.ndarray:
    X_values = X.to_numpy(dtype=float)
    tree_predictions = np.stack([tree.predict(X_values) for tree in model.estimators_], axis=0)
    return np.quantile(tree_predictions, quantile, axis=0)


def train_seq2seq_models(train_data: TensorDataset) -> Dict[str, object]:
    y_train = np.clip(train_data.y, 0.0, 9.0)
    weights = _sample_weight_from_target(y_train)
    mean_model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        max_features=0.70,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    mean_model.fit(train_data.X, y_train, sample_weight=weights)

    envelope_model = ExtraTreesRegressor(
        n_estimators=450,
        min_samples_leaf=2,
        max_features=0.75,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    envelope_model.fit(train_data.X, y_train, sample_weight=weights)

    ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    ridge_model.fit(train_data.X, y_train)
    return {
        "mean_model": mean_model,
        "envelope_model": envelope_model,
        "ridge_model": ridge_model,
        "feature_columns": train_data.feature_columns,
    }


def predict_seq2seq_bundle(bundle: Dict[str, object], X: pd.DataFrame) -> Dict[str, np.ndarray]:
    mean_pred = np.clip(bundle["mean_model"].predict(X), 0.0, 9.0)
    ridge_pred = np.clip(bundle["ridge_model"].predict(X), 0.0, 9.0)
    # Blend a stable linear sequence model with the nonlinear forest to reduce
    # long-lead variance while keeping storm nonlinearity.
    blended_mean = np.clip(0.72 * mean_pred + 0.28 * ridge_pred, 0.0, 9.0)
    p90 = np.clip(_tree_quantile_predictions(bundle["envelope_model"], X, 0.90), 0.0, 9.0)
    p95 = np.clip(_tree_quantile_predictions(bundle["envelope_model"], X, 0.95), 0.0, 9.0)
    return {
        "kp_seq_mean_forecast": blended_mean,
        "kp_seq_p90_envelope": np.maximum(blended_mean, p90),
        "kp_seq_p95_envelope": np.maximum(blended_mean, p95),
    }


def flatten_rolling_predictions(
    feature_df: pd.DataFrame,
    tensor_data: TensorDataset,
    prediction_arrays: Dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for sample_idx, init_idx in enumerate(tensor_data.init_indices):
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


def _prediction_metrics(frame: pd.DataFrame, pred_col: str) -> dict:
    y = frame["observed_kp"].to_numpy(dtype=float)
    pred = frame[pred_col].to_numpy(dtype=float)
    precision5, recall5, f15, _ = precision_recall_fscore_support(y >= 5.0, pred >= 5.0, average="binary", zero_division=0)
    precision6, recall6, f16, _ = precision_recall_fscore_support(y >= 6.0, pred >= 6.0, average="binary", zero_division=0)
    quiet = y < 5.0
    return {
        "n": int(len(frame)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "pearson_corr": _safe_corr(y, pred),
        "bias": float(np.mean(pred - y)),
        "precision_kp_ge_5": float(precision5),
        "recall_kp_ge_5": float(recall5),
        "f1_kp_ge_5": float(f15),
        "precision_kp_ge_6": float(precision6),
        "recall_kp_ge_6": float(recall6),
        "f1_kp_ge_6": float(f16),
        "quiet_false_storm_inflation": float(np.mean(pred[quiet] >= 5.0)) if int(quiet.sum()) else 0.0,
        "max_underprediction": float(np.max(np.maximum(y - pred, 0.0))) if len(y) else 0.0,
    }


def leadtime_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_cols = ["kp_seq_mean_forecast", "kp_seq_p90_envelope", "kp_seq_p95_envelope"]
    for model_col in model_cols:
        rows.append({"model": model_col, "lead_time_bin": "all", "lead_step": 0, **_prediction_metrics(predictions, model_col)})
        for lead_step, group in predictions.groupby("lead_step", sort=True):
            rows.append(
                {
                    "model": model_col,
                    "lead_time_bin": _lead_time_bin(float(group["lead_time_hours"].iloc[0])),
                    "lead_step": int(lead_step),
                    **_prediction_metrics(group, model_col),
                }
            )
    return pd.DataFrame(rows)


def _plot_experiment12(predictions: pd.DataFrame, metrics: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    paths = {
        "timeseries": os.path.join(output_dir, "kp_experiment12_seq2seq_timeseries.png"),
        "leadtime": os.path.join(output_dir, "kp_experiment12_leadtime_curve.png"),
    }
    refreshed = predictions[predictions["lead_step"] == 1].copy()
    refreshed["timestamp"] = pd.to_datetime(refreshed["timestamp"])
    storm_rows = refreshed[refreshed["observed_kp"] >= 5.0]
    if not storm_rows.empty:
        center = storm_rows.iloc[len(storm_rows) // 2]["timestamp"]
        plot_df = refreshed[(refreshed["timestamp"] >= center - pd.Timedelta(days=12)) & (refreshed["timestamp"] <= center + pd.Timedelta(days=12))]
    else:
        plot_df = refreshed.head(320)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(plot_df["timestamp"], plot_df["observed_kp"], color="black", linewidth=2, label="Observed Kp")
    ax.plot(plot_df["timestamp"], plot_df["kp_seq_mean_forecast"], color="tab:blue", linewidth=1.4, label="Seq2Seq mean")
    ax.plot(plot_df["timestamp"], plot_df["kp_seq_p90_envelope"], color="tab:red", linewidth=1.1, label="Seq2Seq p90 envelope")
    ax.fill_between(plot_df["timestamp"], plot_df["kp_seq_mean_forecast"], plot_df["kp_seq_p90_envelope"], color="tab:red", alpha=0.12)
    ax.axhline(5.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Kp")
    ax.set_xlabel("Valid time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths["timeseries"], dpi=180)
    plt.close(fig)

    step_metrics = metrics[(metrics["model"] == "kp_seq_mean_forecast") & (metrics["lead_step"] > 0)].sort_values("lead_step")
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(step_metrics["lead_time_bin"], step_metrics["mae"], marker="o", color="tab:blue", label="MAE")
    ax1.set_ylabel("MAE")
    ax1.set_xlabel("Lead time")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(step_metrics["lead_time_bin"], step_metrics["pearson_corr"], marker="s", color="tab:green", label="Pearson r")
    ax2.set_ylabel("Pearson r")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(paths["leadtime"], dpi=180)
    plt.close(fig)
    return paths


def _load_legacy_metrics(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, "kp_experiment7_dynamic_backtest_metrics.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    legacy = pd.read_csv(path)
    legacy["model"] = "experiment7_dynamic_frozen"
    return legacy


# At every refresh, rebuild the whole horizon from a frozen observation boundary.
# Do not mix predictions produced under different initialization times or splits.
def run_experiment12_unified_refresh(
    raw_df: pd.DataFrame,
    output_dir: str = "testing",
    test_size: float = DEFAULT_TEST_SIZE,
) -> dict:
    _ensure_directory(output_dir)
    feature_df, metadata = prepare_temporal_dataframe(raw_df, return_metadata=True)
    train_df, test_df = chronological_split(feature_df, test_size)
    split_index = len(train_df)
    cadence_hours = _cadence_hours(feature_df)
    if not np.isclose(cadence_hours, 6.0, atol=1.0):
        print(f"WARNING: detected cadence is {cadence_hours:.3f}h, but Experiment 12 expects about 6h.")

    train_tensor = build_sequence_tensor_dataset(feature_df, PAST_STEPS - 1, split_index - FUTURE_STEPS - 1, include_past_kp=True)
    test_tensor = build_sequence_tensor_dataset(feature_df, split_index - 1, len(feature_df) - FUTURE_STEPS - 1, include_past_kp=True)
    no_assim_train = build_sequence_tensor_dataset(feature_df, PAST_STEPS - 1, split_index - FUTURE_STEPS - 1, include_past_kp=False)
    no_assim_test = build_sequence_tensor_dataset(feature_df, split_index - 1, len(feature_df) - FUTURE_STEPS - 1, include_past_kp=False)

    bundle = train_seq2seq_models(train_tensor)
    pred_arrays = predict_seq2seq_bundle(bundle, test_tensor.X)
    predictions = flatten_rolling_predictions(feature_df, test_tensor, pred_arrays)

    no_assim_bundle = train_seq2seq_models(no_assim_train)
    no_assim_pred = predict_seq2seq_bundle(no_assim_bundle, no_assim_test.X)
    no_assim_predictions = flatten_rolling_predictions(feature_df, no_assim_test, no_assim_pred)
    no_assim_predictions = no_assim_predictions.rename(
        columns={
            "kp_seq_mean_forecast": "kp_seq_no_assim_mean",
            "kp_seq_p90_envelope": "kp_seq_no_assim_p90",
            "kp_seq_p95_envelope": "kp_seq_no_assim_p95",
        }
    )
    predictions = predictions.merge(
        no_assim_predictions[["timestamp", "t_init", "kp_seq_no_assim_mean", "kp_seq_no_assim_p90"]],
        on=["timestamp", "t_init"],
        how="left",
    )
    metrics = leadtime_analysis(predictions)
    no_assim_metric = []
    for lead_step, group in predictions.groupby("lead_step", sort=True):
        no_assim_metric.append(
            {
                "model": "kp_seq_no_assim_mean",
                "lead_time_bin": _lead_time_bin(float(group["lead_time_hours"].iloc[0])),
                "lead_step": int(lead_step),
                **_prediction_metrics(group.rename(columns={"kp_seq_no_assim_mean": "tmp_pred"}), "tmp_pred"),
            }
        )
    no_assim_metric.append(
        {
            "model": "kp_seq_no_assim_mean",
            "lead_time_bin": "all",
            "lead_step": 0,
            **_prediction_metrics(predictions.rename(columns={"kp_seq_no_assim_mean": "tmp_pred"}), "tmp_pred"),
        }
    )
    metrics = pd.concat([metrics, pd.DataFrame(no_assim_metric)], ignore_index=True)

    paths = {
        "predictions": os.path.join(output_dir, "kp_experiment12_unified_rolling_predictions.csv"),
        "leadtime_analysis": os.path.join(output_dir, "kp_experiment12_leadtime_analysis.csv"),
        "pi_summary": os.path.join(output_dir, "kp_experiment12_pi_summary.md"),
        "debug": os.path.join(output_dir, "kp_experiment12_debug.json"),
    }
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["leadtime_analysis"], index=False)
    paths.update(_plot_experiment12(predictions, metrics, output_dir))

    legacy = _load_legacy_metrics(output_dir)
    legacy_compare = pd.DataFrame()
    if not legacy.empty:
        seq = metrics[(metrics["model"] == "kp_seq_mean_forecast") & (metrics["lead_step"] > 0)].copy()
        legacy_step = legacy[["lead_time_bin", "mae", "rmse", "pearson_corr"]].rename(
            columns={"mae": "experiment7_mae", "rmse": "experiment7_rmse", "pearson_corr": "experiment7_pearson_corr"}
        )
        legacy_compare = seq.merge(legacy_step, on="lead_time_bin", how="left")
        legacy_compare["mae_delta_vs_experiment7"] = legacy_compare["mae"] - legacy_compare["experiment7_mae"]

    assimilated_all = metrics[(metrics["model"] == "kp_seq_mean_forecast") & (metrics["lead_time_bin"] == "all")].iloc[0]
    no_assim_all = metrics[(metrics["model"] == "kp_seq_no_assim_mean") & (metrics["lead_time_bin"] == "all")].iloc[0]
    assimilation_delta = float(no_assim_all["mae"] - assimilated_all["mae"])
    debug = {
        "cadence_hours": cadence_hours,
        "past_steps": PAST_STEPS,
        "future_steps": FUTURE_STEPS,
        "train_samples": int(len(train_tensor.X)),
        "test_samples": int(len(test_tensor.X)),
        "prediction_rows": int(len(predictions)),
        "past_features": _available(PAST_FEATURES, feature_df),
        "future_features": _available(FUTURE_FEATURES, feature_df),
        "uses_future_observed_kp": bool(predictions["uses_future_observed_kp"].astype(bool).any()),
        "assimilation_mae_delta_vs_no_assim": assimilation_delta,
        "paths": paths,
    }
    with open(paths["debug"], "w", encoding="utf-8") as debug_file:
        json.dump(debug, debug_file, indent=2, default=str)
        debug_file.write("\n")

    all_rows = metrics[metrics["lead_time_bin"] == "all"].copy()
    lead_rows = metrics[(metrics["model"] == "kp_seq_mean_forecast") & (metrics["lead_step"] > 0)].copy()
    summary_sections = [
        "# Experiment 12 Unified Seq2Seq Rolling Refresh",
        "## Architecture\nA single multi-output sequence wrapper ingests a 48h assimilation lookback tensor and a 120h forecasted physics tensor, then emits the full 20-step Kp trajectory in one pass. The p90/p95 envelopes are derived from the tree ensemble distribution.",
        "## Data Assimilation Rule\nObserved Kp appears only in the historical lookback ending at t_init. The future horizon tensor contains forecast-valid physics and Bz turning signatures, with no observed or predicted Kp inside the block.",
        f"## Data Assimilation Value\nAll-row MAE improvement versus the no-Kp-history ablation: `{assimilation_delta:.4f}`. Positive means the 48h assimilation window helped.",
        "## All-Row Metrics\n" + _markdown_table(all_rows),
        "## Mean Forecast Lead-Time Metrics\n" + _markdown_table(lead_rows[["lead_step", "lead_time_bin", "mae", "rmse", "pearson_corr", "recall_kp_ge_5", "quiet_false_storm_inflation"]]),
        "## Legacy Comparison\n" + (_markdown_table(legacy_compare[["lead_step", "lead_time_bin", "mae", "experiment7_mae", "mae_delta_vs_experiment7", "pearson_corr", "experiment7_pearson_corr"]]) if not legacy_compare.empty else "Legacy Experiment 7 dynamic metrics were not found."),
        "## Deployment Blueprint\nAt every 6h refresh, freeze observed Kp through t_init, ingest the next 20 rows of upstream forecasted physics, build the two tensors, and publish the mean Kp curve plus p90/p95 risk envelopes. When the next Kp observation arrives, discard overlapping old short-lead predictions and regenerate the full 5-day curve.",
        "## Artifacts\n" + "\n".join([f"- `{value}`" for value in paths.values()]),
    ]
    with open(paths["pi_summary"], "w", encoding="utf-8") as summary_file:
        summary_file.write("\n\n".join(summary_sections) + "\n")
    return {
        "predictions": predictions,
        "metrics": metrics,
        "legacy_comparison": legacy_compare,
        "paths": paths,
        "debug": debug,
    }


def print_validation_matrix(metrics: pd.DataFrame, legacy_comparison: pd.DataFrame) -> None:
    seq = metrics[(metrics["model"] == "kp_seq_mean_forecast") & (metrics["lead_step"] > 0)].copy()
    display = seq[["lead_step", "lead_time_bin", "mae", "rmse", "pearson_corr", "recall_kp_ge_5"]].copy()
    if not legacy_comparison.empty:
        display = display.merge(
            legacy_comparison[["lead_step", "experiment7_mae", "experiment7_pearson_corr", "mae_delta_vs_experiment7"]],
            on="lead_step",
            how="left",
        )
    print("\nEXPERIMENT 12 VALIDATION MATRIX")
    print(display.to_string(index=False))


def main() -> None:
    data_path = os.environ.get("KP_EXPERIMENT12_DATA", "testing/2008_2025_df.txt")
    output_dir = os.environ.get("KP_EXPERIMENT12_OUTPUT_DIR", "testing")
    raw_df = pd.read_csv(data_path)
    result = run_experiment12_unified_refresh(raw_df, output_dir=output_dir)
    print_validation_matrix(result["metrics"], result["legacy_comparison"])
    print("\nExperiment 12 artifacts:")
    for key, value in result["paths"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
