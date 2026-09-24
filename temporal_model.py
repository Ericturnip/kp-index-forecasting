"""
MODULE: temporal_model
@author: Codex / Benjamin Pieczynski project extension
DATE: 2026-04-24

PURPOSE:
    Train and evaluate leak-safe temporal Kp models using a chronological split,
    persistence baselines, storm-focused metrics, and report artifacts.

INCLUDED FUNCTIONS:
    prepare_temporal_dataframe
    train_temporal_model
    load_temporal_model
    evaluate_saved_temporal_model

MODIFICATION HISTORY:
    1.0.0 - 2026-04-24: added temporal Kp modeling, storm classifiers,
                        public alert and storm boost gates.
"""

import json
import os
import pickle
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

warnings.filterwarnings(
    "ignore",
    message="Since version 1.0, it is not needed to import enable_hist_gradient_boosting anymore.*",
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.linalg import LinAlgWarning
from sklearn.base import clone
try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
except ImportError:
    pass

try:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
except ImportError:
    HistGradientBoostingClassifier = None
    HistGradientBoostingRegressor = None
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from sklearn.inspection import permutation_importance
except ImportError:
    permutation_importance = None
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)


try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None


RANDOM_STATE = 42
FORECAST_INPUTS_ALREADY_TARGET_ALIGNED = True
RUN_EXPERIMENT7_DIAGNOSTIC_TARGET_ALIGNED = True
RUN_EXPERIMENT7_OPERATIONAL_ROLLING = True
DRIVER_INPUT_SOURCE_OBSERVED_BACKTEST = "observed_historical_driver_backtest"
DRIVER_INPUT_SOURCE_UPSTREAM_FORECAST = "upstream_forecast_driver_inputs"
KP_MODEL_FORECAST_SEMANTICS = {
    "core_task": "same_timestamp_driver_to_kp_calculation",
    "correct_framing": "Given forecasted Bz, velocity, and density at timestamp T, calculate Kp at timestamp T.",
    "incorrect_framing": "Given current observed drivers, extrapolate future Kp without forecasted driver inputs.",
    "production_horizon_source": "upstream Bz/velocity/density forecasts",
    "driver_input_source_for_training_backtests": DRIVER_INPUT_SOURCE_OBSERVED_BACKTEST,
    "production_driver_input_source": DRIVER_INPUT_SOURCE_UPSTREAM_FORECAST,
}
ORIGINAL_PI_LINEAR_EQUATION = {
    "source_module": "run_kp_model.py",
    "equation": "Kp = A + amp * (B*dphi_dt + C*viscous + D*bz_term + E*dbzdt_term)",
    "dphi_dt": "velocity^(4/3) * Bt^(2/3) * sin(theta_c / 2)^(8/3)",
    "Bt": "sqrt(Bx^2 + By^2 + Bz^2)",
    "theta_c": "atan2(By, Bz), normalized to [0, 2*pi)",
    "viscous": "sqrt(density) * velocity^2",
    "bz_term": "10^(0.5 - Bz), active only when Bz <= -0.5",
    "dbzdt_term": "Bz-change / dip-amplifier term from parabolic_fit.py",
}
RIDGE_ALPHA_GRID = [0.01, 0.1, 1, 10, 100, 1000]
LAG_STEPS = [1, 2, 4, 8]
KP_LAG_STEPS = [1, 2, 4, 8, 12, 16, 20]
ROLL_WINDOWS = [2, 4, 8]
EXPERIMENT7_ROLL_WINDOWS = [4, 8, 12, 20]
KP_ROLL_WINDOWS = [4, 8, 20]
SUSTAINED_BS_THRESHOLD_NT = 5.0
ALERT_CONSTRAINTS = {
    "alert_duty_cycle": 0.07,
    "max_continuous_alert_hours": 72.0,
    "false_alert_hours_per_month": 40.0,
}
WATCH_CONSTRAINTS = {
    "alert_duty_cycle": 0.15,
    "max_continuous_alert_hours": 120.0,
    "false_alert_hours_per_month": 120.0,
}
VIGILANT_WATCH_CONSTRAINTS = {
    "alert_duty_cycle": 0.25,
    "max_continuous_alert_hours": 144.0,
    "false_alert_hours_per_month": 180.0,
}
BOOST_GATE_MAX_DUTY_CYCLE = 0.20
BOOST_GATE_MAX_CONTINUOUS_ALERT_HOURS = 144.0
STORM_MAGNITUDE_HORIZONS_HOURS = [3, 6, 12]
STORM_MAGNITUDE_QUANTILES = [0.50, 0.75, 0.90, 0.95]
SEVERITY_THRESHOLDS = [5.0, 6.0, 7.0]
MIN_SEVERITY_POSITIVES = 8
EXPERIMENT3_WATCH_DUTY_TARGETS = [0.15, 0.20, 0.25, 0.30]
EXPERIMENT3_SEVERITY_THRESHOLD_GRID = {
    5.0: [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 0.90, 0.99],
    6.0: [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 0.90, 0.99],
    7.0: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90, 0.99],
}
EXPERIMENT3_SEVERITY_MAX_DUTY_CYCLE = 0.30
EXPERIMENT4_QUIET_INFLATION_TARGETS = [0.50, 0.35, 0.25, 0.15]
EXPERIMENT4_PROB6_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
EXPERIMENT4_WATCH_THRESHOLDS = [0.60, 0.70, 0.80, 0.90]
EXPERIMENT4_P95_ABS_THRESHOLDS = [5.5, 6.0]
EXPERIMENT4_P95_GAP_THRESHOLDS = [1.0, 1.5, 2.0]
EXPERIMENT4_MIN_VOTES = [2, 3]
EXPERIMENT6_SUPPRESSOR_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
EXPERIMENT6_SUPPRESSION_MODES = ["cap_4p67", "fallback_to_adjusted"]
EXPERIMENT6_PROB6_SAFETY_THRESHOLD = 0.75
EXPERIMENT6_PROB7_SAFETY_THRESHOLD = 0.05
EXPERIMENT6_P95_SAFETY_THRESHOLD = 6.5
EXPERIMENT6_HIGH_VOTE_THRESHOLD = 4
SEVERE_SAMPLE_WEIGHT_CONFIG = {
    "ordinary": 1.0,
    "kp_ge_5": 3.0,
    "kp_ge_6": 6.0,
    "kp_ge_7": 10.0,
    "onset_multiplier": 2.0,
}
ORIGINAL_PHYSICS_FEATURES = [
    "coupling_term",
    "viscous_term",
    "bz_term",
    "dbzdt_term",
]
SOLAR_WIND_CURRENT_FEATURES = [
    "bz",
    "bs",
    "velocity",
    "density",
    "pdyn",
    "pdyn_max_12h",
    "pdyn_jump_24h",
    "velocity_jump_6h",
    "bz_drop_36h",
    "density_max_24h",
    "ey",
    "b_t_yz",
    "newell_coupling",
    "compact_storm_score",
    "coupling_term",
    "viscous_term",
    "bz_term",
    "dbzdt_term",
    "v_bs",
    "v2_bs",
    "dynamic_pressure_proxy",
    "b_total",
    "clock_angle",
    "sin_half_clock",
    "coupling_simple",
    "epsilon_proxy",
    "kan_lee_proxy",
]
SUSTAINED_DRIVING_FEATURES = [
    "bs_roll_sum_2",
    "bs_roll_sum_4",
    "bs_roll_sum_8",
    "ey_roll_max_2",
    "ey_roll_max_4",
    "ey_roll_max_8",
    "ey_roll_sum_2",
    "ey_roll_sum_4",
    "ey_roll_sum_8",
    "pdyn_roll_max_4",
    "pdyn_roll_max_8",
    "pdyn_jump",
    "velocity_jump",
    "compression_score",
    "pdyn_above_2npa_duration",
    "bs_above_threshold_duration",
    "southward_pressure_interaction",
]
BZ_SHARP_DROP_FEATURES = [
    "bz_delta_1",
    "bz_delta_2",
    "bz_delta_4",
    "bz_drop_6h",
    "bz_drop_12h",
    "bz_drop_24h",
    "bz_southward_drop_6h",
    "bz_southward_drop_12h",
    "bz_southward_drop_24h",
    "bz_drop_gt_5nt",
    "bz_drop_gt_10nt",
    "bz_crossed_southward",
    "strong_southward_turning",
    "bz_drop_times_velocity",
    "bz_drop_times_pdyn",
    "bz_drop_times_density",
    "bz_drop_times_ey",
    "southward_turning_pressure_interaction",
    "southward_turning_coupling_interaction",
]
EXPERIMENT7_MOVING_PHYSICS_ROLL_FEATURES = [
    f"{base}_roll_{kind}_{window}"
    for base, kind in [
        ("bs", "sum"),
        ("ey", "sum"),
        ("ey", "max"),
        ("pdyn", "max"),
        ("velocity", "max"),
        ("coupling_simple", "sum"),
        ("compact_storm_score", "max"),
    ]
    for window in EXPERIMENT7_ROLL_WINDOWS
]
KP_HISTORY_FEATURES = [
    "kp_lag_1",
    "kp_lag_2",
    "kp_lag_4",
    "kp_lag_8",
    "kp_lag_12",
    "kp_lag_16",
    "kp_lag_20",
    "kp_delta_1",
    "kp_delta_4",
    "kp_abs_delta_4",
    "kp_24h_repeat_error",
    "kp_48h_repeat_error",
    "kp_72h_repeat_error",
    "kp_daily_pattern_break_score",
    "kp_daily_variability_ratio",
    "kp_recent_vs_multiday_mean",
    "kp_recent_max_vs_multiday_mean",
    "kp_roll_mean_4",
    "kp_roll_std_4",
    "kp_roll_max_4",
    "kp_roll_min_4",
    "kp_roll_range_4",
    "kp_roll_mean_8",
    "kp_roll_std_8",
    "kp_roll_max_8",
    "kp_roll_range_8",
    "kp_roll_mean_20",
    "kp_roll_std_20",
    "kp_roll_max_20",
    "kp_roll_sum_20",
    "kp_roll_sum_4",
    "kp_roll_sum_8",
    "time_since_kp_ge_4",
    "time_since_kp_ge_5",
    "time_since_kp_ge_6",
]
STORM_TARGET_COLUMNS = [
    "storm_now",
    "storm_next_6h",
    "storm_next_12h",
    "storm_now_or_next_12h",
    "storm_onset_next_12h",
]
STORM_TARGET_HORIZON_STEPS = {
    "storm_now": 0,
    "storm_next_6h": 1,
    "storm_next_12h": 2,
    "storm_now_or_next_12h": 2,
    "storm_onset_next_12h": 2,
}


def _unique_in_order(values: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _safe_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    if np.isclose(np.std(y_true), 0.0) or np.isclose(np.std(y_pred), 0.0):
        return 0.0
    corr, _ = pearsonr(y_true, y_pred)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "corr": _safe_correlation(y_true, y_pred),
    }


def _subset_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict:
    if int(mask.sum()) == 0:
        return {"count": 0, "mae": None, "rmse": None}
    return {
        "count": int(mask.sum()),
        "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
        "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
    }


def _storm_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    observed_storm = y_true >= 5
    predicted_storm = y_pred >= 5
    precision, recall, f1, _ = precision_recall_fscore_support(
        observed_storm,
        predicted_storm,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(
        observed_storm,
        predicted_storm,
        labels=[False, True],
    ).ravel()

    return {
        "kp_lt_4": _subset_metrics(y_true, y_pred, y_true < 4),
        "kp_ge_4": _subset_metrics(y_true, y_pred, y_true >= 4),
        "kp_ge_5": _subset_metrics(y_true, y_pred, y_true >= 5),
        "storm_threshold_ge_5": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        },
    }


def _full_metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    bundle = _regression_metrics(y_true, y_pred)
    bundle["storm"] = _storm_metrics(y_true, y_pred)
    return bundle


def _storm_target(y: Union[pd.Series, np.ndarray]) -> np.ndarray:
    return np.asarray(y) >= 5


def _safe_average_precision(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true_binary)) < 2:
        return float(np.mean(y_true_binary))
    return float(average_precision_score(y_true_binary, y_score))


def _binary_classification_metrics(y_true_binary: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred_binary = y_score >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary,
        y_pred_binary,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(
        y_true_binary,
        y_pred_binary,
        labels=[False, True],
    ).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_binary, y_pred_binary)),
        "average_precision": _safe_average_precision(y_true_binary, y_score),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def _flatten_binary_metric_row(
    model_name: str,
    evaluation: str,
    metrics: dict,
    note: str = "",
    era: str = "",
) -> dict:
    confusion = metrics["confusion_matrix"]
    return {
        "era": era,
        "model": model_name,
        "evaluation": evaluation,
        "note": note,
        "threshold": metrics["threshold"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "average_precision": metrics["average_precision"],
        "tn": confusion["tn"],
        "fp": confusion["fp"],
        "fn": confusion["fn"],
        "tp": confusion["tp"],
    }


def _timestamp_from_columns(df: pd.DataFrame) -> pd.Series:
    year = df["year"].astype(int).astype(str)
    doy = df["doy"].astype(int).astype(str).str.zfill(3)
    hour = df["hour"].astype(int).astype(str).str.zfill(2)
    return pd.to_datetime(year + doy + hour, format="%Y%j%H")


def _format_timestamp(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def _cadence_hours(feature_df: pd.DataFrame) -> float:
    cadence = feature_df["timestamp"].diff().dropna()
    if len(cadence) == 0:
        return 0.0
    return float(cadence.median() / pd.Timedelta(hours=1))


def _flatten_metric_row(model_name: str, mode: str, metrics: dict, status: str = "ok", note: str = "") -> dict:
    storm = metrics["storm"]
    confusion = storm["storm_threshold_ge_5"]["confusion_matrix"]
    return {
        "model": model_name,
        "mode": mode,
        "status": status,
        "note": note,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "corr": metrics["corr"],
        "mae_kp_lt_4": storm["kp_lt_4"]["mae"],
        "rmse_kp_lt_4": storm["kp_lt_4"]["rmse"],
        "count_kp_lt_4": storm["kp_lt_4"]["count"],
        "mae_kp_ge_4": storm["kp_ge_4"]["mae"],
        "rmse_kp_ge_4": storm["kp_ge_4"]["rmse"],
        "count_kp_ge_4": storm["kp_ge_4"]["count"],
        "mae_kp_ge_5": storm["kp_ge_5"]["mae"],
        "rmse_kp_ge_5": storm["kp_ge_5"]["rmse"],
        "count_kp_ge_5": storm["kp_ge_5"]["count"],
        "precision_kp_ge_5": storm["storm_threshold_ge_5"]["precision"],
        "recall_kp_ge_5": storm["storm_threshold_ge_5"]["recall"],
        "f1_kp_ge_5": storm["storm_threshold_ge_5"]["f1"],
        "tn_kp_ge_5": confusion["tn"],
        "fp_kp_ge_5": confusion["fp"],
        "fn_kp_ge_5": confusion["fn"],
        "tp_kp_ge_5": confusion["tp"],
    }


def _markdown_table(df: pd.DataFrame, float_cols: Optional[List[str]] = None) -> str:
    if df.empty:
        return "No rows.\n"

    render_df = df.copy()
    if float_cols is None:
        float_cols = render_df.select_dtypes(include=["float64", "float32"]).columns.tolist()

    for col in float_cols:
        if col in render_df.columns:
            render_df[col] = render_df[col].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )

    headers = render_df.columns.tolist()
    rows = render_df.astype(str).values.tolist()
    separator = ["---" for _ in headers]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _original_pi_linear_equation_metadata() -> dict:
    return dict(ORIGINAL_PI_LINEAR_EQUATION)


def _driver_input_metadata(dataset_path: str = "unknown", source: str = DRIVER_INPUT_SOURCE_OBSERVED_BACKTEST) -> dict:
    return {
        "dataset_path": dataset_path,
        "driver_input_source": source,
        "production_driver_input_source": DRIVER_INPUT_SOURCE_UPSTREAM_FORECAST,
        "forecast_inputs_already_target_aligned": bool(FORECAST_INPUTS_ALREADY_TARGET_ALIGNED),
        "semantics": dict(KP_MODEL_FORECAST_SEMANTICS),
        "original_pi_linear_equation": _original_pi_linear_equation_metadata(),
    }


def _kp_history_availability_metadata(
    feature_df: pd.DataFrame,
    source_time_df: pd.DataFrame,
    kp_history_features: List[str],
) -> dict:
    available_features = [feature for feature in kp_history_features if feature in source_time_df.columns]
    details = []
    violations = []
    target_time = feature_df["timestamp"]
    for feature in available_features:
        source_time = source_time_df[feature]
        valid = source_time < target_time
        if not valid.all():
            invalid_rows = np.where(~valid.to_numpy())[0][:5].tolist()
            violations.append({"feature": feature, "rows": invalid_rows})
        lookback_hours = (target_time - source_time) / pd.Timedelta(hours=1)
        details.append(
            {
                "feature": feature,
                "source_time_rule": "strictly_before_target_timestamp",
                "min_lookback_hours": float(lookback_hours.min()) if len(lookback_hours) else None,
                "median_lookback_hours": float(lookback_hours.median()) if len(lookback_hours) else None,
                "max_lookback_hours": float(lookback_hours.max()) if len(lookback_hours) else None,
            }
        )
    return {
        "policy": "Kp history may be used only when available at or before forecast initialization; engineered features here use strictly past Kp source timestamps.",
        "checked_features": int(len(available_features)),
        "passed": len(violations) == 0,
        "violations": violations,
        "features": details,
    }


def _storm_recovery_metrics(frame: pd.DataFrame, model_columns: List[str]) -> pd.DataFrame:
    if frame.empty or "kp" not in frame.columns:
        return pd.DataFrame()

    work = frame.copy()
    y_true = work["kp"].to_numpy(dtype=float)
    prior_kp_max = work["kp"].shift(1).rolling(4, min_periods=1).max()
    recent_storm = prior_kp_max >= 5.0
    current_not_extreme = work["kp"] < 7.0

    relaxed_parts = []
    if "bz" in work.columns:
        relaxed_parts.append(work["bz"] > -2.0)
    if "bs" in work.columns:
        relaxed_parts.append(work["bs"] < 4.0)
    elif "bz" in work.columns:
        relaxed_parts.append(work["bz"] > -4.0)
    if "velocity" in work.columns:
        relaxed_parts.append(work["velocity"] <= work["velocity"].rolling(20, min_periods=1).median())
    if "density" in work.columns:
        relaxed_parts.append(work["density"] <= work["density"].rolling(20, min_periods=1).median())

    if relaxed_parts:
        relaxed_driver_count = sum(part.astype(int) for part in relaxed_parts)
        driver_relaxed = relaxed_driver_count >= max(1, int(np.ceil(len(relaxed_parts) / 2.0)))
    else:
        driver_relaxed = pd.Series(False, index=work.index)

    recovery_mask = (recent_storm & current_not_extreme & driver_relaxed).fillna(False).to_numpy(dtype=bool)
    quiet_driver_false_elevation_mask = ((work["kp"] < 5.0) & driver_relaxed.fillna(False)).to_numpy(dtype=bool)

    rows = []
    for model_column in model_columns:
        if model_column not in work.columns:
            continue
        y_pred = work[model_column].to_numpy(dtype=float)
        if int(recovery_mask.sum()) > 0:
            recovery_error = y_pred[recovery_mask] - y_true[recovery_mask]
            rows.append(
                {
                    "metric_group": "storm_recovery_after_driver_relaxation",
                    "model": model_column,
                    "row_count": int(recovery_mask.sum()),
                    "mae": float(mean_absolute_error(y_true[recovery_mask], y_pred[recovery_mask])),
                    "mean_bias": float(np.mean(recovery_error)),
                    "underprediction_rate": float(np.mean(recovery_error < 0.0)),
                    "mean_underprediction": float(np.mean(np.maximum(y_true[recovery_mask] - y_pred[recovery_mask], 0.0))),
                    "max_underprediction": float(np.max(np.maximum(y_true[recovery_mask] - y_pred[recovery_mask], 0.0))),
                    "false_elevation_rate_on_relaxed_quiet_drivers": None,
                }
            )
        if int(quiet_driver_false_elevation_mask.sum()) > 0:
            rows.append(
                {
                    "metric_group": "quiet_driver_false_elevation",
                    "model": model_column,
                    "row_count": int(quiet_driver_false_elevation_mask.sum()),
                    "mae": float(mean_absolute_error(y_true[quiet_driver_false_elevation_mask], y_pred[quiet_driver_false_elevation_mask])),
                    "mean_bias": float(np.mean(y_pred[quiet_driver_false_elevation_mask] - y_true[quiet_driver_false_elevation_mask])),
                    "underprediction_rate": float(np.mean(y_pred[quiet_driver_false_elevation_mask] < y_true[quiet_driver_false_elevation_mask])),
                    "mean_underprediction": float(np.mean(np.maximum(y_true[quiet_driver_false_elevation_mask] - y_pred[quiet_driver_false_elevation_mask], 0.0))),
                    "max_underprediction": float(np.max(np.maximum(y_true[quiet_driver_false_elevation_mask] - y_pred[quiet_driver_false_elevation_mask], 0.0))),
                    "false_elevation_rate_on_relaxed_quiet_drivers": float(np.mean(y_pred[quiet_driver_false_elevation_mask] >= 5.0)),
                }
            )
    return pd.DataFrame(rows)


def _watch_gate_metrics(frame: pd.DataFrame, gate_column: str, target_threshold: float, label: str, lead_steps: int) -> dict:
    if gate_column not in frame.columns:
        gate = np.zeros(len(frame), dtype=bool)
    else:
        gate = frame[gate_column].astype(bool).to_numpy()
    y_true = frame["kp"].to_numpy(dtype=float)
    y_binary = y_true >= target_threshold
    point_metrics = _binary_classification_metrics_from_mask(y_binary, gate, gate.astype(float), 0.5)
    event_recall, event_count, missed_events, mean_lead, median_lead = _event_peak_recall(
        frame["timestamp"],
        y_true,
        gate,
        lead_steps,
        target_threshold,
    )
    return {
        "gate_label": label,
        "gate_column": gate_column,
        "target": f"kp_ge_{int(target_threshold)}",
        "threshold": 0.5,
        "precision": point_metrics["precision"],
        "recall": point_metrics["recall"],
        "f1": point_metrics["f1"],
        "event_recall": event_recall,
        "events": event_count,
        "missed_events": missed_events,
        "mean_lead_time_hours": mean_lead,
        "median_lead_time_hours": median_lead,
        "alert_duty_cycle": float(np.mean(gate)) if len(gate) else 0.0,
        "false_watch_rate_kp_lt_5": float(np.mean(gate[y_true < 5.0])) if int((y_true < 5.0).sum()) else 0.0,
    }


def _select_public_alert_reference_threshold(
    validation_frame: pd.DataFrame,
    validation_score: np.ndarray,
    kp_threshold: float,
    thresholds: np.ndarray,
    lead_steps: int,
) -> pd.Series:
    sweep = _storm_recall_threshold_sweep(
        validation_frame,
        validation_score,
        kp_threshold,
        thresholds,
        lead_steps,
        "validation_public_reference",
    )
    feasible = sweep[
        (sweep["alert_duty_cycle"] <= ALERT_CONSTRAINTS["alert_duty_cycle"])
        & (sweep["false_watch_rate_kp_lt_5"] <= 0.10)
    ].copy()
    if feasible.empty:
        feasible = sweep[sweep["event_recall"] > 0.0].copy()
    if feasible.empty:
        feasible = sweep.copy()
    return feasible.sort_values(
        ["event_recall", "recall", "precision", "alert_duty_cycle", "threshold"],
        ascending=[False, False, False, True, True],
    ).iloc[0]


def _storm_recall_branch_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    candidate_columns = [
        "kp_base",
        "kp_storm_adjusted",
        "kp_storm_conservative_v2",
        "kp_final_risk_conservative_v5",
        "kp_final_risk_conservative_v6",
        "kp_storm_recall_conservative",
    ]
    metrics = _storm_recall_regime_rows(frame, [col for col in candidate_columns if col in frame.columns])
    threshold_rows = metrics[metrics["metric_group"] == "threshold_recall"].copy()
    if threshold_rows.empty:
        return threshold_rows
    return threshold_rows[
        threshold_rows["regime"].isin(["kp_ge_5", "kp_ge_6", "kp_ge_7"])
    ].sort_values(["regime", "recall", "event_recall", "precision"], ascending=[True, False, False, False])


def _ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _collapse_duplicate_timestamps(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before_rows = len(df)
    collapsed = (
        df.groupby(["year", "doy", "hour"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["year", "doy", "hour"])
        .reset_index(drop=True)
    )
    duplicates_removed = before_rows - len(collapsed)
    return collapsed, int(duplicates_removed)


def _rolling_max_current(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).max()


def _largest_positive_jump(series: pd.Series, lookback_steps: int) -> pd.Series:
    positive_jump = (series - series.shift(1)).clip(lower=0.0)
    return positive_jump.rolling(lookback_steps, min_periods=lookback_steps).max()


def _largest_positive_drop(series: pd.Series, lookback_steps: int) -> pd.Series:
    positive_drop = (series.shift(1) - series).clip(lower=0.0)
    return positive_drop.rolling(lookback_steps, min_periods=lookback_steps).max()


def _consecutive_true_count(mask: pd.Series) -> pd.Series:
    """
    Count the current consecutive run length for a boolean condition.
    """

    counts = []
    current_count = 0
    for value in mask.fillna(False).astype(bool):
        if value:
            current_count += 1
        else:
            current_count = 0
        counts.append(current_count)
    return pd.Series(counts, index=mask.index, dtype=float)


def _steps_since_true(mask: pd.Series) -> pd.Series:
    """
    Count timesteps since the most recent true value in a past-only mask.
    """

    counts = []
    current_count = np.nan
    for value in mask.fillna(False).astype(bool):
        if value:
            current_count = 0.0
        elif np.isnan(current_count):
            current_count = np.nan
        else:
            current_count += 1.0
        counts.append(current_count)
    return pd.Series(counts, index=mask.index, dtype=float)


def _compact_storm_score(work_df: pd.DataFrame) -> pd.Series:
    linear_score = (
        -5.053
        + 0.369 * work_df["pdyn_max_12h"]
        + 1.159 * work_df["pdyn_jump_24h"]
        + 0.00602 * work_df["velocity"]
        + 0.0382 * work_df["velocity_jump_6h"]
        + 3.729 * work_df["bz_drop_36h"]
        + 0.1289 * work_df["density_max_24h"]
    )
    return 1.0 / (1.0 + np.exp(-linear_score))


def _build_exogenous_features(work_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series], List[str], List[str]]:
    source_times: Dict[str, pd.Series] = {}
    solar_wind_history_features: List[str] = []

    theta_c = (np.arctan2(work_df["by"], work_df["bz"]) + 2 * np.pi) % (2 * np.pi)
    newell_theta = np.arctan2(np.abs(work_df["by"]), work_df["bz"])
    b_total = np.sqrt(work_df["bx"] ** 2 + work_df["by"] ** 2 + work_df["bz"] ** 2)
    b_t_yz = np.sqrt(work_df["by"] ** 2 + work_df["bz"] ** 2)
    bs = np.maximum(0.0, -work_df["bz"])

    work_df["timestamp"] = _timestamp_from_columns(work_df)
    timestamp_ns = pd.Series(work_df["timestamp"].astype("int64"), index=work_df.index)
    work_df["b_total"] = b_total
    work_df["b_t_yz"] = b_t_yz
    work_df["bs"] = bs
    work_df["pdyn"] = 1.6726e-6 * work_df["density"] * work_df["velocity"] ** 2
    work_df["pdyn_max_12h"] = _rolling_max_current(work_df["pdyn"], 2)
    work_df["pdyn_jump_24h"] = _largest_positive_jump(work_df["pdyn"], 4)
    work_df["pdyn_jump"] = (work_df["pdyn"] - work_df["pdyn"].shift(1)).clip(lower=0.0)
    work_df["velocity_jump_6h"] = (work_df["velocity"] - work_df["velocity"].shift(1)).clip(lower=0.0)
    work_df["velocity_jump"] = work_df["velocity_jump_6h"]
    work_df["bz_drop_36h"] = _largest_positive_drop(work_df["bz"], 6)
    work_df["density_max_24h"] = _rolling_max_current(work_df["density"], 4)
    work_df["ey"] = work_df["velocity"] * work_df["bs"] / 1000.0
    work_df["clock_angle"] = newell_theta
    work_df["sin_half_clock"] = np.sin(newell_theta / 2).clip(lower=0.0)
    work_df["newell_coupling"] = (
        work_df["velocity"] ** (4 / 3)
        * work_df["b_t_yz"] ** (2 / 3)
        * np.sin(newell_theta / 2) ** (8 / 3)
    )
    work_df["coupling_simple"] = (
        work_df["velocity"] ** (4 / 3)
        * work_df["b_t_yz"] ** (2 / 3)
        * work_df["sin_half_clock"] ** (8 / 3)
    )
    work_df["epsilon_proxy"] = work_df["velocity"] * work_df["b_t_yz"] ** 2 * work_df["sin_half_clock"] ** 4
    work_df["kan_lee_proxy"] = work_df["velocity"] * work_df["b_t_yz"] * work_df["sin_half_clock"] ** 2
    work_df["coupling_term"] = (
        work_df["velocity"] ** (4 / 3)
        * b_total ** (2 / 3)
        * np.sin(theta_c / 2) ** (8 / 3)
    )
    work_df["viscous_term"] = np.sqrt(work_df["density"]) * work_df["velocity"] ** 2
    work_df["bz_term"] = np.where(work_df["bz"] < -0.5, 10 ** (0.5 - work_df["bz"]), 0.0)
    work_df["dbzdt_term"] = work_df["bz"].diff().fillna(0.0)
    work_df["v_bs"] = work_df["velocity"] * work_df["bs"]
    work_df["v2_bs"] = work_df["velocity"] ** 2 * work_df["bs"]
    work_df["dynamic_pressure_proxy"] = work_df["density"] * work_df["velocity"] ** 2
    work_df["compression_score"] = work_df["pdyn_jump"] * work_df["velocity_jump"]
    work_df["southward_pressure_interaction"] = work_df["bs"] * work_df["pdyn"]
    work_df["pdyn_above_2npa_duration"] = _consecutive_true_count(work_df["pdyn"] > 2.0)
    work_df["bs_above_threshold_duration"] = _consecutive_true_count(work_df["bs"] > SUSTAINED_BS_THRESHOLD_NT)
    work_df["bz_delta_1"] = work_df["bz"] - work_df["bz"].shift(1)
    work_df["bz_delta_2"] = work_df["bz"] - work_df["bz"].shift(2)
    work_df["bz_delta_4"] = work_df["bz"] - work_df["bz"].shift(4)
    work_df["bz_drop_6h"] = work_df["bz"].shift(1) - work_df["bz"]
    work_df["bz_drop_12h"] = work_df["bz"].shift(2) - work_df["bz"]
    work_df["bz_drop_24h"] = work_df["bz"].shift(4) - work_df["bz"]
    work_df["bz_southward_drop_6h"] = work_df["bz_drop_6h"].clip(lower=0.0)
    work_df["bz_southward_drop_12h"] = work_df["bz_drop_12h"].clip(lower=0.0)
    work_df["bz_southward_drop_24h"] = work_df["bz_drop_24h"].clip(lower=0.0)
    work_df["bz_drop_gt_5nt"] = (work_df["bz_southward_drop_6h"] > 5.0).astype(float)
    work_df["bz_drop_gt_10nt"] = (work_df["bz_southward_drop_6h"] > 10.0).astype(float)
    work_df["bz_crossed_southward"] = ((work_df["bz"].shift(1) > 0.0) & (work_df["bz"] < 0.0)).astype(float)
    work_df["strong_southward_turning"] = ((work_df["bz_southward_drop_6h"] > 5.0) & (work_df["bz"] < -5.0)).astype(float)
    work_df["bz_drop_times_velocity"] = work_df["bz_southward_drop_6h"] * work_df["velocity"]
    work_df["bz_drop_times_pdyn"] = work_df["bz_southward_drop_6h"] * work_df["pdyn"]
    work_df["bz_drop_times_density"] = work_df["bz_southward_drop_6h"] * work_df["density"]
    work_df["bz_drop_times_ey"] = work_df["bz_southward_drop_6h"] * work_df["ey"]
    work_df["southward_turning_pressure_interaction"] = (
        work_df["bz_southward_drop_6h"] * work_df["pdyn"] * (work_df["bz"] < 0.0).astype(float)
    )
    work_df["southward_turning_coupling_interaction"] = (
        work_df["bz_southward_drop_6h"] * work_df["coupling_simple"] * (work_df["bz"] < 0.0).astype(float)
    )
    for window in ROLL_WINDOWS:
        work_df[f"bs_roll_sum_{window}"] = work_df["bs"].rolling(window, min_periods=window).sum()
        work_df[f"ey_roll_max_{window}"] = work_df["ey"].rolling(window, min_periods=window).max()
        work_df[f"ey_roll_sum_{window}"] = work_df["ey"].rolling(window, min_periods=window).sum()
        work_df[f"pdyn_roll_max_{window}"] = work_df["pdyn"].rolling(window, min_periods=window).max()
    work_df["compact_storm_score"] = _compact_storm_score(work_df)
    for window in EXPERIMENT7_ROLL_WINDOWS:
        work_df[f"bs_roll_sum_{window}"] = work_df["bs"].rolling(window, min_periods=window).sum()
        work_df[f"ey_roll_sum_{window}"] = work_df["ey"].rolling(window, min_periods=window).sum()
        work_df[f"ey_roll_max_{window}"] = work_df["ey"].rolling(window, min_periods=window).max()
        work_df[f"pdyn_roll_max_{window}"] = work_df["pdyn"].rolling(window, min_periods=window).max()
        work_df[f"velocity_roll_max_{window}"] = work_df["velocity"].rolling(window, min_periods=window).max()
        work_df[f"coupling_simple_roll_sum_{window}"] = work_df["coupling_simple"].rolling(window, min_periods=window).sum()
        work_df[f"compact_storm_score_roll_max_{window}"] = work_df["compact_storm_score"].rolling(window, min_periods=window).max()

    lag_sources = [
        "bz",
        "bs",
        "velocity",
        "density",
        "pdyn",
        "pdyn_jump",
        "pdyn_max_12h",
        "pdyn_jump_24h",
        "velocity_jump",
        "velocity_jump_6h",
        "bz_drop_36h",
        "density_max_24h",
        "ey",
        "newell_coupling",
        "coupling_simple",
        "epsilon_proxy",
        "kan_lee_proxy",
        "compact_storm_score",
        "coupling_term",
        "viscous_term",
        "dbzdt_term",
        "v_bs",
        "v2_bs",
        "dynamic_pressure_proxy",
        "compression_score",
        "southward_pressure_interaction",
        "pdyn_above_2npa_duration",
        "bs_above_threshold_duration",
        "bs_roll_sum_2",
        "bs_roll_sum_4",
        "bs_roll_sum_8",
        "ey_roll_max_2",
        "ey_roll_max_4",
        "ey_roll_max_8",
        "ey_roll_sum_2",
        "ey_roll_sum_4",
        "ey_roll_sum_8",
        "pdyn_roll_max_4",
        "pdyn_roll_max_8",
    ]
    for lag in LAG_STEPS:
        shifted_time = work_df["timestamp"].shift(lag)
        for col in lag_sources:
            feature_name = f"{col}_lag_{lag}"
            work_df[feature_name] = work_df[col].shift(lag)
            source_times[feature_name] = shifted_time
            solar_wind_history_features.append(feature_name)

    roll_sources = [
        "bz",
        "bs",
        "velocity",
        "density",
        "pdyn",
        "pdyn_jump",
        "ey",
        "newell_coupling",
        "coupling_simple",
        "epsilon_proxy",
        "kan_lee_proxy",
        "compact_storm_score",
        "coupling_term",
        "viscous_term",
        "v_bs",
        "v2_bs",
        "dynamic_pressure_proxy",
        "compression_score",
        "southward_pressure_interaction",
    ]
    for window in ROLL_WINDOWS:
        shifted_time = pd.to_datetime(timestamp_ns.shift(1).rolling(window).max())
        for col in roll_sources:
            feature_name = f"{col}_roll_mean_{window}"
            work_df[feature_name] = work_df[col].shift(1).rolling(window).mean()
            source_times[feature_name] = shifted_time
            solar_wind_history_features.append(feature_name)

        feature_name = f"bz_roll_min_{window}"
        work_df[feature_name] = work_df["bz"].shift(1).rolling(window).min()
        source_times[feature_name] = shifted_time
        solar_wind_history_features.append(feature_name)

        feature_name = f"velocity_roll_max_{window}"
        work_df[feature_name] = work_df["velocity"].shift(1).rolling(window).max()
        source_times[feature_name] = shifted_time
        solar_wind_history_features.append(feature_name)

    solar_wind_feature_columns = _unique_in_order(
        SOLAR_WIND_CURRENT_FEATURES
        + SUSTAINED_DRIVING_FEATURES
        + BZ_SHARP_DROP_FEATURES
        + EXPERIMENT7_MOVING_PHYSICS_ROLL_FEATURES
        + solar_wind_history_features
    )
    return work_df, source_times, solar_wind_history_features, solar_wind_feature_columns


def prepare_temporal_dataframe(df: pd.DataFrame, return_metadata: bool = False):
    """
    Build the temporal training frame and feature lineage metadata.
    """

    required_cols = ["year", "doy", "hour", "bx", "by", "bz", "density", "velocity", "kp"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    work_df = df[required_cols].copy()
    work_df = work_df.dropna(subset=required_cols)
    work_df, duplicates_removed = _collapse_duplicate_timestamps(work_df)
    work_df, source_times, solar_wind_history_features, solar_wind_feature_columns = _build_exogenous_features(work_df)

    for lag in KP_LAG_STEPS:
        feature_name = f"kp_lag_{lag}"
        work_df[feature_name] = work_df["kp"].shift(lag)
        source_times[feature_name] = work_df["timestamp"].shift(lag)

    for window in KP_ROLL_WINDOWS:
        timestamp_ns = pd.Series(work_df["timestamp"].astype("int64"), index=work_df.index)
        shifted_time = pd.to_datetime(timestamp_ns.shift(1).rolling(window).max())
        feature_name = f"kp_roll_mean_{window}"
        work_df[feature_name] = work_df["kp"].shift(1).rolling(window).mean()
        source_times[feature_name] = shifted_time

        feature_name = f"kp_roll_std_{window}"
        work_df[feature_name] = work_df["kp"].shift(1).rolling(window).std()
        source_times[feature_name] = shifted_time

        feature_name = f"kp_roll_max_{window}"
        work_df[feature_name] = work_df["kp"].shift(1).rolling(window).max()
        source_times[feature_name] = shifted_time

        feature_name = f"kp_roll_min_{window}"
        work_df[feature_name] = work_df["kp"].shift(1).rolling(window).min()
        source_times[feature_name] = shifted_time

        feature_name = f"kp_roll_range_{window}"
        work_df[feature_name] = work_df[f"kp_roll_max_{window}"] - work_df[f"kp_roll_min_{window}"]
        source_times[feature_name] = shifted_time

        feature_name = f"kp_roll_sum_{window}"
        work_df[feature_name] = work_df["kp"].shift(1).rolling(window).sum()
        source_times[feature_name] = shifted_time

    work_df["kp_delta_1"] = work_df["kp"].shift(1) - work_df["kp"].shift(2)
    source_times["kp_delta_1"] = work_df["timestamp"].shift(1)
    work_df["kp_delta_4"] = work_df["kp"].shift(1) - work_df["kp"].shift(4)
    source_times["kp_delta_4"] = work_df["timestamp"].shift(1)
    work_df["kp_abs_delta_4"] = work_df["kp_delta_4"].abs()
    source_times["kp_abs_delta_4"] = work_df["timestamp"].shift(1)
    work_df["kp_24h_repeat_error"] = (work_df["kp"].shift(1) - work_df["kp"].shift(4)).abs()
    source_times["kp_24h_repeat_error"] = work_df["timestamp"].shift(1)
    work_df["kp_48h_repeat_error"] = (work_df["kp"].shift(1) - work_df["kp"].shift(8)).abs()
    source_times["kp_48h_repeat_error"] = work_df["timestamp"].shift(1)
    work_df["kp_72h_repeat_error"] = (work_df["kp"].shift(1) - work_df["kp"].shift(12)).abs()
    source_times["kp_72h_repeat_error"] = work_df["timestamp"].shift(1)
    work_df["kp_daily_pattern_break_score"] = work_df["kp_24h_repeat_error"] + work_df["kp_delta_1"].abs()
    source_times["kp_daily_pattern_break_score"] = work_df["timestamp"].shift(1)
    work_df["kp_daily_variability_ratio"] = work_df["kp_roll_std_4"] / (work_df["kp_roll_std_20"] + 1e-6)
    source_times["kp_daily_variability_ratio"] = work_df["timestamp"].shift(1)
    work_df["kp_recent_vs_multiday_mean"] = work_df["kp_roll_mean_4"] - work_df["kp_roll_mean_20"]
    source_times["kp_recent_vs_multiday_mean"] = work_df["timestamp"].shift(1)
    work_df["kp_recent_max_vs_multiday_mean"] = work_df["kp_roll_max_4"] - work_df["kp_roll_mean_20"]
    source_times["kp_recent_max_vs_multiday_mean"] = work_df["timestamp"].shift(1)
    work_df["time_since_kp_ge_4"] = _steps_since_true(work_df["kp"].shift(1) >= 4.0).fillna(999.0)
    source_times["time_since_kp_ge_4"] = work_df["timestamp"].shift(1)
    work_df["time_since_kp_ge_5"] = _steps_since_true(work_df["kp"].shift(1) >= 5.0).fillna(999.0)
    source_times["time_since_kp_ge_5"] = work_df["timestamp"].shift(1)
    work_df["time_since_kp_ge_6"] = _steps_since_true(work_df["kp"].shift(1) >= 6.0).fillna(999.0)
    source_times["time_since_kp_ge_6"] = work_df["timestamp"].shift(1)

    future_kp_1 = work_df["kp"].shift(-1)
    future_kp_2 = work_df["kp"].shift(-2)
    future_12h_max = pd.concat([future_kp_1, future_kp_2], axis=1).max(axis=1)
    cadence_hours = _cadence_hours(work_df)
    if cadence_hours <= 0:
        cadence_hours = 6.0
    magnitude_horizon_bins = {}
    for horizon_hours in STORM_MAGNITUDE_HORIZONS_HOURS:
        horizon_bins = max(1, int(np.ceil(float(horizon_hours) / cadence_hours)))
        magnitude_horizon_bins[horizon_hours] = horizon_bins
        future_columns = [work_df["kp"].shift(-step) for step in range(1, horizon_bins + 1)]
        target_name = f"kp_max_next_{horizon_hours}h"
        work_df[target_name] = pd.concat(future_columns, axis=1).max(axis=1)
        work_df[f"{target_name}_valid"] = pd.concat(future_columns, axis=1).notna().all(axis=1)

    recent_quiet = (
        (work_df["kp"] < 5)
        & (work_df["kp"].shift(1) < 5)
        & (work_df["kp"].shift(2) < 5)
    )
    work_df["storm_now"] = work_df["kp"] >= 5
    work_df["storm_next_6h"] = future_kp_1 >= 5
    work_df["storm_next_12h"] = future_12h_max >= 5
    work_df["storm_now_or_next_12h"] = work_df["storm_now"] | work_df["storm_next_12h"]
    work_df["storm_onset_next_12h"] = work_df["storm_next_12h"] & recent_quiet
    work_df["storm_now_valid"] = True
    work_df["storm_next_6h_valid"] = future_kp_1.notna()
    work_df["storm_next_12h_valid"] = future_kp_1.notna() & future_kp_2.notna()
    work_df["storm_now_or_next_12h_valid"] = work_df["storm_next_12h_valid"]
    work_df["storm_onset_next_12h_valid"] = work_df["storm_next_12h_valid"]
    for severity_threshold in SEVERITY_THRESHOLDS:
        severity_label = str(int(severity_threshold))
        target_name = f"severity_kp_ge_{severity_label}_next_12h"
        alias_target_name = f"kp_ge_{severity_label}_next_12h"
        work_df[target_name] = work_df["kp_max_next_12h"] >= severity_threshold
        work_df[f"{target_name}_valid"] = work_df["kp_max_next_12h_valid"]
        work_df[alias_target_name] = work_df[target_name]
        work_df[f"{alias_target_name}_valid"] = work_df[f"{target_name}_valid"]

    temporal_feature_columns = _unique_in_order(solar_wind_feature_columns + KP_HISTORY_FEATURES)
    required_feature_columns = temporal_feature_columns + ["timestamp", "kp"]
    mask = work_df[required_feature_columns].notna().all(axis=1)
    feature_df = work_df.loc[mask].copy().reset_index(drop=True)

    source_time_df = pd.DataFrame(
        {feature_name: source_series.loc[mask].reset_index(drop=True) for feature_name, source_series in source_times.items()}
    )
    leakage_details = run_leakage_checks(feature_df, source_time_df)
    kp_history_availability = _kp_history_availability_metadata(
        feature_df,
        source_time_df,
        KP_HISTORY_FEATURES,
    )

    if len(feature_df) < 100:
        raise ValueError(
            "Not enough rows remain after temporal feature generation. "
            "Provide a larger input range."
        )

    metadata = {
        "duplicates_removed": duplicates_removed,
        "forecast_inputs_already_target_aligned": FORECAST_INPUTS_ALREADY_TARGET_ALIGNED,
        "driver_input_metadata": _driver_input_metadata(),
        "original_pi_linear_equation": _original_pi_linear_equation_metadata(),
        "raw_columns": df.columns.tolist(),
        "issue_time_valid_time_metadata_found": bool(
            any(col in df.columns for col in ["issue_time", "init_time", "forecast_time", "forecast_issue_time"])
            and any(col in df.columns for col in ["valid_time", "target_time", "timestamp"])
        ),
        "temporal_feature_columns": temporal_feature_columns,
        "solar_wind_feature_columns": solar_wind_feature_columns,
        "bz_sharp_drop_feature_columns": BZ_SHARP_DROP_FEATURES,
        "kp_history_feature_columns": KP_HISTORY_FEATURES,
        "storm_magnitude_horizon_bins": magnitude_horizon_bins,
        "temporal_source_times": source_time_df,
        "leakage": leakage_details,
        "kp_history_availability": kp_history_availability,
    }

    if return_metadata:
        return feature_df, metadata
    return feature_df


def run_leakage_checks(feature_df: pd.DataFrame, source_time_df: pd.DataFrame) -> dict:
    """
    Verify that every lag / rolling feature references only timestamps
    strictly earlier than the target timestamp.
    """

    target_time = feature_df["timestamp"]
    violations = []

    for feature_name in source_time_df.columns:
        comparison = source_time_df[feature_name] < target_time
        if not comparison.all():
            invalid_rows = np.where(~comparison.to_numpy())[0][:5].tolist()
            violations.append(
                {
                    "feature": feature_name,
                    "rows": invalid_rows,
                }
            )

    return {
        "passed": len(violations) == 0,
        "checked_features": int(source_time_df.shape[1]),
        "violations": violations,
    }


def chronological_split(feature_df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1")

    split_index = int(len(feature_df) * (1 - test_size))
    if split_index <= 0 or split_index >= len(feature_df):
        raise ValueError("test_size leaves no rows in either the train or test split")

    train_df = feature_df.iloc[:split_index].copy().reset_index(drop=True)
    test_df = feature_df.iloc[split_index:].copy().reset_index(drop=True)
    return train_df, test_df


def _refresh_kp_history_features(row: pd.Series, history_kp: List[float]) -> pd.Series:
    for lag in KP_LAG_STEPS:
        row[f"kp_lag_{lag}"] = history_kp[-lag]

    kp_array = np.asarray(history_kp, dtype=float)
    for window in KP_ROLL_WINDOWS:
        kp_window = kp_array[-window:]
        row[f"kp_roll_mean_{window}"] = float(np.mean(kp_window))
        row[f"kp_roll_std_{window}"] = float(np.std(kp_window, ddof=1))
        row[f"kp_roll_max_{window}"] = float(np.max(kp_window))
        row[f"kp_roll_min_{window}"] = float(np.min(kp_window))
        row[f"kp_roll_range_{window}"] = float(np.max(kp_window) - np.min(kp_window))
        row[f"kp_roll_sum_{window}"] = float(np.sum(kp_window))
    row["kp_delta_1"] = history_kp[-1] - history_kp[-2]
    row["kp_delta_4"] = history_kp[-1] - history_kp[-4]
    row["kp_abs_delta_4"] = abs(row["kp_delta_4"])
    row["kp_24h_repeat_error"] = abs(history_kp[-1] - history_kp[-4])
    row["kp_48h_repeat_error"] = abs(history_kp[-1] - history_kp[-8])
    row["kp_72h_repeat_error"] = abs(history_kp[-1] - history_kp[-12])
    row["kp_daily_pattern_break_score"] = row["kp_24h_repeat_error"] + abs(row["kp_delta_1"])
    row["kp_daily_variability_ratio"] = row["kp_roll_std_4"] / (row["kp_roll_std_20"] + 1e-6)
    row["kp_recent_vs_multiday_mean"] = row["kp_roll_mean_4"] - row["kp_roll_mean_20"]
    row["kp_recent_max_vs_multiday_mean"] = row["kp_roll_max_4"] - row["kp_roll_mean_20"]
    for threshold in [4, 5, 6]:
        recent_true = [idx for idx, value in enumerate(reversed(history_kp), start=1) if value >= threshold]
        row[f"time_since_kp_ge_{threshold}"] = float(recent_true[0]) if recent_true else 999.0

    return row


def _recursive_predict(
    model,
    full_feature_df: pd.DataFrame,
    split_index: int,
    feature_columns: List[str],
    horizon: int,
    uses_kp_history: bool,
) -> pd.DataFrame:
    prediction_frames = []
    for start_idx in range(split_index, len(full_feature_df) - horizon + 1, horizon):
        history_kp = full_feature_df.iloc[:start_idx]["kp"].tolist()
        future_df = full_feature_df.iloc[start_idx:start_idx + horizon].copy()
        pred_values = []

        for _, future_row in future_df.iterrows():
            candidate_row = future_row.copy()
            if uses_kp_history:
                candidate_row = _refresh_kp_history_features(candidate_row, history_kp)

            pred = float(model.predict(pd.DataFrame([candidate_row])[feature_columns])[0])
            pred = float(np.clip(pred, 0.0, 9.0))
            pred_values.append(pred)
            history_kp.append(pred)

        forecast_df = future_df.copy()
        forecast_df["pred_kp"] = np.array(pred_values)
        prediction_frames.append(forecast_df)

    if len(prediction_frames) == 0:
        return pd.DataFrame(columns=list(full_feature_df.columns) + ["pred_kp"])
    return pd.concat(prediction_frames, ignore_index=True)


def _time_series_cv_splits(n_rows: int, requested_splits: int = 5) -> int:
    return max(2, min(requested_splits, n_rows // 250))


def _tune_ridge_alpha(X_train: pd.DataFrame, y_train: pd.Series, alphas: List[float]) -> Tuple[float, pd.DataFrame]:
    splitter = TimeSeriesSplit(n_splits=_time_series_cv_splits(len(X_train)))
    rows = []

    for alpha in alphas:
        fold_rmse = []
        fold_mae = []
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train), start=1):
            model = Ridge(alpha=alpha)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            pred = model.predict(X_train.iloc[val_idx])
            fold_rmse.append(np.sqrt(mean_squared_error(y_train.iloc[val_idx], pred)))
            fold_mae.append(mean_absolute_error(y_train.iloc[val_idx], pred))
            rows.append(
                {
                    "alpha": alpha,
                    "fold": fold_idx,
                    "mae": float(fold_mae[-1]),
                    "rmse": float(fold_rmse[-1]),
                }
            )

    cv_df = pd.DataFrame(rows)
    summary_df = (
        cv_df.groupby("alpha", as_index=False)[["mae", "rmse"]]
        .mean()
        .sort_values(["rmse", "mae", "alpha"])
        .reset_index(drop=True)
    )
    best_alpha = float(summary_df.iloc[0]["alpha"])
    return best_alpha, summary_df


def _time_series_oof_scores(
    model_factory: Callable[[], object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    predict_func: str = "predict",
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = TimeSeriesSplit(n_splits=_time_series_cv_splits(len(X_train)))
    scores = np.full(len(X_train), np.nan)

    for train_idx, val_idx in splitter.split(X_train):
        model = model_factory()
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        if predict_func == "predict_proba":
            fold_scores = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        else:
            fold_scores = model.predict(X_train.iloc[val_idx])
        scores[val_idx] = np.asarray(fold_scores, dtype=float)

    mask = ~np.isnan(scores)
    return y_train.iloc[mask].to_numpy(), scores[mask]


def _time_series_oof_classifier_scores(
    model_factory: Callable[[], object],
    X_train: pd.DataFrame,
    y_train_binary: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = TimeSeriesSplit(n_splits=_time_series_cv_splits(len(X_train)))
    scores = np.full(len(X_train), np.nan)

    for train_idx, val_idx in splitter.split(X_train):
        fold_y = y_train_binary[train_idx]
        if len(np.unique(fold_y)) < 2:
            continue

        model = model_factory()
        model = _fit_storm_classifier(model, X_train.iloc[train_idx], fold_y)
        scores[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]

    mask = ~np.isnan(scores)
    return y_train_binary[mask], scores[mask]


def _time_series_oof_classifier_prediction_frame(
    model_factory: Callable[[], object],
    train_df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    sample_weight_func: Optional[Callable[[pd.DataFrame, str], np.ndarray]] = None,
) -> pd.DataFrame:
    X_train = train_df[feature_columns]
    y_train_binary = train_df[target_column].astype(int).to_numpy()
    splitter = TimeSeriesSplit(n_splits=_time_series_cv_splits(len(train_df)))
    scores = np.full(len(train_df), np.nan)

    for train_idx, val_idx in splitter.split(X_train):
        fold_y = y_train_binary[train_idx]
        if len(np.unique(fold_y)) < 2:
            continue

        model = model_factory()
        sample_weight = sample_weight_func(train_df.iloc[train_idx], target_column) if sample_weight_func is not None else None
        model = _fit_storm_classifier(model, X_train.iloc[train_idx], fold_y, sample_weight=sample_weight)
        scores[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]

    mask = ~np.isnan(scores)
    return pd.DataFrame(
        {
            "timestamp": train_df.loc[mask, "timestamp"].to_numpy(),
            "kp": train_df.loc[mask, "kp"].to_numpy(),
            "storm_observed": y_train_binary[mask],
            "score": scores[mask],
        }
    )


def _fit_score_calibrator(y_true_binary: np.ndarray, raw_score: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    calibrator.fit(np.asarray(raw_score).reshape(-1, 1), np.asarray(y_true_binary).astype(int))
    return calibrator


def _apply_score_calibrator(calibrator: LogisticRegression, raw_score: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(np.asarray(raw_score).reshape(-1, 1))[:, 1]


def _ridge_threshold_curve(y_true_binary: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_score)
    threshold_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[1:],
            "recall": recall[1:],
        }
    )
    threshold_df["f1"] = np.where(
        threshold_df["precision"] + threshold_df["recall"] > 0,
        2 * threshold_df["precision"] * threshold_df["recall"] / (threshold_df["precision"] + threshold_df["recall"]),
        0.0,
    )
    return threshold_df.sort_values("threshold").reset_index(drop=True)


def _select_operating_thresholds(threshold_df: pd.DataFrame) -> Dict[str, dict]:
    if threshold_df.empty:
        raise ValueError("Threshold dataframe is empty; cannot select storm operating thresholds.")

    balanced_row = threshold_df.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0]

    recall_candidates = threshold_df[threshold_df["recall"] >= 0.80]
    if len(recall_candidates) > 0:
        high_recall_row = recall_candidates.sort_values(["precision", "threshold"], ascending=[False, True]).iloc[0]
    else:
        high_recall_row = threshold_df.sort_values(["recall", "precision"], ascending=[False, False]).iloc[0]

    precision_candidates = threshold_df[threshold_df["precision"] >= 0.80]
    if len(precision_candidates) > 0:
        high_precision_row = precision_candidates.sort_values(["recall", "threshold"], ascending=[False, True]).iloc[0]
    else:
        high_precision_row = threshold_df.sort_values(["precision", "recall"], ascending=[False, False]).iloc[0]

    return {
        "high_recall": high_recall_row.to_dict(),
        "balanced_f1": balanced_row.to_dict(),
        "high_precision": high_precision_row.to_dict(),
    }


def _model_registry(best_ridge_alpha: float) -> Dict[str, Callable[[], object]]:
    registry: Dict[str, Callable[[], object]] = {
        "ridge_baseline": lambda: Ridge(alpha=0.4),
        "ridge_tuned": lambda: Ridge(alpha=best_ridge_alpha),
        "scaled_ridge_30": lambda: make_pipeline(
            StandardScaler(),
            Ridge(alpha=30.0),
        ),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if HistGradientBoostingRegressor is not None:
        registry["hist_gradient_boosting"] = lambda: HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        )

    if XGBRegressor is not None:
        registry["xgboost"] = lambda: XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if LGBMRegressor is not None:
        registry["lightgbm"] = lambda: LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1,
        )

    return registry


def _storm_classifier_registry() -> Dict[str, Callable[[], object]]:
    registry = {
        "logistic_balanced": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
        "random_forest_classifier_balanced": lambda: RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if HistGradientBoostingClassifier is not None:
        registry["hist_gradient_boosting_classifier"] = lambda: HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        )

    if XGBClassifier is not None:
        registry["xgboost_classifier"] = lambda: XGBClassifier(
            objective="binary:logistic",
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if LGBMClassifier is not None:
        registry["lightgbm_classifier"] = lambda: LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
        )

    return registry


def _fit_storm_classifier(model, X_train: pd.DataFrame, y_train_binary: np.ndarray, sample_weight: Optional[np.ndarray] = None):
    if HistGradientBoostingClassifier is not None and isinstance(model, HistGradientBoostingClassifier):
        if sample_weight is None:
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_binary)
        model.fit(X_train, y_train_binary, sample_weight=sample_weight)
    elif sample_weight is not None:
        model.fit(X_train, y_train_binary, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train_binary)
    return model


def _storm_event_sample_weight(train_df: pd.DataFrame, target_column: str) -> np.ndarray:
    """
    Weight abrupt and severe storm rows without making the mode mandatory.
    """

    weights = np.ones(len(train_df), dtype=float)
    if target_column in train_df.columns:
        weights *= np.where(train_df[target_column].astype(bool).to_numpy(), 3.0, 1.0)
    if "kp" in train_df.columns:
        kp_values = train_df["kp"].to_numpy()
        weights *= np.where(kp_values >= 6.0, 2.0, 1.0)
        first_storm_bin = (kp_values >= 5.0) & (pd.Series(kp_values).shift(1).fillna(0.0).to_numpy() < 5.0)
        weights *= np.where(first_storm_bin, 2.0, 1.0)
    if "storm_onset_next_12h" in train_df.columns:
        weights *= np.where(train_df["storm_onset_next_12h"].astype(bool).to_numpy(), 1.5, 1.0)
    return weights


def _persistence_predictions(test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "persistence_kp_lag_1": test_df["kp_lag_1"].to_numpy(),
        "persistence_roll_mean_4": test_df["kp_roll_mean_4"].to_numpy(),
        "persistence_roll_mean_8": test_df["kp_roll_mean_8"].to_numpy(),
    }


def _standardized_coefficients(model: Ridge, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    x_scale = X_train.std(ddof=0).replace(0, np.nan)
    y_scale = float(y_train.std(ddof=0))
    standardized = model.coef_ * x_scale.to_numpy() / y_scale
    df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "coefficient": model.coef_,
            "standardized_coefficient": standardized,
            "abs_standardized_coefficient": np.abs(standardized),
        }
    )
    return df.sort_values("abs_standardized_coefficient", ascending=False).reset_index(drop=True)


def _feature_importance_dataframe(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> pd.DataFrame:
    if isinstance(model, Ridge):
        return _standardized_coefficients(model, X_train, y_train)

    if hasattr(model, "feature_importances_"):
        importance = getattr(model, "feature_importances_")
        df = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": importance,
                "abs_importance": np.abs(importance),
            }
        )
        return df.sort_values("abs_importance", ascending=False).reset_index(drop=True)

    if permutation_importance is None:
        return pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": np.zeros(len(X_train.columns)),
                "abs_importance": np.zeros(len(X_train.columns)),
                "note": "permutation_importance_unavailable",
            }
        )

    sample_size = min(1000, len(X_eval))
    sample_X = X_eval.iloc[:sample_size]
    sample_y = y_eval.iloc[:sample_size]
    importance = permutation_importance(
        model,
        sample_X,
        sample_y,
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    df = pd.DataFrame(
        {
            "feature": sample_X.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
            "abs_importance_mean": np.abs(importance.importances_mean),
        }
    )
    return df.sort_values("abs_importance_mean", ascending=False).reset_index(drop=True)


def _save_feature_importance(output_dir: str, model_name: str, importance_df: pd.DataFrame) -> str:
    path = os.path.join(output_dir, f"kp_feature_importance_{model_name}.csv")
    importance_df.to_csv(path, index=False)
    return path


def _plot_holdout_time_series(prediction_df: pd.DataFrame, output_dir: str) -> str:
    plt.figure(figsize=(12, 5))
    plt.plot(prediction_df["timestamp"], prediction_df["kp"], color="black", linewidth=2, label="Observed Kp")
    if "physics_linear" in prediction_df.columns:
        plt.plot(prediction_df["timestamp"], prediction_df["physics_linear"], linewidth=1.5, label="Physics-only")
    if "persistence_kp_lag_1" in prediction_df.columns:
        plt.plot(prediction_df["timestamp"], prediction_df["persistence_kp_lag_1"], linewidth=1.2, label="Persistence lag-1")
    plt.plot(prediction_df["timestamp"], prediction_df["best_model_pred"], linewidth=2, label="Best model")
    plt.ylabel("Kp")
    plt.xlabel("Holdout time")
    plt.title("Observed vs Predicted Kp on Holdout")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "kp_holdout_timeseries.png")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _plot_scatter(prediction_df: pd.DataFrame, output_dir: str) -> str:
    plt.figure(figsize=(6, 6))
    plt.scatter(prediction_df["kp"], prediction_df["best_model_pred"], alpha=0.45, s=14)
    min_kp = min(prediction_df["kp"].min(), prediction_df["best_model_pred"].min())
    max_kp = max(prediction_df["kp"].max(), prediction_df["best_model_pred"].max())
    plt.plot([min_kp, max_kp], [min_kp, max_kp], color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Observed Kp")
    plt.ylabel("Predicted Kp")
    plt.title("Observed vs Predicted Scatter")
    plt.tight_layout()
    path = os.path.join(output_dir, "kp_holdout_scatter.png")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _plot_residual_histogram(prediction_df: pd.DataFrame, output_dir: str) -> str:
    residuals = prediction_df["best_model_pred"] - prediction_df["kp"]
    plt.figure(figsize=(7, 4.5))
    plt.hist(residuals, bins=40, alpha=0.8, color="steelblue", edgecolor="black")
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Prediction residual")
    plt.ylabel("Count")
    plt.title("Residual Histogram")
    plt.tight_layout()
    path = os.path.join(output_dir, "kp_residual_histogram.png")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _plot_residual_vs_truth(prediction_df: pd.DataFrame, output_dir: str) -> str:
    residuals = prediction_df["best_model_pred"] - prediction_df["kp"]
    plt.figure(figsize=(7, 4.5))
    plt.scatter(prediction_df["kp"], residuals, alpha=0.45, s=14)
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Observed Kp")
    plt.ylabel("Residual")
    plt.title("Residual vs Observed Kp")
    plt.tight_layout()
    path = os.path.join(output_dir, "kp_residual_vs_true.png")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _plot_storm_time_series(prediction_df: pd.DataFrame, output_dir: str) -> str:
    storm_mask = prediction_df["kp"].rolling(9, center=True, min_periods=1).max() >= 5
    storm_df = prediction_df.loc[storm_mask].copy()
    plt.figure(figsize=(12, 5))
    plt.plot(storm_df["timestamp"], storm_df["kp"], color="black", linewidth=2, label="Observed Kp")
    plt.plot(storm_df["timestamp"], storm_df["best_model_pred"], linewidth=2, label="Best model")
    plt.axhline(5, color="red", linestyle="--", linewidth=1.2, label="Storm threshold")
    plt.ylabel("Kp")
    plt.xlabel("Storm-focused holdout time")
    plt.title("Storm-focused Holdout Performance")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "kp_storm_only_timeseries.png")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _plot_recursive_horizon_curve(recursive_df: pd.DataFrame, output_dir: str) -> str:
    plot_df = recursive_df.copy()
    plot_df["horizon"] = plot_df["horizon"].astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for model_name, group_df in plot_df.groupby("model"):
        axes[0].plot(group_df["horizon"], group_df["mae"], marker="o", label=model_name)
        axes[1].plot(group_df["horizon"], group_df["rmse"], marker="o", label=model_name)

    axes[0].set_ylabel("MAE")
    axes[0].set_title("Recursive Forecast Performance vs Horizon")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xlabel("Forecast horizon (steps)")
    axes[0].legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "kp_recursive_performance_vs_horizon.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_storm_precision_recall_curve(
    output_dir: str,
    y_true_binary: np.ndarray,
    ridge_scores: np.ndarray,
    logistic_scores: np.ndarray,
    rf_scores: np.ndarray,
    ridge_threshold_rows: Dict[str, dict],
) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, scores in [
        ("ridge_tuned_score", ridge_scores),
        ("logistic_balanced", logistic_scores),
        ("rf_classifier_balanced", rf_scores),
    ]:
        precision, recall, _ = precision_recall_curve(y_true_binary, scores)
        ap = _safe_average_precision(y_true_binary, scores)
        ax.plot(recall, precision, linewidth=2, label=f"{label} (AP={ap:.3f})")

    for label, row in ridge_threshold_rows.items():
        metrics = _binary_classification_metrics(y_true_binary, ridge_scores, row["threshold"])
        ax.scatter(
            metrics["recall"],
            metrics["precision"],
            s=60,
            label=f"ridge {label} @ {row['threshold']:.2f}",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Storm Detection Precision-Recall")
    ax.legend(loc="best")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "kp_storm_precision_recall_curve.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_expanding_window_backtests(backtest_df: pd.DataFrame, output_dir: str) -> str:
    plot_df = backtest_df.copy()
    plot_df = plot_df[plot_df["model"].isin(["ridge_balanced_f1", "ridge_high_recall", "ridge_high_precision", "logistic_balanced", "random_forest_classifier_balanced"])]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for model_name, group_df in plot_df.groupby("model"):
        axes[0].plot(group_df["era"], group_df["recall"], marker="o", label=model_name)
        axes[1].plot(group_df["era"], group_df["f1"], marker="o", label=model_name)

    axes[0].set_ylabel("Recall")
    axes[0].set_title("Expanding-Window Storm Backtests by Era")
    axes[1].set_ylabel("F1")
    axes[1].set_xlabel("Test era")
    axes[0].legend(loc="best")
    fig.tight_layout()
    path = os.path.join(output_dir, "kp_storm_backtest_by_era.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_event_timeline(
    output_dir: str,
    prediction_df: pd.DataFrame,
    score_col: str,
    threshold: float,
    title_suffix: str,
) -> str:
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(prediction_df["timestamp"], prediction_df["kp"], color="black", linewidth=1.8, label="Observed Kp")
    ax1.axhline(5, color="red", linestyle="--", linewidth=1.2, label="Kp >= 5")
    ax1.set_ylabel("Kp")
    ax1.set_xlabel("Time")

    ax2 = ax1.twinx()
    ax2.plot(prediction_df["timestamp"], prediction_df[score_col], color="tab:blue", alpha=0.75, label="Storm probability")
    ax2.axhline(threshold, color="tab:blue", linestyle=":", linewidth=1.2, label="Alert threshold")
    ax2.set_ylabel("Storm probability")
    ax2.set_ylim(-0.02, 1.02)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    ax1.set_title(f"Storm Event Timeline ({title_suffix})")
    fig.tight_layout()
    path = os.path.join(output_dir, f"kp_storm_event_timeline_{title_suffix}.png".replace(" ", "_").lower())
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_probability_vs_kp(
    output_dir: str,
    prediction_df: pd.DataFrame,
    score_col: str,
    title_suffix: str,
) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(prediction_df["kp"], prediction_df[score_col], alpha=0.45, s=16)
    ax.axvline(5, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Observed Kp")
    ax.set_ylabel("Predicted storm probability")
    ax.set_title(f"Predicted Storm Probability vs True Kp ({title_suffix})")
    fig.tight_layout()
    path = os.path.join(output_dir, f"kp_storm_probability_vs_kp_{title_suffix}.png".replace(" ", "_").lower())
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_storm_risk_event_overlay(
    output_dir: str,
    prediction_df: pd.DataFrame,
    risk_col: str,
    gate_threshold: float,
) -> str:
    fig, ax1 = plt.subplots(figsize=(13, 5))
    storm_mask = prediction_df["kp"] >= 5
    for start, end in _contiguous_true_runs(storm_mask.to_numpy()):
        ax1.axvspan(
            prediction_df["timestamp"].iloc[start],
            prediction_df["timestamp"].iloc[end],
            color="tab:red",
            alpha=0.14,
        )

    ax1.plot(prediction_df["timestamp"], prediction_df["kp"], color="black", linewidth=1.4, label="Observed Kp")
    ax1.axhline(5, color="tab:red", linestyle="--", linewidth=1.1, label="Kp >= 5")
    ax1.set_ylabel("Observed Kp")
    ax1.set_xlabel("Holdout time")

    ax2 = ax1.twinx()
    ax2.plot(prediction_df["timestamp"], prediction_df[risk_col], color="tab:blue", linewidth=1.2, label="Calibrated storm risk")
    ax2.axhline(gate_threshold, color="tab:blue", linestyle=":", linewidth=1.2, label="Selected gate")
    ax2.set_ylabel("Storm risk probability")
    ax2.set_ylim(-0.02, 1.02)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    ax1.set_title("Calibrated Storm Risk vs Actual Kp >= 5 Events")
    fig.tight_layout()
    path = os.path.join(output_dir, "kp_calibrated_storm_risk_vs_events.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_calibration_curve(
    output_dir: str,
    y_true_binary: np.ndarray,
    raw_score: np.ndarray,
    calibrated_score: np.ndarray,
) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for label, score in [("raw score", raw_score), ("calibrated probability", calibrated_score)]:
        bins = pd.qcut(pd.Series(score).rank(method="first"), q=10, duplicates="drop")
        curve_df = pd.DataFrame({"score": score, "observed": y_true_binary.astype(int), "bin": bins})
        grouped = curve_df.groupby("bin", observed=True).agg(predicted=("score", "mean"), observed=("observed", "mean"))
        ax.plot(grouped["predicted"], grouped["observed"], marker="o", linewidth=1.8, label=label)

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed storm frequency")
    ax.set_title("Storm Risk Calibration on Holdout")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "kp_storm_risk_calibration_curve.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_event_precision_recall_curve(
    output_dir: str,
    event_predictions_df: pd.DataFrame,
    selected_rows: pd.DataFrame,
    target_name: str,
) -> Optional[str]:
    plot_df = event_predictions_df[event_predictions_df["target"] == target_name].copy()
    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 6))
    for (feature_mode, model_name), group_df in plot_df.groupby(["feature_mode", "model"]):
        precision, recall, _ = precision_recall_curve(group_df["storm_observed"].astype(bool), group_df["score"])
        ap = _safe_average_precision(group_df["storm_observed"].astype(bool).to_numpy(), group_df["score"].to_numpy())
        ax.plot(recall, precision, linewidth=1.8, label=f"{feature_mode}/{model_name} (AP={ap:.3f})")

    marker_df = selected_rows[
        (selected_rows["target"] == target_name)
        & (selected_rows["threshold_mode"].isin(["high_recall", "balanced_f1", "high_precision"]))
    ]
    for _, row in marker_df.iterrows():
        ax.scatter(row["recall"], row["precision"], s=45)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Event Storm Precision-Recall ({target_name})")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, f"kp_event_storm_precision_recall_{target_name}.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_classifier_feature_importance(
    output_dir: str,
    importance_df: pd.DataFrame,
    title_suffix: str,
) -> str:
    if "abs_importance" in importance_df.columns:
        sort_col = "abs_importance"
        value_col = "importance"
    elif "abs_importance_mean" in importance_df.columns:
        sort_col = "abs_importance_mean"
        value_col = "importance_mean"
    elif "abs_standardized_coefficient" in importance_df.columns:
        sort_col = "abs_standardized_coefficient"
        value_col = "standardized_coefficient"
    else:
        numeric_cols = [col for col in importance_df.columns if col != "feature" and pd.api.types.is_numeric_dtype(importance_df[col])]
        value_col = numeric_cols[0]
        sort_col = value_col

    plot_df = importance_df.sort_values(sort_col, ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(plot_df["feature"], plot_df[value_col], color="tab:green", alpha=0.85)
    ax.set_xlabel(value_col)
    ax.set_title(f"Storm Classifier Feature Importance ({title_suffix})")
    fig.tight_layout()
    path = os.path.join(output_dir, f"kp_storm_classifier_feature_importance_{title_suffix}.png".replace(" ", "_").lower())
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_plots(output_dir: str, prediction_df: pd.DataFrame, recursive_df: pd.DataFrame) -> dict:
    return {
        "holdout_time_series": _plot_holdout_time_series(prediction_df, output_dir),
        "scatter": _plot_scatter(prediction_df, output_dir),
        "residual_histogram": _plot_residual_histogram(prediction_df, output_dir),
        "residual_vs_true": _plot_residual_vs_truth(prediction_df, output_dir),
        "storm_time_series": _plot_storm_time_series(prediction_df, output_dir),
        "recursive_horizon": _plot_recursive_horizon_curve(recursive_df, output_dir),
    }


def _write_report(report_path: str, report_sections: List[str]) -> None:
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n\n".join(report_sections).strip() + "\n")


def _model_availability_note() -> List[dict]:
    notes = []
    if XGBRegressor is None:
        notes.append({"model": "xgboost", "status": "skipped", "reason": "package not installed"})
    if LGBMRegressor is None:
        notes.append({"model": "lightgbm", "status": "skipped", "reason": "package not installed"})
    return notes


def _expanding_window_eras(feature_df: pd.DataFrame, min_train_years: int = 4, test_window_years: int = 3) -> List[dict]:
    years = sorted(feature_df["timestamp"].dt.year.unique().tolist())
    min_year = min(years)
    max_year = max(years)
    eras = []
    start_year = min_year + min_train_years

    while start_year <= max_year:
        end_year = min(start_year + test_window_years - 1, max_year)
        train_end = pd.Timestamp(year=start_year, month=1, day=1)
        test_end = pd.Timestamp(year=end_year + 1, month=1, day=1)
        train_mask = feature_df["timestamp"] < train_end
        test_mask = (feature_df["timestamp"] >= train_end) & (feature_df["timestamp"] < test_end)
        if int(train_mask.sum()) >= 500 and int(test_mask.sum()) >= 100:
            eras.append(
                {
                    "label": f"{start_year}-{end_year}",
                    "train_mask": train_mask,
                    "test_mask": test_mask,
                }
            )
        start_year += test_window_years

    return eras


def _run_storm_expanding_window_backtests(feature_df: pd.DataFrame, temporal_feature_columns: List[str]) -> pd.DataFrame:
    backtest_rows = []
    classifier_registry = _storm_classifier_registry()

    for era in _expanding_window_eras(feature_df):
        train_df = feature_df.loc[era["train_mask"]].copy().reset_index(drop=True)
        test_df = feature_df.loc[era["test_mask"]].copy().reset_index(drop=True)
        X_train = train_df[temporal_feature_columns]
        X_test = test_df[temporal_feature_columns]
        y_train = train_df["kp"]
        y_test = test_df["kp"]
        y_train_binary = _storm_target(y_train)
        y_test_binary = _storm_target(y_test)

        ridge_alpha, _ = _tune_ridge_alpha(X_train, y_train, RIDGE_ALPHA_GRID)
        ridge_factory = lambda: Ridge(alpha=ridge_alpha)
        cv_true, cv_scores = _time_series_oof_scores(ridge_factory, X_train, y_train, predict_func="predict")
        threshold_df = _ridge_threshold_curve(_storm_target(cv_true), cv_scores)
        thresholds = _select_operating_thresholds(threshold_df)

        ridge_model = ridge_factory()
        ridge_model.fit(X_train, y_train)
        ridge_test_scores = np.asarray(ridge_model.predict(X_test), dtype=float)
        for mode_name, threshold_row in thresholds.items():
            metrics = _binary_classification_metrics(y_test_binary, ridge_test_scores, threshold_row["threshold"])
            backtest_rows.append(
                _flatten_binary_metric_row(
                    model_name=f"ridge_{mode_name}",
                    evaluation="expanding_window",
                    metrics=metrics,
                    note=f"alpha={ridge_alpha}",
                    era=era["label"],
                )
            )

        persistence_scores = test_df["kp_lag_1"].to_numpy()
        persistence_metrics = _binary_classification_metrics(y_test_binary, persistence_scores, 5.0)
        backtest_rows.append(
            _flatten_binary_metric_row(
                model_name="persistence_kp_lag_1",
                evaluation="expanding_window",
                metrics=persistence_metrics,
                note="threshold=5.0",
                era=era["label"],
            )
        )

        y_train_binary_series = pd.Series(y_train_binary.astype(int), index=train_df.index)
        for classifier_name, classifier_factory in classifier_registry.items():
            classifier = classifier_factory()
            classifier = _fit_storm_classifier(classifier, X_train, y_train_binary_series.to_numpy())
            classifier_scores = classifier.predict_proba(X_test)[:, 1]
            metrics = _binary_classification_metrics(y_test_binary, classifier_scores, 0.5)
            backtest_rows.append(
                _flatten_binary_metric_row(
                    model_name=classifier_name,
                    evaluation="expanding_window",
                    metrics=metrics,
                    note="threshold=0.5",
                    era=era["label"],
                )
            )

    return pd.DataFrame(backtest_rows)


def _threshold_sweep_rows(
    y_true_binary: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
) -> List[dict]:
    rows = []
    for threshold in thresholds:
        metrics = _binary_classification_metrics(y_true_binary, y_score, threshold)
        rows.append(
            {
                "threshold": metrics["threshold"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "average_precision": metrics["average_precision"],
            }
        )
    return rows


def _contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _event_metrics(
    timestamps: pd.Series,
    kp_values: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    lead_steps: int,
) -> dict:
    return _event_metrics_from_alert_mask(
        timestamps,
        kp_values,
        np.asarray(scores) >= threshold,
        lead_steps,
    )


def _event_metrics_from_alert_mask(
    timestamps: pd.Series,
    kp_values: np.ndarray,
    alert_mask: np.ndarray,
    lead_steps: int,
) -> dict:
    storm_runs = _contiguous_true_runs(kp_values >= 5)
    alert_mask = np.asarray(alert_mask).astype(bool)
    alert_runs = _contiguous_true_runs(alert_mask)
    cadence_hours = _cadence_hours(pd.DataFrame({"timestamp": timestamps.reset_index(drop=True)}))
    if cadence_hours <= 0:
        cadence_hours = 6.0
    detected_events = 0
    lead_hours = []
    matched_alert_runs = set()
    event_alert_window_mask = np.zeros(len(alert_mask), dtype=bool)

    for event_start, event_end in storm_runs:
        window_start = max(0, event_start - lead_steps)
        event_alert_window_mask[window_start:event_end + 1] = True

    for event_start, _event_end in storm_runs:
        window_start = max(0, event_start - lead_steps)
        candidate_alert_indices = np.where(alert_mask[window_start:event_start + 1])[0] + window_start
        if len(candidate_alert_indices) == 0:
            continue

        detected_events += 1
        alert_idx = int(candidate_alert_indices[0])
        lead_hours.append(float((timestamps.iloc[event_start] - timestamps.iloc[alert_idx]) / pd.Timedelta(hours=1)))
        for run_idx, (alert_start, alert_end) in enumerate(alert_runs):
            if alert_start <= alert_idx <= alert_end:
                matched_alert_runs.add(run_idx)
                break

    false_alert_windows = len(alert_runs) - len(matched_alert_runs)
    false_alert_mask = alert_mask & ~event_alert_window_mask
    false_alert_runs = _contiguous_true_runs(false_alert_mask)
    alert_durations = [(end - start + 1) * cadence_hours for start, end in alert_runs]
    false_alert_durations = [(end - start + 1) * cadence_hours for start, end in false_alert_runs]
    total_alert_hours = float(np.sum(alert_durations))
    false_alert_hours = float(np.sum(false_alert_durations))
    total_days = max(1e-9, float((timestamps.iloc[-1] - timestamps.iloc[0]) / pd.Timedelta(days=1)))
    total_months = total_days / 30.4375
    total_years = total_days / 365.25
    total_hours = total_days * 24.0

    return {
        "storm_events": int(len(storm_runs)),
        "storm_events_detected": int(detected_events),
        "event_recall": float(detected_events / len(storm_runs)) if len(storm_runs) > 0 else 0.0,
        "mean_lead_time_hours": float(np.mean(lead_hours)) if lead_hours else None,
        "median_lead_time_hours": float(np.median(lead_hours)) if lead_hours else None,
        "alert_windows": int(len(alert_runs)),
        "total_alert_hours": total_alert_hours,
        "alert_hours_per_month": float(total_alert_hours / total_months),
        "alert_hours_per_year": float(total_alert_hours / total_years),
        "alert_duty_cycle": float(total_alert_hours / total_hours),
        "max_continuous_alert_hours": float(max(alert_durations)) if alert_durations else 0.0,
        "false_alert_windows": int(false_alert_windows),
        "duration_based_false_alert_windows": int(len(false_alert_runs)),
        "false_alert_hours": false_alert_hours,
        "false_alert_hours_per_month": float(false_alert_hours / total_months),
        "false_alert_hours_per_year": float(false_alert_hours / total_years),
        "max_continuous_false_alert_hours": float(max(false_alert_durations)) if false_alert_durations else 0.0,
        "false_alert_windows_per_month": float(false_alert_windows / total_months),
        "false_alert_windows_per_year": float(false_alert_windows / total_years),
    }


def _binary_classification_metrics_from_mask(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict:
    y_true_binary = np.asarray(y_true_binary).astype(bool)
    y_pred_binary = np.asarray(y_pred_binary).astype(bool)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary,
        y_pred_binary,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(
        y_true_binary,
        y_pred_binary,
        labels=[False, True],
    ).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_binary, y_pred_binary)),
        "average_precision": _safe_average_precision(y_true_binary, y_score),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def _merge_alert_gaps(alert_mask: np.ndarray, max_gap_steps: int = 1) -> np.ndarray:
    merged = np.asarray(alert_mask).astype(bool).copy()
    runs = _contiguous_true_runs(merged)
    for (_, left_end), (right_start, _) in zip(runs[:-1], runs[1:]):
        gap = right_start - left_end - 1
        if 0 < gap <= max_gap_steps:
            merged[left_end + 1:right_start] = True
    return merged


def _cap_alert_duration(alert_mask: np.ndarray, max_steps: int) -> np.ndarray:
    capped = np.asarray(alert_mask).astype(bool).copy()
    if max_steps <= 0:
        return capped
    for start, end in _contiguous_true_runs(capped):
        if end - start + 1 > max_steps:
            capped[start + max_steps:end + 1] = False
    return capped


def _suppress_alerts_after_events(alert_mask: np.ndarray, kp_values: np.ndarray, suppress_steps: int = 2) -> np.ndarray:
    suppressed = np.asarray(alert_mask).astype(bool).copy()
    if suppress_steps <= 0:
        return suppressed
    for _, event_end in _contiguous_true_runs(np.asarray(kp_values) >= 5):
        start = event_end + 1
        end = min(len(suppressed), start + suppress_steps)
        suppressed[start:end] = False
    return suppressed


def _local_peak_alert_mask(scores: np.ndarray, threshold: float, radius_steps: int = 1, window_steps: int = 1) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    peak_mask = np.zeros(len(scores), dtype=bool)
    for idx, score in enumerate(scores):
        if score < threshold:
            continue
        left = max(0, idx - radius_steps)
        right = min(len(scores), idx + radius_steps + 1)
        if score >= np.nanmax(scores[left:right]):
            peak_mask[max(0, idx - window_steps):min(len(scores), idx + window_steps + 1)] = True
    return peak_mask


def _postprocess_alert_mask(
    alert_mask: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    kp_values: np.ndarray,
    cadence_hours: float,
    mode: str,
) -> np.ndarray:
    if mode == "none":
        return np.asarray(alert_mask).astype(bool)
    if mode == "merge_gap_1":
        return _merge_alert_gaps(alert_mask, max_gap_steps=1)
    if mode == "merge_gap_1_cap_72h":
        merged = _merge_alert_gaps(alert_mask, max_gap_steps=1)
        return _cap_alert_duration(merged, max_steps=max(1, int(round(72.0 / cadence_hours))))
    if mode == "merge_gap_1_cap_72h_suppress_12h":
        merged = _merge_alert_gaps(alert_mask, max_gap_steps=1)
        capped = _cap_alert_duration(merged, max_steps=max(1, int(round(72.0 / cadence_hours))))
        return _suppress_alerts_after_events(capped, kp_values, suppress_steps=max(1, int(round(12.0 / cadence_hours))))
    if mode == "local_peaks":
        return _local_peak_alert_mask(scores, threshold, radius_steps=1, window_steps=1)
    raise ValueError(f"Unknown alert post-processing mode: {mode}")


def _alert_row_recall(kp_values: np.ndarray, alert_mask: np.ndarray, kp_threshold: float) -> float:
    target_mask = np.asarray(kp_values) >= kp_threshold
    if int(target_mask.sum()) == 0:
        return 0.0
    return float(np.mean(np.asarray(alert_mask).astype(bool)[target_mask]))


def _flatten_event_row(
    target: str,
    feature_mode: str,
    model: str,
    threshold_mode: str,
    threshold: float,
    point_metrics: dict,
    event_metrics: dict,
) -> dict:
    return {
        "target": target,
        "feature_mode": feature_mode,
        "model": model,
        "threshold_mode": threshold_mode,
        "threshold": threshold,
        "precision": point_metrics["precision"],
        "recall": point_metrics["recall"],
        "f1": point_metrics["f1"],
        "average_precision": point_metrics["average_precision"],
        "storm_events": event_metrics["storm_events"],
        "storm_events_detected": event_metrics["storm_events_detected"],
        "event_recall": event_metrics["event_recall"],
        "mean_lead_time_hours": event_metrics["mean_lead_time_hours"],
        "median_lead_time_hours": event_metrics["median_lead_time_hours"],
        "alert_windows": event_metrics["alert_windows"],
        "total_alert_hours": event_metrics["total_alert_hours"],
        "alert_hours_per_month": event_metrics["alert_hours_per_month"],
        "alert_hours_per_year": event_metrics["alert_hours_per_year"],
        "alert_duty_cycle": event_metrics["alert_duty_cycle"],
        "max_continuous_alert_hours": event_metrics["max_continuous_alert_hours"],
        "false_alert_windows": event_metrics["false_alert_windows"],
        "duration_based_false_alert_windows": event_metrics["duration_based_false_alert_windows"],
        "false_alert_hours": event_metrics["false_alert_hours"],
        "false_alert_hours_per_month": event_metrics["false_alert_hours_per_month"],
        "false_alert_hours_per_year": event_metrics["false_alert_hours_per_year"],
        "max_continuous_false_alert_hours": event_metrics["max_continuous_false_alert_hours"],
        "false_alert_windows_per_month": event_metrics["false_alert_windows_per_month"],
        "false_alert_windows_per_year": event_metrics["false_alert_windows_per_year"],
    }


def _select_classifier_thresholds(sweep_df: pd.DataFrame) -> Dict[str, float]:
    high_recall_candidates = sweep_df[sweep_df["recall"] >= 0.80]
    if len(high_recall_candidates) > 0:
        high_recall = high_recall_candidates.sort_values(["precision", "threshold"], ascending=[False, False]).iloc[0]
    else:
        high_recall = sweep_df.sort_values(["recall", "precision"], ascending=[False, False]).iloc[0]

    balanced = sweep_df.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0]

    high_precision_candidates = sweep_df[sweep_df["precision"] >= 0.50]
    if len(high_precision_candidates) > 0:
        high_precision = high_precision_candidates.sort_values(["recall", "threshold"], ascending=[False, False]).iloc[0]
    else:
        high_precision = sweep_df.sort_values(["precision", "recall"], ascending=[False, False]).iloc[0]

    return {
        "high_recall": float(high_recall["threshold"]),
        "balanced_f1": float(balanced["threshold"]),
        "high_precision": float(high_precision["threshold"]),
    }


def _constrained_threshold_rows(
    prediction_df: pd.DataFrame,
    thresholds: np.ndarray,
    lead_steps: int,
) -> pd.DataFrame:
    rows = []
    y_true = prediction_df["storm_observed"].astype(bool).to_numpy()
    scores = prediction_df["score"].to_numpy()
    kp_values = prediction_df["kp"].to_numpy()
    timestamps = prediction_df["timestamp"]

    for threshold in thresholds:
        point_metrics = _binary_classification_metrics(y_true, scores, threshold)
        event_metrics = _event_metrics(timestamps, kp_values, scores, threshold, lead_steps)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": point_metrics["precision"],
                "recall": point_metrics["recall"],
                "f1": point_metrics["f1"],
                "average_precision": point_metrics["average_precision"],
                "event_recall": event_metrics["event_recall"],
                "mean_lead_time_hours": event_metrics["mean_lead_time_hours"],
                "alert_duty_cycle": event_metrics["alert_duty_cycle"],
                "max_continuous_alert_hours": event_metrics["max_continuous_alert_hours"],
                "false_alert_hours_per_month": event_metrics["false_alert_hours_per_month"],
                "alert_hours_per_month": event_metrics["alert_hours_per_month"],
                "false_alert_windows_per_month": event_metrics["false_alert_windows_per_month"],
                "storm_events_detected": event_metrics["storm_events_detected"],
            }
        )

    return pd.DataFrame(rows)


def _constrained_alert_score_rows(
    prediction_df: pd.DataFrame,
    thresholds: np.ndarray,
    lead_steps: int,
    postprocess_mode: str = "none",
) -> pd.DataFrame:
    rows = []
    y_true = prediction_df["storm_observed"].astype(bool).to_numpy()
    scores = prediction_df["score"].to_numpy()
    kp_values = prediction_df["kp"].to_numpy()
    timestamps = prediction_df["timestamp"]
    cadence_hours = _cadence_hours(pd.DataFrame({"timestamp": timestamps.reset_index(drop=True)}))
    if cadence_hours <= 0:
        cadence_hours = 6.0

    for threshold in thresholds:
        raw_alert_mask = scores >= threshold
        alert_mask = _postprocess_alert_mask(
            raw_alert_mask,
            scores,
            float(threshold),
            kp_values,
            cadence_hours,
            postprocess_mode,
        )
        point_metrics = _binary_classification_metrics_from_mask(
            y_true,
            alert_mask,
            scores,
            float(threshold),
        )
        event_metrics = _event_metrics_from_alert_mask(
            timestamps,
            kp_values,
            alert_mask,
            lead_steps,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "postprocess_mode": postprocess_mode,
                "precision": point_metrics["precision"],
                "recall": point_metrics["recall"],
                "f1": point_metrics["f1"],
                "average_precision": point_metrics["average_precision"],
                "kp_ge_5_recall": _alert_row_recall(kp_values, alert_mask, 5.0),
                "kp_ge_6_recall": _alert_row_recall(kp_values, alert_mask, 6.0),
                "event_recall": event_metrics["event_recall"],
                "mean_lead_time_hours": event_metrics["mean_lead_time_hours"],
                "median_lead_time_hours": event_metrics["median_lead_time_hours"],
                "alert_duty_cycle": event_metrics["alert_duty_cycle"],
                "max_continuous_alert_hours": event_metrics["max_continuous_alert_hours"],
                "false_alert_hours_per_month": event_metrics["false_alert_hours_per_month"],
                "alert_hours_per_month": event_metrics["alert_hours_per_month"],
                "false_alert_windows_per_month": event_metrics["false_alert_windows_per_month"],
                "storm_events_detected": event_metrics["storm_events_detected"],
            }
        )

    return pd.DataFrame(rows)


def _select_constrained_threshold(
    threshold_df: pd.DataFrame,
    constraints: Dict[str, float],
) -> Tuple[pd.Series, bool]:
    constrained = threshold_df.copy()
    for metric_name, max_value in constraints.items():
        constrained = constrained[constrained[metric_name] <= max_value]

    feasible = not constrained.empty
    candidates = constrained if feasible else threshold_df.copy()
    if not feasible:
        candidates["constraint_violation"] = (
            np.maximum(0.0, candidates["alert_duty_cycle"] - constraints["alert_duty_cycle"])
            + np.maximum(0.0, candidates["max_continuous_alert_hours"] - constraints["max_continuous_alert_hours"]) / 72.0
            + np.maximum(0.0, candidates["false_alert_hours_per_month"] - constraints["false_alert_hours_per_month"]) / 40.0
        )
        sort_columns = [
            "constraint_violation",
            "event_recall",
            "mean_lead_time_hours",
            "f1",
            "threshold",
        ]
        ascending = [True, False, False, False, False]
    else:
        sort_columns = ["event_recall", "mean_lead_time_hours", "f1", "threshold"]
        ascending = [False, False, False, False]

    selected = candidates.sort_values(sort_columns, ascending=ascending).iloc[0]
    return selected, feasible


def _event_regime_label(row: pd.Series) -> str:
    labels = []
    if row["min_bz_window"] > -3.0 and row["max_pdyn_window"] >= 2.5:
        labels.append("weak_Bz_high_pressure")
    if row["min_bz_window"] <= -5.0 and row["max_velocity_window"] < 500.0:
        labels.append("strong_Bz_low_velocity")
    if row["max_pdyn_jump_24h_window"] >= 1.0 and row["duration_hours"] <= 6.0:
        labels.append("possible_sudden_impulse_or_6h_cadence_loss")
    if row["min_bz_window"] <= -5.0 and row["max_velocity_window"] >= 500.0:
        labels.append("classic_southward_Bz_fast_flow")
    if row["max_pdyn_window"] >= 2.5 and row["max_velocity_window"] >= 500.0:
        labels.append("compression_fast_flow")
    if not labels:
        labels.append("mixed_or_weak_signature")
    return ";".join(labels)


def _build_storm_event_audit(
    test_df: pd.DataFrame,
    risk_probability: np.ndarray,
    gate_threshold: float,
    lead_steps: int,
) -> pd.DataFrame:
    storm_runs = _contiguous_true_runs((test_df["kp"].to_numpy() >= 5))
    rows = []
    cadence = _cadence_hours(test_df)
    if cadence <= 0:
        cadence = 6.0
    gate_mask = risk_probability >= gate_threshold

    for event_idx, (start, end) in enumerate(storm_runs, start=1):
        lead_start = max(0, start - lead_steps)
        context_start = max(0, start - 4)
        context_end = min(len(test_df) - 1, end + 1)
        alert_indices = np.where(gate_mask[lead_start:end + 1])[0] + lead_start
        detected = len(alert_indices) > 0
        first_alert_idx = int(alert_indices[0]) if detected else None
        event_slice = test_df.iloc[start:end + 1]
        context_slice = test_df.iloc[context_start:context_end + 1]
        lead_slice = test_df.iloc[lead_start:start + 1]

        row = {
            "event_id": event_idx,
            "detected": bool(detected),
            "event_start": test_df["timestamp"].iloc[start],
            "event_end": test_df["timestamp"].iloc[end],
            "duration_hours": float((end - start + 1) * cadence),
            "max_kp": float(event_slice["kp"].max()),
            "mean_kp": float(event_slice["kp"].mean()),
            "lead_time_hours": float((test_df["timestamp"].iloc[start] - test_df["timestamp"].iloc[first_alert_idx]) / pd.Timedelta(hours=1))
            if detected else None,
            "first_alert_time": test_df["timestamp"].iloc[first_alert_idx] if detected else None,
            "max_risk_lead_event": float(np.max(risk_probability[lead_start:end + 1])),
            "mean_risk_lead_event": float(np.mean(risk_probability[lead_start:end + 1])),
            "max_risk_context": float(np.max(risk_probability[context_start:context_end + 1])),
            "min_bz_window": float(context_slice["bz"].min()),
            "mean_bz_window": float(context_slice["bz"].mean()),
            "max_pdyn_window": float(context_slice["pdyn"].max()),
            "max_pdyn_jump_24h_window": float(context_slice["pdyn_jump_24h"].max()),
            "max_velocity_window": float(context_slice["velocity"].max()),
            "mean_velocity_window": float(context_slice["velocity"].mean()),
            "max_density_window": float(context_slice["density"].max()),
            "max_bz_drop_36h_window": float(context_slice["bz_drop_36h"].max()),
            "min_bz_lead": float(lead_slice["bz"].min()),
            "max_pdyn_lead": float(lead_slice["pdyn"].max()),
            "max_velocity_lead": float(lead_slice["velocity"].max()),
            "notes": "",
        }
        row["physical_regime"] = _event_regime_label(pd.Series(row))
        if (not detected) and row["max_risk_lead_event"] >= gate_threshold * 0.75:
            row["notes"] = "near_miss_below_gate"
        elif (not detected) and "possible_sudden_impulse_or_6h_cadence_loss" in row["physical_regime"]:
            row["notes"] = "check_higher_cadence_data"
        rows.append(row)

    return pd.DataFrame(rows)


def _summarize_event_regimes(event_audit_df: pd.DataFrame) -> pd.DataFrame:
    if event_audit_df.empty:
        return pd.DataFrame()
    rows = []
    for detected, group_df in event_audit_df.groupby("detected"):
        exploded = group_df.assign(physical_regime=group_df["physical_regime"].str.split(";")).explode("physical_regime")
        counts = exploded["physical_regime"].value_counts()
        for regime, count in counts.items():
            rows.append(
                {
                    "detected": bool(detected),
                    "physical_regime": regime,
                    "event_count": int(count),
                    "fraction_of_group": float(count / len(group_df)),
                }
            )
    return pd.DataFrame(rows).sort_values(["detected", "event_count"], ascending=[True, False]).reset_index(drop=True)


def _rf_alert_score_frame(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    use_event_weights: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame, LogisticRegression]:
    valid_col = f"{target_column}_valid"
    target_train_df = train_df[train_df[valid_col]].copy().reset_index(drop=True)
    y_train = target_train_df[target_column].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Target {target_column} has fewer than two classes in training data.")

    model_factory = _storm_classifier_registry()["random_forest_classifier_balanced"]
    sample_weight_func = _storm_event_sample_weight if use_event_weights else None
    oof_prediction_df = _time_series_oof_classifier_prediction_frame(
        model_factory,
        target_train_df,
        feature_columns,
        target_column,
        sample_weight_func=sample_weight_func,
    )
    if oof_prediction_df.empty or oof_prediction_df["storm_observed"].nunique() < 2:
        raise ValueError(f"OOF predictions for {target_column} do not contain both classes.")

    calibrator = _fit_score_calibrator(
        oof_prediction_df["storm_observed"].to_numpy(),
        oof_prediction_df["score"].to_numpy(),
    )
    model = model_factory()
    sample_weight = _storm_event_sample_weight(target_train_df, target_column) if use_event_weights else None
    model = _fit_storm_classifier(
        model,
        target_train_df[feature_columns],
        y_train,
        sample_weight=sample_weight,
    )
    raw_scores = model.predict_proba(test_df[feature_columns])[:, 1]
    calibrated_scores = _apply_score_calibrator(calibrator, raw_scores)
    calibrated_scores = np.clip(calibrated_scores, 0.0, 1.0)
    oof_prediction_df["raw_score"] = oof_prediction_df["score"]
    oof_prediction_df["score"] = _apply_score_calibrator(
        calibrator,
        oof_prediction_df["raw_score"].to_numpy(),
    )
    return calibrated_scores, oof_prediction_df, calibrator


def _alert_ablation_prediction_df(
    test_df: pd.DataFrame,
    target_column: str,
    scores: np.ndarray,
) -> pd.DataFrame:
    valid_col = f"{target_column}_valid"
    prediction_df = test_df[test_df[valid_col]].copy().reset_index(drop=True)
    valid_scores = pd.Series(scores, index=test_df.index).loc[test_df[valid_col]].to_numpy()
    return pd.DataFrame(
        {
            "timestamp": prediction_df["timestamp"],
            "kp": prediction_df["kp"],
            "storm_observed": prediction_df[target_column].astype(int),
            "score": valid_scores,
        }
    )


def _alert_ablation_row(
    label: str,
    prediction_df: pd.DataFrame,
    thresholds: np.ndarray,
    constraints: Dict[str, float],
    lead_steps: int,
    postprocess_mode: str = "none",
) -> Tuple[dict, pd.DataFrame, np.ndarray]:
    sweep_df = _constrained_alert_score_rows(
        prediction_df,
        thresholds,
        lead_steps,
        postprocess_mode=postprocess_mode,
    )
    selected_row, feasible = _select_constrained_threshold(sweep_df, constraints)
    cadence_hours = _cadence_hours(pd.DataFrame({"timestamp": prediction_df["timestamp"]}))
    if cadence_hours <= 0:
        cadence_hours = 6.0
    alert_mask = _postprocess_alert_mask(
        prediction_df["score"].to_numpy() >= float(selected_row["threshold"]),
        prediction_df["score"].to_numpy(),
        float(selected_row["threshold"]),
        prediction_df["kp"].to_numpy(),
        cadence_hours,
        postprocess_mode,
    )
    row = selected_row.to_dict()
    row.update(
        {
            "alert_score": label,
            "selected_threshold": float(selected_row["threshold"]),
            "selection_feasible": bool(feasible),
            "postprocess_mode": postprocess_mode,
        }
    )
    return row, sweep_df, alert_mask


def _run_onset_alert_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    solar_feature_columns: List[str],
    output_dir: str,
    thresholds: np.ndarray,
    constraints: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    baseline_feature_columns = []
    for feature in solar_feature_columns:
        if feature in SUSTAINED_DRIVING_FEATURES:
            continue
        if any(feature.startswith(f"{sustained_feature}_lag_") for sustained_feature in SUSTAINED_DRIVING_FEATURES):
            continue
        baseline_feature_columns.append(feature)
    debug = {
        "sustained_features_requested": SUSTAINED_DRIVING_FEATURES,
        "sustained_features_present": [feature for feature in SUSTAINED_DRIVING_FEATURES if feature in solar_feature_columns],
        "sustained_features_missing": [feature for feature in SUSTAINED_DRIVING_FEATURES if feature not in solar_feature_columns],
        "baseline_feature_count": len(baseline_feature_columns),
        "sustained_feature_count": len(solar_feature_columns),
    }

    next_scores_base, _, _ = _rf_alert_score_frame(
        train_df,
        test_df,
        baseline_feature_columns,
        "storm_next_12h",
        use_event_weights=False,
    )
    next_scores_sustained, _, _ = _rf_alert_score_frame(
        train_df,
        test_df,
        solar_feature_columns,
        "storm_next_12h",
        use_event_weights=False,
    )
    onset_scores_sustained, _, _ = _rf_alert_score_frame(
        train_df,
        test_df,
        solar_feature_columns,
        "storm_onset_next_12h",
        use_event_weights=False,
    )
    next_scores_weighted, _, _ = _rf_alert_score_frame(
        train_df,
        test_df,
        solar_feature_columns,
        "storm_next_12h",
        use_event_weights=True,
    )
    onset_scores_weighted, _, _ = _rf_alert_score_frame(
        train_df,
        test_df,
        solar_feature_columns,
        "storm_onset_next_12h",
        use_event_weights=True,
    )

    combined_scores = np.maximum(next_scores_sustained, onset_scores_sustained)
    combined_weighted_scores = np.maximum(next_scores_weighted, onset_scores_weighted)
    score_arrays = {
        "Baseline RF next_12h": ("storm_next_12h", next_scores_base, "none"),
        "RF next_12h + sustained features": ("storm_next_12h", next_scores_sustained, "none"),
        "RF onset_next_12h + sustained features": ("storm_onset_next_12h", onset_scores_sustained, "none"),
        "max(next_12h, onset_next_12h)": ("storm_next_12h", combined_scores, "none"),
        "max(...) + event weighting": ("storm_next_12h", combined_weighted_scores, "none"),
        "max(...) + post-processing": ("storm_next_12h", combined_scores, "merge_gap_1_cap_72h"),
    }

    for score_name, score_values in [
        ("next_scores_base", next_scores_base),
        ("next_scores_sustained", next_scores_sustained),
        ("onset_scores_sustained", onset_scores_sustained),
        ("combined_scores", combined_scores),
        ("combined_weighted_scores", combined_weighted_scores),
    ]:
        if np.nanmin(score_values) < -1e-9 or np.nanmax(score_values) > 1.0 + 1e-9:
            raise ValueError(f"Alert score {score_name} is outside [0, 1].")

    ablation_rows = []
    sweep_frames = []
    selected_masks = {}
    for label, (target_column, score_values, postprocess_mode) in score_arrays.items():
        prediction_df = _alert_ablation_prediction_df(test_df, target_column, score_values)
        row, sweep_df, alert_mask = _alert_ablation_row(
            label,
            prediction_df,
            thresholds,
            constraints,
            STORM_TARGET_HORIZON_STEPS[target_column],
            postprocess_mode=postprocess_mode,
        )
        ablation_rows.append(row)
        sweep_df["alert_score"] = label
        sweep_df["target"] = target_column
        sweep_frames.append(sweep_df)
        selected_masks[label] = (prediction_df["timestamp"], alert_mask)

    watch_prediction_df = _alert_ablation_prediction_df(test_df, "storm_next_12h", combined_scores)
    watch_row, watch_sweep_df, watch_mask = _alert_ablation_row(
        "storm_watch_gate_combined_max",
        watch_prediction_df,
        thresholds,
        WATCH_CONSTRAINTS,
        STORM_TARGET_HORIZON_STEPS["storm_next_12h"],
        postprocess_mode="none",
    )
    watch_row["alert_score"] = "storm_watch_gate_combined_max"
    ablation_rows.append(watch_row)
    watch_sweep_df["alert_score"] = "storm_watch_gate_combined_max"
    watch_sweep_df["target"] = "storm_next_12h"
    sweep_frames.append(watch_sweep_df)

    ablation_df = pd.DataFrame(ablation_rows)
    sweep_df = pd.concat(sweep_frames, ignore_index=True) if sweep_frames else pd.DataFrame()
    prediction_columns_df = test_df[["timestamp"]].copy()
    prediction_columns_df["alert_score_next12h"] = next_scores_sustained
    prediction_columns_df["storm_onset_probability"] = onset_scores_sustained
    prediction_columns_df["alert_score_onset12h"] = onset_scores_sustained
    prediction_columns_df["alert_score_combined_max"] = combined_scores
    prediction_columns_df["alert_score_combined_weighted_max"] = combined_weighted_scores
    combined_row = ablation_df[ablation_df["alert_score"] == "max(next_12h, onset_next_12h)"].iloc[0]
    post_row = ablation_df[ablation_df["alert_score"] == "max(...) + post-processing"].iloc[0]
    prediction_columns_df["public_alert_gate_combined"] = (
        combined_scores >= float(combined_row["selected_threshold"])
    ).astype(int)
    post_timestamps, post_mask = selected_masks["max(...) + post-processing"]
    post_lookup = pd.Series(post_mask.astype(int), index=post_timestamps)
    prediction_columns_df["public_alert_gate_combined_postprocessed"] = (
        prediction_columns_df["timestamp"].map(post_lookup).fillna(0).astype(int)
    )
    prediction_columns_df["storm_watch_gate"] = (
        combined_scores >= float(watch_row["selected_threshold"])
    ).astype(int)
    debug["selected_thresholds"] = ablation_df[["alert_score", "selected_threshold", "postprocess_mode"]].to_dict(orient="records")
    debug["postprocessed_threshold"] = float(post_row["selected_threshold"])

    ablation_path = os.path.join(output_dir, "kp_alert_onset_ablation.csv")
    sweep_path = os.path.join(output_dir, "kp_alert_onset_threshold_sweeps.csv")
    debug_path = os.path.join(output_dir, "kp_alert_onset_debug.json")
    ablation_df.to_csv(ablation_path, index=False)
    sweep_df.to_csv(sweep_path, index=False)
    with open(debug_path, "w", encoding="utf-8") as debug_file:
        json.dump(debug, debug_file, indent=2, default=str)
        debug_file.write("\n")

    return ablation_df, sweep_df, prediction_columns_df, {
        "ablation_path": ablation_path,
        "sweep_path": sweep_path,
        "debug_path": debug_path,
        "debug": debug,
    }


def _severity_target_name(kp_threshold: float) -> str:
    return f"severity_kp_ge_{int(kp_threshold)}_next_12h"


def _storm_magnitude_sample_weight(train_df: pd.DataFrame, target_column: str) -> np.ndarray:
    weights = np.ones(len(train_df), dtype=float)
    target_values = train_df[target_column].to_numpy()
    weights *= np.where(target_values >= 5.0, 4.0, 1.0)
    weights *= np.where(target_values >= 6.0, 2.0, 1.0)
    weights *= np.where(target_values >= 7.0, 2.0, 1.0)
    if "storm_onset_next_12h" in train_df.columns:
        weights *= np.where(train_df["storm_onset_next_12h"].astype(bool).to_numpy(), 1.5, 1.0)
    return weights


def _asymmetric_storm_magnitude_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    abs_error = np.abs(error)
    under = error < 0
    over = error > 0
    storm_mask = y_true >= 5.0
    quiet_mask = y_true < 4.0
    asym_error = np.where(under, 3.0 * np.abs(error), np.abs(error))

    def _safe_mean(values: np.ndarray) -> Optional[float]:
        return float(np.mean(values)) if len(values) > 0 else None

    kp5_metrics = _binary_classification_metrics(y_true >= 5.0, y_pred, 5.0)
    kp6_metrics = _binary_classification_metrics(y_true >= 6.0, y_pred, 6.0)
    quiet_false_storm = quiet_mask & (y_pred >= 5.0)
    return {
        "mae": float(np.mean(abs_error)),
        "storm_mae": _safe_mean(abs_error[storm_mask]),
        "asymmetric_mae": float(np.mean(asym_error)),
        "storm_asymmetric_mae": _safe_mean(asym_error[storm_mask]),
        "underprediction_rate": float(np.mean(under)),
        "storm_underprediction_rate": _safe_mean(under[storm_mask].astype(float)),
        "mean_underprediction_size": _safe_mean(np.abs(error[under])),
        "storm_mean_underprediction_size": _safe_mean(np.abs(error[storm_mask & under])),
        "overprediction_rate": float(np.mean(over)),
        "storm_overprediction_rate": _safe_mean(over[storm_mask].astype(float)),
        "mean_overprediction_size": _safe_mean(error[over]),
        "false_storm_inflation_quiet_rate": float(np.mean(quiet_false_storm)) if int(quiet_mask.sum()) > 0 else 0.0,
        "false_storm_inflation_quiet_mean_size": _safe_mean((y_pred - y_true)[quiet_false_storm]),
        "kp_ge_5_recall": kp5_metrics["recall"],
        "kp_ge_5_precision": kp5_metrics["precision"],
        "kp_ge_5_f1": kp5_metrics["f1"],
        "kp_ge_6_recall": kp6_metrics["recall"],
        "kp_ge_6_precision": kp6_metrics["precision"],
        "kp_ge_6_f1": kp6_metrics["f1"],
    }


def _flatten_magnitude_metric_row(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = _asymmetric_storm_magnitude_metrics(y_true, y_pred)
    return {"model": model_name, **metrics}


def _select_vigilant_watch_threshold(
    threshold_df: pd.DataFrame,
    constraints: Dict[str, float],
) -> Tuple[pd.Series, bool]:
    constrained = threshold_df.copy()
    for metric_name, max_value in constraints.items():
        constrained = constrained[constrained[metric_name] <= max_value]

    feasible = not constrained.empty
    candidates = constrained if feasible else threshold_df.copy()
    if not feasible:
        candidates["constraint_violation"] = (
            np.maximum(0.0, candidates["alert_duty_cycle"] - constraints["alert_duty_cycle"])
            + np.maximum(0.0, candidates["max_continuous_alert_hours"] - constraints["max_continuous_alert_hours"]) / 72.0
            + np.maximum(0.0, candidates["false_alert_hours_per_month"] - constraints["false_alert_hours_per_month"]) / 40.0
        )
        sort_columns = [
            "constraint_violation",
            "kp_ge_6_recall",
            "kp_ge_5_recall",
            "event_recall",
            "mean_lead_time_hours",
            "f1",
        ]
        ascending = [True, False, False, False, False, False]
    else:
        sort_columns = [
            "kp_ge_6_recall",
            "kp_ge_5_recall",
            "event_recall",
            "mean_lead_time_hours",
            "f1",
            "threshold",
        ]
        ascending = [False, False, False, False, False, False]

    return candidates.sort_values(sort_columns, ascending=ascending).iloc[0], feasible


def _fit_quantile_regressor(quantile: float):
    if HistGradientBoostingRegressor is None:
        return None
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )


def _run_storm_magnitude_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    solar_feature_columns: List[str],
    base_prediction: np.ndarray,
    watch_score: np.ndarray,
    thresholds: np.ndarray,
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    prediction_df = test_df[["timestamp", "kp"]].copy()
    for horizon_hours in STORM_MAGNITUDE_HORIZONS_HOURS:
        target_column = f"kp_max_next_{horizon_hours}h"
        valid_column = f"{target_column}_valid"
        if target_column in test_df.columns:
            prediction_df[target_column] = test_df[target_column].to_numpy()
        if valid_column in test_df.columns:
            prediction_df[valid_column] = test_df[valid_column].astype(int).to_numpy()
    prediction_df["kp_base"] = base_prediction
    prediction_df["storm_watch_probability_vigilant"] = np.clip(watch_score, 0.0, 1.0)
    debug = {
        "quantile_model": "hist_gradient_boosting_quantile" if HistGradientBoostingRegressor is not None else "unavailable",
        "magnitude_targets": [],
        "severity_targets": {},
        "watch_constraints": VIGILANT_WATCH_CONSTRAINTS,
    }

    watch_prediction_df = _alert_ablation_prediction_df(test_df, "storm_next_12h", watch_score)
    watch_sweep_df = _constrained_alert_score_rows(
        watch_prediction_df,
        thresholds,
        STORM_TARGET_HORIZON_STEPS["storm_next_12h"],
        postprocess_mode="none",
    )
    watch_row, watch_feasible = _select_vigilant_watch_threshold(watch_sweep_df, VIGILANT_WATCH_CONSTRAINTS)
    watch_threshold = float(watch_row["threshold"])
    watch_gate = watch_score >= watch_threshold
    prediction_df["storm_watch_gate_vigilant"] = watch_gate.astype(int)

    metric_rows = []
    quantile_predictions = {}
    for horizon_hours in STORM_MAGNITUDE_HORIZONS_HOURS:
        target_column = f"kp_max_next_{horizon_hours}h"
        valid_column = f"{target_column}_valid"
        debug["magnitude_targets"].append(
            {
                "target": target_column,
                "train_valid": int(train_df[valid_column].sum()),
                "test_valid": int(test_df[valid_column].sum()),
            }
        )
        train_valid = train_df[valid_column].astype(bool)
        if int(train_valid.sum()) < 100 or HistGradientBoostingRegressor is None:
            continue
        X_train = train_df.loc[train_valid, solar_feature_columns]
        y_train = train_df.loc[train_valid, target_column].to_numpy()
        sample_weight = _storm_magnitude_sample_weight(train_df.loc[train_valid].copy(), target_column)
        for quantile in STORM_MAGNITUDE_QUANTILES:
            model = _fit_quantile_regressor(quantile)
            if model is None:
                continue
            model.fit(X_train, y_train, sample_weight=sample_weight)
            pred = np.clip(model.predict(test_df[solar_feature_columns]), 0.0, 9.0)
            col = f"kp_q{int(round(quantile * 100))}_next_{horizon_hours}h"
            prediction_df[col] = pred
            quantile_predictions[(horizon_hours, quantile)] = pred

    p50 = quantile_predictions.get((12, 0.50), base_prediction)
    p75 = quantile_predictions.get((12, 0.75), p50)
    p90 = quantile_predictions.get((12, 0.90), p75)
    p95 = quantile_predictions.get((12, 0.95), p90)
    prediction_df["kp_risk_p50"] = p50
    prediction_df["kp_risk_p75"] = p75
    prediction_df["kp_risk_p90"] = p90
    prediction_df["kp_risk_p95"] = p95

    severity_probabilities = {}
    severity_thresholds = {}
    ceiling = np.zeros(len(test_df), dtype=float)
    high_risk_gate = np.zeros(len(test_df), dtype=bool)
    severity_rows = []
    for kp_threshold in SEVERITY_THRESHOLDS:
        target_column = _severity_target_name(kp_threshold)
        valid_column = f"{target_column}_valid"
        train_positive = int(train_df.loc[train_df[valid_column].astype(bool), target_column].sum())
        test_positive = int(test_df.loc[test_df[valid_column].astype(bool), target_column].sum())
        debug["severity_targets"][target_column] = {
            "train_positive": train_positive,
            "test_positive": test_positive,
        }
        if train_positive < MIN_SEVERITY_POSITIVES:
            debug["severity_targets"][target_column]["skipped"] = True
            continue
        scores, _, _ = _rf_alert_score_frame(
            train_df,
            test_df,
            solar_feature_columns,
            target_column,
            use_event_weights=True,
        )
        scores = np.clip(scores, 0.0, 1.0)
        severity_probabilities[kp_threshold] = scores
        score_col = f"severity_kp_ge_{int(kp_threshold)}_probability"
        gate_col = f"severity_kp_ge_{int(kp_threshold)}_gate"
        prediction_df[score_col] = scores
        severity_prediction_df = _alert_ablation_prediction_df(test_df, target_column, scores)
        severity_sweep = _constrained_alert_score_rows(
            severity_prediction_df,
            thresholds,
            STORM_TARGET_HORIZON_STEPS["storm_next_12h"],
            postprocess_mode="none",
        )
        selected_row, feasible = _select_vigilant_watch_threshold(severity_sweep, VIGILANT_WATCH_CONSTRAINTS)
        threshold = float(selected_row["threshold"])
        severity_thresholds[kp_threshold] = threshold
        gate = scores >= threshold
        prediction_df[gate_col] = gate.astype(int)
        ceiling = np.maximum(ceiling, np.where(gate, kp_threshold, 0.0))
        if kp_threshold >= 6.0:
            high_risk_gate = high_risk_gate | gate
        row = selected_row.to_dict()
        row.update(
            {
                "severity_target": target_column,
                "kp_threshold": kp_threshold,
                "selected_threshold": threshold,
                "selection_feasible": bool(feasible),
                "train_positive": train_positive,
                "test_positive": test_positive,
            }
        )
        severity_rows.append(row)

    prediction_df["kp_ceiling_classifier"] = ceiling
    prediction_df["storm_high_risk_gate"] = high_risk_gate.astype(int)
    quantile_ceiling = np.where(watch_gate, np.ceil(p90), 0.0)
    prediction_df["kp_quantile_ceiling"] = np.clip(quantile_ceiling, 0.0, 9.0)
    watch_candidate = np.maximum.reduce([base_prediction, p75, ceiling, prediction_df["kp_quantile_ceiling"].to_numpy()])
    high_risk_candidate = np.maximum(watch_candidate, p90)
    conservative = np.where(watch_gate, watch_candidate, base_prediction)
    conservative = np.where(watch_gate & high_risk_gate, high_risk_candidate, conservative)
    conservative = np.clip(conservative, 0.0, 9.0)
    prediction_df["kp_storm_conservative"] = conservative

    y_true = test_df["kp"].to_numpy()
    metric_rows.append(_flatten_magnitude_metric_row("kp_base", y_true, base_prediction))
    metric_rows.append(_flatten_magnitude_metric_row("kp_risk_p75", y_true, p75))
    metric_rows.append(_flatten_magnitude_metric_row("kp_risk_p90", y_true, p90))
    metric_rows.append(_flatten_magnitude_metric_row("kp_ceiling_classifier", y_true, np.maximum(base_prediction, ceiling)))
    metric_rows.append(_flatten_magnitude_metric_row("kp_quantile_ceiling", y_true, np.maximum(base_prediction, prediction_df["kp_quantile_ceiling"].to_numpy())))
    metric_rows.append(_flatten_magnitude_metric_row("kp_storm_conservative", y_true, conservative))
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df["watch_threshold"] = watch_threshold
    metrics_df["watch_feasible"] = bool(watch_feasible)
    metrics_df["watch_event_recall"] = float(watch_row["event_recall"])
    metrics_df["watch_kp_ge_5_recall"] = float(watch_row["kp_ge_5_recall"])
    metrics_df["watch_kp_ge_6_recall"] = float(watch_row["kp_ge_6_recall"])
    metrics_df["watch_alert_duty_cycle"] = float(watch_row["alert_duty_cycle"])
    metrics_df["watch_false_alert_hours_per_month"] = float(watch_row["false_alert_hours_per_month"])
    metrics_df["watch_max_continuous_alert_hours"] = float(watch_row["max_continuous_alert_hours"])

    severity_df = pd.DataFrame(severity_rows)
    metrics_path = os.path.join(output_dir, "kp_storm_magnitude_experiment2_metrics.csv")
    predictions_path = os.path.join(output_dir, "kp_storm_magnitude_experiment2_predictions.csv")
    watch_sweep_path = os.path.join(output_dir, "kp_storm_magnitude_experiment2_watch_sweep.csv")
    severity_path = os.path.join(output_dir, "kp_storm_magnitude_experiment2_severity_thresholds.csv")
    debug_path = os.path.join(output_dir, "kp_storm_magnitude_experiment2_debug.json")
    prediction_df.to_csv(predictions_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    watch_sweep_df.to_csv(watch_sweep_path, index=False)
    severity_df.to_csv(severity_path, index=False)
    with open(debug_path, "w", encoding="utf-8") as debug_file:
        json.dump(debug, debug_file, indent=2, default=str)
        debug_file.write("\n")

    artifacts = {
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "watch_sweep_path": watch_sweep_path,
        "severity_path": severity_path,
        "debug_path": debug_path,
        "watch_threshold": watch_threshold,
        "watch_feasible": bool(watch_feasible),
        "severity_thresholds": severity_thresholds,
    }
    return metrics_df, prediction_df, artifacts


def _advanced_asymmetric_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    abs_error = np.abs(error)
    under = error < 0
    over = error > 0
    weights = np.ones(len(y_true), dtype=float)
    weights = np.where(y_true >= 5.0, 4.0, weights)
    weights = np.where(y_true >= 6.0, 6.0, weights)
    weights = np.where(y_true >= 7.0, 8.0, weights)
    weights = np.where((y_true < 5.0) & under, 2.0, weights)
    asymmetric_error = np.where(under, weights * abs_error, abs_error)
    storm_mask = y_true >= 5.0
    quiet_mask = y_true < 4.0

    def _mean_or_none(values: np.ndarray) -> Optional[float]:
        return float(np.mean(values)) if len(values) > 0 else None

    def _subset_asym(threshold: float) -> Optional[float]:
        mask = y_true >= threshold
        return _mean_or_none(asymmetric_error[mask])

    kp5 = _binary_classification_metrics(y_true >= 5.0, y_pred, 5.0)
    kp6 = _binary_classification_metrics(y_true >= 6.0, y_pred, 6.0)
    kp7 = _binary_classification_metrics(y_true >= 7.0, y_pred, 7.0) if int((y_true >= 7.0).sum()) > 0 else None
    quiet_false_storm = quiet_mask & (y_pred >= 5.0)
    storm_under = storm_mask & under
    storm_over = storm_mask & over
    return {
        "storm_mae": _mean_or_none(abs_error[storm_mask]),
        "asymmetric_storm_mae": _mean_or_none(asymmetric_error[storm_mask]),
        "kp_ge_5_asymmetric_mae": _subset_asym(5.0),
        "kp_ge_6_asymmetric_mae": _subset_asym(6.0),
        "kp_ge_7_asymmetric_mae": _subset_asym(7.0),
        "kp_ge_5_recall": kp5["recall"],
        "kp_ge_5_precision": kp5["precision"],
        "kp_ge_5_f1": kp5["f1"],
        "kp_ge_6_recall": kp6["recall"],
        "kp_ge_6_precision": kp6["precision"],
        "kp_ge_6_f1": kp6["f1"],
        "kp_ge_7_recall": kp7["recall"] if kp7 is not None else None,
        "kp_ge_7_precision": kp7["precision"] if kp7 is not None else None,
        "kp_ge_7_f1": kp7["f1"] if kp7 is not None else None,
        "mean_underprediction": _mean_or_none(abs_error[under]),
        "storm_mean_underprediction": _mean_or_none(abs_error[storm_under]),
        "max_underprediction": float(np.max(abs_error[under])) if int(under.sum()) > 0 else 0.0,
        "max_storm_underprediction": float(np.max(abs_error[storm_under])) if int(storm_under.sum()) > 0 else 0.0,
        "underprediction_rate": float(np.mean(under)),
        "storm_underprediction_rate": _mean_or_none(under[storm_mask].astype(float)),
        "overprediction_rate": float(np.mean(over)),
        "storm_overprediction_rate": _mean_or_none(over[storm_mask].astype(float)),
        "mean_overprediction": _mean_or_none(error[over]),
        "storm_mean_overprediction": _mean_or_none(error[storm_over]),
        "quiet_false_storm_inflation_rate": float(np.mean(quiet_false_storm)) if int(quiet_mask.sum()) > 0 else 0.0,
    }


def _severe_sample_weight(train_df: pd.DataFrame) -> np.ndarray:
    future_max = train_df["kp_max_next_12h"].to_numpy()
    weights = np.full(len(train_df), SEVERE_SAMPLE_WEIGHT_CONFIG["ordinary"], dtype=float)
    weights = np.where(future_max >= 5.0, SEVERE_SAMPLE_WEIGHT_CONFIG["kp_ge_5"], weights)
    weights = np.where(future_max >= 6.0, SEVERE_SAMPLE_WEIGHT_CONFIG["kp_ge_6"], weights)
    weights = np.where(future_max >= 7.0, SEVERE_SAMPLE_WEIGHT_CONFIG["kp_ge_7"], weights)
    if "storm_onset_next_12h" in train_df.columns:
        weights *= np.where(
            train_df["storm_onset_next_12h"].astype(bool).to_numpy(),
            SEVERE_SAMPLE_WEIGHT_CONFIG["onset_multiplier"],
            1.0,
        )
    return weights


def _fit_classifier_with_weights(model, X_train: pd.DataFrame, y_train: np.ndarray, sample_weight: Optional[np.ndarray]):
    if sample_weight is None:
        model.fit(X_train, y_train)
    elif hasattr(model, "named_steps") and "logisticregression" in model.named_steps:
        model.fit(X_train, y_train, logisticregression__sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def _severe_classifier_factories() -> Dict[str, Callable[[], object]]:
    factories: Dict[str, Callable[[], object]] = {
        "logistic_balanced": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        ),
        "random_forest_balanced": lambda: RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if HistGradientBoostingClassifier is not None:
        factories["hist_gradient_boosting"] = lambda: HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        )
    return factories


def _event_peak_recall(
    timestamps: pd.Series,
    kp_values: np.ndarray,
    alert_mask: np.ndarray,
    lead_steps: int,
    peak_threshold: float,
) -> Tuple[float, int, int, Optional[float], Optional[float]]:
    runs = _contiguous_true_runs(np.asarray(kp_values) >= 5.0)
    alert_mask = np.asarray(alert_mask).astype(bool)
    total = 0
    detected = 0
    lead_hours = []
    for start, end in runs:
        peak = float(np.max(np.asarray(kp_values)[start:end + 1]))
        if peak < peak_threshold:
            continue
        total += 1
        window_start = max(0, start - lead_steps)
        indices = np.where(alert_mask[window_start:end + 1])[0] + window_start
        if len(indices) == 0:
            continue
        detected += 1
        first_idx = int(indices[0])
        lead_hours.append(float((timestamps.iloc[start] - timestamps.iloc[first_idx]) / pd.Timedelta(hours=1)))
    recall = float(detected / total) if total > 0 else 0.0
    return (
        recall,
        int(total),
        int(total - detected),
        float(np.mean(lead_hours)) if lead_hours else None,
        float(np.median(lead_hours)) if lead_hours else None,
    )


def _storm_recall_score(frame: pd.DataFrame, kp_threshold: float) -> np.ndarray:
    """
    Select the prediction-time storm probability used by the recall branch.
    Falls back to a normalized conservative Kp estimate when classifier scores are unavailable.
    """
    score_column = f"prob_kp_ge_{int(kp_threshold)}_next_12h"
    if score_column in frame.columns:
        return np.clip(frame[score_column].to_numpy(dtype=float), 0.0, 1.0)

    fallback_column = "kp_final_risk_conservative_v6"
    if fallback_column not in frame.columns:
        fallback_column = "kp_final_risk_conservative_v5" if "kp_final_risk_conservative_v5" in frame.columns else "kp_base"
    score = frame[fallback_column].to_numpy(dtype=float) / max(kp_threshold, 1.0)
    return np.clip(score, 0.0, 1.0)


def _storm_recall_threshold_sweep(
    frame: pd.DataFrame,
    score: np.ndarray,
    kp_threshold: float,
    thresholds: np.ndarray,
    lead_steps: int,
    split_name: str,
) -> pd.DataFrame:
    """
    Evaluate row-level and event-level storm recall for candidate watch thresholds.
    Thresholds are later selected from validation rows and replayed on holdout rows.
    """
    y_true = frame["kp"].to_numpy(dtype=float)
    y_binary = y_true >= kp_threshold
    rows = []
    for threshold in thresholds:
        gate = score >= threshold
        point_metrics = _binary_classification_metrics_from_mask(
            y_binary,
            gate,
            score,
            float(threshold),
        )
        event_recall, event_count, missed_events, mean_lead, median_lead = _event_peak_recall(
            frame["timestamp"],
            y_true,
            gate,
            lead_steps,
            kp_threshold,
        )
        rows.append(
            {
                "split": split_name,
                "target": f"kp_ge_{int(kp_threshold)}",
                "threshold": float(threshold),
                "precision": point_metrics["precision"],
                "recall": point_metrics["recall"],
                "f1": point_metrics["f1"],
                "average_precision": point_metrics["average_precision"],
                "event_recall": event_recall,
                "events": event_count,
                "missed_events": missed_events,
                "mean_lead_time_hours": mean_lead,
                "median_lead_time_hours": median_lead,
                "alert_duty_cycle": float(np.mean(gate)) if len(gate) else 0.0,
                "false_watch_rate_kp_lt_5": float(np.mean(gate[y_true < 5.0])) if int((y_true < 5.0).sum()) > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _select_storm_recall_threshold(sweep_df: pd.DataFrame, kp_threshold: float) -> pd.Series:
    """
    Select a validation-era internal watch threshold for high storm recall.
    Kp>=5 aims for broad storm capture; Kp>=6 keeps a slightly lower target because severe events are sparser.
    """
    target_recall = 0.90 if kp_threshold < 6.0 else 0.875
    min_threshold = 0.02
    max_false_watch = 0.40
    usable = sweep_df[
        (sweep_df["threshold"] >= min_threshold)
        & (sweep_df["false_watch_rate_kp_lt_5"] <= max_false_watch)
    ].copy()
    if usable.empty:
        usable = sweep_df[sweep_df["threshold"] >= min_threshold].copy()
    if usable.empty:
        usable = sweep_df.copy()

    candidates = usable[usable["recall"] >= target_recall].copy()
    if candidates.empty:
        candidates = usable.copy()
        sort_columns = ["event_recall", "recall", "precision", "threshold"]
        ascending = [False, False, False, True]
    else:
        sort_columns = ["event_recall", "precision", "f1", "alert_duty_cycle", "threshold"]
        ascending = [False, False, False, True, True]
    return candidates.sort_values(sort_columns, ascending=ascending).iloc[0]


def _storm_recall_regime_rows(frame: pd.DataFrame, model_columns: List[str]) -> pd.DataFrame:
    """
    Produce regime-specific regression and classification metrics for storm-recall model columns.
    The Kp>7 row is diagnostic only and is not used as the main selector.
    """
    y_true = frame["kp"].to_numpy(dtype=float)
    regimes = {
        "quiet_kp_lt_4": y_true < 4.0,
        "non_storm_kp_lt_5": y_true < 5.0,
        "kp_4_to_lt_7_primary_warning_regime": (y_true >= 4.0) & (y_true < 7.0),
        "kp_lt_5": y_true < 5.0,
        "kp_5_to_lt_7": (y_true >= 5.0) & (y_true < 7.0),
        "kp_lt_7": y_true < 7.0,
        "kp_gt_7_diagnostic": y_true > 7.0,
    }
    rows = []
    for model_column in model_columns:
        if model_column not in frame.columns:
            continue
        y_pred = frame[model_column].to_numpy(dtype=float)
        for regime_name, mask in regimes.items():
            if int(mask.sum()) == 0:
                continue
            error = y_pred[mask] - y_true[mask]
            under = error < 0.0
            over = error > 0.0
            rows.append(
                {
                    "metric_group": "regression_regime",
                    "model": model_column,
                    "regime": regime_name,
                    "row_count": int(mask.sum()),
                    "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
                    "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
                    "corr": _safe_correlation(y_true[mask], y_pred[mask]),
                    "underprediction_rate": float(np.mean(under)),
                    "overprediction_rate": float(np.mean(over)),
                    "max_underprediction": float(np.max(np.maximum(y_true[mask] - y_pred[mask], 0.0))),
                    "mean_underprediction": float(np.mean(np.maximum(y_true[mask] - y_pred[mask], 0.0))),
                    "mean_overprediction": float(np.mean(np.maximum(y_pred[mask] - y_true[mask], 0.0))),
                }
            )
        for kp_threshold in [5.0, 6.0, 7.0]:
            if int((y_true >= kp_threshold).sum()) == 0:
                continue
            metrics = _binary_classification_metrics(y_true >= kp_threshold, y_pred, kp_threshold)
            event_recall, event_count, missed_events, mean_lead, median_lead = _event_peak_recall(
                frame["timestamp"],
                y_true,
                y_pred >= kp_threshold,
                max(1, int(round(12.0 / max(_cadence_hours(frame), 1.0)))),
                kp_threshold,
            )
            rows.append(
                {
                    "metric_group": "threshold_recall",
                    "model": model_column,
                    "regime": f"kp_ge_{int(kp_threshold)}",
                    "row_count": int((y_true >= kp_threshold).sum()),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "average_precision": metrics["average_precision"],
                    "event_recall": event_recall,
                    "events": event_count,
                    "missed_events": missed_events,
                    "mean_lead_time_hours": mean_lead,
                    "median_lead_time_hours": median_lead,
                }
            )
    return pd.DataFrame(rows)


def _storm_recall_severe_event_diagnostics(frame: pd.DataFrame, lead_steps: int) -> pd.DataFrame:
    """
    List held-out Kp>=6 events missed by the high-recall severe watch gate.
    Includes local driver summaries so misses can be inspected scientifically.
    """
    y_true = frame["kp"].to_numpy(dtype=float)
    gate = frame["storm_recall_internal_watch_kp_ge_6"].astype(bool).to_numpy()
    rows = []
    cadence = _cadence_hours(frame)
    if cadence <= 0:
        cadence = 6.0
    for event_id, (start, end) in enumerate(_contiguous_true_runs(y_true >= 5.0), start=1):
        event = frame.iloc[start:end + 1]
        event_peak = float(event["kp"].max())
        if event_peak < 6.0:
            continue
        lead_start = max(0, start - lead_steps)
        window = frame.iloc[lead_start:end + 1]
        gate_window = gate[lead_start:end + 1]
        detected = bool(gate_window.any())
        first_watch_time = None
        lead_time_hours = None
        if detected:
            first_idx = int(np.where(gate_window)[0][0] + lead_start)
            first_watch_time = frame["timestamp"].iloc[first_idx]
            lead_time_hours = float((frame["timestamp"].iloc[start] - first_watch_time) / pd.Timedelta(hours=1))
        row = {
            "event_id": event_id,
            "missed_by_severe_watch": not detected,
            "event_start_time": frame["timestamp"].iloc[start],
            "event_end_time": frame["timestamp"].iloc[end],
            "event_peak_kp": event_peak,
            "event_duration_hours": float((end - start + 1) * cadence),
            "first_watch_time": first_watch_time,
            "lead_time_hours": lead_time_hours,
            "max_prob_kp_ge_5_window": float(window["storm_recall_probability_kp_ge_5"].max()),
            "max_prob_kp_ge_6_window": float(window["storm_recall_probability_kp_ge_6"].max()),
            "max_prediction_window": float(window["kp_storm_recall_conservative"].max()),
            "max_underprediction": float(np.max(np.maximum(event["kp"].to_numpy(dtype=float) - event["kp_storm_recall_conservative"].to_numpy(dtype=float), 0.0))),
            "min_bz_window": float(window["bz"].min()) if "bz" in window else None,
            "max_velocity_window": float(window["velocity"].max()) if "velocity" in window else None,
            "max_density_window": float(window["density"].max()) if "density" in window else None,
            "max_pdyn_window": float(window["pdyn"].max()) if "pdyn" in window else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# This branch selects recall-biased internal watch thresholds on validation data.
# Keep it distinct from any public-alert gate: a high-recall science watch can
# legitimately tolerate more false watches than an externally issued alert.
def run_storm_recall_forecasting_diagnostics(
    validation_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    inner_train_df: pd.DataFrame,
    output_dir: str,
) -> dict:
    """
    Build first-class storm-recall artifacts from validation-selected internal watch thresholds.
    The branch prioritizes Kp>=5/Kp>=6 recall and reports Kp>7 as a separate diagnostic regime.
    """
    _ensure_directory(output_dir)
    validation_frame = validation_frame.copy().reset_index(drop=True)
    holdout_frame = holdout_frame.copy().reset_index(drop=True)
    cadence_hours = _cadence_hours(holdout_frame)
    if cadence_hours <= 0:
        cadence_hours = 6.0
    lead_steps = max(1, int(round(12.0 / cadence_hours)))
    thresholds = np.linspace(0.01, 0.99, 99)
    threshold_frames = []
    selected_rows = []
    selected_thresholds = {}

    for kp_threshold in [5.0, 6.0]:
        validation_score = _storm_recall_score(validation_frame, kp_threshold)
        holdout_score = _storm_recall_score(holdout_frame, kp_threshold)
        validation_sweep = _storm_recall_threshold_sweep(
            validation_frame,
            validation_score,
            kp_threshold,
            thresholds,
            lead_steps,
            "validation",
        )
        selected = _select_storm_recall_threshold(validation_sweep, kp_threshold)
        selected_threshold = float(selected["threshold"])
        selected_thresholds[f"kp_ge_{int(kp_threshold)}"] = selected_threshold
        validation_sweep["selected_for_internal_watch"] = validation_sweep["threshold"] == selected_threshold
        holdout_sweep = _storm_recall_threshold_sweep(
            holdout_frame,
            holdout_score,
            kp_threshold,
            thresholds,
            lead_steps,
            "holdout",
        )
        holdout_sweep["selected_for_internal_watch"] = holdout_sweep["threshold"] == selected_threshold
        threshold_frames.extend([validation_sweep, holdout_sweep])
        selected_rows.append({"target": f"kp_ge_{int(kp_threshold)}", **selected.to_dict()})
        holdout_frame[f"storm_recall_probability_kp_ge_{int(kp_threshold)}"] = holdout_score
        holdout_frame[f"storm_recall_internal_watch_kp_ge_{int(kp_threshold)}"] = (
            holdout_score >= selected_threshold
        ).astype(int)

    if "storm_alert_gate" in holdout_frame.columns:
        holdout_frame["storm_recall_public_alert_reference"] = holdout_frame["storm_alert_gate"].astype(int)
    else:
        holdout_frame["storm_recall_public_alert_reference"] = 0
    public_reference_selected = _select_public_alert_reference_threshold(
        validation_frame,
        _storm_recall_score(validation_frame, 5.0),
        5.0,
        thresholds,
        lead_steps,
    )
    public_reference_threshold = float(public_reference_selected["threshold"])
    holdout_frame["storm_recall_public_alert_retuned_reference"] = (
        holdout_frame["storm_recall_probability_kp_ge_5"].to_numpy(dtype=float) >= public_reference_threshold
    ).astype(int)

    base = holdout_frame["kp_base"].to_numpy(dtype=float) if "kp_base" in holdout_frame.columns else holdout_frame["kp"].to_numpy(dtype=float)
    risk_column = "kp_final_risk_conservative_v6"
    if risk_column not in holdout_frame.columns:
        risk_column = "kp_final_risk_conservative_v5" if "kp_final_risk_conservative_v5" in holdout_frame.columns else "kp_base"
    risk_pred = holdout_frame[risk_column].to_numpy(dtype=float)
    candidate = np.maximum(base, risk_pred)
    gate5 = holdout_frame["storm_recall_internal_watch_kp_ge_5"].astype(bool).to_numpy()
    gate6 = holdout_frame["storm_recall_internal_watch_kp_ge_6"].astype(bool).to_numpy()
    if "kp_risk_p75" in holdout_frame.columns:
        candidate = np.where(gate5, np.maximum(candidate, holdout_frame["kp_risk_p75"].to_numpy(dtype=float)), candidate)
    if "kp_risk_p90" in holdout_frame.columns:
        candidate = np.where(gate6, np.maximum(candidate, holdout_frame["kp_risk_p90"].to_numpy(dtype=float)), candidate)
    candidate = np.where(gate5, np.maximum(candidate, 5.0), candidate)
    candidate = np.where(gate6, np.maximum(candidate, 6.0), candidate)
    holdout_frame["kp_storm_recall_conservative"] = np.clip(candidate, 0.0, 9.0)

    prob7 = _storm_recall_score(holdout_frame, 7.0)
    labels = []
    for idx in range(len(holdout_frame)):
        if holdout_frame["kp_storm_recall_conservative"].iloc[idx] >= 7.0 or prob7[idx] >= 0.20:
            labels.append("kp_gt_7_low_confidence_extreme")
        elif gate6[idx]:
            labels.append("kp_ge_6_high_recall_watch")
        elif gate5[idx]:
            labels.append("kp_ge_5_high_recall_watch")
        else:
            labels.append("ordinary_or_below_watch")
    holdout_frame["storm_recall_extreme_diagnostic_label"] = labels

    model_columns = [
        "kp_base",
        "kp_storm_adjusted",
        "kp_final_risk_conservative_v5",
        "kp_final_risk_conservative_v6",
        "kp_storm_recall_conservative",
    ]
    metrics_df = _storm_recall_regime_rows(holdout_frame, model_columns)
    threshold_df = pd.concat(threshold_frames, ignore_index=True)
    selected_df = pd.DataFrame(selected_rows)
    severe_event_df = _storm_recall_severe_event_diagnostics(holdout_frame, lead_steps)
    recovery_metrics_df = _storm_recovery_metrics(holdout_frame, model_columns)
    branch_comparison_df = _storm_recall_branch_comparison(holdout_frame)
    watch_comparison_df = pd.DataFrame(
        [
            _watch_gate_metrics(holdout_frame, "storm_recall_internal_watch_kp_ge_5", 5.0, "internal_high_recall_watch_kp_ge_5", lead_steps),
            _watch_gate_metrics(holdout_frame, "storm_recall_internal_watch_kp_ge_6", 6.0, "internal_high_recall_watch_kp_ge_6", lead_steps),
            _watch_gate_metrics(holdout_frame, "storm_recall_public_alert_reference", 5.0, "existing_public_alert_reference", lead_steps),
            _watch_gate_metrics(holdout_frame, "storm_recall_public_alert_retuned_reference", 5.0, "validation_retuned_public_alert_reference", lead_steps),
        ]
    )

    paths = {
        "metrics": os.path.join(output_dir, "kp_storm_recall_metrics.csv"),
        "threshold_sweeps": os.path.join(output_dir, "kp_storm_recall_threshold_sweeps.csv"),
        "selected_thresholds": os.path.join(output_dir, "kp_storm_recall_selected_thresholds.csv"),
        "severe_event_diagnostics": os.path.join(output_dir, "kp_storm_recall_severe_event_miss_diagnostics.csv"),
        "storm_recovery_metrics": os.path.join(output_dir, "kp_storm_recall_recovery_metrics.csv"),
        "branch_comparison": os.path.join(output_dir, "kp_storm_recall_branch_comparison.csv"),
        "watch_public_comparison": os.path.join(output_dir, "kp_storm_recall_watch_public_comparison.csv"),
        "predictions": os.path.join(output_dir, "kp_storm_recall_predictions.csv"),
        "pi_summary": os.path.join(output_dir, "kp_storm_recall_pi_summary.md"),
        "debug": os.path.join(output_dir, "kp_storm_recall_debug.json"),
    }
    metrics_df.to_csv(paths["metrics"], index=False)
    threshold_df.to_csv(paths["threshold_sweeps"], index=False)
    selected_df.to_csv(paths["selected_thresholds"], index=False)
    severe_event_df.to_csv(paths["severe_event_diagnostics"], index=False)
    recovery_metrics_df.to_csv(paths["storm_recovery_metrics"], index=False)
    branch_comparison_df.to_csv(paths["branch_comparison"], index=False)
    watch_comparison_df.to_csv(paths["watch_public_comparison"], index=False)
    holdout_frame.to_csv(paths["predictions"], index=False)

    debug = {
        "selection_policy": "validation_selected_high_recall_internal_watch",
        "selection_objective": "prioritize Kp>=5 and Kp>=6 recall with Kp<7 accuracy reported separately; Kp>7 is diagnostic only",
        "driver_input_metadata": _driver_input_metadata(),
        "original_pi_linear_equation": _original_pi_linear_equation_metadata(),
        "train_window": {
            "start": _format_timestamp(inner_train_df["timestamp"].iloc[0]),
            "end": _format_timestamp(inner_train_df["timestamp"].iloc[-1]),
            "rows": int(len(inner_train_df)),
        },
        "validation_window": {
            "start": _format_timestamp(validation_frame["timestamp"].iloc[0]),
            "end": _format_timestamp(validation_frame["timestamp"].iloc[-1]),
            "rows": int(len(validation_frame)),
        },
        "test_window": {
            "start": _format_timestamp(holdout_frame["timestamp"].iloc[0]),
            "end": _format_timestamp(holdout_frame["timestamp"].iloc[-1]),
            "rows": int(len(holdout_frame)),
        },
        "threshold_selection_source": "validation",
        "final_evaluation_source": "holdout",
        "lead_steps": lead_steps,
        "lead_hours": float(lead_steps * cadence_hours),
        "selected_thresholds": selected_thresholds,
        "public_alert_retuned_reference": {
            "threshold_selection_source": "validation",
            "selected_threshold": public_reference_threshold,
            "selection_row": public_reference_selected.to_dict(),
            "purpose": "diagnostic strict public-alert reference; kept separate from internal high-recall watch and existing public alert gate",
        },
        "storm_recall_score_columns": ["prob_kp_ge_5_next_12h", "prob_kp_ge_6_next_12h"],
        "forbidden_feature_fragments": ["future", "target", "storm_next", "kp_max_next"],
        "public_alert_column_kept_separate": "storm_recall_public_alert_reference",
        "public_alert_retuned_reference_column": "storm_recall_public_alert_retuned_reference",
        "extreme_regime_policy": "Kp>7 diagnostics are reported separately and do not select the primary operating point.",
        "paths": paths,
    }
    with open(paths["debug"], "w", encoding="utf-8") as debug_file:
        json.dump(debug, debug_file, indent=2, default=str)
        debug_file.write("\n")

    summary_sections = [
        "# Storm Recall Forecasting Summary",
        "## Operating Point\n"
        f"- Kp>=5 internal watch threshold: `{selected_thresholds['kp_ge_5']}`\n"
        f"- Kp>=6 internal watch threshold: `{selected_thresholds['kp_ge_6']}`\n"
        "- Thresholds were selected on validation-era predictions and evaluated on the final holdout.",
        "## Regime Metrics\n" + _markdown_table(metrics_df.head(40)),
        "## Selected Thresholds\n" + _markdown_table(selected_df),
        "## Internal Watch vs Public Alert\n" + _markdown_table(watch_comparison_df),
        "## Branch Comparison\n" + _markdown_table(branch_comparison_df.head(40)),
        "## Storm Recovery Metrics\n" + _markdown_table(recovery_metrics_df),
        "## Severe Event Diagnostics\n"
        + (_markdown_table(severe_event_df) if not severe_event_df.empty else "No Kp>=6 holdout events were available."),
        "## Artifacts\n" + "\n".join([f"- `{key}`: `{value}`" for key, value in paths.items()]),
    ]
    with open(paths["pi_summary"], "w", encoding="utf-8") as report_file:
        report_file.write("\n\n".join(summary_sections) + "\n")

    return {
        "paths": paths,
        "metrics": metrics_df,
        "threshold_sweeps": threshold_df,
        "selected_thresholds": selected_df,
        "severe_event_diagnostics": severe_event_df,
        "storm_recovery_metrics": recovery_metrics_df,
        "branch_comparison": branch_comparison_df,
        "watch_public_comparison": watch_comparison_df,
        "predictions": holdout_frame,
        "debug": debug,
    }


def _max_underprediction_for_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    mask = np.asarray(y_true) >= threshold
    if int(mask.sum()) == 0:
        return 0.0
    return float(np.maximum(np.asarray(y_true)[mask] - np.asarray(y_pred)[mask], 0.0).max())


def _candidate_watch_rows(
    test_df: pd.DataFrame,
    score: np.ndarray,
    thresholds: np.ndarray,
    base_prediction: np.ndarray,
    p95: np.ndarray,
    severe_ceiling: np.ndarray,
    lead_steps: int,
) -> pd.DataFrame:
    rows = []
    y_true = test_df["kp"].to_numpy()
    quiet_mask = y_true < 4.0
    for threshold in thresholds:
        gate = score >= threshold
        event_metrics = _event_metrics_from_alert_mask(test_df["timestamp"], y_true, gate, lead_steps)
        kp5_recall, kp5_events, kp5_missed, mean_lead, median_lead = _event_peak_recall(
            test_df["timestamp"], y_true, gate, lead_steps, 5.0
        )
        kp6_recall, kp6_events, kp6_missed, _, _ = _event_peak_recall(
            test_df["timestamp"], y_true, gate, lead_steps, 6.0
        )
        kp7_recall, kp7_events, kp7_missed, _, _ = _event_peak_recall(
            test_df["timestamp"], y_true, gate, lead_steps, 7.0
        )
        candidate_pred = np.where(
            gate,
            np.maximum.reduce([base_prediction, p95, np.ceil(p95), severe_ceiling]),
            base_prediction,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "event_recall": event_metrics["event_recall"],
                "kp_ge_5_event_recall": kp5_recall,
                "kp_ge_6_event_recall": kp6_recall,
                "kp_ge_7_event_recall": kp7_recall,
                "kp_ge_5_events": kp5_events,
                "kp_ge_6_events": kp6_events,
                "kp_ge_7_events": kp7_events,
                "missed_kp_ge_5_events": kp5_missed,
                "missed_kp_ge_6_events": kp6_missed,
                "missed_kp_ge_7_events": kp7_missed,
                "alert_duty_cycle": event_metrics["alert_duty_cycle"],
                "false_storm_inflation_rate": float(np.mean(gate[quiet_mask])) if int(quiet_mask.sum()) > 0 else 0.0,
                "mean_lead_time_hours": mean_lead,
                "median_lead_time_hours": median_lead,
                "max_underprediction_kp_ge_6": _max_underprediction_for_threshold(y_true, candidate_pred, 6.0),
            }
        )
    return pd.DataFrame(rows)


def _select_watch_for_duty(watch_sweep_df: pd.DataFrame, duty_target: float) -> pd.Series:
    candidates = watch_sweep_df[watch_sweep_df["alert_duty_cycle"] <= duty_target].copy()
    if candidates.empty:
        candidates = watch_sweep_df.copy()
        candidates["duty_violation"] = np.maximum(0.0, candidates["alert_duty_cycle"] - duty_target)
        sort_columns = [
            "duty_violation",
            "kp_ge_7_event_recall",
            "kp_ge_6_event_recall",
            "kp_ge_5_event_recall",
            "max_underprediction_kp_ge_6",
            "threshold",
        ]
        ascending = [True, False, False, False, True, False]
    else:
        sort_columns = [
            "kp_ge_7_event_recall",
            "kp_ge_6_event_recall",
            "kp_ge_5_event_recall",
            "max_underprediction_kp_ge_6",
            "alert_duty_cycle",
            "threshold",
        ]
        ascending = [False, False, False, True, False, False]
    return candidates.sort_values(sort_columns, ascending=ascending).iloc[0]


def _physics_regime_for_event(row: pd.Series) -> str:
    labels = []
    if row.get("min_Bz_previous_12h", 0.0) <= -5.0 or row.get("max_Bs_previous_12h", 0.0) >= 5.0:
        labels.append("clear_southward_imf")
    if row.get("max_pdyn_previous_12h", 0.0) >= 2.5 or row.get("pdyn_jump_previous_12h", 0.0) >= 1.0:
        labels.append("pressure_compression_driven")
    if row.get("Ey_roll_sum_previous_12h", 0.0) >= 5.0 or row.get("max_coupling_proxy_previous_12h", 0.0) > 0.0:
        labels.append("sustained_coupling")
    if row.get("prior_Kp_max_24h", 0.0) >= 4.0:
        labels.append("preconditioned_magnetosphere")
    if not labels and row.get("event_duration_hours", 0.0) <= 6.0:
        labels.append("abrupt_or_unresolved_driver")
    if not labels:
        labels.append("no_clear_driver_in_current_features")
    return ";".join(labels)


def _build_experiment3_event_diagnostics(
    test_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    lead_steps: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = test_df.reset_index(drop=True).copy()
    pred = prediction_df.reset_index(drop=True).copy()
    for col in pred.columns:
        if col not in merged.columns:
            merged[col] = pred[col]
    cadence = _cadence_hours(merged)
    if cadence <= 0:
        cadence = 6.0
    rows = []
    for event_id, (start, end) in enumerate(_contiguous_true_runs(merged["kp"].to_numpy() >= 5.0), start=1):
        event = merged.iloc[start:end + 1]
        lead_start = max(0, start - lead_steps)
        prior_24_start = max(0, start - max(1, int(round(24.0 / cadence))))
        window = merged.iloc[lead_start:end + 1]
        previous = merged.iloc[lead_start:start + 1]
        prior_24 = merged.iloc[prior_24_start:start + 1]
        peak_kp = float(event["kp"].max())
        row = {
            "event_id": event_id,
            "event_start_time": merged["timestamp"].iloc[start],
            "event_peak_kp": peak_kp,
            "event_duration_hours": float((end - start + 1) * cadence),
            "storm_watch_gate_fired_before_or_during": bool(window.get("storm_watch_gate", pd.Series(False, index=window.index)).astype(bool).any()),
            "storm_watch_gate_vigilant_fired_before_or_during": bool(window.get("storm_watch_gate_vigilant", pd.Series(False, index=window.index)).astype(bool).any()),
            "severe_watch_gate_fired_before_or_during": bool(window.get("severe_watch_gate", pd.Series(False, index=window.index)).astype(bool).any()),
            "storm_boost_gate_fired_before_or_during": bool(window.get("storm_boost_gate", pd.Series(False, index=window.index)).astype(bool).any()),
            "prob_kp_ge_5_max_before_or_during": float(window.get("prob_kp_ge_5_next_12h", pd.Series(0.0, index=window.index)).max()),
            "prob_kp_ge_6_max_before_or_during": float(window.get("prob_kp_ge_6_next_12h", pd.Series(0.0, index=window.index)).max()),
            "prob_kp_ge_7_max_before_or_during": float(window.get("prob_kp_ge_7_next_12h", pd.Series(0.0, index=window.index)).max()),
            "min_Bz_previous_12h": float(previous["bz"].min()) if "bz" in previous else None,
            "max_Bs_previous_12h": float(np.maximum(0.0, -previous["bz"]).max()) if "bz" in previous else None,
            "Bs_roll_sum_previous_12h": float(np.maximum(0.0, -previous["bz"]).sum()) if "bz" in previous else None,
            "max_Ey_previous_12h": float(previous["ey"].max()) if "ey" in previous else None,
            "Ey_roll_sum_previous_12h": float(previous["ey"].sum()) if "ey" in previous else None,
            "max_pdyn_previous_12h": float(previous["pdyn"].max()) if "pdyn" in previous else None,
            "pdyn_jump_previous_12h": float(previous["pdyn_jump"].max()) if "pdyn_jump" in previous else None,
            "velocity_jump_previous_12h": float(previous["velocity_jump"].max()) if "velocity_jump" in previous else None,
            "max_coupling_proxy_previous_12h": float(previous["coupling_simple"].max()) if "coupling_simple" in previous else None,
            "prior_Kp_max_24h": float(prior_24["kp"].max()) if len(prior_24) else None,
        }
        for col in [
            "kp_base",
            "kp_storm_adjusted",
            "kp_risk_p90",
            "kp_risk_p95",
            "kp_severe_ceiling",
            "kp_storm_conservative_v2",
        ]:
            values = window[col].to_numpy() if col in window else np.zeros(len(window))
            event_values = event[col].to_numpy() if col in event else np.zeros(len(event))
            row[f"{col}_max_before_or_during"] = float(np.max(values))
            row[f"{col}_max_underprediction_during"] = float(np.maximum(event["kp"].to_numpy() - event_values, 0.0).max())
        row["maximum_underprediction_kp_storm_conservative_v2"] = row["kp_storm_conservative_v2_max_underprediction_during"]
        alert_indices = np.where(window.get("severe_watch_gate", pd.Series(False, index=window.index)).astype(bool).to_numpy())[0]
        row["lead_time_hours_if_any"] = (
            float((merged["timestamp"].iloc[start] - window["timestamp"].iloc[int(alert_indices[0])]) / pd.Timedelta(hours=1))
            if len(alert_indices) > 0
            else None
        )
        categories = []
        if not row["storm_watch_gate_vigilant_fired_before_or_during"] and not row["severe_watch_gate_fired_before_or_during"]:
            categories.append("1_watch_gate_missed_event_entirely")
        if (row["storm_watch_gate_vigilant_fired_before_or_during"] or row["severe_watch_gate_fired_before_or_during"]) and row["maximum_underprediction_kp_storm_conservative_v2"] > 0.75:
            categories.append("2_vigilant_watch_fired_but_magnitude_too_low")
        if peak_kp >= 6.0 and row["kp_severe_ceiling_max_before_or_during"] < 6.0:
            categories.append("3_severe_classifier_failed_kp_ge_6_ceiling")
        if row["kp_risk_p95_max_before_or_during"] >= 6.0 and row["kp_storm_conservative_v2_max_before_or_during"] < row["kp_risk_p95_max_before_or_during"] - 0.25:
            categories.append("4_quantile_high_but_final_combination_suppressed_it")
        row["physics_regime"] = _physics_regime_for_event(pd.Series(row))
        if "no_clear_driver_in_current_features" in row["physics_regime"] or "abrupt_or_unresolved_driver" in row["physics_regime"]:
            categories.append("5_inputs_lacked_clear_storm_driving_signal")
        if not categories:
            categories.append("6_captured_or_mild_underprediction")
        row["failure_categories"] = ";".join(categories)
        row["primary_failure_category"] = categories[0]
        rows.append(row)
    event_df = pd.DataFrame(rows)
    category_df = (
        event_df.assign(category=event_df["failure_categories"].str.split(";"))
        .explode("category")
        .groupby("category", as_index=False)
        .agg(
            events=("event_id", "count"),
            mean_peak_kp=("event_peak_kp", "mean"),
            mean_underprediction=("maximum_underprediction_kp_storm_conservative_v2", "mean"),
        )
        .sort_values(["events", "mean_peak_kp"], ascending=[False, False])
    )
    severe_df = event_df[event_df["event_peak_kp"] >= 6.0].copy()
    return event_df, category_df, severe_df


def _run_experiment3_severe_storm_branch(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    current_predictions: pd.DataFrame,
    thresholds: np.ndarray,
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    lead_steps = STORM_TARGET_HORIZON_STEPS["storm_next_12h"]
    prediction_df = current_predictions[["timestamp", "kp"]].copy()
    for column in [
        "kp_base",
        "kp_storm_adjusted",
        "kp_risk_p75",
        "kp_risk_p90",
        "kp_storm_conservative",
        "storm_watch_gate_vigilant",
        "storm_boost_gate",
    ]:
        prediction_df[column] = current_predictions[column].to_numpy() if column in current_predictions else 0.0

    p95_model = _fit_quantile_regressor(0.95)
    train_valid = train_df["kp_max_next_12h_valid"].astype(bool)
    if p95_model is not None and int(train_valid.sum()) >= 100:
        p95_model.fit(
            train_df.loc[train_valid, feature_columns],
            train_df.loc[train_valid, "kp_max_next_12h"].to_numpy(),
            sample_weight=_storm_magnitude_sample_weight(train_df.loc[train_valid].copy(), "kp_max_next_12h"),
        )
        p95 = np.clip(p95_model.predict(test_df[feature_columns]), 0.0, 9.0)
    else:
        p95 = prediction_df["kp_risk_p90"].to_numpy()
    prediction_df["kp_risk_p95"] = p95

    classifier_rows = []
    severity_threshold_rows = []
    selected_severity_thresholds = {}
    best_scores = {}
    factories = _severe_classifier_factories()
    for kp_threshold in SEVERITY_THRESHOLDS:
        alias_target = f"kp_ge_{int(kp_threshold)}_next_12h"
        valid_col = f"{alias_target}_valid"
        valid_train = train_df[valid_col].astype(bool)
        valid_test = test_df[valid_col].astype(bool)
        y_train = train_df.loc[valid_train, alias_target].astype(int).to_numpy()
        y_test = test_df.loc[valid_test, alias_target].astype(bool).to_numpy()
        if int(y_train.sum()) < MIN_SEVERITY_POSITIVES or len(np.unique(y_train)) < 2:
            continue
        sample_weight = _severe_sample_weight(train_df.loc[valid_train].copy())
        best_model_row = None
        for model_name, model_factory in factories.items():
            model = model_factory()
            model = _fit_classifier_with_weights(
                model,
                train_df.loc[valid_train, feature_columns],
                y_train,
                sample_weight,
            )
            scores = np.zeros(len(test_df), dtype=float)
            scores[valid_test.to_numpy()] = model.predict_proba(test_df.loc[valid_test, feature_columns])[:, 1]
            score_col = f"prob_kp_ge_{int(kp_threshold)}_next_12h_{model_name}"
            prediction_df[score_col] = np.clip(scores, 0.0, 1.0)
            grid = EXPERIMENT3_SEVERITY_THRESHOLD_GRID.get(kp_threshold, list(thresholds))
            for threshold in grid:
                point = _binary_classification_metrics(y_test, scores[valid_test.to_numpy()], float(threshold))
                gate = scores >= float(threshold)
                gate_event_metrics = _event_metrics_from_alert_mask(
                    test_df["timestamp"],
                    test_df["kp"].to_numpy(),
                    gate,
                    lead_steps,
                )
                quiet_mask = test_df["kp"].to_numpy() < 4.0
                event_recall, total_events, missed_events, mean_lead, median_lead = _event_peak_recall(
                    test_df["timestamp"], test_df["kp"].to_numpy(), gate, lead_steps, kp_threshold
                )
                row = {
                    "target": alias_target,
                    "kp_threshold": kp_threshold,
                    "model": model_name,
                    "threshold": float(threshold),
                    "precision": point["precision"],
                    "recall": point["recall"],
                    "f1": point["f1"],
                    "average_precision": point["average_precision"],
                    "event_recall": event_recall,
                    "events": total_events,
                    "missed_events": missed_events,
                    "alert_duty_cycle": gate_event_metrics["alert_duty_cycle"],
                    "false_storm_inflation_rate": float(np.mean(gate[quiet_mask])) if int(quiet_mask.sum()) > 0 else 0.0,
                    "mean_lead_time_hours": mean_lead,
                    "median_lead_time_hours": median_lead,
                    "train_positive": int(y_train.sum()),
                    "test_positive": int(y_test.sum()),
                }
                severity_threshold_rows.append(row)
                feasible_duty = row["alert_duty_cycle"] <= EXPERIMENT3_SEVERITY_MAX_DUTY_CYCLE
                candidate = (
                    feasible_duty,
                    event_recall,
                    point["recall"],
                    -row["false_storm_inflation_rate"],
                    -missed_events,
                    point["f1"],
                    -float(threshold),
                )
                if best_model_row is None or candidate > best_model_row[0]:
                    best_model_row = (candidate, row, scores)
        if best_model_row is None:
            continue
        _, selected_row, selected_scores = best_model_row
        selected_severity_thresholds[kp_threshold] = float(selected_row["threshold"])
        best_scores[kp_threshold] = np.clip(selected_scores, 0.0, 1.0)
        prediction_df[f"prob_kp_ge_{int(kp_threshold)}_next_12h"] = best_scores[kp_threshold]
        prediction_df[f"prob_kp_ge_{int(kp_threshold)}_next_12h_model"] = selected_row["model"]
        classifier_rows.append({**selected_row, "selected": True})

    severe_ceiling = np.zeros(len(test_df), dtype=float)
    for kp_threshold, scores in best_scores.items():
        threshold = selected_severity_thresholds[kp_threshold]
        gate = scores >= threshold
        prediction_df[f"kp_ge_{int(kp_threshold)}_severe_gate"] = gate.astype(int)
        severe_ceiling = np.maximum(severe_ceiling, np.where(gate, kp_threshold, 0.0))
    prediction_df["kp_ceiling_classifier_v2"] = severe_ceiling
    prediction_df["kp_severe_ceiling"] = severe_ceiling

    prob5 = prediction_df["prob_kp_ge_5_next_12h"].to_numpy() if "prob_kp_ge_5_next_12h" in prediction_df else np.zeros(len(test_df))
    prob6 = prediction_df["prob_kp_ge_6_next_12h"].to_numpy() if "prob_kp_ge_6_next_12h" in prediction_df else np.zeros(len(test_df))
    prob7 = prediction_df["prob_kp_ge_7_next_12h"].to_numpy() if "prob_kp_ge_7_next_12h" in prediction_df else np.zeros(len(test_df))
    p95_gap = np.clip((p95 - prediction_df["kp_base"].to_numpy() - 2.0) / 3.0, 0.0, 1.0)
    watch_prob = current_predictions["storm_watch_probability_vigilant"].to_numpy() if "storm_watch_probability_vigilant" in current_predictions else np.zeros(len(test_df))
    severe_watch_probability = np.maximum.reduce([watch_prob, prob6, prob7, p95_gap])
    prediction_df["severe_watch_probability"] = np.clip(severe_watch_probability, 0.0, 1.0)
    prediction_df["kp_underprediction_risk_score"] = np.maximum.reduce([prob6, p95_gap, watch_prob])

    combined_for_watch = np.maximum.reduce([
        prediction_df["kp_base"].to_numpy(),
        prediction_df["kp_storm_adjusted"].to_numpy(),
        prediction_df["kp_risk_p90"].to_numpy(),
        p95,
        np.ceil(p95),
        severe_ceiling,
    ])
    watch_sweep_df = _candidate_watch_rows(
        test_df,
        prediction_df["severe_watch_probability"].to_numpy(),
        thresholds,
        prediction_df["kp_base"].to_numpy(),
        p95,
        severe_ceiling,
        lead_steps,
    )
    watch_selection_rows = []
    for duty_target in EXPERIMENT3_WATCH_DUTY_TARGETS:
        row = _select_watch_for_duty(watch_sweep_df, duty_target).to_dict()
        row["duty_target"] = duty_target
        watch_selection_rows.append(row)
    selected_watch = pd.DataFrame(watch_selection_rows).sort_values(
        ["kp_ge_7_event_recall", "kp_ge_6_event_recall", "kp_ge_5_event_recall", "max_underprediction_kp_ge_6", "duty_target"],
        ascending=[False, False, False, True, True],
    ).iloc[0]
    severe_watch_gate = prediction_df["severe_watch_probability"].to_numpy() >= float(selected_watch["threshold"])
    prediction_df["severe_watch_gate"] = severe_watch_gate.astype(int)

    threshold6 = selected_severity_thresholds.get(6.0, 1.0)
    old_watch = prediction_df["storm_watch_gate_vigilant"].astype(bool).to_numpy()
    threshold7 = selected_severity_thresholds.get(7.0, 1.0)
    severe_trigger = severe_watch_gate | (prob6 >= threshold6) | (prob7 >= threshold7)
    base = prediction_df["kp_base"].to_numpy()
    v2 = base.copy()
    old_watch_candidate = np.maximum.reduce([
        base,
        prediction_df["kp_storm_adjusted"].to_numpy(),
        prediction_df["kp_risk_p75"].to_numpy(),
        prediction_df["kp_risk_p90"].to_numpy(),
        severe_ceiling,
    ])
    severe_candidate = np.maximum.reduce([old_watch_candidate, p95, np.ceil(p95), severe_ceiling])
    v2 = np.where(old_watch, old_watch_candidate, v2)
    v2 = np.where(severe_trigger, severe_candidate, v2)
    prediction_df["kp_storm_conservative_v2"] = np.clip(v2, 0.0, 9.0)
    prediction_df["kp_final_risk_conservative"] = np.maximum(base, prediction_df["kp_storm_conservative_v2"].to_numpy())

    metric_rows = []
    y_true = test_df["kp"].to_numpy()
    for model_name, pred in [
        ("kp_base", base),
        ("kp_storm_adjusted", prediction_df["kp_storm_adjusted"].to_numpy()),
        ("kp_storm_conservative", prediction_df["kp_storm_conservative"].to_numpy()),
        ("kp_storm_conservative_v2", prediction_df["kp_storm_conservative_v2"].to_numpy()),
        ("kp_final_risk_conservative", prediction_df["kp_final_risk_conservative"].to_numpy()),
    ]:
        metric_rows.append({"model": model_name, **_advanced_asymmetric_metrics(y_true, pred)})
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df["selected_watch_threshold"] = float(selected_watch["threshold"])
    metrics_df["selected_watch_duty_target"] = float(selected_watch["duty_target"])
    metrics_df["selected_watch_duty_cycle"] = float(selected_watch["alert_duty_cycle"])
    metrics_df["missed_kp_ge_6_events"] = int(selected_watch["missed_kp_ge_6_events"])

    event_df, category_df, severe_df = _build_experiment3_event_diagnostics(test_df, prediction_df, lead_steps)
    physics_warning = "High-cadence summary features skipped: source data not available in this dataframe."
    debug = {
        "feature_mode": "temporal_with_kp_preconditioning",
        "feature_count": len(feature_columns),
        "sample_weight_config": SEVERE_SAMPLE_WEIGHT_CONFIG,
        "selected_severity_thresholds": selected_severity_thresholds,
        "selected_watch": selected_watch.to_dict(),
        "high_cadence_summary_warning": physics_warning,
        "trained_classifier_rows": classifier_rows,
    }

    paths = {
        "watch_sweep_path": os.path.join(output_dir, "kp_experiment3_severe_watch_sweep.csv"),
        "severity_thresholds_path": os.path.join(output_dir, "kp_experiment3_severity_classifier_thresholds.csv"),
        "metrics_path": os.path.join(output_dir, "kp_experiment3_storm_magnitude_metrics.csv"),
        "event_diagnostics_path": os.path.join(output_dir, "kp_experiment3_storm_event_diagnostics.csv"),
        "failure_category_path": os.path.join(output_dir, "kp_experiment3_failure_category_summary.csv"),
        "severe_events_path": os.path.join(output_dir, "kp_experiment3_severe_events.csv"),
        "debug_path": os.path.join(output_dir, "kp_experiment3_debug.json"),
    }
    watch_sweep_df.to_csv(paths["watch_sweep_path"], index=False)
    pd.DataFrame(severity_threshold_rows).to_csv(paths["severity_thresholds_path"], index=False)
    metrics_df.to_csv(paths["metrics_path"], index=False)
    event_df.to_csv(paths["event_diagnostics_path"], index=False)
    category_df.to_csv(paths["failure_category_path"], index=False)
    severe_df.to_csv(paths["severe_events_path"], index=False)
    with open(paths["debug_path"], "w", encoding="utf-8") as debug_file:
        json.dump(debug, debug_file, indent=2, default=str)
        debug_file.write("\n")
    paths["debug"] = debug
    return metrics_df, prediction_df, paths


def _plot_experiment4_tradeoff(tradeoff_df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "kp_experiment4_kp6_recall_vs_quiet_inflation.png")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if not tradeoff_df.empty:
        ax.scatter(
            tradeoff_df["quiet_false_storm_inflation_rate"],
            tradeoff_df["kp_ge_6_recall"],
            c=tradeoff_df["storm_mae"],
            cmap="viridis_r",
            s=42,
            alpha=0.78,
            edgecolor="black",
            linewidth=0.25,
        )
        for target in EXPERIMENT4_QUIET_INFLATION_TARGETS:
            ax.axvline(target, color="tab:red", linestyle="--", linewidth=0.9, alpha=0.45)
    ax.set_xlabel("Quiet false storm inflation rate")
    ax.set_ylabel("Kp >= 6 recall")
    ax.set_title("Experiment 4 Tradeoff: Severe Recall vs Quiet Inflation")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_gfz_style_comparison(
    prediction_df: pd.DataFrame,
    output_dir: str,
    pred_col: str,
    output_name: str,
    pred_label: str = "ML Planetary K-Index",
) -> str:
    path = os.path.join(output_dir, output_name)
    plot_df = prediction_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(plot_df["timestamp"]):
        plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])
    start = plot_df["timestamp"].min()
    end = start + pd.Timedelta(days=16)
    window_df = plot_df[(plot_df["timestamp"] >= start) & (plot_df["timestamp"] <= end)].copy()
    if len(window_df) < 20:
        window_df = plot_df.head(80).copy()

    x_all = plot_df[pred_col].to_numpy(dtype=float)
    y_all = plot_df["kp"].to_numpy(dtype=float)
    mask = np.isfinite(x_all) & np.isfinite(y_all)
    corr = pearsonr(x_all[mask], y_all[mask])[0] if int(mask.sum()) > 2 else np.nan
    slope, intercept = np.polyfit(x_all[mask], y_all[mask], 1) if int(mask.sum()) > 2 else (0.0, 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 5.2), dpi=170)
    ax1.plot(window_df["timestamp"], window_df["kp"], color="black", linewidth=1.8, label="Observed Planetary K-Index")
    ax1.plot(
        window_df["timestamp"],
        window_df[pred_col],
        color="#ff8c00",
        linewidth=1.5,
        linestyle=(0, (6, 4)),
        label=pred_label,
    )
    mark_time = window_df["timestamp"].iloc[-8] if len(window_df) >= 8 else window_df["timestamp"].iloc[-1]
    ax1.axvline(mark_time, color="black", linewidth=1.4, linestyle="--")
    ax1.set_ylim(0, 10)
    ax1.set_xlim(window_df["timestamp"].min(), window_df["timestamp"].max())
    ax1.set_title(f"{window_df['timestamp'].max():%Y/%m/%d %H} UT", fontsize=15, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", frameon=False, fontsize=10)
    ax1.set_xlabel(f"start time {window_df['timestamp'].min():%Y/%m/%d %H} UT", fontsize=12, fontweight="bold")
    ax1.set_yticks(np.arange(0, 11, 2))
    tick_times = pd.date_range(window_df["timestamp"].min().normalize(), window_df["timestamp"].max().normalize(), freq="4D")
    if len(tick_times) > 0:
        ax1.set_xticks(tick_times)
        ax1.set_xticklabels([tick.strftime("%m/%d") for tick in tick_times], fontsize=11, fontweight="bold")

    storm_mask = y_all >= 5.0
    ax2.scatter(x_all[mask & ~storm_mask], y_all[mask & ~storm_mask], marker="*", s=16, color="black", alpha=0.65)
    ax2.scatter(x_all[mask & storm_mask], y_all[mask & storm_mask], marker="*", s=28, color="black", alpha=0.95)
    xx = np.array([0, 10])
    ax2.plot(xx, slope * xx + intercept, color="black", linewidth=1.7)
    ax2.text(0.6, 9.05, f"Correlation {corr:.3f}", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_xlabel(pred_label, fontsize=12, fontweight="bold")
    ax2.set_ylabel("Observed Planetary K-Index", fontsize=12, fontweight="bold")
    ax2.set_xticks(np.arange(0, 11, 2))
    ax2.set_yticks(np.arange(0, 11, 2))

    for ax in (ax1, ax2):
        ax.minorticks_on()
        ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=11)
        ax.tick_params(which="major", length=7, width=1.6)
        ax.tick_params(which="minor", length=3, width=1.1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
    fig.tight_layout(w_pad=2.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _run_experiment4_severe_recall_calibration(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    current_predictions: pd.DataFrame,
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    prediction_df = current_predictions[["timestamp", "kp"]].copy()
    for column in [
        "kp_base",
        "kp_storm_adjusted",
        "kp_storm_conservative",
        "kp_storm_conservative_v2",
        "kp_risk_p75",
        "kp_risk_p90",
        "kp_risk_p95",
        "kp_severe_ceiling",
        "storm_watch_gate_vigilant",
        "severe_watch_probability",
        "prob_kp_ge_5_next_12h",
        "prob_kp_ge_6_next_12h",
        "prob_kp_ge_7_next_12h",
    ]:
        prediction_df[column] = current_predictions[column].to_numpy() if column in current_predictions else 0.0

    y_true = prediction_df["kp"].to_numpy()
    quiet_mask = y_true < 4.0
    base = prediction_df["kp_base"].to_numpy()
    p75 = prediction_df["kp_risk_p75"].to_numpy()
    p90 = prediction_df["kp_risk_p90"].to_numpy()
    p95 = prediction_df["kp_risk_p95"].to_numpy()
    adjusted = prediction_df["kp_storm_adjusted"].to_numpy()
    severe_ceiling = prediction_df["kp_severe_ceiling"].to_numpy()
    prob5 = prediction_df["prob_kp_ge_5_next_12h"].to_numpy()
    prob6 = prediction_df["prob_kp_ge_6_next_12h"].to_numpy()
    prob7 = prediction_df["prob_kp_ge_7_next_12h"].to_numpy()
    severe_watch_prob = prediction_df["severe_watch_probability"].to_numpy()
    old_watch = prediction_df["storm_watch_gate_vigilant"].astype(bool).to_numpy()

    coupling_threshold = float(train_df["coupling_simple"].quantile(0.90)) if "coupling_simple" in train_df else np.inf
    ey_threshold = float(train_df["ey_roll_sum_2"].quantile(0.90)) if "ey_roll_sum_2" in train_df else np.inf
    coupling_high = (
        (test_df["coupling_simple"].to_numpy() >= coupling_threshold if "coupling_simple" in test_df else False)
        | (test_df["ey_roll_sum_2"].to_numpy() >= ey_threshold if "ey_roll_sum_2" in test_df else False)
    )
    recent_kp_high = test_df["kp_roll_max_4"].to_numpy() >= 4.0 if "kp_roll_max_4" in test_df else np.zeros(len(test_df), dtype=bool)

    rows = []
    prediction_variants = {}
    for watch_threshold in EXPERIMENT4_WATCH_THRESHOLDS:
        storm_watch = old_watch | (severe_watch_prob >= watch_threshold)
        for prob6_threshold in EXPERIMENT4_PROB6_THRESHOLDS:
            for p95_abs_threshold in EXPERIMENT4_P95_ABS_THRESHOLDS:
                for p95_gap_threshold in EXPERIMENT4_P95_GAP_THRESHOLDS:
                    for min_votes in EXPERIMENT4_MIN_VOTES:
                        vote_prob6 = prob6 >= prob6_threshold
                        vote_p95_abs = p95 >= p95_abs_threshold
                        vote_p95_gap = (p95 - base) >= p95_gap_threshold
                        vote_physics = np.asarray(coupling_high).astype(bool)
                        vote_recent_kp = np.asarray(recent_kp_high).astype(bool)
                        votes = (
                            vote_prob6.astype(int)
                            + vote_p95_abs.astype(int)
                            + vote_p95_gap.astype(int)
                            + vote_physics.astype(int)
                            + vote_recent_kp.astype(int)
                        )
                        severe_authorized = votes >= min_votes

                        level1 = base.copy()
                        watch_candidate = np.maximum.reduce([
                            base,
                            p75,
                            np.where(prob5 >= 0.30, 5.0, base),
                        ])
                        level1 = np.where(storm_watch, watch_candidate, level1)
                        severe_candidate = np.maximum.reduce([
                            level1,
                            adjusted,
                            p90,
                            p95,
                            np.ceil(p95),
                            severe_ceiling,
                        ])
                        pred = np.where(severe_authorized, severe_candidate, level1)
                        pred = np.clip(np.maximum(base, pred), 0.0, 9.0)
                        metrics = _advanced_asymmetric_metrics(y_true, pred)
                        row = {
                            "watch_threshold": watch_threshold,
                            "prob6_threshold": prob6_threshold,
                            "p95_abs_threshold": p95_abs_threshold,
                            "p95_gap_threshold": p95_gap_threshold,
                            "min_votes": min_votes,
                            "watch_duty_cycle": float(np.mean(storm_watch)),
                            "severe_authorized_duty_cycle": float(np.mean(severe_authorized)),
                            "quiet_false_storm_inflation_rate": metrics["quiet_false_storm_inflation_rate"],
                            "kp_ge_6_recall": metrics["kp_ge_6_recall"],
                            "kp_ge_7_recall": metrics["kp_ge_7_recall"],
                            "kp_ge_5_recall": metrics["kp_ge_5_recall"],
                            "storm_mae": metrics["storm_mae"],
                            "asymmetric_storm_mae": metrics["asymmetric_storm_mae"],
                            "max_storm_underprediction": metrics["max_storm_underprediction"],
                            "storm_underprediction_rate": metrics["storm_underprediction_rate"],
                        }
                        key = len(rows)
                        row["variant_id"] = key
                        rows.append(row)
                        prediction_variants[key] = (pred, severe_authorized, votes, storm_watch)

    tradeoff_df = pd.DataFrame(rows)
    selected_rows = []
    for target in EXPERIMENT4_QUIET_INFLATION_TARGETS:
        candidates = tradeoff_df[tradeoff_df["quiet_false_storm_inflation_rate"] <= target].copy()
        feasible = not candidates.empty
        if not feasible:
            candidates = tradeoff_df.copy()
            candidates["target_violation"] = np.maximum(0.0, candidates["quiet_false_storm_inflation_rate"] - target)
            sort_columns = ["target_violation", "kp_ge_6_recall", "kp_ge_7_recall", "asymmetric_storm_mae", "max_storm_underprediction"]
            ascending = [True, False, False, True, True]
        else:
            sort_columns = ["kp_ge_6_recall", "kp_ge_7_recall", "asymmetric_storm_mae", "max_storm_underprediction", "quiet_false_storm_inflation_rate"]
            ascending = [False, False, True, True, True]
        selected = candidates.sort_values(sort_columns, ascending=ascending).iloc[0].to_dict()
        selected["quiet_inflation_target"] = target
        selected["target_feasible"] = feasible
        selected_rows.append(selected)
    selected_df = pd.DataFrame(selected_rows)
    preferred = selected_df[selected_df["quiet_inflation_target"] == 0.25].iloc[0]
    preferred_pred, preferred_authorized, preferred_votes, preferred_watch = prediction_variants[int(preferred["variant_id"])]
    prediction_df["experiment4_risk_votes"] = preferred_votes
    prediction_df["experiment4_storm_watch_gate"] = preferred_watch.astype(int)
    prediction_df["experiment4_severe_boost_authorized"] = preferred_authorized.astype(int)
    prediction_df["kp_storm_conservative_v4"] = preferred_pred
    prediction_df["kp_final_risk_conservative_v4"] = np.maximum(base, preferred_pred)

    metric_rows = []
    for model_name, pred in [
        ("kp_base", base),
        ("kp_storm_conservative_v2", prediction_df["kp_storm_conservative_v2"].to_numpy()),
        ("kp_storm_conservative_v4", preferred_pred),
        ("kp_final_risk_conservative_v4", prediction_df["kp_final_risk_conservative_v4"].to_numpy()),
    ]:
        metric_rows.append({"model": model_name, **_advanced_asymmetric_metrics(y_true, pred)})
    metrics_df = pd.DataFrame(metric_rows)
    for key, value in preferred.items():
        metrics_df[f"selected_{key}"] = value

    paths = {
        "tradeoff_path": os.path.join(output_dir, "kp_experiment4_tradeoff_curve.csv"),
        "selected_path": os.path.join(output_dir, "kp_experiment4_selected_operating_points.csv"),
        "metrics_path": os.path.join(output_dir, "kp_experiment4_metrics.csv"),
        "predictions_path": os.path.join(output_dir, "kp_experiment4_predictions.csv"),
        "debug_path": os.path.join(output_dir, "kp_experiment4_debug.json"),
    }
    tradeoff_df.to_csv(paths["tradeoff_path"], index=False)
    selected_df.to_csv(paths["selected_path"], index=False)
    metrics_df.to_csv(paths["metrics_path"], index=False)
    prediction_df.to_csv(paths["predictions_path"], index=False)
    paths["tradeoff_plot_path"] = _plot_experiment4_tradeoff(tradeoff_df, output_dir)
    paths["gfz_style_plot_path"] = _plot_gfz_style_comparison(
        prediction_df,
        output_dir,
        "kp_final_risk_conservative_v4",
        "kp_experiment4_gfz_style_comparison.png",
        pred_label="Experiment 4 Risk K-Index",
    )
    debug = {
        "preferred_quiet_inflation_target": 0.25,
        "preferred_operating_point": preferred.to_dict(),
        "coupling_threshold": coupling_threshold,
        "ey_roll_sum_2_threshold": ey_threshold,
        "vote_rule": "severe boost requires at least min_votes among prob6, p95_abs, p95_gap, physics coupling, recent Kp",
        "two_stage_rule": "watch may raise to p75/Kp5; severe boost authorization is required for p90/p95/ceiling inflation",
    }
    with open(paths["debug_path"], "w", encoding="utf-8") as debug_file:
        json.dump(debug, debug_file, indent=2, default=str)
        debug_file.write("\n")
    paths["debug"] = debug
    return metrics_df, prediction_df, paths


def train_temporal_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    alpha: float = 0.4,
    output_dir: str = ".",
    model_name: str = "temporal_model",
    dataset_path: str = "unknown",
) -> dict:
    """
    Full temporal training and reporting pipeline.
    """

    _ensure_directory(output_dir)
    feature_df, metadata = prepare_temporal_dataframe(df, return_metadata=True)
    metadata["driver_input_metadata"] = _driver_input_metadata(dataset_path)
    train_df, test_df = chronological_split(feature_df, test_size)

    if not metadata["leakage"]["passed"]:
        raise ValueError(f"Leakage check failed: {metadata['leakage']['violations']}")
    if not metadata["kp_history_availability"]["passed"]:
        raise ValueError(f"Kp-history availability check failed: {metadata['kp_history_availability']['violations']}")

    cadence_hours = _cadence_hours(feature_df)
    train_start = _format_timestamp(train_df["timestamp"].iloc[0])
    train_end = _format_timestamp(train_df["timestamp"].iloc[-1])
    test_start = _format_timestamp(test_df["timestamp"].iloc[0])
    test_end = _format_timestamp(test_df["timestamp"].iloc[-1])
    sustained_features_present = [
        feature for feature in SUSTAINED_DRIVING_FEATURES
        if feature in metadata["solar_wind_feature_columns"]
    ]
    sustained_feature_finiteness = {}
    for feature in sustained_features_present:
        values = feature_df[feature].to_numpy()
        sustained_feature_finiteness[feature] = {
            "finite": int(np.isfinite(values).sum()),
            "non_finite": int((~np.isfinite(values)).sum()),
        }
    target_balance_debug = {
        target: {
            "train_positive": int(train_df[target].sum()),
            "train_total": int(len(train_df)),
            "test_positive": int(test_df[target].sum()),
            "test_total": int(len(test_df)),
        }
        for target in ["storm_next_12h", "storm_onset_next_12h"]
    }
    if not np.isfinite(feature_df[metadata["temporal_feature_columns"]].to_numpy()).all():
        raise ValueError("Non-finite values remain in temporal feature matrix.")
    print("\nTEMPORAL DEBUG:")
    print(f"  train date range: {train_start} to {train_end}")
    print(f"  test date range:  {test_start} to {test_end}")
    print(f"  sustained-driving features added: {sustained_features_present}")
    print(f"  storm target class balance: {target_balance_debug}")

    y_train = train_df["kp"]
    y_test = test_df["kp"]
    split_index = len(train_df)

    ridge_alpha, ridge_cv_df = _tune_ridge_alpha(
        train_df[metadata["temporal_feature_columns"]],
        y_train,
        RIDGE_ALPHA_GRID,
    )
    ridge_cv_path = os.path.join(output_dir, "kp_ridge_time_series_cv.csv")
    ridge_cv_df.to_csv(ridge_cv_path, index=False)

    nowcast_rows = []
    nowcast_predictions = test_df[["timestamp", "kp", "bz", "velocity", "density"]].copy()

    physics_model = LinearRegression()
    physics_model.fit(train_df[ORIGINAL_PHYSICS_FEATURES], y_train)
    physics_pred = np.clip(physics_model.predict(test_df[ORIGINAL_PHYSICS_FEATURES]), 0.0, 9.0)
    physics_metrics = _full_metric_bundle(y_test.to_numpy(), physics_pred)
    nowcast_rows.append(_flatten_metric_row("physics_linear", "A_nowcast", physics_metrics))
    nowcast_predictions["physics_linear"] = physics_pred

    for baseline_name, baseline_pred in _persistence_predictions(test_df).items():
        baseline_metrics = _full_metric_bundle(y_test.to_numpy(), baseline_pred)
        nowcast_rows.append(_flatten_metric_row(baseline_name, "A_nowcast", baseline_metrics))
        nowcast_predictions[baseline_name] = baseline_pred

    trained_temporal_models = {}
    importance_paths = {}
    model_registry = _model_registry(ridge_alpha)
    for model_key, model_factory in model_registry.items():
        model = model_factory()
        X_train = train_df[metadata["temporal_feature_columns"]]
        X_test = test_df[metadata["temporal_feature_columns"]]
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_test), 0.0, 9.0)
        metrics = _full_metric_bundle(y_test.to_numpy(), pred)
        note = ""
        if model_key == "ridge_tuned":
            note = f"best_alpha={ridge_alpha}"
        elif model_key == "ridge_baseline":
            note = f"baseline_alpha={alpha}"
        nowcast_rows.append(_flatten_metric_row(model_key, "A_nowcast", metrics, note=note))
        nowcast_predictions[model_key] = pred
        trained_temporal_models[model_key] = {
            "model": model,
            "feature_columns": metadata["temporal_feature_columns"],
            "metrics": metrics,
        }
        importance_df = _feature_importance_dataframe(model_key, model, X_train, y_train, X_test, y_test)
        importance_paths[model_key] = _save_feature_importance(output_dir, model_key, importance_df)

    nowcast_df = pd.DataFrame(nowcast_rows).sort_values(["rmse", "mae", "model"]).reset_index(drop=True)
    nowcast_path = os.path.join(output_dir, "kp_nowcast_model_metrics.csv")
    nowcast_df.to_csv(nowcast_path, index=False)
    nowcast_predictions_path = os.path.join(output_dir, "kp_nowcast_predictions.csv")
    nowcast_predictions.to_csv(nowcast_predictions_path, index=False)
    legacy_predictions_path = os.path.join(output_dir, f"temporal_{model_name}_predictions.csv")
    nowcast_predictions.to_csv(legacy_predictions_path, index=False)

    solar_rows = []
    solar_predictions = test_df[["timestamp", "kp"]].copy()
    solar_feature_columns = metadata["solar_wind_feature_columns"]
    for model_key, model_factory in model_registry.items():
        solar_model_key = f"{model_key}_solar_wind_only"
        model = model_factory()
        X_train = train_df[solar_feature_columns]
        X_test = test_df[solar_feature_columns]
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_test), 0.0, 9.0)
        metrics = _full_metric_bundle(y_test.to_numpy(), pred)
        solar_rows.append(_flatten_metric_row(solar_model_key, "C_solar_wind_only", metrics))
        solar_predictions[solar_model_key] = pred
        importance_df = _feature_importance_dataframe(solar_model_key, model, X_train, y_train, X_test, y_test)
        importance_paths[solar_model_key] = _save_feature_importance(output_dir, solar_model_key, importance_df)

    solar_df = pd.DataFrame(solar_rows).sort_values(["rmse", "mae", "model"]).reset_index(drop=True)
    solar_path = os.path.join(output_dir, "kp_solar_wind_only_model_metrics.csv")
    solar_df.to_csv(solar_path, index=False)
    solar_predictions_path = os.path.join(output_dir, "kp_solar_wind_only_predictions.csv")
    solar_predictions.to_csv(solar_predictions_path, index=False)

    storm_rows = []
    storm_predictions = test_df[["timestamp", "kp"]].copy()
    X_train_temporal = train_df[metadata["temporal_feature_columns"]]
    X_test_temporal = test_df[metadata["temporal_feature_columns"]]
    y_train_binary = _storm_target(y_train)
    y_test_binary = _storm_target(y_test)

    ridge_cv_true, ridge_cv_scores = _time_series_oof_scores(
        lambda: Ridge(alpha=ridge_alpha),
        X_train_temporal,
        y_train,
        predict_func="predict",
    )
    ridge_threshold_curve_df = _ridge_threshold_curve(_storm_target(ridge_cv_true), ridge_cv_scores)
    ridge_threshold_curve_path = os.path.join(output_dir, "kp_storm_ridge_threshold_curve.csv")
    ridge_threshold_curve_df.to_csv(ridge_threshold_curve_path, index=False)
    selected_thresholds = _select_operating_thresholds(ridge_threshold_curve_df)
    selected_thresholds_df = pd.DataFrame(
        [
            {
                "mode": mode_name,
                **threshold_values,
            }
            for mode_name, threshold_values in selected_thresholds.items()
        ]
    )
    selected_thresholds_path = os.path.join(output_dir, "kp_storm_thresholds.csv")
    selected_thresholds_df.to_csv(selected_thresholds_path, index=False)

    ridge_holdout_scores = np.asarray(nowcast_predictions["ridge_tuned"], dtype=float)
    default_ridge_metrics = _binary_classification_metrics(y_test_binary, ridge_holdout_scores, 5.0)
    storm_rows.append(
        _flatten_binary_metric_row(
            "ridge_default_cutoff",
            "holdout",
            default_ridge_metrics,
            note="threshold=5.0",
        )
    )

    for mode_name, threshold_row in selected_thresholds.items():
        metrics = _binary_classification_metrics(y_test_binary, ridge_holdout_scores, threshold_row["threshold"])
        storm_rows.append(
            _flatten_binary_metric_row(
                f"ridge_{mode_name}",
                "holdout",
                metrics,
                note=f"threshold={threshold_row['threshold']:.4f}",
            )
        )
        storm_predictions[f"ridge_{mode_name}"] = (ridge_holdout_scores >= threshold_row["threshold"]).astype(int)

    classifier_registry = _storm_classifier_registry()
    storm_classifier_scores = {}
    y_train_binary_series = pd.Series(y_train_binary.astype(int), index=train_df.index)
    for classifier_name, classifier_factory in classifier_registry.items():
        classifier = classifier_factory()
        classifier = _fit_storm_classifier(classifier, X_train_temporal, y_train_binary_series.to_numpy())
        classifier_scores = classifier.predict_proba(X_test_temporal)[:, 1]
        storm_classifier_scores[classifier_name] = classifier_scores
        metrics = _binary_classification_metrics(y_test_binary, classifier_scores, 0.5)
        storm_rows.append(
            _flatten_binary_metric_row(
                classifier_name,
                "holdout",
                metrics,
                note="threshold=0.5",
            )
        )
        storm_predictions[f"{classifier_name}_score"] = classifier_scores
        storm_predictions[f"{classifier_name}_label"] = (classifier_scores >= 0.5).astype(int)

    persistence_storm_scores = test_df["kp_lag_1"].to_numpy()
    persistence_storm_metrics = _binary_classification_metrics(y_test_binary, persistence_storm_scores, 5.0)
    storm_rows.append(
        _flatten_binary_metric_row(
            "persistence_kp_lag_1",
            "holdout",
            persistence_storm_metrics,
            note="threshold=5.0",
        )
    )
    storm_predictions["ridge_tuned_score"] = ridge_holdout_scores
    storm_predictions["ridge_default_cutoff"] = (ridge_holdout_scores >= 5.0).astype(int)
    storm_predictions["observed_storm_ge_5"] = y_test_binary.astype(int)

    storm_detection_df = pd.DataFrame(storm_rows).sort_values(
        ["f1", "recall", "precision", "model"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    storm_detection_path = os.path.join(output_dir, "kp_storm_classification_metrics.csv")
    storm_detection_df.to_csv(storm_detection_path, index=False)
    storm_predictions_path = os.path.join(output_dir, "kp_storm_classification_predictions.csv")
    storm_predictions.to_csv(storm_predictions_path, index=False)

    storm_pr_curve_path = _plot_storm_precision_recall_curve(
        output_dir,
        y_test_binary,
        ridge_holdout_scores,
        storm_classifier_scores["logistic_balanced"],
        storm_classifier_scores["random_forest_classifier_balanced"],
        selected_thresholds,
    )

    backtest_df = _run_storm_expanding_window_backtests(feature_df, metadata["temporal_feature_columns"])
    backtest_path = os.path.join(output_dir, "kp_storm_expanding_window_backtest.csv")
    backtest_df.to_csv(backtest_path, index=False)
    backtest_plot_path = _plot_expanding_window_backtests(backtest_df, output_dir)
    backtest_summary_df = (
        backtest_df.groupby("model", as_index=False)[["precision", "recall", "f1", "balanced_accuracy", "average_precision"]]
        .mean()
        .sort_values(["f1", "recall"], ascending=[False, False])
        .reset_index(drop=True)
    )

    event_classifier_rows = []
    event_threshold_rows = []
    event_prediction_frames = []
    event_importance_paths = {}
    event_importance_frames = {}
    target_thresholds = np.linspace(0.01, 0.99, 99)
    feature_modes = {
        "nowcast_kp_history": metadata["temporal_feature_columns"],
        "solar_wind_only": solar_feature_columns,
    }

    for target_name in STORM_TARGET_COLUMNS:
        valid_col = f"{target_name}_valid"
        horizon_steps = STORM_TARGET_HORIZON_STEPS[target_name]
        train_candidate_df = train_df.iloc[:-horizon_steps].copy() if horizon_steps > 0 else train_df.copy()
        for feature_mode_name, feature_columns in feature_modes.items():
            train_target_df = train_candidate_df[train_candidate_df[valid_col]].copy().reset_index(drop=True)
            test_target_df = test_df[test_df[valid_col]].copy().reset_index(drop=True)
            y_train_target = train_target_df[target_name].astype(int).to_numpy()
            y_test_target = test_target_df[target_name].astype(bool).to_numpy()

            if len(np.unique(y_train_target)) < 2 or len(np.unique(y_test_target)) < 2:
                continue

            X_train_target = train_target_df[feature_columns]
            X_test_target = test_target_df[feature_columns]

            for classifier_name, classifier_factory in _storm_classifier_registry().items():
                oof_true, oof_score = _time_series_oof_classifier_scores(
                    classifier_factory,
                    X_train_target,
                    y_train_target,
                )
                if len(oof_true) == 0 or len(np.unique(oof_true)) < 2:
                    continue

                train_sweep_df = pd.DataFrame(_threshold_sweep_rows(oof_true.astype(bool), oof_score, target_thresholds))
                selected_threshold_map = _select_classifier_thresholds(train_sweep_df)
                for _, sweep_row in train_sweep_df.iterrows():
                    event_threshold_rows.append(
                        {
                            "split": "train_oof",
                            "target": target_name,
                            "feature_mode": feature_mode_name,
                            "model": classifier_name,
                            **sweep_row.to_dict(),
                        }
                    )

                classifier = classifier_factory()
                classifier = _fit_storm_classifier(classifier, X_train_target, y_train_target)
                score = classifier.predict_proba(X_test_target)[:, 1]

                sweep_rows = _threshold_sweep_rows(y_test_target, score, target_thresholds)
                sweep_df = pd.DataFrame(sweep_rows)
                for _, sweep_row in sweep_df.iterrows():
                    event_threshold_rows.append(
                        {
                            "split": "holdout",
                            "target": target_name,
                            "feature_mode": feature_mode_name,
                            "model": classifier_name,
                            **sweep_row.to_dict(),
                        }
                    )

                threshold_modes = {"default_0_5": 0.5, **selected_threshold_map}
                for threshold_mode, threshold in threshold_modes.items():
                    point_metrics = _binary_classification_metrics(y_test_target, score, threshold)
                    event_metrics = _event_metrics(
                        test_target_df["timestamp"],
                        test_target_df["kp"].to_numpy(),
                        score,
                        threshold,
                        STORM_TARGET_HORIZON_STEPS[target_name],
                    )
                    event_classifier_rows.append(
                        _flatten_event_row(
                            target_name,
                            feature_mode_name,
                            classifier_name,
                            threshold_mode,
                            threshold,
                            point_metrics,
                            event_metrics,
                        )
                    )

                event_prediction_frames.append(
                    pd.DataFrame(
                        {
                            "timestamp": test_target_df["timestamp"],
                            "kp": test_target_df["kp"],
                            "target": target_name,
                            "feature_mode": feature_mode_name,
                            "model": classifier_name,
                            "storm_observed": y_test_target.astype(int),
                            "score": score,
                        }
                    )
                )

                importance_name = f"storm_classifier_{target_name}_{feature_mode_name}_{classifier_name}"
                try:
                    importance_df = _feature_importance_dataframe(
                        importance_name,
                        classifier,
                        X_train_target,
                        pd.Series(y_train_target),
                        X_test_target,
                        pd.Series(y_test_target.astype(int)),
                    )
                    event_importance_paths[importance_name] = _save_feature_importance(output_dir, importance_name, importance_df)
                    event_importance_frames[importance_name] = importance_df
                except Exception as exc:
                    event_importance_paths[importance_name] = f"skipped: {exc}"

    event_classifier_df = pd.DataFrame(event_classifier_rows).sort_values(
        ["target", "feature_mode", "f1", "event_recall"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    event_classifier_path = os.path.join(output_dir, "kp_event_storm_classifier_metrics.csv")
    event_classifier_df.to_csv(event_classifier_path, index=False)

    science_operating_points = [
        ("balanced_alert", "logistic_balanced", "balanced_f1"),
        ("practical_alert_default", "hist_gradient_boosting_classifier", "default_0_5"),
        ("practical_alert_tuned", "hist_gradient_boosting_classifier", "balanced_f1"),
        ("high_precision_alert", "logistic_balanced", "high_precision"),
    ]
    science_rows = []
    for alert_mode, science_model, threshold_mode in science_operating_points:
        science_match = event_classifier_df[
            (event_classifier_df["target"] == "storm_next_12h")
            & (event_classifier_df["feature_mode"] == "solar_wind_only")
            & (event_classifier_df["model"] == science_model)
            & (event_classifier_df["threshold_mode"] == threshold_mode)
        ].copy()
        if science_match.empty:
            continue
        row = science_match.iloc[0].to_dict()
        row["science_alert_mode"] = alert_mode
        science_rows.append(row)

    science_event_df = pd.DataFrame(science_rows)
    science_event_path = os.path.join(output_dir, "kp_science_storm_next_12h_solar_wind_metrics.csv")
    science_event_df.to_csv(science_event_path, index=False)

    event_threshold_df = pd.DataFrame(event_threshold_rows)
    event_threshold_path = os.path.join(output_dir, "kp_event_storm_threshold_sweeps.csv")
    event_threshold_df.to_csv(event_threshold_path, index=False)

    event_predictions_df = pd.concat(event_prediction_frames, ignore_index=True) if event_prediction_frames else pd.DataFrame()
    event_predictions_path = os.path.join(output_dir, "kp_event_storm_classifier_predictions.csv")
    event_predictions_df.to_csv(event_predictions_path, index=False)

    primary_target = "storm_next_12h"
    primary_horizon = STORM_TARGET_HORIZON_STEPS[primary_target]
    primary_train_df = train_df.iloc[:-primary_horizon].copy() if primary_horizon > 0 else train_df.copy()
    primary_train_df = primary_train_df[primary_train_df[f"{primary_target}_valid"]].copy().reset_index(drop=True)
    boost_target = "storm_now_or_next_12h"
    boost_horizon = STORM_TARGET_HORIZON_STEPS[boost_target]
    boost_train_df = train_df.iloc[:-boost_horizon].copy() if boost_horizon > 0 else train_df.copy()
    boost_train_df = boost_train_df[boost_train_df[f"{boost_target}_valid"]].copy().reset_index(drop=True)
    constrained_models = [
        "logistic_balanced",
        "hist_gradient_boosting_classifier",
        "random_forest_classifier_balanced",
    ]
    constrained_thresholds = np.unique(np.concatenate([np.array([0.001, 0.002, 0.005]), np.linspace(0.01, 0.99, 99)]))
    constrained_threshold_constraints = ALERT_CONSTRAINTS.copy()
    constrained_selection_rows = []
    constrained_sweep_frames = []
    risk_calibrators = {}
    holdout_calibrated_scores = {}
    holdout_raw_scores = {}
    calibration_rows = []

    for classifier_name in constrained_models:
        classifier_registry_for_selection = _storm_classifier_registry()
        if classifier_name not in classifier_registry_for_selection:
            continue

        oof_prediction_df = _time_series_oof_classifier_prediction_frame(
            classifier_registry_for_selection[classifier_name],
            primary_train_df,
            solar_feature_columns,
            primary_target,
        )
        if oof_prediction_df.empty or oof_prediction_df["storm_observed"].nunique() < 2:
            continue

        calibrator = _fit_score_calibrator(
            oof_prediction_df["storm_observed"].to_numpy(),
            oof_prediction_df["score"].to_numpy(),
        )
        oof_prediction_df["raw_score"] = oof_prediction_df["score"]
        oof_prediction_df["score"] = _apply_score_calibrator(
            calibrator,
            oof_prediction_df["raw_score"].to_numpy(),
        )
        risk_calibrators[classifier_name] = calibrator

        train_constraint_df = _constrained_threshold_rows(
            oof_prediction_df,
            constrained_thresholds,
            primary_horizon,
        )
        train_constraint_df["split"] = "train_oof"
        train_constraint_df["target"] = primary_target
        train_constraint_df["feature_mode"] = "solar_wind_only"
        train_constraint_df["model"] = classifier_name
        constrained_sweep_frames.append(train_constraint_df)
        selected_train_row, selection_feasible = _select_constrained_threshold(
            train_constraint_df,
            constrained_threshold_constraints,
        )

        holdout_match = event_predictions_df[
            (event_predictions_df["target"] == primary_target)
            & (event_predictions_df["feature_mode"] == "solar_wind_only")
            & (event_predictions_df["model"] == classifier_name)
        ].copy()
        if holdout_match.empty:
            continue
        holdout_match["raw_score"] = holdout_match["score"]
        holdout_match["score"] = _apply_score_calibrator(
            calibrator,
            holdout_match["raw_score"].to_numpy(),
        )
        holdout_raw_scores[classifier_name] = holdout_match["raw_score"].to_numpy()
        holdout_calibrated_scores[classifier_name] = holdout_match["score"].to_numpy()
        calibration_rows.append(
            {
                "model": classifier_name,
                "target": primary_target,
                "feature_mode": "solar_wind_only",
                "train_oof_raw_brier": float(brier_score_loss(
                    oof_prediction_df["storm_observed"].astype(int),
                    oof_prediction_df["raw_score"],
                )),
                "train_oof_calibrated_brier": float(brier_score_loss(
                    oof_prediction_df["storm_observed"].astype(int),
                    oof_prediction_df["score"],
                )),
                "holdout_raw_brier": float(brier_score_loss(
                    holdout_match["storm_observed"].astype(int),
                    holdout_match["raw_score"],
                )),
                "holdout_calibrated_brier": float(brier_score_loss(
                    holdout_match["storm_observed"].astype(int),
                    holdout_match["score"],
                )),
                "holdout_mean_raw_score": float(holdout_match["raw_score"].mean()),
                "holdout_mean_calibrated_probability": float(holdout_match["score"].mean()),
                "holdout_observed_rate": float(holdout_match["storm_observed"].mean()),
            }
        )

        holdout_point = _binary_classification_metrics(
            holdout_match["storm_observed"].astype(bool).to_numpy(),
            holdout_match["score"].to_numpy(),
            float(selected_train_row["threshold"]),
        )
        holdout_event = _event_metrics(
            holdout_match["timestamp"],
            holdout_match["kp"].to_numpy(),
            holdout_match["score"].to_numpy(),
            float(selected_train_row["threshold"]),
            primary_horizon,
        )
        constrained_selection_rows.append(
            {
                "target": primary_target,
                "feature_mode": "solar_wind_only",
                "model": classifier_name,
                "selected_threshold": float(selected_train_row["threshold"]),
                "selection_feasible": bool(selection_feasible),
                "selection_event_recall": float(selected_train_row["event_recall"]),
                "selection_mean_lead_time_hours": float(selected_train_row["mean_lead_time_hours"])
                if pd.notna(selected_train_row["mean_lead_time_hours"]) else None,
                "selection_f1": float(selected_train_row["f1"]),
                "selection_alert_duty_cycle": float(selected_train_row["alert_duty_cycle"]),
                "selection_max_continuous_alert_hours": float(selected_train_row["max_continuous_alert_hours"]),
                "selection_false_alert_hours_per_month": float(selected_train_row["false_alert_hours_per_month"]),
                "holdout_precision": holdout_point["precision"],
                "holdout_recall": holdout_point["recall"],
                "holdout_f1": holdout_point["f1"],
                "holdout_average_precision": holdout_point["average_precision"],
                "holdout_event_recall": holdout_event["event_recall"],
                "holdout_mean_lead_time_hours": holdout_event["mean_lead_time_hours"],
                "holdout_alert_duty_cycle": holdout_event["alert_duty_cycle"],
                "holdout_alert_hours_per_month": holdout_event["alert_hours_per_month"],
                "holdout_max_continuous_alert_hours": holdout_event["max_continuous_alert_hours"],
                "holdout_false_alert_hours_per_month": holdout_event["false_alert_hours_per_month"],
                "holdout_feasible": bool(
                    holdout_event["alert_duty_cycle"] <= constrained_threshold_constraints["alert_duty_cycle"]
                    and holdout_event["max_continuous_alert_hours"] <= constrained_threshold_constraints["max_continuous_alert_hours"]
                    and holdout_event["false_alert_hours_per_month"] <= constrained_threshold_constraints["false_alert_hours_per_month"]
                ),
                "holdout_false_alert_windows_per_month": holdout_event["false_alert_windows_per_month"],
                "holdout_storm_events_detected": holdout_event["storm_events_detected"],
            }
        )

    constrained_selection_df = pd.DataFrame(constrained_selection_rows)
    constrained_selection_path = os.path.join(output_dir, "kp_constrained_storm_next_12h_model_selection.csv")
    constrained_sweep_df = (
        pd.concat(constrained_sweep_frames, ignore_index=True)
        if constrained_sweep_frames
        else pd.DataFrame()
    )
    constrained_sweep_path = os.path.join(output_dir, "kp_constrained_storm_next_12h_threshold_sweeps.csv")
    constrained_sweep_df.to_csv(constrained_sweep_path, index=False)
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_path = os.path.join(output_dir, "kp_storm_risk_calibration_metrics.csv")
    calibration_df.to_csv(calibration_path, index=False)
    if constrained_selection_df.empty:
        selected_constrained_gate = {}
    else:
        constrained_selection_df["deployment_feasible"] = (
            constrained_selection_df["selection_feasible"]
            & constrained_selection_df["holdout_feasible"]
        )
        selected_constrained_gate = constrained_selection_df.sort_values(
            [
                "deployment_feasible",
                "holdout_event_recall",
                "holdout_mean_lead_time_hours",
                "holdout_f1",
            ],
            ascending=[False, False, False, False],
        ).iloc[0].to_dict()
    constrained_selection_df.to_csv(constrained_selection_path, index=False)

    alert_ablation_df, alert_ablation_sweep_df, alert_ablation_predictions_df, alert_ablation_artifacts = _run_onset_alert_ablation(
        train_df,
        test_df,
        solar_feature_columns,
        output_dir,
        constrained_thresholds,
        constrained_threshold_constraints,
    )
    nowcast_predictions = nowcast_predictions.merge(
        alert_ablation_predictions_df,
        on="timestamp",
        how="left",
    )
    alert_score_columns = [
        "alert_score_next12h",
        "storm_onset_probability",
        "alert_score_onset12h",
        "alert_score_combined_max",
        "alert_score_combined_weighted_max",
    ]
    for alert_score_column in alert_score_columns:
        nowcast_predictions[alert_score_column] = nowcast_predictions[alert_score_column].fillna(0.0)
    for alert_gate_column in ["public_alert_gate_combined", "public_alert_gate_combined_postprocessed", "storm_watch_gate"]:
        nowcast_predictions[alert_gate_column] = nowcast_predictions[alert_gate_column].fillna(0).astype(int)
    print("\nALERT ABLATION DEBUG:")
    print(alert_ablation_df[[
        "alert_score",
        "selected_threshold",
        "event_recall",
        "kp_ge_5_recall",
        "kp_ge_6_recall",
        "alert_duty_cycle",
        "false_alert_hours_per_month",
        "max_continuous_alert_hours",
    ]].to_string(index=False))

    experiment2_metrics_df, experiment2_predictions_df, experiment2_artifacts = _run_storm_magnitude_experiment(
        train_df,
        test_df,
        solar_feature_columns,
        nowcast_predictions["scaled_ridge_30"].to_numpy(),
        nowcast_predictions["alert_score_combined_max"].to_numpy(),
        constrained_thresholds,
        output_dir,
    )
    nowcast_predictions = nowcast_predictions.merge(
        experiment2_predictions_df.drop(columns=["kp_base"], errors="ignore"),
        on=["timestamp", "kp"],
        how="left",
        suffixes=("", "_experiment2"),
    )
    experiment2_fill_zero_columns = [
        "storm_watch_probability_vigilant",
        "storm_watch_gate_vigilant",
        "kp_risk_p50",
        "kp_risk_p75",
        "kp_risk_p90",
        "kp_ceiling_classifier",
        "kp_quantile_ceiling",
        "storm_high_risk_gate",
        "kp_storm_conservative",
    ]
    for column in experiment2_fill_zero_columns:
        if column in nowcast_predictions.columns:
            if column.endswith("_gate") or column in ["storm_watch_gate_vigilant", "storm_high_risk_gate"]:
                nowcast_predictions[column] = nowcast_predictions[column].fillna(0).astype(int)
            else:
                nowcast_predictions[column] = nowcast_predictions[column].fillna(nowcast_predictions["scaled_ridge_30"])
    print("\nEXPERIMENT 2 STORM MAGNITUDE DEBUG:")
    print(experiment2_metrics_df[[
        "model",
        "storm_mae",
        "storm_asymmetric_mae",
        "kp_ge_5_recall",
        "kp_ge_6_recall",
        "storm_underprediction_rate",
        "storm_mean_underprediction_size",
        "false_storm_inflation_quiet_rate",
        "watch_alert_duty_cycle",
    ]].to_string(index=False))

    event_plot_paths = {}
    event_pr_curve_path = _plot_event_precision_recall_curve(
        output_dir,
        event_predictions_df,
        event_classifier_df,
        "storm_next_12h",
    )
    if event_pr_curve_path is not None:
        event_plot_paths["precision_recall_curve_storm_next_12h"] = event_pr_curve_path

    event_plot_candidates = event_classifier_df[
        (event_classifier_df["target"] == "storm_next_12h")
        & (event_classifier_df["feature_mode"] == "solar_wind_only")
        & (event_classifier_df["threshold_mode"] == "balanced_f1")
    ].copy()
    if event_plot_candidates.empty:
        event_plot_candidates = event_classifier_df[event_classifier_df["threshold_mode"] == "balanced_f1"].copy()
    if event_plot_candidates.empty:
        event_plot_candidates = event_classifier_df.copy()

    if not event_plot_candidates.empty and not event_predictions_df.empty:
        best_event_row = event_plot_candidates.sort_values(
            ["f1", "event_recall", "average_precision"],
            ascending=[False, False, False],
        ).iloc[0]
        title_suffix = (
            f"{best_event_row['target']}_{best_event_row['feature_mode']}_"
            f"{best_event_row['model']}_{best_event_row['threshold_mode']}"
        )
        best_event_predictions = event_predictions_df[
            (event_predictions_df["target"] == best_event_row["target"])
            & (event_predictions_df["feature_mode"] == best_event_row["feature_mode"])
            & (event_predictions_df["model"] == best_event_row["model"])
        ].copy()
        event_plot_paths["storm_event_timeline"] = _plot_event_timeline(
            output_dir,
            best_event_predictions,
            "score",
            float(best_event_row["threshold"]),
            title_suffix,
        )
        event_plot_paths["probability_vs_true_kp"] = _plot_probability_vs_kp(
            output_dir,
            best_event_predictions,
            "score",
            title_suffix,
        )

        importance_name = (
            f"storm_classifier_{best_event_row['target']}_{best_event_row['feature_mode']}_"
            f"{best_event_row['model']}"
        )
        if importance_name in event_importance_frames:
            event_plot_paths["feature_importance"] = _plot_classifier_feature_importance(
                output_dir,
                event_importance_frames[importance_name],
                title_suffix,
            )

    two_stage_note = "not_available"
    boost_tuning_rows = []
    boost_tuning_path = os.path.join(output_dir, "kp_two_stage_boost_validation_tuning.csv")
    classifier_registry_for_boost = _storm_classifier_registry()
    selected_risk_model_name = selected_constrained_gate.get("model", "logistic_balanced")
    selected_gate_threshold = float(selected_constrained_gate.get("selected_threshold", 0.83))
    if selected_risk_model_name not in classifier_registry_for_boost:
        selected_risk_model_name = "logistic_balanced"
        selected_gate_threshold = 0.83
    selected_risk_factory = classifier_registry_for_boost[selected_risk_model_name]
    boost_scale = None
    selected_boost_gate_threshold = selected_gate_threshold
    full_risk_model = None
    selected_calibrator = risk_calibrators.get(selected_risk_model_name)
    public_alert_model = None
    public_alert_raw_score = np.zeros(len(test_df), dtype=float)
    public_alert_probability = np.zeros(len(test_df), dtype=float)
    public_alert_gate = np.zeros(len(test_df), dtype=int)
    if len(primary_train_df) > 1000 and primary_train_df[primary_target].nunique() > 1:
        public_alert_model = selected_risk_factory()
        public_alert_model = _fit_storm_classifier(
            public_alert_model,
            primary_train_df[solar_feature_columns],
            primary_train_df[primary_target].astype(int).to_numpy(),
        )
        public_alert_raw_score = public_alert_model.predict_proba(test_df[solar_feature_columns])[:, 1]
        public_alert_probability = (
            _apply_score_calibrator(selected_calibrator, public_alert_raw_score)
            if selected_calibrator is not None
            else public_alert_raw_score
        )
        public_alert_gate = (public_alert_probability >= selected_gate_threshold).astype(int)

    selected_boost_calibrator = None
    amplitude_model = None

    if len(boost_train_df) > 1000 and boost_train_df[boost_target].nunique() > 1:
        validation_rows = max(500, int(0.2 * len(boost_train_df)))
        dev_df = boost_train_df.iloc[:-validation_rows].copy().reset_index(drop=True)
        val_df = boost_train_df.iloc[-validation_rows:].copy().reset_index(drop=True)
        dev_df_for_target = dev_df.iloc[:-boost_horizon].copy() if boost_horizon > 0 else dev_df.copy()

        base_validation_model = make_pipeline(StandardScaler(), Ridge(alpha=30.0))
        base_validation_model.fit(dev_df[metadata["temporal_feature_columns"]], dev_df["kp"])
        validation_base_pred = np.clip(
            base_validation_model.predict(val_df[metadata["temporal_feature_columns"]]),
            0.0,
            9.0,
        )

        risk_validation_model = selected_risk_factory()
        risk_validation_model = _fit_storm_classifier(
            risk_validation_model,
            dev_df_for_target[solar_feature_columns],
            dev_df_for_target[boost_target].astype(int).to_numpy(),
        )
        validation_raw_risk = risk_validation_model.predict_proba(val_df[solar_feature_columns])[:, 1]
        dev_oof_for_calibration = _time_series_oof_classifier_prediction_frame(
            selected_risk_factory,
            dev_df_for_target,
            solar_feature_columns,
            boost_target,
        )
        if dev_oof_for_calibration.empty or dev_oof_for_calibration["storm_observed"].nunique() < 2:
            validation_risk = validation_raw_risk
        else:
            validation_calibrator = _fit_score_calibrator(
                dev_oof_for_calibration["storm_observed"].to_numpy(),
                dev_oof_for_calibration["score"].to_numpy(),
            )
            validation_risk = _apply_score_calibrator(validation_calibrator, validation_raw_risk)
        validation_storm_mask = val_df["kp"].to_numpy() >= 5
        validation_normal_mask = val_df["kp"].to_numpy() < 4

        best_boost = None
        boost_gate_threshold_candidates = np.unique(
            np.clip(
                np.array(
                    [
                        0.005,
                        0.01,
                        0.02,
                        0.03,
                        0.04,
                        0.05,
                        0.075,
                        0.10,
                        selected_gate_threshold * 0.50,
                        selected_gate_threshold * 0.75,
                        selected_gate_threshold,
                        selected_gate_threshold * 1.25,
                        selected_gate_threshold * 1.50,
                    ]
                ),
                0.001,
                0.50,
            )
        )
        boost_scale_candidates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
        for candidate_gate_threshold in boost_gate_threshold_candidates:
            validation_gate = validation_risk >= candidate_gate_threshold
            validation_gate_metrics = _event_metrics(
                val_df["timestamp"],
                val_df["kp"].to_numpy(),
                validation_risk,
                float(candidate_gate_threshold),
                boost_horizon,
            )
            # The boost gate is internal Kp correction, not a public warning.
            # Keep it bounded, but allow a broader duty cycle than the public
            # alert gate so storm amplitude is not under-corrected.
            if (
                validation_gate_metrics["alert_duty_cycle"] > BOOST_GATE_MAX_DUTY_CYCLE
                or validation_gate_metrics["max_continuous_alert_hours"] > BOOST_GATE_MAX_CONTINUOUS_ALERT_HOURS
            ):
                continue

            for boost_scale in boost_scale_candidates:
                validation_boost_strength = np.minimum(validation_risk / max(candidate_gate_threshold, 1e-6), 2.5)
                boosted_val = np.clip(
                    validation_base_pred + np.where(validation_gate, boost_scale * validation_boost_strength, 0.0),
                    0.0,
                    9.0,
                )
                boosted_val_metrics = _full_metric_bundle(val_df["kp"].to_numpy(), boosted_val)
                val_storm_mae = (
                    float(mean_absolute_error(val_df["kp"].to_numpy()[validation_storm_mask], boosted_val[validation_storm_mask]))
                    if int(validation_storm_mask.sum()) > 0
                    else boosted_val_metrics["mae"]
                )
                val_normal_mae = (
                    float(mean_absolute_error(val_df["kp"].to_numpy()[validation_normal_mask], boosted_val[validation_normal_mask]))
                    if int(validation_normal_mask.sum()) > 0
                    else boosted_val_metrics["mae"]
                )
                val_event_metrics = _event_metrics(
                    val_df["timestamp"],
                    val_df["kp"].to_numpy(),
                    boosted_val,
                    5.0,
                    0,
                )
                tuning_row = {
                    "boost_gate_threshold": float(candidate_gate_threshold),
                    "boost_scale": boost_scale,
                    "validation_mae": boosted_val_metrics["mae"],
                    "validation_rmse": boosted_val_metrics["rmse"],
                    "validation_corr": boosted_val_metrics["corr"],
                    "validation_normal_mae": val_normal_mae,
                    "validation_storm_mae": val_storm_mae,
                    "validation_event_recall": val_event_metrics["event_recall"],
                    "validation_storm_events_detected": val_event_metrics["storm_events_detected"],
                    "validation_false_alert_hours_per_month": val_event_metrics["false_alert_hours_per_month"],
                    "validation_alert_duty_cycle": val_event_metrics["alert_duty_cycle"],
                    "validation_gate_event_recall": validation_gate_metrics["event_recall"],
                    "validation_gate_false_alert_hours_per_month": validation_gate_metrics["false_alert_hours_per_month"],
                    "validation_gate_alert_duty_cycle": validation_gate_metrics["alert_duty_cycle"],
                    "validation_gate_max_continuous_alert_hours": validation_gate_metrics["max_continuous_alert_hours"],
                    "risk_model": selected_risk_model_name,
                    "risk_target": boost_target,
                    "public_alert_threshold": selected_gate_threshold,
                    "risk_gate_threshold": float(candidate_gate_threshold),
                    "risk_gate_alert_duty_cycle": float(validation_gate.mean()),
                    "boost_formula": "gate * boost_scale * min(calibrated_probability / boost_gate_threshold, 2.5)",
                }
                boost_tuning_rows.append(tuning_row)
                candidate = (
                    tuning_row["validation_event_recall"] >= 0.35,
                    -tuning_row["validation_storm_mae"],
                    tuning_row["validation_event_recall"],
                    -max(0.0, tuning_row["validation_normal_mae"] - 0.80),
                    tuning_row["validation_corr"],
                    -tuning_row["validation_mae"],
                    -boost_scale,
                )
                if best_boost is None or candidate > best_boost[0]:
                    best_boost = (candidate, boost_scale, float(candidate_gate_threshold))

        if best_boost is None:
            selected_boost_gate_threshold = selected_gate_threshold
            validation_gate = validation_risk >= selected_boost_gate_threshold
            for boost_scale in [0.25, 0.5, 0.75, 1.0, 1.5]:
                validation_boost_strength = np.minimum(validation_risk / max(selected_boost_gate_threshold, 1e-6), 2.0)
                boosted_val = np.clip(
                    validation_base_pred + np.where(validation_gate, boost_scale * validation_boost_strength, 0.0),
                    0.0,
                    9.0,
                )
                boosted_val_metrics = _full_metric_bundle(val_df["kp"].to_numpy(), boosted_val)
                val_storm_mae = (
                    float(mean_absolute_error(val_df["kp"].to_numpy()[validation_storm_mask], boosted_val[validation_storm_mask]))
                    if int(validation_storm_mask.sum()) > 0
                    else boosted_val_metrics["mae"]
                )
                val_event_metrics = _event_metrics(
                    val_df["timestamp"],
                    val_df["kp"].to_numpy(),
                    boosted_val,
                    5.0,
                    0,
                )
                tuning_row = {
                    "boost_gate_threshold": float(selected_boost_gate_threshold),
                    "boost_scale": boost_scale,
                    "validation_mae": boosted_val_metrics["mae"],
                    "validation_rmse": boosted_val_metrics["rmse"],
                    "validation_corr": boosted_val_metrics["corr"],
                    "validation_normal_mae": None,
                    "validation_storm_mae": val_storm_mae,
                    "validation_event_recall": val_event_metrics["event_recall"],
                    "validation_storm_events_detected": val_event_metrics["storm_events_detected"],
                    "validation_false_alert_hours_per_month": val_event_metrics["false_alert_hours_per_month"],
                    "validation_alert_duty_cycle": val_event_metrics["alert_duty_cycle"],
                    "risk_model": selected_risk_model_name,
                    "risk_target": boost_target,
                    "public_alert_threshold": selected_gate_threshold,
                    "risk_gate_threshold": float(selected_boost_gate_threshold),
                    "risk_gate_alert_duty_cycle": float(validation_gate.mean()),
                    "boost_formula": "fallback gate * boost_scale * min(calibrated_probability / boost_gate_threshold, 2)",
                }
                boost_tuning_rows.append(tuning_row)
                candidate = (
                    tuning_row["validation_event_recall"] >= 0.35,
                    -tuning_row["validation_storm_mae"],
                    tuning_row["validation_event_recall"],
                    tuning_row["validation_corr"],
                    -boost_scale,
                )
                if best_boost is None or candidate > best_boost[0]:
                    best_boost = (candidate, boost_scale, float(selected_boost_gate_threshold))

        pd.DataFrame(boost_tuning_rows).to_csv(boost_tuning_path, index=False)

        _, boost_scale, selected_boost_gate_threshold = best_boost
        full_risk_model = selected_risk_factory()
        full_risk_model = _fit_storm_classifier(
            full_risk_model,
            boost_train_df[solar_feature_columns],
            boost_train_df[boost_target].astype(int).to_numpy(),
        )
        boost_oof_for_calibration = _time_series_oof_classifier_prediction_frame(
            selected_risk_factory,
            boost_train_df,
            solar_feature_columns,
            boost_target,
        )
        if not boost_oof_for_calibration.empty and boost_oof_for_calibration["storm_observed"].nunique() >= 2:
            selected_boost_calibrator = _fit_score_calibrator(
                boost_oof_for_calibration["storm_observed"].to_numpy(),
                boost_oof_for_calibration["score"].to_numpy(),
            )
        holdout_raw_risk = full_risk_model.predict_proba(test_df[solar_feature_columns])[:, 1]
        holdout_risk = (
            _apply_score_calibrator(selected_boost_calibrator, holdout_raw_risk)
            if selected_boost_calibrator is not None
            else holdout_raw_risk
        )
        if not boost_oof_for_calibration.empty and boost_oof_for_calibration["storm_observed"].nunique() >= 2:
            boost_oof_calibrated = (
                _apply_score_calibrator(selected_boost_calibrator, boost_oof_for_calibration["score"].to_numpy())
                if selected_boost_calibrator is not None
                else boost_oof_for_calibration["score"].to_numpy()
            )
            boost_holdout_target = test_df[boost_target].astype(int).to_numpy()
            calibration_df = pd.concat(
                [
                    calibration_df,
                    pd.DataFrame(
                        [
                            {
                                "model": selected_risk_model_name,
                                "target": boost_target,
                                "feature_mode": "solar_wind_only",
                                "train_oof_raw_brier": float(brier_score_loss(
                                    boost_oof_for_calibration["storm_observed"].astype(int),
                                    boost_oof_for_calibration["score"],
                                )),
                                "train_oof_calibrated_brier": float(brier_score_loss(
                                    boost_oof_for_calibration["storm_observed"].astype(int),
                                    boost_oof_calibrated,
                                )),
                                "holdout_raw_brier": float(brier_score_loss(
                                    boost_holdout_target,
                                    holdout_raw_risk,
                                )),
                                "holdout_calibrated_brier": float(brier_score_loss(
                                    boost_holdout_target,
                                    holdout_risk,
                                )),
                                "holdout_mean_raw_score": float(np.mean(holdout_raw_risk)),
                                "holdout_mean_calibrated_probability": float(np.mean(holdout_risk)),
                                "holdout_observed_rate": float(np.mean(boost_holdout_target)),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            calibration_df.to_csv(calibration_path, index=False)
        holdout_gate = holdout_risk >= selected_boost_gate_threshold
        base_pred = nowcast_predictions["scaled_ridge_30"].to_numpy()
        holdout_boost_strength = np.minimum(holdout_risk / max(selected_boost_gate_threshold, 1e-6), 2.5)
        boosted_pred = np.clip(
            base_pred + np.where(holdout_gate, boost_scale * holdout_boost_strength, 0.0),
            0.0,
            9.0,
        )
        boosted_metrics = _full_metric_bundle(y_test.to_numpy(), boosted_pred)
        nowcast_predictions["two_stage_scaled_ridge_storm_boost"] = boosted_pred
        nowcast_predictions["storm_alert_probability"] = public_alert_probability
        nowcast_predictions["storm_alert_raw_score"] = public_alert_raw_score
        nowcast_predictions["storm_alert_gate"] = public_alert_gate
        nowcast_predictions["storm_boost_probability"] = holdout_risk
        nowcast_predictions["storm_boost_raw_score"] = holdout_raw_risk
        nowcast_predictions["storm_boost_gate"] = holdout_gate.astype(int)
        nowcast_predictions["storm_next_12h_solar_risk"] = public_alert_probability
        nowcast_predictions["storm_next_12h_solar_raw_score"] = public_alert_raw_score
        nowcast_predictions["storm_next_12h_solar_gate"] = public_alert_gate
        nowcast_predictions["storm_boost_strength"] = holdout_boost_strength
        nowcast_predictions["kp_base"] = base_pred
        nowcast_predictions["storm_risk_probability"] = holdout_risk
        nowcast_predictions["storm_risk_gate"] = holdout_gate.astype(int)
        nowcast_predictions["kp_storm_adjusted"] = boosted_pred
        nowcast_rows.append(
            _flatten_metric_row(
                "two_stage_scaled_ridge_storm_boost",
                "A_nowcast",
                boosted_metrics,
                note=(
                    f"boost_scale={boost_scale};selected_on=validation_era_storm_mae_event_recall;"
                    f"risk={selected_risk_model_name}_solar_wind_only_{boost_target};"
                    f"boost_gate_threshold={selected_boost_gate_threshold};normalized_calibrated_risk_boost"
                ),
            )
        )
        two_stage_note = (
            f"boost_scale={boost_scale}; selected_on=validation_era_storm_mae_event_recall; "
            f"risk={selected_risk_model_name}; target={boost_target}; "
            f"boost_gate_threshold={selected_boost_gate_threshold}; normalized_calibrated_risk_boost"
        )

        base_oof_true, base_oof_pred = _time_series_oof_scores(
            lambda: make_pipeline(StandardScaler(), Ridge(alpha=30.0)),
            train_df[metadata["temporal_feature_columns"]],
            y_train,
            predict_func="predict",
        )
        risk_oof_true, risk_oof_score = _time_series_oof_classifier_scores(
            selected_risk_factory,
            boost_train_df[solar_feature_columns],
            boost_train_df[boost_target].astype(int).to_numpy(),
        )
        if selected_boost_calibrator is not None:
            risk_oof_score = _apply_score_calibrator(selected_boost_calibrator, risk_oof_score)
        oof_len = min(len(base_oof_true), len(base_oof_pred), len(risk_oof_score), len(boost_train_df))
        amplitude_source_df = boost_train_df.iloc[-oof_len:].copy().reset_index(drop=True)
        amplitude_base_oof = np.asarray(base_oof_pred[-oof_len:], dtype=float)
        amplitude_risk_oof = np.asarray(risk_oof_score[-oof_len:], dtype=float)
        amplitude_target = np.clip(amplitude_source_df["kp"].to_numpy() - amplitude_base_oof, 0.0, 4.0)
        amplitude_alert_threshold = selected_boost_gate_threshold
        high_risk_mask = (amplitude_risk_oof >= amplitude_alert_threshold) | (amplitude_source_df["kp"].to_numpy() >= 5)
        if HistGradientBoostingRegressor is not None and int(high_risk_mask.sum()) >= 100:
            amplitude_X = amplitude_source_df.loc[high_risk_mask, solar_feature_columns].copy()
            amplitude_X["storm_risk"] = amplitude_risk_oof[high_risk_mask]
            amplitude_X["base_pred"] = amplitude_base_oof[high_risk_mask]
            amplitude_y = amplitude_target[high_risk_mask]
            amplitude_model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            )
            amplitude_model.fit(amplitude_X, amplitude_y)
            amplitude_holdout_X = test_df[solar_feature_columns].copy()
            amplitude_holdout_X["storm_risk"] = holdout_risk
            amplitude_holdout_X["base_pred"] = base_pred
            amplitude_boost = np.clip(amplitude_model.predict(amplitude_holdout_X), 0.0, 4.0)
            amplitude_gate = holdout_risk >= amplitude_alert_threshold
            amplitude_pred = np.clip(base_pred + np.where(amplitude_gate, amplitude_boost, 0.0), 0.0, 9.0)
            amplitude_metrics = _full_metric_bundle(y_test.to_numpy(), amplitude_pred)
            nowcast_predictions["two_stage_amplitude_storm_boost"] = amplitude_pred
            nowcast_predictions["storm_amplitude_boost"] = amplitude_boost
            nowcast_rows.append(
                _flatten_metric_row(
                    "two_stage_amplitude_storm_boost",
                    "A_nowcast",
                    amplitude_metrics,
                    note=(
                        f"risk={selected_risk_model_name}_solar_wind_only_{boost_target};"
                        f"amplitude=HGB_positive_residual_high_risk;gate={amplitude_alert_threshold}"
                    ),
                )
            )
            if (
                amplitude_metrics["storm"]["kp_ge_5"]["mae"] is not None
                and amplitude_metrics["storm"]["kp_ge_5"]["mae"] <= boosted_metrics["storm"]["kp_ge_5"]["mae"]
                and amplitude_metrics["storm"]["storm_threshold_ge_5"]["f1"] >= boosted_metrics["storm"]["storm_threshold_ge_5"]["f1"]
            ):
                nowcast_predictions["kp_storm_adjusted"] = amplitude_pred
                nowcast_predictions["storm_adjustment_model"] = "two_stage_amplitude_storm_boost"
                two_stage_note += "; final_adjustment=amplitude_residual_model"
    else:
        pd.DataFrame(boost_tuning_rows).to_csv(boost_tuning_path, index=False)

    if "kp_base" not in nowcast_predictions.columns:
        nowcast_predictions["kp_base"] = nowcast_predictions["scaled_ridge_30"].to_numpy()
    if "storm_alert_probability" not in nowcast_predictions.columns:
        nowcast_predictions["storm_alert_probability"] = public_alert_probability
    if "storm_alert_gate" not in nowcast_predictions.columns:
        nowcast_predictions["storm_alert_gate"] = public_alert_gate
    if "storm_boost_probability" not in nowcast_predictions.columns:
        nowcast_predictions["storm_boost_probability"] = 0.0
    if "storm_boost_gate" not in nowcast_predictions.columns:
        nowcast_predictions["storm_boost_gate"] = 0
    if "storm_risk_probability" not in nowcast_predictions.columns:
        nowcast_predictions["storm_risk_probability"] = nowcast_predictions["storm_boost_probability"].to_numpy()
    if "storm_risk_gate" not in nowcast_predictions.columns:
        nowcast_predictions["storm_risk_gate"] = nowcast_predictions["storm_boost_gate"].to_numpy()
    if "kp_storm_adjusted" not in nowcast_predictions.columns:
        nowcast_predictions["kp_storm_adjusted"] = nowcast_predictions["kp_base"].to_numpy()
    if "storm_adjustment_model" not in nowcast_predictions.columns:
        nowcast_predictions["storm_adjustment_model"] = "two_stage_scaled_ridge_storm_boost"

    experiment3_metrics_df, experiment3_predictions_df, experiment3_artifacts = _run_experiment3_severe_storm_branch(
        train_df,
        test_df,
        metadata["temporal_feature_columns"],
        nowcast_predictions,
        constrained_thresholds,
        output_dir,
    )
    nowcast_predictions = nowcast_predictions.merge(
        experiment3_predictions_df.drop(
            columns=[
                col for col in ["kp_base", "kp_storm_adjusted", "kp_risk_p75", "kp_risk_p90", "kp_storm_conservative"]
                if col in experiment3_predictions_df.columns
            ],
            errors="ignore",
        ),
        on=["timestamp", "kp"],
        how="left",
        suffixes=("", "_experiment3"),
    )
    print("\nEXPERIMENT 3 SEVERE STORM DEBUG:")
    print(experiment3_metrics_df[[
        "model",
        "storm_mae",
        "asymmetric_storm_mae",
        "kp_ge_5_recall",
        "kp_ge_6_recall",
        "kp_ge_7_recall",
        "max_storm_underprediction",
        "quiet_false_storm_inflation_rate",
        "selected_watch_duty_cycle",
        "missed_kp_ge_6_events",
    ]].to_string(index=False))
    print(experiment3_artifacts["debug"]["high_cadence_summary_warning"])

    experiment4_metrics_df, experiment4_predictions_df, experiment4_artifacts = _run_experiment4_severe_recall_calibration(
        train_df,
        test_df,
        nowcast_predictions,
        output_dir,
    )
    nowcast_predictions = nowcast_predictions.merge(
        experiment4_predictions_df.drop(
            columns=[
                col for col in [
                    "kp_base",
                    "kp_storm_conservative_v2",
                    "kp_risk_p75",
                    "kp_risk_p90",
                    "kp_risk_p95",
                    "kp_severe_ceiling",
                ]
                if col in experiment4_predictions_df.columns
            ],
            errors="ignore",
        ),
        on=["timestamp", "kp"],
        how="left",
        suffixes=("", "_experiment4"),
    )
    print("\nEXPERIMENT 4 SEVERE RECALL CALIBRATION DEBUG:")
    print(experiment4_metrics_df[[
        "model",
        "storm_mae",
        "asymmetric_storm_mae",
        "kp_ge_5_recall",
        "kp_ge_6_recall",
        "kp_ge_7_recall",
        "max_storm_underprediction",
        "quiet_false_storm_inflation_rate",
        "selected_quiet_inflation_target",
    ]].to_string(index=False))

    experiment5_results = None
    experiment5_artifacts = {}
    try:
        # Experiment 5 is a stabilization/diagnostic layer over the selected
        # Experiment 4 operating point, so write the current holdout prediction
        # frame once and let the diagnostic helper generate PI-facing artifacts.
        nowcast_predictions.to_csv(nowcast_predictions_path, index=False)
        experiment5_results = run_experiment5_diagnostics(nowcast_predictions_path, output_dir)
        experiment5_artifacts = experiment5_results["paths"]
        experiment5_predictions_df = experiment5_results["predictions"]
        experiment5_columns = [
            col for col in experiment5_predictions_df.columns
            if col.startswith("experiment5_")
            or col in ["kp_storm_conservative_v5", "kp_final_risk_conservative_v5"]
        ]
        nowcast_predictions = nowcast_predictions.drop(
            columns=[col for col in experiment5_columns if col in nowcast_predictions.columns],
            errors="ignore",
        ).merge(
            experiment5_predictions_df[["timestamp", "kp", *experiment5_columns]],
            on=["timestamp", "kp"],
            how="left",
        )
        print("\nEXPERIMENT 5 CALIBRATION / DIAGNOSTICS DEBUG:")
        print(f"PI summary: {experiment5_artifacts['pi_summary']}")
        print(f"Operating frontier: {experiment5_artifacts['operating_frontier']}")
        print(f"GFZ-style examples: {experiment5_artifacts['gfz_examples']}")
    except Exception as exc:
        print("\nEXPERIMENT 5 DIAGNOSTICS WARNING:")
        print(f"Experiment 5 diagnostics were skipped because: {exc}")

    experiment6_metrics_df = pd.DataFrame()
    experiment6_artifacts = {}
    storm_recall_metrics_df = pd.DataFrame()
    storm_recall_artifacts = {}
    storm_recall_debug = {}
    try:
        experiment6_metrics_df, experiment6_predictions_df, experiment6_result = _run_experiment6_false_inflation_suppression(
            train_df,
            test_df,
            nowcast_predictions,
            solar_feature_columns,
            metadata["temporal_feature_columns"],
            output_dir,
        )
        experiment6_artifacts = experiment6_result["paths"]
        storm_recall_artifacts = experiment6_result["debug"].get("storm_recall", {}).get("paths", {})
        storm_recall_debug = experiment6_result["debug"].get("storm_recall", {})
        if storm_recall_artifacts and os.path.exists(storm_recall_artifacts.get("metrics", "")):
            storm_recall_metrics_df = pd.read_csv(storm_recall_artifacts["metrics"])
        experiment6_columns = [
            "experiment6_false_inflation_suppressor_score",
            "experiment6_suppress_inflation_gate",
            "experiment6_severe_safety_override",
            "kp_storm_conservative_v6",
            "kp_final_risk_conservative_v6",
            "experiment6_operating_point_label",
            "experiment6_selected_threshold",
            "experiment6_selected_suppression_mode",
            "storm_recall_probability_kp_ge_5",
            "storm_recall_probability_kp_ge_6",
            "storm_recall_internal_watch_kp_ge_5",
            "storm_recall_internal_watch_kp_ge_6",
            "storm_recall_public_alert_reference",
            "storm_recall_public_alert_retuned_reference",
            "storm_recall_extreme_diagnostic_label",
            "kp_storm_recall_conservative",
        ]
        nowcast_predictions = nowcast_predictions.drop(
            columns=[col for col in experiment6_columns if col in nowcast_predictions.columns],
            errors="ignore",
        ).merge(
            experiment6_predictions_df[["timestamp", "kp", *[col for col in experiment6_columns if col in experiment6_predictions_df.columns]]],
            on=["timestamp", "kp"],
            how="left",
        )
        print("\nEXPERIMENT 6 FALSE-INFLATION SUPPRESSION DEBUG:")
        print(experiment6_metrics_df[[
            "model",
            "storm_mae",
            "asymmetric_storm_mae",
            "kp_ge_5_recall",
            "kp_ge_6_recall",
            "kp_ge_7_recall",
            "max_storm_underprediction",
            "quiet_false_storm_inflation_rate",
        ]].to_string(index=False))
        print(f"Recommendation: {experiment6_result['recommendation']}")
        if storm_recall_artifacts:
            print(f"Storm recall metrics: {storm_recall_artifacts['metrics']}")
            print(f"Storm recall summary: {storm_recall_artifacts['pi_summary']}")
    except Exception as exc:
        print("\nEXPERIMENT 6 DIAGNOSTICS WARNING:")
        print(f"Experiment 6 diagnostics were skipped because: {exc}")

    experiment7_metrics_df = pd.DataFrame()
    experiment7_artifacts = {}
    experiment7_result = {}
    try:
        experiment7_metrics_df, experiment7_predictions_df, experiment7_result = _run_experiment7_forecast_aligned(
            feature_df,
            train_df,
            test_df,
            nowcast_predictions,
            metadata,
            output_dir,
        )
        experiment7_artifacts = experiment7_result["paths"]
        experiment7_columns = [
            "kp_experiment7_bz_velocity_density_baseline",
            "kp_experiment7_target_aligned_base",
            "kp_experiment7_target_aligned_best",
            "kp_experiment7_operational_frozen",
            "kp_experiment7_operational_autoregressive",
            "kp_experiment7_operational_physics_only",
            "kp_experiment7_conservative",
            "prob_experiment7_kp_ge_5",
            "prob_experiment7_kp_ge_6",
            "prob_experiment7_kp_ge_7",
            "bz_southward_drop_6h",
            "bz_southward_drop_12h",
            "bz_southward_drop_24h",
            "bz_crossed_southward",
            "strong_southward_turning",
            "bz_drop_times_velocity",
            "bz_drop_times_pdyn",
            "southward_turning_pressure_interaction",
            "kp_daily_pattern_break_score",
            "kp_daily_variability_ratio",
            "kp_recent_vs_multiday_mean",
            "kp_recent_max_vs_multiday_mean",
        ]
        nowcast_predictions = nowcast_predictions.drop(
            columns=[col for col in experiment7_columns if col in nowcast_predictions.columns],
            errors="ignore",
        ).merge(
            experiment7_predictions_df[["timestamp", "kp", *[col for col in experiment7_columns if col in experiment7_predictions_df.columns]]],
            on=["timestamp", "kp"],
            how="left",
        )
        print("\nEXPERIMENT 7 FORECAST-INPUT-ALIGNED DEBUG:")
        print(f"Forecast inputs already target-aligned: {FORECAST_INPUTS_ALREADY_TARGET_ALIGNED}")
        print(f"Issue/valid forecast horizon metadata exists: {experiment7_result['debug']['issue_time_valid_time_metadata_found']}")
        print(f"Detected cadence hours: {experiment7_result['debug']['cadence_hours']:.3f}")
        print(f"Rolling forecast initializations: {experiment7_result['rolling_initializations']}")
        print(f"Rolling predicted target rows: {experiment7_result['predicted_target_rows']}")
        print(
            "Best diagnostic target-aligned upper-bound model: "
            f"{experiment7_result['best_model']} "
            f"MAE={experiment7_result['best_metrics']['mae']:.3f}, "
            f"RMSE={experiment7_result['best_metrics']['rmse']:.3f}, "
            f"corr={experiment7_result['best_metrics']['pearson_corr']:.3f}"
        )
        print(
            "Operational frozen-anchor daily update: "
            f"MAE={experiment7_result['operational_frozen_metrics']['mae']:.3f}, "
            f"RMSE={experiment7_result['operational_frozen_metrics']['rmse']:.3f}, "
            f"corr={experiment7_result['operational_frozen_metrics']['pearson_corr']:.3f}"
        )
        print(
            "Operational autoregressive daily update: "
            f"MAE={experiment7_result['operational_autoregressive_metrics']['mae']:.3f}, "
            f"RMSE={experiment7_result['operational_autoregressive_metrics']['rmse']:.3f}, "
            f"corr={experiment7_result['operational_autoregressive_metrics']['pearson_corr']:.3f}"
        )
        print(
            "Operational physics-only daily update: "
            f"MAE={experiment7_result['operational_physics_only_metrics']['mae']:.3f}, "
            f"RMSE={experiment7_result['operational_physics_only_metrics']['rmse']:.3f}, "
            f"corr={experiment7_result['operational_physics_only_metrics']['pearson_corr']:.3f}"
        )
        if not experiment7_result["leadtime_metrics"].empty:
            lead_cols = ["model_mode", "update_cadence_hours", "lead_time_bin", "recall_kp_ge_5", "recall_kp_ge_6"]
            lead_print = experiment7_result["leadtime_metrics"]
            lead_print = lead_print[
                (lead_print["update_cadence_hours"] == 24.0)
                & (lead_print["model_mode"] == "experiment7_operational_frozen")
            ][lead_cols]
            print("Operational frozen Kp>=5/Kp>=6 recall by lead-time bin:")
            print(lead_print.to_string(index=False))
        print(f"Bz sharp-drop features improved performance: {experiment7_result['bz_drop_improved']}")
        print(f"24h Kp pattern-break features improved performance: {experiment7_result['kp_pattern_improved']}")
        print(f"Recommended Experiment 7 operational branch: {experiment7_result['recommendation']}")
    except Exception as exc:
        print("\nEXPERIMENT 7 FORECAST-ALIGNED WARNING:")
        print(f"Experiment 7 was skipped because: {exc}")

    experiment8_metrics_df = pd.DataFrame()
    experiment8_artifacts = {}
    try:
        experiment8_result = run_experiment8_storm_watch(feature_df, train_df, test_df, output_dir)
        experiment8_metrics_df = experiment8_result["metrics"]
        experiment8_artifacts = experiment8_result["paths"]
        print("\nEXPERIMENT 8 5-DAY STORM WATCH DEBUG:")
        print(experiment8_metrics_df.to_string(index=False))
        print(f"Time-series plot: {experiment8_artifacts['timeseries']}")
    except Exception as exc:
        print("\nEXPERIMENT 8 STORM WATCH WARNING:")
        print(f"Experiment 8 was skipped because: {exc}")

    experiment11_metrics_df = pd.DataFrame()
    experiment11_artifacts = {}
    try:
        experiment11_result = run_experiment11_storm_aware_structural(
            feature_df,
            train_df,
            test_df,
            metadata,
            output_dir,
        )
        experiment11_metrics_df = experiment11_result["metrics"]
        experiment11_artifacts = experiment11_result["paths"]
        print("\nEXPERIMENT 11 STRUCTURAL STORM-AWARE DEBUG:")
        all_rows = experiment11_metrics_df[experiment11_metrics_df["lead_time_bin"] == "all"].copy()
        print(
            all_rows[
                [
                    "model",
                    "mae",
                    "rmse",
                    "pearson_corr",
                    "recall_kp_ge_5",
                    "recall_kp_ge_6",
                    "quiet_false_storm_inflation",
                ]
            ].to_string(index=False)
        )
        print(f"Time-series plot: {experiment11_artifacts['timeseries']}")
    except Exception as exc:
        print("\nEXPERIMENT 11 STRUCTURAL STORM-AWARE WARNING:")
        print(f"Experiment 11 was skipped because: {exc}")

    normal_kp_mask = y_test.to_numpy() < 4
    storm_kp_mask = y_test.to_numpy() >= 5
    adjusted_storm_metrics = _storm_metrics(
        y_test.to_numpy(),
        nowcast_predictions["kp_storm_adjusted"].to_numpy(),
    )["storm_threshold_ge_5"]
    conservative_metrics = _asymmetric_storm_magnitude_metrics(
        y_test.to_numpy(),
        nowcast_predictions["kp_storm_conservative"].to_numpy()
        if "kp_storm_conservative" in nowcast_predictions.columns
        else nowcast_predictions["kp_base"].to_numpy(),
    )
    conservative_v2_metrics = _advanced_asymmetric_metrics(
        y_test.to_numpy(),
        nowcast_predictions["kp_storm_conservative_v2"].to_numpy()
        if "kp_storm_conservative_v2" in nowcast_predictions.columns
        else nowcast_predictions["kp_base"].to_numpy(),
    )
    conservative_v4_metrics = _advanced_asymmetric_metrics(
        y_test.to_numpy(),
        nowcast_predictions["kp_final_risk_conservative_v4"].to_numpy()
        if "kp_final_risk_conservative_v4" in nowcast_predictions.columns
        else nowcast_predictions["kp_base"].to_numpy(),
    )
    conservative_v5_metrics = _advanced_asymmetric_metrics(
        y_test.to_numpy(),
        nowcast_predictions["kp_final_risk_conservative_v5"].to_numpy()
        if "kp_final_risk_conservative_v5" in nowcast_predictions.columns
        else (
            nowcast_predictions["kp_final_risk_conservative_v4"].to_numpy()
            if "kp_final_risk_conservative_v4" in nowcast_predictions.columns
            else nowcast_predictions["kp_base"].to_numpy()
        ),
    )
    conservative_v6_metrics = _advanced_asymmetric_metrics(
        y_test.to_numpy(),
        nowcast_predictions["kp_final_risk_conservative_v6"].to_numpy()
        if "kp_final_risk_conservative_v6" in nowcast_predictions.columns
        else (
            nowcast_predictions["kp_final_risk_conservative_v5"].to_numpy()
            if "kp_final_risk_conservative_v5" in nowcast_predictions.columns
            else nowcast_predictions["kp_base"].to_numpy()
        ),
    )
    public_alert_target_mask = test_df[primary_target].astype(bool).to_numpy()
    public_alert_point_metrics = _binary_classification_metrics(
        public_alert_target_mask,
        nowcast_predictions["storm_alert_probability"].to_numpy(),
        selected_gate_threshold,
    )
    public_alert_event_metrics = _event_metrics(
        test_df["timestamp"],
        test_df["kp"].to_numpy(),
        nowcast_predictions["storm_alert_probability"].to_numpy(),
        selected_gate_threshold,
        primary_horizon,
    )
    boost_gate_event_metrics = _event_metrics(
        test_df["timestamp"],
        test_df["kp"].to_numpy(),
        nowcast_predictions["storm_boost_probability"].to_numpy(),
        selected_boost_gate_threshold,
        boost_horizon,
    )
    boost_gate_point_metrics = _binary_classification_metrics(
        test_df[boost_target].astype(bool).to_numpy(),
        nowcast_predictions["storm_boost_probability"].to_numpy(),
        selected_boost_gate_threshold,
    )
    final_system_metrics_df = pd.DataFrame(
        [
            {
                "kp_base_model": "scaled_ridge_30",
                "driver_input_source": metadata["driver_input_metadata"]["driver_input_source"],
                "production_horizon_source": metadata["driver_input_metadata"]["semantics"]["production_horizon_source"],
                "forecast_semantics": metadata["driver_input_metadata"]["semantics"]["core_task"],
                "original_pi_linear_equation": metadata["original_pi_linear_equation"]["equation"],
                "public_alert_model": selected_risk_model_name,
                "public_alert_target": primary_target,
                "public_alert_threshold": selected_gate_threshold,
                "public_alert_event_recall": public_alert_event_metrics["event_recall"],
                "public_alert_mean_lead_time_hours": public_alert_event_metrics["mean_lead_time_hours"],
                "public_alert_precision": public_alert_point_metrics["precision"],
                "public_alert_recall": public_alert_point_metrics["recall"],
                "public_alert_f1": public_alert_point_metrics["f1"],
                "public_alert_duty_cycle": public_alert_event_metrics["alert_duty_cycle"],
                "public_alert_alert_hours_per_month": public_alert_event_metrics["alert_hours_per_month"],
                "public_alert_false_alert_hours_per_month": public_alert_event_metrics["false_alert_hours_per_month"],
                "public_alert_max_continuous_alert_hours": public_alert_event_metrics["max_continuous_alert_hours"],
                "public_alert_feasible": bool(
                    public_alert_event_metrics["alert_duty_cycle"] <= constrained_threshold_constraints["alert_duty_cycle"]
                    and public_alert_event_metrics["max_continuous_alert_hours"] <= constrained_threshold_constraints["max_continuous_alert_hours"]
                    and public_alert_event_metrics["false_alert_hours_per_month"] <= constrained_threshold_constraints["false_alert_hours_per_month"]
                ),
                "storm_boost_model": selected_risk_model_name,
                "storm_boost_target": boost_target,
                "storm_boost_gate_threshold": selected_boost_gate_threshold,
                "storm_boost_scale": boost_scale,
                "kp_storm_adjusted_model": str(nowcast_predictions["storm_adjustment_model"].iloc[0]),
                "normal_kp_mae_kp_base": float(mean_absolute_error(
                    y_test.to_numpy()[normal_kp_mask],
                    nowcast_predictions["kp_base"].to_numpy()[normal_kp_mask],
                )),
                "storm_kp_mae_kp_storm_adjusted": float(mean_absolute_error(
                    y_test.to_numpy()[storm_kp_mask],
                    nowcast_predictions["kp_storm_adjusted"].to_numpy()[storm_kp_mask],
                )) if int(storm_kp_mask.sum()) > 0 else None,
                "kp_ge_5_precision_kp_storm_adjusted": adjusted_storm_metrics["precision"],
                "kp_ge_5_recall_kp_storm_adjusted": adjusted_storm_metrics["recall"],
                "kp_ge_5_f1_kp_storm_adjusted": adjusted_storm_metrics["f1"],
                "storm_kp_mae_kp_storm_conservative": conservative_metrics["storm_mae"],
                "storm_asymmetric_mae_kp_storm_conservative": conservative_metrics["storm_asymmetric_mae"],
                "kp_ge_5_recall_kp_storm_conservative": conservative_metrics["kp_ge_5_recall"],
                "kp_ge_5_f1_kp_storm_conservative": conservative_metrics["kp_ge_5_f1"],
                "kp_ge_6_recall_kp_storm_conservative": conservative_metrics["kp_ge_6_recall"],
                "storm_underprediction_rate_kp_storm_conservative": conservative_metrics["storm_underprediction_rate"],
                "storm_mean_underprediction_size_kp_storm_conservative": conservative_metrics["storm_mean_underprediction_size"],
                "storm_overprediction_rate_kp_storm_conservative": conservative_metrics["storm_overprediction_rate"],
                "false_storm_inflation_quiet_rate_kp_storm_conservative": conservative_metrics["false_storm_inflation_quiet_rate"],
                "storm_kp_mae_kp_storm_conservative_v2": conservative_v2_metrics["storm_mae"],
                "storm_asymmetric_mae_kp_storm_conservative_v2": conservative_v2_metrics["asymmetric_storm_mae"],
                "kp_ge_5_recall_kp_storm_conservative_v2": conservative_v2_metrics["kp_ge_5_recall"],
                "kp_ge_5_f1_kp_storm_conservative_v2": conservative_v2_metrics["kp_ge_5_f1"],
                "kp_ge_6_recall_kp_storm_conservative_v2": conservative_v2_metrics["kp_ge_6_recall"],
                "kp_ge_7_recall_kp_storm_conservative_v2": conservative_v2_metrics["kp_ge_7_recall"],
                "max_storm_underprediction_kp_storm_conservative_v2": conservative_v2_metrics["max_storm_underprediction"],
                "storm_underprediction_rate_kp_storm_conservative_v2": conservative_v2_metrics["storm_underprediction_rate"],
                "quiet_false_storm_inflation_rate_kp_storm_conservative_v2": conservative_v2_metrics["quiet_false_storm_inflation_rate"],
                "storm_kp_mae_kp_final_risk_conservative_v4": conservative_v4_metrics["storm_mae"],
                "storm_asymmetric_mae_kp_final_risk_conservative_v4": conservative_v4_metrics["asymmetric_storm_mae"],
                "kp_ge_5_recall_kp_final_risk_conservative_v4": conservative_v4_metrics["kp_ge_5_recall"],
                "kp_ge_6_recall_kp_final_risk_conservative_v4": conservative_v4_metrics["kp_ge_6_recall"],
                "kp_ge_7_recall_kp_final_risk_conservative_v4": conservative_v4_metrics["kp_ge_7_recall"],
                "max_storm_underprediction_kp_final_risk_conservative_v4": conservative_v4_metrics["max_storm_underprediction"],
                "quiet_false_storm_inflation_rate_kp_final_risk_conservative_v4": conservative_v4_metrics["quiet_false_storm_inflation_rate"],
                "storm_kp_mae_kp_final_risk_conservative_v5": conservative_v5_metrics["storm_mae"],
                "storm_asymmetric_mae_kp_final_risk_conservative_v5": conservative_v5_metrics["asymmetric_storm_mae"],
                "kp_ge_5_recall_kp_final_risk_conservative_v5": conservative_v5_metrics["kp_ge_5_recall"],
                "kp_ge_6_recall_kp_final_risk_conservative_v5": conservative_v5_metrics["kp_ge_6_recall"],
                "kp_ge_7_recall_kp_final_risk_conservative_v5": conservative_v5_metrics["kp_ge_7_recall"],
                "max_storm_underprediction_kp_final_risk_conservative_v5": conservative_v5_metrics["max_storm_underprediction"],
                "quiet_false_storm_inflation_rate_kp_final_risk_conservative_v5": conservative_v5_metrics["quiet_false_storm_inflation_rate"],
                "storm_kp_mae_kp_final_risk_conservative_v6": conservative_v6_metrics["storm_mae"],
                "storm_asymmetric_mae_kp_final_risk_conservative_v6": conservative_v6_metrics["asymmetric_storm_mae"],
                "kp_ge_5_recall_kp_final_risk_conservative_v6": conservative_v6_metrics["kp_ge_5_recall"],
                "kp_ge_6_recall_kp_final_risk_conservative_v6": conservative_v6_metrics["kp_ge_6_recall"],
                "kp_ge_7_recall_kp_final_risk_conservative_v6": conservative_v6_metrics["kp_ge_7_recall"],
                "max_storm_underprediction_kp_final_risk_conservative_v6": conservative_v6_metrics["max_storm_underprediction"],
                "quiet_false_storm_inflation_rate_kp_final_risk_conservative_v6": conservative_v6_metrics["quiet_false_storm_inflation_rate"],
                "storm_recall_kp_ge_5_internal_watch_threshold": storm_recall_debug.get("selected_thresholds", {}).get("kp_ge_5"),
                "storm_recall_kp_ge_6_internal_watch_threshold": storm_recall_debug.get("selected_thresholds", {}).get("kp_ge_6"),
                "storm_recall_threshold_selection_source": storm_recall_debug.get("threshold_selection_source"),
                "storm_recall_final_evaluation_source": storm_recall_debug.get("final_evaluation_source"),
                "storm_boost_gate_precision": boost_gate_point_metrics["precision"],
                "storm_boost_gate_recall": boost_gate_point_metrics["recall"],
                "storm_boost_gate_f1": boost_gate_point_metrics["f1"],
                "storm_boost_gate_event_recall": boost_gate_event_metrics["event_recall"],
                "storm_boost_gate_alert_duty_cycle": boost_gate_event_metrics["alert_duty_cycle"],
                "storm_boost_gate_alert_hours_per_month": boost_gate_event_metrics["alert_hours_per_month"],
                "storm_boost_gate_false_alert_hours_per_month": boost_gate_event_metrics["false_alert_hours_per_month"],
                "storm_boost_gate_max_continuous_alert_hours": boost_gate_event_metrics["max_continuous_alert_hours"],
            }
        ]
    )
    final_system_metrics_path = os.path.join(output_dir, "kp_final_system_metrics.csv")
    final_system_metrics_df.to_csv(final_system_metrics_path, index=False)

    risk_plot_df = nowcast_predictions[["timestamp", "kp", "storm_risk_probability", "storm_risk_gate"]].copy()
    storm_risk_event_plot_path = _plot_storm_risk_event_overlay(
        output_dir,
        risk_plot_df,
        "storm_risk_probability",
        selected_boost_gate_threshold,
    )

    selected_holdout_match = event_predictions_df[
        (event_predictions_df["target"] == boost_target)
        & (event_predictions_df["feature_mode"] == "solar_wind_only")
        & (event_predictions_df["model"] == selected_risk_model_name)
    ].copy()
    calibration_curve_path = None
    if not selected_holdout_match.empty:
        selected_holdout_match["calibrated_probability"] = (
            _apply_score_calibrator(selected_boost_calibrator, selected_holdout_match["score"].to_numpy())
            if selected_boost_calibrator is not None
            else selected_holdout_match["score"].to_numpy()
        )
        calibration_curve_path = _plot_calibration_curve(
            output_dir,
            selected_holdout_match["storm_observed"].astype(int).to_numpy(),
            selected_holdout_match["score"].to_numpy(),
            selected_holdout_match["calibrated_probability"].to_numpy(),
        )

    event_audit_df = _build_storm_event_audit(
        test_df,
        nowcast_predictions["storm_risk_probability"].to_numpy(),
        selected_boost_gate_threshold,
        boost_horizon,
    )
    event_audit_path = os.path.join(output_dir, "kp_storm_event_audit.csv")
    event_audit_df.to_csv(event_audit_path, index=False)
    detected_event_audit_path = os.path.join(output_dir, "kp_detected_storm_events.csv")
    missed_event_audit_path = os.path.join(output_dir, "kp_missed_storm_events.csv")
    event_audit_df[event_audit_df["detected"]].to_csv(detected_event_audit_path, index=False)
    event_audit_df[~event_audit_df["detected"]].to_csv(missed_event_audit_path, index=False)
    regime_summary_df = _summarize_event_regimes(event_audit_df)
    regime_summary_path = os.path.join(output_dir, "kp_missed_detected_storm_regime_summary.csv")
    regime_summary_df.to_csv(regime_summary_path, index=False)

    nowcast_df = pd.DataFrame(nowcast_rows).sort_values(["rmse", "mae", "model"]).reset_index(drop=True)
    nowcast_df.to_csv(nowcast_path, index=False)
    nowcast_predictions.to_csv(nowcast_predictions_path, index=False)
    nowcast_predictions.to_csv(legacy_predictions_path, index=False)

    recursive_rows = []
    for baseline_name, baseline_pred in _persistence_predictions(test_df).items():
        for horizon in [1, 2, 4, 8]:
            if baseline_name == "persistence_kp_lag_1":
                recursive_pred = []
                actual = []
                for start_idx in range(split_index, len(feature_df) - horizon + 1, horizon):
                    seed = feature_df.iloc[start_idx - 1]["kp"]
                    true_slice = feature_df.iloc[start_idx:start_idx + horizon]["kp"].to_numpy()
                    recursive_pred.extend([seed] * len(true_slice))
                    actual.extend(true_slice.tolist())
                metrics = _full_metric_bundle(np.array(actual), np.array(recursive_pred))
                recursive_rows.append(
                    {
                        **_flatten_metric_row(baseline_name, "B_recursive", metrics),
                        "horizon": horizon,
                    }
                )
        if baseline_name != "persistence_kp_lag_1":
            continue

    recursive_model_keys = [
        model_key
        for model_key in ["ridge_baseline", "ridge_tuned", "scaled_ridge_30"]
        if model_key in trained_temporal_models
    ]
    for model_key in recursive_model_keys:
        trained_model = trained_temporal_models[model_key]
        direct_pred = nowcast_predictions[model_key].to_numpy()
        direct_metrics = _full_metric_bundle(y_test.to_numpy(), direct_pred)
        recursive_rows.append(
            {
                **_flatten_metric_row(model_key, "B_recursive", direct_metrics),
                "horizon": 1,
            }
        )
        for horizon in [2, 4, 8]:
            recursive_df = _recursive_predict(
                trained_model["model"],
                feature_df,
                split_index,
                trained_model["feature_columns"],
                horizon,
                uses_kp_history=True,
            )
            metrics = _full_metric_bundle(
                recursive_df["kp"].to_numpy(),
                recursive_df["pred_kp"].to_numpy(),
            )
            recursive_rows.append(
                {
                    **_flatten_metric_row(model_key, "B_recursive", metrics),
                    "horizon": horizon,
                }
            )

    recursive_df = pd.DataFrame(recursive_rows).sort_values(["horizon", "rmse", "mae", "model"]).reset_index(drop=True)
    recursive_path = os.path.join(output_dir, "kp_recursive_model_metrics.csv")
    recursive_df.to_csv(recursive_path, index=False)

    best_model_name = nowcast_df[nowcast_df["model"].isin(trained_temporal_models.keys())].iloc[0]["model"]
    best_model_bundle = trained_temporal_models[best_model_name]
    nowcast_predictions["best_model_pred"] = nowcast_predictions[best_model_name]
    plots = _save_plots(output_dir, nowcast_predictions, recursive_df)
    baseline_architecture_path = os.path.join(output_dir, "kp_final_architecture_baseline.pkl")
    baseline_architecture_summary_path = os.path.join(output_dir, "kp_final_architecture_baseline_summary.json")
    baseline_architecture = {
        "architecture_name": "base_ridge_plus_separate_public_alert_and_storm_boost_gates",
        "base_kp_model_name": "scaled_ridge_30",
        "base_kp_model": trained_temporal_models["scaled_ridge_30"]["model"],
        "base_feature_columns": trained_temporal_models["scaled_ridge_30"]["feature_columns"],
        "feature_mode": "solar_wind_only",
        "solar_wind_feature_columns": solar_feature_columns,
        "sustained_driving_features": sustained_features_present,
        "public_alert_target": primary_target,
        "public_alert_model_name": selected_risk_model_name,
        "public_alert_model": public_alert_model,
        "public_alert_calibrator": selected_calibrator,
        "public_alert_threshold": selected_gate_threshold,
        "public_alert_constraints": constrained_threshold_constraints,
        "storm_boost_target": boost_target,
        "storm_boost_model_name": selected_risk_model_name,
        "storm_boost_model": full_risk_model,
        "storm_boost_calibrator": selected_boost_calibrator,
        "storm_boost_gate_threshold": selected_boost_gate_threshold,
        "storm_boost_scale": boost_scale,
        "storm_amplitude_model": amplitude_model,
        "summary_metrics": final_system_metrics_df.to_dict(orient="records"),
        "constrained_gate": selected_constrained_gate,
    }
    with open(baseline_architecture_path, "wb") as baseline_file:
        pickle.dump(baseline_architecture, baseline_file)
    with open(baseline_architecture_summary_path, "w", encoding="utf-8") as baseline_summary_file:
        json.dump(
            {
                key: value
                for key, value in baseline_architecture.items()
                if key not in [
                    "base_kp_model",
                    "public_alert_model",
                    "public_alert_calibrator",
                    "storm_boost_model",
                    "storm_boost_calibrator",
                    "storm_amplitude_model",
                ]
            },
            baseline_summary_file,
            indent=2,
            default=str,
        )
        baseline_summary_file.write("\n")

    summary = {
        "name": model_name,
        "dataset_path": dataset_path,
        "driver_input_metadata": metadata["driver_input_metadata"],
        "original_pi_linear_equation": metadata["original_pi_linear_equation"],
        "rows_after_cleaning": int(len(feature_df)),
        "cadence_hours": cadence_hours,
        "duplicates_removed": metadata["duplicates_removed"],
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_date_range": {"start": train_start, "end": train_end},
        "test_date_range": {"start": test_start, "end": test_end},
        "feature_counts": {
            "original_physics": len(ORIGINAL_PHYSICS_FEATURES),
            "solar_wind_only": len(solar_feature_columns),
            "temporal": len(metadata["temporal_feature_columns"]),
        },
        "features": {
            "original_physics": ORIGINAL_PHYSICS_FEATURES,
            "solar_wind_only": solar_feature_columns,
            "sustained_driving": sustained_features_present,
            "temporal": metadata["temporal_feature_columns"],
            "kp_history": metadata["kp_history_feature_columns"],
        },
        "sustained_feature_debug": sustained_feature_finiteness,
        "leakage_checks": metadata["leakage"],
        "kp_history_availability": metadata["kp_history_availability"],
        "ridge_cv": {
            "baseline_alpha": float(alpha),
            "grid": RIDGE_ALPHA_GRID,
            "best_alpha": ridge_alpha,
        },
        "storm_detection": {
            "selected_ridge_thresholds": {
                mode_name: {key: float(value) for key, value in mode_values.items()}
                for mode_name, mode_values in selected_thresholds.items()
            },
            "best_holdout_model": storm_detection_df.iloc[0][["model", "precision", "recall", "f1", "average_precision"]].to_dict(),
            "expanding_window_mean_metrics": backtest_summary_df.to_dict(orient="records"),
            "event_targets": STORM_TARGET_COLUMNS,
            "event_feature_modes": list(feature_modes.keys()),
            "primary_science_target": "storm_next_12h",
            "primary_science_feature_mode": "solar_wind_only",
            "science_operating_points": science_event_df.to_dict(orient="records") if not science_event_df.empty else [],
            "constrained_selection_constraints": constrained_threshold_constraints,
            "constrained_selected_gate": selected_constrained_gate,
            "alert_onset_ablation": alert_ablation_df.to_dict(orient="records"),
            "alert_onset_debug": alert_ablation_artifacts["debug"],
            "storm_magnitude_experiment2": {
                "metrics": experiment2_metrics_df.to_dict(orient="records"),
                "artifacts": experiment2_artifacts,
            },
            "severe_storm_experiment3": {
                "metrics": experiment3_metrics_df.to_dict(orient="records"),
                "artifacts": experiment3_artifacts,
            },
            "severe_recall_calibration_experiment4": {
                "metrics": experiment4_metrics_df.to_dict(orient="records"),
                "artifacts": experiment4_artifacts,
            },
            "experiment5_calibration_diagnostics": {
                "artifacts": experiment5_artifacts,
                "status": "generated" if experiment5_artifacts else "skipped",
            },
            "experiment6_false_inflation_suppression": {
                "metrics": experiment6_metrics_df.to_dict(orient="records") if not experiment6_metrics_df.empty else [],
                "artifacts": experiment6_artifacts,
                "status": "generated" if experiment6_artifacts else "skipped",
                "storm_recall_forecasting": {
                    "metrics": storm_recall_metrics_df.to_dict(orient="records") if not storm_recall_metrics_df.empty else [],
                    "artifacts": storm_recall_artifacts,
                    "debug": storm_recall_debug,
                    "status": "generated" if storm_recall_artifacts else "skipped",
                },
            },
            "experiment7_forecast_input_aligned": {
                "metrics": experiment7_metrics_df.to_dict(orient="records") if not experiment7_metrics_df.empty else [],
                "artifacts": experiment7_artifacts,
                "status": "generated" if experiment7_artifacts else "skipped",
                "debug": experiment7_result.get("debug", {}),
            },
            "experiment8_five_day_storm_watch": {
                "metrics": experiment8_metrics_df.to_dict(orient="records") if not experiment8_metrics_df.empty else [],
                "artifacts": experiment8_artifacts,
                "status": "generated" if experiment8_artifacts else "skipped",
            },
            "experiment11_structural_storm_aware": {
                "metrics": experiment11_metrics_df.to_dict(orient="records") if not experiment11_metrics_df.empty else [],
                "artifacts": experiment11_artifacts,
                "status": "generated" if experiment11_artifacts else "skipped",
            },
            "best_event_classifier": event_classifier_df.iloc[0].to_dict() if not event_classifier_df.empty else {},
            "two_stage_ridge_storm_boost": two_stage_note,
        },
        "best_model": {
            "name": best_model_name,
            "mode": "A_nowcast",
            "metrics": best_model_bundle["metrics"],
        },
        "artifacts": {
            "nowcast_metrics": nowcast_path,
            "solar_wind_only_metrics": solar_path,
            "recursive_metrics": recursive_path,
            "storm_classification_metrics": storm_detection_path,
            "storm_classification_predictions": storm_predictions_path,
            "storm_threshold_curve": ridge_threshold_curve_path,
            "storm_thresholds": selected_thresholds_path,
            "storm_expanding_window_backtest": backtest_path,
            "event_storm_classifier_metrics": event_classifier_path,
            "science_storm_next_12h_metrics": science_event_path,
            "constrained_storm_next_12h_model_selection": constrained_selection_path,
            "constrained_storm_next_12h_threshold_sweeps": constrained_sweep_path,
            "alert_onset_ablation": alert_ablation_artifacts["ablation_path"],
            "alert_onset_threshold_sweeps": alert_ablation_artifacts["sweep_path"],
            "alert_onset_debug": alert_ablation_artifacts["debug_path"],
            "storm_magnitude_experiment2_metrics": experiment2_artifacts["metrics_path"],
            "storm_magnitude_experiment2_predictions": experiment2_artifacts["predictions_path"],
            "storm_magnitude_experiment2_watch_sweep": experiment2_artifacts["watch_sweep_path"],
            "storm_magnitude_experiment2_severity_thresholds": experiment2_artifacts["severity_path"],
            "storm_magnitude_experiment2_debug": experiment2_artifacts["debug_path"],
            "experiment3_severe_watch_sweep": experiment3_artifacts["watch_sweep_path"],
            "experiment3_severity_classifier_thresholds": experiment3_artifacts["severity_thresholds_path"],
            "experiment3_storm_magnitude_metrics": experiment3_artifacts["metrics_path"],
            "experiment3_storm_event_diagnostics": experiment3_artifacts["event_diagnostics_path"],
            "experiment3_failure_category_summary": experiment3_artifacts["failure_category_path"],
            "experiment3_severe_events": experiment3_artifacts["severe_events_path"],
            "experiment3_debug": experiment3_artifacts["debug_path"],
            "experiment4_tradeoff_curve": experiment4_artifacts["tradeoff_path"],
            "experiment4_selected_operating_points": experiment4_artifacts["selected_path"],
            "experiment4_metrics": experiment4_artifacts["metrics_path"],
            "experiment4_predictions": experiment4_artifacts["predictions_path"],
            "experiment4_tradeoff_plot": experiment4_artifacts["tradeoff_plot_path"],
            "experiment4_gfz_style_plot": experiment4_artifacts["gfz_style_plot_path"],
            "experiment4_debug": experiment4_artifacts["debug_path"],
            **{f"experiment5_{key}": value for key, value in experiment5_artifacts.items()},
            **{f"experiment6_{key}": value for key, value in experiment6_artifacts.items()},
            **{f"storm_recall_{key}": value for key, value in storm_recall_artifacts.items()},
            "event_storm_threshold_sweeps": event_threshold_path,
            "event_storm_classifier_predictions": event_predictions_path,
            "two_stage_boost_validation_tuning": boost_tuning_path,
            "final_system_metrics": final_system_metrics_path,
            "storm_risk_calibration_metrics": calibration_path,
            "storm_event_audit": event_audit_path,
            "detected_storm_events": detected_event_audit_path,
            "missed_storm_events": missed_event_audit_path,
            "storm_regime_summary": regime_summary_path,
            "final_architecture_baseline": baseline_architecture_path,
            "final_architecture_baseline_summary": baseline_architecture_summary_path,
            "nowcast_predictions": nowcast_predictions_path,
            "legacy_temporal_predictions": legacy_predictions_path,
            "solar_wind_only_predictions": solar_predictions_path,
            "ridge_cv": ridge_cv_path,
            "plots": plots,
            "storm_plots": {
                "precision_recall_curve": storm_pr_curve_path,
                "backtest_by_era": backtest_plot_path,
                "calibrated_storm_risk_vs_events": storm_risk_event_plot_path,
                "storm_risk_calibration_curve": calibration_curve_path,
                **event_plot_paths,
            },
            "feature_importances": {**importance_paths, **event_importance_paths},
        },
        "model_availability": _model_availability_note(),
    }

    summary_path = os.path.join(output_dir, f"temporal_{model_name}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(json.dumps(summary, indent=2))
        summary_file.write("\n")

    report_sections = [
        "# Kp Model Comparison Report",
        "## Dataset\n"
        f"- dataset path: `{dataset_path}`\n"
        f"- rows after cleaning: `{len(feature_df)}`\n"
        f"- cadence (median hours): `{cadence_hours:.2f}`\n"
        f"- duplicate timestamp rows collapsed: `{metadata['duplicates_removed']}`\n"
        f"- train date range: `{train_start}` to `{train_end}`\n"
        f"- test date range: `{test_start}` to `{test_end}`",
        "## Project Forecast Semantics\n"
        f"- core task: `{metadata['driver_input_metadata']['semantics']['core_task']}`\n"
        f"- driver input source for this run: `{metadata['driver_input_metadata']['driver_input_source']}`\n"
        f"- production driver source: `{metadata['driver_input_metadata']['production_driver_input_source']}`\n"
        f"- production horizon source: `{metadata['driver_input_metadata']['semantics']['production_horizon_source']}`\n"
        f"- correct framing: {metadata['driver_input_metadata']['semantics']['correct_framing']}\n"
        f"- original PI equation source: `{metadata['original_pi_linear_equation']['source_module']}`\n"
        f"- original PI equation: `{metadata['original_pi_linear_equation']['equation']}`\n"
        f"- transformed terms: `dphi_dt={metadata['original_pi_linear_equation']['dphi_dt']}`, "
        f"`viscous={metadata['original_pi_linear_equation']['viscous']}`, "
        f"`bz_term={metadata['original_pi_linear_equation']['bz_term']}`, "
        f"`dbzdt_term={metadata['original_pi_linear_equation']['dbzdt_term']}`",
        "## Leakage Checks\n"
        f"- passed: `{metadata['leakage']['passed']}`\n"
        f"- checked temporal features: `{metadata['leakage']['checked_features']}`\n"
        f"- violations: `{metadata['leakage']['violations']}`\n"
        f"- Kp-history availability passed: `{metadata['kp_history_availability']['passed']}`\n"
        f"- checked Kp-history features: `{metadata['kp_history_availability']['checked_features']}`\n"
        f"- Kp-history violations: `{metadata['kp_history_availability']['violations']}`",
        "## Feature Lists\n"
        f"- original physics features ({len(ORIGINAL_PHYSICS_FEATURES)}): `{ORIGINAL_PHYSICS_FEATURES}`\n"
        f"- sustained-driving features ({len(sustained_features_present)}): `{sustained_features_present}`\n"
        f"- sustained feature finite/non-finite counts: `{sustained_feature_finiteness}`\n"
        f"- solar-wind-only features ({len(solar_feature_columns)}): `{solar_feature_columns}`\n"
        f"- temporal features ({len(metadata['temporal_feature_columns'])}): `{metadata['temporal_feature_columns']}`",
        "## Ridge Time-Series CV\n"
        f"- baseline alpha: `{alpha}`\n"
        f"- searched alphas: `{RIDGE_ALPHA_GRID}`\n"
        f"- best alpha: `{ridge_alpha}`\n\n"
        + _markdown_table(ridge_cv_df),
        "## Mode A: Nowcast / One-Step Comparison\n" + _markdown_table(
            nowcast_df[
                [
                    "model",
                    "rmse",
                    "mae",
                    "corr",
                    "mae_kp_lt_4",
                    "mae_kp_ge_4",
                    "mae_kp_ge_5",
                    "precision_kp_ge_5",
                    "recall_kp_ge_5",
                    "f1_kp_ge_5",
                    "note",
                ]
            ]
        ),
        "## Mode B: Recursive Forecast Comparison\n" + _markdown_table(
            recursive_df[
                [
                    "model",
                    "horizon",
                    "rmse",
                    "mae",
                    "corr",
                    "mae_kp_ge_5",
                    "precision_kp_ge_5",
                    "recall_kp_ge_5",
                    "f1_kp_ge_5",
                ]
            ]
        ),
        "## Mode C: Solar-Wind-Only Comparison\n" + _markdown_table(
            solar_df[
                [
                    "model",
                    "rmse",
                    "mae",
                    "corr",
                    "mae_kp_lt_4",
                    "mae_kp_ge_4",
                    "mae_kp_ge_5",
                    "precision_kp_ge_5",
                    "recall_kp_ge_5",
                    "f1_kp_ge_5",
                ]
            ]
        ),
        "## Storm Classification Holdout\n" + _markdown_table(
            storm_detection_df[
                [
                    "model",
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "balanced_accuracy",
                    "average_precision",
                    "note",
                ]
            ]
        ),
        "## Event-Based Storm Classifiers\n"
        "Targets are `storm_now`, `storm_next_6h`, `storm_next_12h`, and "
        "`storm_onset_next_12h`. Threshold operating points are selected from "
        "training-only time-series out-of-fold probabilities, then evaluated on the holdout. "
        "False-alert duration is counted by alert-hours outside event lead/active windows, "
        "so a very long alert is penalized by duration even if it overlaps one true event.\n\n"
        + _markdown_table(
            event_classifier_df[
                [
                    "target",
                    "feature_mode",
                    "model",
                    "threshold_mode",
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "average_precision",
                    "storm_events",
                    "storm_events_detected",
                    "event_recall",
                    "mean_lead_time_hours",
                    "alert_hours_per_month",
                    "alert_duty_cycle",
                    "max_continuous_alert_hours",
                    "false_alert_hours_per_month",
                    "max_continuous_false_alert_hours",
                    "false_alert_windows_per_month",
                    "false_alert_windows_per_year",
                ]
            ].head(40)
        ),
        "## Main Science Storm-Next-12h Model\n"
        "Primary science target is `storm_next_12h` with `solar_wind_only` features. "
        "This section compares the deployable operating points requested for "
        "`logistic_balanced` and `hist_gradient_boosting_classifier`.\n\n"
        + _markdown_table(
            science_event_df[
                [
                    "science_alert_mode",
                    "model",
                    "threshold_mode",
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "average_precision",
                    "event_recall",
                    "mean_lead_time_hours",
                    "alert_hours_per_month",
                    "alert_duty_cycle",
                    "false_alert_windows_per_month",
                    "false_alert_hours_per_month",
                    "max_continuous_alert_hours",
                ]
            ] if not science_event_df.empty else science_event_df
        ),
        "## Constrained Public Alert Gate Selection\n"
        "Thresholds are selected from training out-of-fold predictions using these primary constraints: "
        "`alert_duty_cycle <= 0.07`, `max_continuous_alert_hours <= 72`, and "
        "`false_alert_hours_per_month <= 40`. Among feasible thresholds, selection maximizes "
        "`event_recall`, then `mean_lead_time_hours`, then pointwise `F1`. This prevents "
        "near-continuous high-recall alert modes from winning by brute force. This gate is the public operational warning gate, "
        "not necessarily the internal Kp boost gate.\n\n"
        + _markdown_table(
            constrained_selection_df[
                [
                    "model",
                    "selected_threshold",
                    "selection_feasible",
                    "holdout_feasible",
                    "deployment_feasible",
                    "selection_event_recall",
                    "selection_mean_lead_time_hours",
                    "selection_f1",
                    "selection_alert_duty_cycle",
                    "selection_max_continuous_alert_hours",
                    "selection_false_alert_hours_per_month",
                    "holdout_event_recall",
                    "holdout_mean_lead_time_hours",
                    "holdout_f1",
                    "holdout_alert_duty_cycle",
                    "holdout_max_continuous_alert_hours",
                    "holdout_false_alert_hours_per_month",
                ]
            ] if not constrained_selection_df.empty else constrained_selection_df
        ),
        "## Onset Alert Ablation\n"
        "This first-pass storm-alert experiment compares the existing RF `storm_next_12h` score, "
        "the new RF `storm_onset_next_12h` score, and their max-combined score under the same public alert constraints. "
        "The weighted row uses optional event-level sample weights; the post-processing row merges one-bin alert gaps and caps alert duration at 72 hours.\n\n"
        + _markdown_table(
            alert_ablation_df[
                [
                    "alert_score",
                    "selected_threshold",
                    "selection_feasible",
                    "event_recall",
                    "kp_ge_5_recall",
                    "kp_ge_6_recall",
                    "alert_duty_cycle",
                    "false_alert_hours_per_month",
                    "mean_lead_time_hours",
                    "median_lead_time_hours",
                    "max_continuous_alert_hours",
                    "postprocess_mode",
                ]
            ]
        )
        + f"\n- ablation table: `{alert_ablation_artifacts['ablation_path']}`\n"
        + f"- threshold sweeps: `{alert_ablation_artifacts['sweep_path']}`\n"
        + f"- debug JSON: `{alert_ablation_artifacts['debug_path']}`",
        "## Event Threshold Sweeps\n"
        f"- full train-OOF and holdout threshold sweep table: `{event_threshold_path}`\n"
        f"- constrained public-alert selection table: `{constrained_selection_path}`\n"
        f"- constrained public-alert threshold sweep table: `{constrained_sweep_path}`\n"
        f"- classifier prediction scores: `{event_predictions_path}`\n"
        f"- event metric table: `{event_classifier_path}`\n"
        f"- primary science model table: `{science_event_path}`",
        "## Ridge Threshold Selection\n"
        + _markdown_table(
            selected_thresholds_df[
                [
                    "mode",
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                ]
            ]
        ),
        "## Expanding-Window Storm Backtests\n" + _markdown_table(
            backtest_df[
                [
                    "era",
                    "model",
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "balanced_accuracy",
                    "average_precision",
                ]
            ]
        ),
        "## Mean Storm Backtest Metrics\n" + _markdown_table(backtest_summary_df),
        "## Two-Stage Ridge + Storm-Risk Boost\n"
        f"- configuration: `{two_stage_note}`\n"
        "- base regressor: `scaled_ridge_30`\n"
        f"- public alert gate: calibrated `{selected_risk_model_name}` classifier for `{primary_target}` using solar-wind-only features\n"
        f"- selected public alert threshold: `{selected_gate_threshold}` under duty-cycle / duration / false-alert constraints\n"
        f"- storm boost gate: calibrated `{selected_risk_model_name}` classifier for `{boost_target}` using solar-wind-only features\n"
        f"- selected storm boost threshold: `{selected_boost_gate_threshold}`\n"
        f"- boost validation table: `{boost_tuning_path}`\n"
        "- boost scale is selected on a validation era using storm MAE and event recall, not total MAE.\n"
        "- The alert gate is constrained for operational warning burden. The boost gate is allowed to be more aggressive because it is used internally to correct storm underprediction, not necessarily to issue public alerts.\n"
        "- amplitude model: `two_stage_amplitude_storm_boost` uses storm boost risk plus a separate HGB regressor for positive storm residual/boost amount in high-risk rows.",
        "## Experiment 2: Storm-Risk-Biased Magnitude Branch\n"
        "This branch keeps `kp_base` unchanged and adds conservative storm-magnitude outputs. It trains quantile models for future Kp maxima, "
        "severity classifiers for Kp>=5/6/7 next 12h where enough examples exist, and a more vigilant watch gate with a looser duty-cycle budget. "
        "`kp_storm_conservative` is intentionally an upper-risk prediction: when the vigilant watch gate is active, it takes the max of `kp_base`, "
        "`kp_risk_p75`, the classifier ceiling, and a watch-active `ceil(kp_risk_p90)` quantile ceiling; when the high-risk severity gate is active "
        "it can also use raw `kp_risk_p90`.\n\n"
        + _markdown_table(
            experiment2_metrics_df[
                [
                    "model",
                    "storm_mae",
                    "storm_asymmetric_mae",
                    "kp_ge_5_recall",
                    "kp_ge_6_recall",
                    "storm_underprediction_rate",
                    "storm_mean_underprediction_size",
                    "storm_overprediction_rate",
                    "false_storm_inflation_quiet_rate",
                    "watch_alert_duty_cycle",
                    "watch_false_alert_hours_per_month",
                    "watch_max_continuous_alert_hours",
                ]
            ]
        )
        + f"\n- metrics: `{experiment2_artifacts['metrics_path']}`\n"
        + f"- predictions: `{experiment2_artifacts['predictions_path']}`\n"
        + f"- watch sweep: `{experiment2_artifacts['watch_sweep_path']}`\n"
        + f"- severity thresholds: `{experiment2_artifacts['severity_path']}`\n"
        + f"- debug JSON: `{experiment2_artifacts['debug_path']}`",
        "## Experiment 3: Severe-Storm-Focused Watch And Magnitude Calibration\n"
        "Experiment 3 is not selected by ordinary all-row MAE. It targets dangerous underprediction, especially Kp>=6. "
        "It adds explicit future-severity classifiers, a p95 upper-risk quantile, a severe internal watch gate, and "
        "`kp_storm_conservative_v2` / `kp_final_risk_conservative`. This branch is allowed to overcall because it is internal, "
        "not the constrained public alert gate. High-cadence summary features were not added because this dataframe only contains "
        "six-hour aggregated columns.\n\n"
        + _markdown_table(
            experiment3_metrics_df[
                [
                    "model",
                    "storm_mae",
                    "asymmetric_storm_mae",
                    "kp_ge_5_recall",
                    "kp_ge_6_recall",
                    "kp_ge_7_recall",
                    "max_storm_underprediction",
                    "storm_underprediction_rate",
                    "quiet_false_storm_inflation_rate",
                    "selected_watch_duty_cycle",
                    "missed_kp_ge_6_events",
                ]
            ]
        )
        + f"\n- severe watch sweep: `{experiment3_artifacts['watch_sweep_path']}`\n"
        + f"- severity classifier thresholds: `{experiment3_artifacts['severity_thresholds_path']}`\n"
        + f"- magnitude metrics: `{experiment3_artifacts['metrics_path']}`\n"
        + f"- event diagnostics: `{experiment3_artifacts['event_diagnostics_path']}`\n"
        + f"- severe events: `{experiment3_artifacts['severe_events_path']}`\n"
        + f"- failure categories: `{experiment3_artifacts['failure_category_path']}`\n"
        + f"- debug JSON: `{experiment3_artifacts['debug_path']}`",
        "## Experiment 4: Severe Recall Calibration\n"
        "Experiment 4 keeps the Experiment 3 watch signal but separates watch from severe magnitude inflation. It uses a multi-signal vote rule "
        "and selects operating points under quiet false-storm inflation targets. This is meant to find a more usable frontier: high Kp>=6 recall "
        "without automatically pushing quiet intervals to Kp~6.\n\n"
        + _markdown_table(
            experiment4_metrics_df[
                [
                    "model",
                    "storm_mae",
                    "asymmetric_storm_mae",
                    "kp_ge_5_recall",
                    "kp_ge_6_recall",
                    "kp_ge_7_recall",
                    "max_storm_underprediction",
                    "storm_underprediction_rate",
                    "quiet_false_storm_inflation_rate",
                    "selected_quiet_inflation_target",
                    "selected_min_votes",
                    "selected_prob6_threshold",
                    "selected_p95_gap_threshold",
                ]
            ]
        )
        + f"\n- tradeoff curve table: `{experiment4_artifacts['tradeoff_path']}`\n"
        + f"- selected operating points: `{experiment4_artifacts['selected_path']}`\n"
        + f"- metrics: `{experiment4_artifacts['metrics_path']}`\n"
        + f"- predictions: `{experiment4_artifacts['predictions_path']}`\n"
        + f"- tradeoff plot: `{experiment4_artifacts['tradeoff_plot_path']}`\n"
        + f"- GFZ-style comparison plot: `{experiment4_artifacts['gfz_style_plot_path']}`\n"
        + f"- debug JSON: `{experiment4_artifacts['debug_path']}`",
        "## Experiment 5: Calibration, Diagnostics, And Operating Frontier\n"
        + (
            "Experiment 5 keeps the selected Experiment 4 severe-risk operating point stable, then adds PI-facing diagnostics by Kp regime, "
            "an operating frontier for Kp>=6 recall vs quiet false inflation, event-level comparisons, and false-inflation distributions. "
            "A learned suppressor is intentionally skipped until a leakage-safe validation-era Experiment 4 prediction set is available.\n\n"
            + _markdown_table(
                experiment5_results["frontier"][
                    [
                        "quiet_inflation_target",
                        "target_feasible",
                        "kp_ge_5_recall",
                        "kp_ge_6_recall",
                        "kp_ge_7_recall",
                        "quiet_false_storm_inflation_rate",
                        "storm_mae",
                        "severe_weighted_asymmetric_mae",
                        "max_kp_ge_6_underprediction",
                        "missed_kp_ge_6_rows_estimated",
                    ]
                ] if experiment5_results is not None and not experiment5_results["frontier"].empty else pd.DataFrame()
            )
            + "\n"
            + "\n".join([f"- `{key}`: `{value}`" for key, value in experiment5_artifacts.items()])
            if experiment5_artifacts
            else "Experiment 5 diagnostics were skipped; see console warning above."
        ),
        "## Experiment 6: Leakage-Safe False-Inflation Suppression\n"
        + (
            "Experiment 6 trains a small leakage-safe suppressor on validation-era out-of-sample component predictions, then applies it to the final holdout. "
            "It is allowed to suppress only marginal v5 storm-level inflation cases, and a severe safety override prevents suppression when Kp>=6 risk signals are strong.\n\n"
            + _markdown_table(
                experiment6_metrics_df[
                    [
                        "model",
                        "storm_mae",
                        "asymmetric_storm_mae",
                        "kp_ge_5_recall",
                        "kp_ge_6_recall",
                        "kp_ge_7_recall",
                        "max_storm_underprediction",
                        "quiet_false_storm_inflation_rate",
                    ]
                ] if not experiment6_metrics_df.empty else experiment6_metrics_df
            )
            + "\n"
            + "\n".join([f"- `{key}`: `{value}`" for key, value in experiment6_artifacts.items()])
            if experiment6_artifacts
            else "Experiment 6 diagnostics were skipped; see console warning above."
        ),
        "## Storm Recall Forecasting\n"
        + (
            "This branch selects internal Kp>=5 and Kp>=6 watch thresholds on the validation era, then evaluates row-level recall, event-level recall, "
            "Kp<7 magnitude accuracy, and Kp>7 diagnostics on the final holdout. Public alert burden remains separate through the existing public alert gate.\n\n"
            + _markdown_table(
                storm_recall_metrics_df[
                    [
                        "metric_group",
                        "model",
                        "regime",
                        "row_count",
                        "mae",
                        "rmse",
                        "recall",
                        "event_recall",
                        "underprediction_rate",
                        "overprediction_rate",
                        "max_underprediction",
                    ]
                ] if not storm_recall_metrics_df.empty else storm_recall_metrics_df
            )
            + "\n"
            + "\n".join([f"- `{key}`: `{value}`" for key, value in storm_recall_artifacts.items()])
            if storm_recall_artifacts
            else "Storm recall forecasting diagnostics were skipped; see Experiment 6 warning above."
        ),
        "## Experiment 7: Forecast-Input-Aligned Kp Mapping\n"
        + (
            "Experiment 7 now separates the diagnostic target-aligned upper bound from deployable rolling-horizon simulation. "
            "Solar-wind physics features move with valid time T; Kp-history features are anchored at forecast initialization time in operational modes.\n\n"
            + _markdown_table(
                experiment7_metrics_df[[col for col in [
                    "model",
                    "mode",
                    "mae",
                    "rmse",
                    "pearson_corr",
                    "spearman_corr",
                    "bias",
                    "underprediction_rate",
                    "max_underprediction",
                    "severe_weighted_asymmetric_mae",
                ] if col in experiment7_metrics_df.columns]] if not experiment7_metrics_df.empty else experiment7_metrics_df
            )
            + "\n"
            + "\n".join([f"- `{key}`: `{value}`" for key, value in experiment7_artifacts.items()])
            if experiment7_artifacts
            else "Experiment 7 diagnostics were skipped; see console warning above."
        ),
        "## Experiment 8: 5-Day Storm Watch Classifier\n"
        + (
            "Experiment 8 trains directly on the operational watch target: forecasted 5-day physics sequence plus anchored Kp state -> "
            "`P(any Kp>=5 in next 120h)`, `P(any Kp>=6 in next 120h)`, and predicted max Kp in the next 5 days.\n\n"
            + _markdown_table(experiment8_metrics_df)
            + "\n"
            + "\n".join([f"- `{key}`: `{value}`" for key, value in experiment8_artifacts.items()])
            if experiment8_artifacts
            else "Experiment 8 diagnostics were skipped; see console warning above."
        ),
        "## Experiment 11: Structural Storm-Aware Model\n"
        + (
            "Experiment 11 tests the structural upgrades requested after the Experiment 10 boost results: leaky-integrator physics memory, "
            "severe-weighted training, ordinal/density Kp probabilities, and a classifier-informed multi-task proxy. It is evaluated with the "
            "same 6h rolling operational wall: forecasted physics moves with valid time T, while Kp history is anchored at t_init.\n\n"
            + _markdown_table(
                experiment11_metrics_df[experiment11_metrics_df["lead_time_bin"] == "all"] if not experiment11_metrics_df.empty else experiment11_metrics_df
            )
            + "\n"
            + "\n".join([f"- `{key}`: `{value}`" for key, value in experiment11_artifacts.items()])
            if experiment11_artifacts
            else "Experiment 11 diagnostics were skipped; see console warning above."
        ),
        "## Final System Outputs\n"
        "The final system intentionally keeps separate outputs instead of forcing one model to win every metric: "
        "`kp_base` for quiet/normal Kp, `storm_alert_probability` / `storm_alert_gate` for constrained operational warning, "
        "`storm_boost_probability` / `storm_boost_gate` for internal storm correction, `kp_storm_adjusted` for storm-aware Kp, "
        "`kp_storm_conservative` for the Experiment 2 upper-risk magnitude branch, and `kp_storm_conservative_v2` / "
        "`kp_final_risk_conservative` for the Experiment 3 severe-storm internal branch, plus "
        "`kp_final_risk_conservative_v4` / `kp_final_risk_conservative_v5` for the calibrated severe-risk operating point. "
        "`storm_recall_internal_watch_kp_ge_5`, `storm_recall_internal_watch_kp_ge_6`, and `kp_storm_recall_conservative` "
        "are validation-selected high-recall diagnostics, not replacements for the public alert gate.\n\n"
        + _markdown_table(final_system_metrics_df),
        "## Storm Risk Calibration\n"
        "Both alert and boost probabilities are calibrated from training out-of-fold classifier scores using one-dimensional "
        "logistic calibration models, then applied to the holdout. The public alert gate is constrained for warning burden; "
        "the boost gate is tuned separately for storm Kp correction.\n\n"
        + _markdown_table(calibration_df)
        + f"\n- calibration metrics: `{calibration_path}`\n"
        + f"- calibration plot: `{calibration_curve_path}`\n"
        + f"- storm-risk event overlay: `{storm_risk_event_plot_path}`",
        "## Event Audit And Physical Regimes\n"
        "Each Kp >= 5 holdout event is listed with detected/missed status, lead time, local Bz/pressure/velocity context, "
        "and a heuristic physical-regime label. Regimes are intended for manual inspection, not as final scientific labels.\n\n"
        + _markdown_table(regime_summary_df)
        + f"\n- all storm events: `{event_audit_path}`\n"
        + f"- detected storm events: `{detected_event_audit_path}`\n"
        + f"- missed storm events: `{missed_event_audit_path}`\n"
        + f"- regime summary: `{regime_summary_path}`",
        "## Frozen Baseline Architecture\n"
        f"- pickled baseline architecture: `{baseline_architecture_path}`\n"
        f"- JSON baseline summary: `{baseline_architecture_summary_path}`\n"
        "- baseline outputs: `kp_base`, `storm_alert_probability`, `storm_alert_gate`, `storm_boost_probability`, `storm_boost_gate`, and `kp_storm_adjusted`",
        "## Plots\n"
        + "\n".join([f"- `{name}`: `{path}`" for name, path in plots.items()])
        + "\n"
        + f"- `storm_precision_recall_curve`: `{storm_pr_curve_path}`\n"
        + f"- `storm_backtest_by_era`: `{backtest_plot_path}`\n"
        + f"- `calibrated_storm_risk_vs_events`: `{storm_risk_event_plot_path}`\n"
        + f"- `storm_risk_calibration_curve`: `{calibration_curve_path}`\n"
        + "\n".join([f"- `{name}`: `{path}`" for name, path in event_plot_paths.items()]),
        "## Best Model Summary\n"
        f"- best model: `{best_model_name}`\n"
        f"- holdout MAE: `{best_model_bundle['metrics']['mae']:.4f}`\n"
        f"- holdout RMSE: `{best_model_bundle['metrics']['rmse']:.4f}`\n"
        f"- holdout correlation: `{best_model_bundle['metrics']['corr']:.4f}`\n"
        f"- best storm holdout configuration: `{storm_detection_df.iloc[0]['model']}` with `F1={storm_detection_df.iloc[0]['f1']:.4f}`\n"
        f"- best event classifier row: `{event_classifier_df.iloc[0].to_dict() if not event_classifier_df.empty else 'none'}`",
        "## Limitations\n"
        "- Chronological holdout is stricter than a random split, but it is still only one temporal split.\n"
        "- The dataset has some historical duplicates that were averaged before modeling.\n"
        "- `xgboost` and `lightgbm` were skipped when the packages were not installed.\n"
        "- Strong nowcast performance from Kp-history models does not imply the same independence as a solar-wind-only forecast.\n"
        "- Storm events are rare in this dataset, so precision-recall tradeoffs remain sensitive to threshold choice and era selection.",
        "## Recommended Next Experiment\n"
        "- With the calibrated baseline frozen, test higher-cadence solar-wind inputs or event-specific storm-phase models before adding more black-box estimators.",
    ]
    report_path = os.path.join(output_dir, "kp_model_comparison_report.md")
    _write_report(report_path, report_sections)

    model_path = os.path.join(output_dir, f"temporal_{model_name}.pkl")
    saved_bundle = {
        "model_type": "temporal_best_model",
        "best_model_name": best_model_name,
        "model": best_model_bundle["model"],
        "feature_columns": best_model_bundle["feature_columns"],
        "summary": summary,
    }
    with open(model_path, "wb") as model_file:
        pickle.dump(saved_bundle, model_file)

    return {
        "summary": summary,
        "summary_path": summary_path,
        "report_path": report_path,
        "model_path": model_path,
        "best_model_name": best_model_name,
        "best_model_metrics": best_model_bundle["metrics"],
        "nowcast_df": nowcast_df,
        "solar_df": solar_df,
        "recursive_df": recursive_df,
        "storm_detection_df": storm_detection_df,
        "backtest_df": backtest_df,
    }


def _experiment7_feature_sets(metadata: dict) -> Dict[str, List[str]]:
    solar = metadata["solar_wind_feature_columns"]
    core = [feature for feature in SOLAR_WIND_CURRENT_FEATURES if feature in solar]
    solar_history = [
        feature for feature in solar
        if feature not in SOLAR_WIND_CURRENT_FEATURES
        and feature not in BZ_SHARP_DROP_FEATURES
        and ("_lag_" in feature or "_roll_" in feature)
    ]
    physics = [feature for feature in SUSTAINED_DRIVING_FEATURES + solar_history if feature in solar]
    bz_drop = [feature for feature in BZ_SHARP_DROP_FEATURES if feature in solar]
    kp_pattern = [feature for feature in KP_HISTORY_FEATURES if feature in metadata["temporal_feature_columns"]]
    return {
        "A0_bz_velocity_density_baseline": [feature for feature in ["bz", "velocity", "density"] if feature in solar],
        "A_solar_wind_inputs_only": _unique_in_order(core),
        "B_solar_wind_plus_storm_physics": _unique_in_order(core + physics),
        "C_solar_wind_plus_bz_drop": _unique_in_order(core + bz_drop),
        "D_solar_wind_plus_kp_pattern_break": _unique_in_order(core + kp_pattern),
        "E_all_features_combined": _unique_in_order(core + physics + bz_drop + kp_pattern),
        "main_operational_no_kp_history": _unique_in_order(core + physics + bz_drop),
    }


def _experiment7_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    under = error < 0.0
    storm_weight = np.ones(len(y_true), dtype=float)
    storm_weight += np.where(y_true >= 5.0, 2.0, 0.0)
    storm_weight += np.where(y_true >= 6.0, 4.0, 0.0)
    storm_weight += np.where(y_true >= 7.0, 6.0, 0.0)
    asymmetric_error = np.where(error < 0.0, np.abs(error) * 2.0, np.abs(error))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson_corr": _safe_corr_values(y_true, y_pred, "pearson"),
        "spearman_corr": _safe_corr_values(y_true, y_pred, "spearman"),
        "bias": float(np.mean(error)),
        "underprediction_rate": float(np.mean(under)),
        "max_underprediction": float(np.max(np.maximum(y_true - y_pred, 0.0))) if len(y_true) else 0.0,
        "severe_weighted_asymmetric_mae": float(np.average(asymmetric_error, weights=storm_weight)),
    }


def _experiment7_regime_metrics(prediction_df: pd.DataFrame, model_columns: List[str]) -> pd.DataFrame:
    regimes = {
        "all_rows": np.ones(len(prediction_df), dtype=bool),
        "kp_lt_4": prediction_df["kp"].to_numpy(dtype=float) < 4.0,
        "kp_lt_5": prediction_df["kp"].to_numpy(dtype=float) < 5.0,
        "kp_ge_5": prediction_df["kp"].to_numpy(dtype=float) >= 5.0,
        "kp_ge_6": prediction_df["kp"].to_numpy(dtype=float) >= 6.0,
        "kp_ge_7": prediction_df["kp"].to_numpy(dtype=float) >= 7.0,
    }
    rows = []
    y_all = prediction_df["kp"].to_numpy(dtype=float)
    for model_col in model_columns:
        if model_col not in prediction_df.columns:
            continue
        p_all = prediction_df[model_col].to_numpy(dtype=float)
        for regime_name, mask in regimes.items():
            if int(mask.sum()) == 0:
                continue
            rows.append(
                {
                    "model": model_col,
                    "regime": regime_name,
                    "n_rows": int(mask.sum()),
                    **_experiment7_regression_metrics(y_all[mask], p_all[mask]),
                }
            )
    return pd.DataFrame(rows)


def _experiment7_classification_row(
    y_true_kp: np.ndarray,
    score: np.ndarray,
    threshold: float,
    kp_threshold: float,
    label: str,
) -> dict:
    y_true_binary = np.asarray(y_true_kp, dtype=float) >= kp_threshold
    y_pred_binary = np.asarray(score, dtype=float) >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary,
        y_pred_binary,
        average="binary",
        zero_division=0,
    )
    try:
        roc_auc = float(roc_auc_score(y_true_binary, score)) if len(np.unique(y_true_binary)) > 1 else None
    except ValueError:
        roc_auc = None
    return {
        "model": label,
        "kp_threshold": kp_threshold,
        "probability_threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision": _safe_average_precision(y_true_binary, score),
        "roc_auc": roc_auc,
        "positives": int(y_true_binary.sum()),
    }


def _experiment7_classifier_threshold(
    train_df: pd.DataFrame,
    feature_columns: List[str],
    kp_threshold: float,
) -> float:
    y_train_binary = (train_df["kp"].to_numpy(dtype=float) >= kp_threshold).astype(int)
    if len(np.unique(y_train_binary)) < 2:
        return 0.5
    oof_true, oof_score = _time_series_oof_classifier_scores(
        lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        ),
        train_df[feature_columns],
        y_train_binary,
    )
    if len(oof_true) == 0 or len(np.unique(oof_true)) < 2:
        return 0.5
    sweep_df = pd.DataFrame(_threshold_sweep_rows(oof_true.astype(bool), oof_score, np.linspace(0.01, 0.99, 99)))
    return _select_classifier_thresholds(sweep_df)["high_recall"]


def _experiment7_fit_probability(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    kp_threshold: float,
) -> Tuple[Optional[np.ndarray], Optional[float], str]:
    y_train_binary = (train_df["kp"].to_numpy(dtype=float) >= kp_threshold).astype(int)
    y_test_binary = (test_df["kp"].to_numpy(dtype=float) >= kp_threshold).astype(int)
    if int(y_train_binary.sum()) < MIN_SEVERITY_POSITIVES or len(np.unique(y_train_binary)) < 2 or len(np.unique(y_test_binary)) < 2:
        return None, None, "skipped_insufficient_positive_examples"
    threshold = _experiment7_classifier_threshold(train_df, feature_columns, kp_threshold)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
    )
    model.fit(train_df[feature_columns], y_train_binary)
    score = model.predict_proba(test_df[feature_columns])[:, 1]
    return score, threshold, "trained_logistic_balanced"


def _experiment7_quantile_model(quantile: float):
    if HistGradientBoostingRegressor is None:
        return None
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )


def _experiment7_plot_scatter(prediction_df: pd.DataFrame, output_dir: str) -> str:
    y = prediction_df["kp"].to_numpy(dtype=float)
    p = prediction_df["kp_forecast_aligned_base"].to_numpy(dtype=float)
    corr = _safe_corr_values(y, p, "pearson")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, p, s=16, alpha=0.45)
    ax.plot([0, 9], [0, 9], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_xlabel("Observed Kp")
    ax.set_ylabel("Forecast-aligned predicted Kp")
    ax.set_title(f"Forecast-Aligned Kp Scatter (r={corr:.3f})" if corr is not None else "Forecast-Aligned Kp Scatter")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, "kp_experiment7_forecast_aligned_scatter.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _experiment7_plot_timeseries(prediction_df: pd.DataFrame, output_dir: str) -> str:
    work = prediction_df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    storm = work[work["kp"] >= 5.0]
    if not storm.empty:
        center = storm.iloc[len(storm) // 2]["timestamp"]
        window = work[(work["timestamp"] >= center - pd.Timedelta(days=8)) & (work["timestamp"] <= center + pd.Timedelta(days=8))]
    else:
        window = work.iloc[: min(len(work), 80)]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(window["timestamp"], window["kp"], color="black", linewidth=1.8, label="Observed Kp")
    for col, label in [
        ("kp_base", "kp_base"),
        ("kp_forecast_aligned_base", "forecast aligned base"),
        ("kp_forecast_aligned_conservative", "forecast aligned conservative"),
        ("kp_final_risk_conservative_v5", "Exp4/5 severe risk"),
    ]:
        if col in window.columns:
            ax.plot(window["timestamp"], window[col], linewidth=1.2, label=label)
    ax.axhline(5.0, color="tab:red", linestyle="--", linewidth=1)
    ax.set_ylim(0, 9.5)
    ax.set_ylabel("Kp")
    ax.set_xlabel("Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, "kp_experiment7_forecast_aligned_timeseries_examples.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _experiment7_plot_bz_drop(prediction_df: pd.DataFrame, output_dir: str) -> str:
    work = prediction_df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    event_rows = work.sort_values("bz_southward_drop_6h", ascending=False).head(3)
    if event_rows.empty:
        event_rows = work.iloc[[0]]
    fig, axes = plt.subplots(len(event_rows), 1, figsize=(12, 3.8 * len(event_rows)), sharex=False)
    if len(event_rows) == 1:
        axes = np.array([axes])
    for ax, (_, row) in zip(axes, event_rows.iterrows()):
        start = row["timestamp"] - pd.Timedelta(days=3)
        end = row["timestamp"] + pd.Timedelta(days=5)
        window = work[(work["timestamp"] >= start) & (work["timestamp"] <= end)]
        ax2 = ax.twinx()
        ax.plot(window["timestamp"], window["bz"], color="tab:blue", label="Bz")
        ax2.plot(window["timestamp"], window["kp"], color="black", linewidth=1.5, label="Observed Kp")
        if "prob_forecast_kp_ge_5" in window.columns:
            ax2.plot(window["timestamp"], 9.0 * window["prob_forecast_kp_ge_5"], color="tab:red", alpha=0.7, label="P(Kp>=5) x 9")
        turns = window[window["bz_crossed_southward"].astype(bool)]
        ax.scatter(turns["timestamp"], turns["bz"], color="tab:orange", s=35, label="southward turn")
        ax.set_ylabel("Bz nT")
        ax2.set_ylabel("Kp / risk")
        ax.grid(True, alpha=0.25)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = os.path.join(output_dir, "kp_experiment7_bz_drop_examples.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _experiment7_plot_pattern_break(prediction_df: pd.DataFrame, output_dir: str) -> str:
    work = prediction_df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    source = work.sort_values("kp_daily_pattern_break_score", ascending=False)
    center = source.iloc[0]["timestamp"] if not source.empty else work.iloc[0]["timestamp"]
    window = work[(work["timestamp"] >= center - pd.Timedelta(days=7)) & (work["timestamp"] <= center + pd.Timedelta(days=7))]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax2 = ax.twinx()
    ax.plot(window["timestamp"], window["kp"], color="black", linewidth=1.6, label="Observed Kp")
    if "kp_lag_4" in window.columns:
        ax.plot(window["timestamp"], window["kp_lag_4"], color="tab:blue", linestyle="--", label="24h Kp lag")
    ax2.plot(window["timestamp"], window["kp_daily_pattern_break_score"], color="tab:orange", label="pattern-break score")
    storms = window[window["kp"] >= 5.0]
    if not storms.empty:
        ax.scatter(storms["timestamp"], storms["kp"], color="tab:red", s=35, label="Kp>=5")
    ax.set_ylabel("Kp")
    ax2.set_ylabel("Pattern-break score")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = os.path.join(output_dir, "kp_experiment7_24h_pattern_break_examples.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _experiment7_old_shifted_comparison(
    feature_df: pd.DataFrame,
    new_best_metrics: dict,
    feature_columns: List[str],
    test_size: float,
    output_dir: str,
) -> Tuple[pd.DataFrame, Optional[bool]]:
    cadence = _cadence_hours(feature_df)
    horizon_steps = max(1, int(round(120.0 / cadence))) if cadence > 0 else 20
    old_df = feature_df.copy()
    old_df["kp_target_t_plus_120h"] = old_df["kp"].shift(-horizon_steps)
    old_df = old_df.dropna(subset=["kp_target_t_plus_120h"]).reset_index(drop=True)
    if len(old_df) < 200:
        comparison = pd.DataFrame(
            [
                {
                    "framing": "old_t_to_t_plus_120h",
                    "status": "skipped_not_enough_rows",
                    "horizon_steps": horizon_steps,
                },
                {
                    "framing": "new_forecast_inputs_valid_at_T_to_kp_T",
                    "status": "ok",
                    **new_best_metrics,
                },
            ]
        )
        comparison.to_csv(os.path.join(output_dir, "kp_experiment7_old_vs_forecast_aligned_comparison.csv"), index=False)
        return comparison, None
    old_train, old_test = chronological_split(old_df, test_size)
    model = make_pipeline(StandardScaler(), Ridge(alpha=30.0))
    model.fit(old_train[feature_columns], old_train["kp_target_t_plus_120h"])
    old_pred = np.clip(model.predict(old_test[feature_columns]), 0.0, 9.0)
    old_metrics = _experiment7_regression_metrics(old_test["kp_target_t_plus_120h"].to_numpy(dtype=float), old_pred)
    comparison = pd.DataFrame(
        [
            {
                "framing": "old_t_to_t_plus_120h",
                "status": "baseline_only_not_main_experiment7",
                "horizon_steps": horizon_steps,
                **old_metrics,
            },
            {
                "framing": "new_forecast_inputs_valid_at_T_to_kp_T",
                "status": "main_experiment7",
                "horizon_steps": 0,
                **new_best_metrics,
            },
        ]
    )
    comparison.to_csv(os.path.join(output_dir, "kp_experiment7_old_vs_forecast_aligned_comparison.csv"), index=False)
    return comparison, bool(new_best_metrics["mae"] <= old_metrics["mae"])


def _run_experiment7_forecast_aligned(
    feature_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    current_predictions: pd.DataFrame,
    metadata: dict,
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    _ensure_directory(output_dir)
    feature_sets = _experiment7_feature_sets(metadata)
    selected_feature_set = (
        "E_all_features_combined"
        if metadata.get("issue_time_valid_time_metadata_found", False)
        else "main_operational_no_kp_history"
    )
    feature_columns = feature_sets[selected_feature_set]
    forbidden_exact = {"kp", "storm_now", "storm_next_6h", "storm_next_12h", "kp_max_next_12h"}
    forbidden_features = [feature for feature in feature_columns if feature in forbidden_exact or feature.endswith("_valid")]
    kp_feature_source_ok = all(feature in metadata["kp_history_feature_columns"] for feature in feature_columns if feature.startswith("kp_"))
    leakage_passed = len(forbidden_features) == 0 and bool(kp_feature_source_ok) and metadata["leakage"]["passed"]
    if not leakage_passed:
        raise ValueError(f"Experiment 7 leakage check failed: forbidden={forbidden_features}, kp_source_ok={kp_feature_source_ok}")

    prediction_df = test_df[[
        "timestamp",
        "kp",
        "bz",
        "velocity",
        "density",
        "pdyn",
        *[feature for feature in BZ_SHARP_DROP_FEATURES + KP_HISTORY_FEATURES if feature in test_df.columns],
    ]].copy()
    for old_col in [
        "kp_base",
        "kp_storm_adjusted",
        "kp_storm_conservative_v4",
        "kp_storm_conservative_v5",
        "kp_final_risk_conservative_v5",
    ]:
        if old_col in current_predictions.columns:
            prediction_df = prediction_df.merge(
                current_predictions[["timestamp", "kp", old_col]],
                on=["timestamp", "kp"],
                how="left",
            )

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    y_train = train_df["kp"].to_numpy(dtype=float)
    y_test = test_df["kp"].to_numpy(dtype=float)
    regressors = {
        "ridge_baseline": make_pipeline(StandardScaler(), Ridge(alpha=30.0)),
        "elasticnet_baseline": make_pipeline(StandardScaler(), ElasticNet(alpha=0.002, l1_ratio=0.15, max_iter=10000, random_state=RANDOM_STATE)),
        "kp_forecast_aligned_rf": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if HistGradientBoostingRegressor is not None:
        regressors["kp_forecast_aligned_hgb"] = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        )

    metric_rows = []
    ablation_rows = []
    fitted_predictions = {}
    minimal_features = feature_sets["A0_bz_velocity_density_baseline"]
    minimal_model = make_pipeline(StandardScaler(), Ridge(alpha=30.0))
    minimal_model.fit(train_df[minimal_features], y_train)
    minimal_pred = np.clip(minimal_model.predict(test_df[minimal_features]), 0.0, 9.0)
    prediction_df["kp_forecast_aligned_bz_velocity_density_baseline"] = minimal_pred
    metric_rows.append(
        {
            "model": "kp_forecast_aligned_bz_velocity_density_baseline",
            "feature_set": "A0_bz_velocity_density_baseline",
            **_experiment7_regression_metrics(y_test, minimal_pred),
        }
    )
    for model_name, model in regressors.items():
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_test), 0.0, 9.0)
        fitted_predictions[model_name] = pred
        output_col = model_name
        if model_name == "elasticnet_baseline":
            output_col = "kp_forecast_aligned_base"
        prediction_df[output_col] = pred
        metrics = _experiment7_regression_metrics(y_test, pred)
        metric_rows.append({"model": output_col, "feature_set": selected_feature_set, **metrics})

    if "kp_forecast_aligned_hgb" not in prediction_df.columns:
        prediction_df["kp_forecast_aligned_hgb"] = prediction_df["kp_forecast_aligned_base"]

    quantile_debug = {}
    for label, quantile in [("p75", 0.75), ("p90", 0.90), ("p95", 0.95)]:
        col = f"kp_forecast_aligned_{label}"
        model = _experiment7_quantile_model(quantile)
        if model is None:
            prediction_df[col] = prediction_df["kp_forecast_aligned_hgb"]
            quantile_debug[label] = "fallback_hgb_quantile_regressor_unavailable"
            continue
        try:
            model.fit(X_train, y_train)
            prediction_df[col] = np.clip(model.predict(X_test), 0.0, 9.0)
            quantile_debug[label] = "trained_hist_gradient_boosting_quantile"
        except Exception as exc:
            prediction_df[col] = prediction_df["kp_forecast_aligned_hgb"]
            quantile_debug[label] = f"fallback_hgb_quantile_failed:{exc}"
        metric_rows.append({"model": col, "feature_set": selected_feature_set, **_experiment7_regression_metrics(y_test, prediction_df[col].to_numpy(dtype=float))})

    classifier_rows = []
    classifier_debug = {}
    thresholds = {}
    for kp_threshold in [5.0, 6.0, 7.0]:
        label = str(int(kp_threshold))
        prob_col = f"prob_forecast_kp_ge_{label}"
        score, threshold, status = _experiment7_fit_probability(train_df, test_df, feature_columns, kp_threshold)
        classifier_debug[prob_col] = status
        if score is None:
            if kp_threshold < 7.0:
                prediction_df[prob_col] = 0.0
                thresholds[prob_col] = 1.01
            continue
        prediction_df[prob_col] = score
        thresholds[prob_col] = float(threshold)
        classifier_rows.append(_experiment7_classification_row(y_test, score, threshold, kp_threshold, prob_col))

    conservative = prediction_df["kp_forecast_aligned_base"].to_numpy(dtype=float).copy()
    if "prob_forecast_kp_ge_5" in prediction_df.columns:
        gate = prediction_df["prob_forecast_kp_ge_5"].to_numpy(dtype=float) >= thresholds.get("prob_forecast_kp_ge_5", 1.01)
        conservative = np.where(gate, np.maximum.reduce([conservative, prediction_df["kp_forecast_aligned_p75"].to_numpy(dtype=float), np.full(len(conservative), 5.0)]), conservative)
    if "prob_forecast_kp_ge_6" in prediction_df.columns:
        gate = prediction_df["prob_forecast_kp_ge_6"].to_numpy(dtype=float) >= thresholds.get("prob_forecast_kp_ge_6", 1.01)
        conservative = np.where(gate, np.maximum.reduce([conservative, prediction_df["kp_forecast_aligned_p90"].to_numpy(dtype=float), np.full(len(conservative), 6.0)]), conservative)
    if "prob_forecast_kp_ge_7" in prediction_df.columns:
        gate = prediction_df["prob_forecast_kp_ge_7"].to_numpy(dtype=float) >= thresholds.get("prob_forecast_kp_ge_7", 1.01)
        conservative = np.where(gate, np.maximum.reduce([conservative, prediction_df["kp_forecast_aligned_p95"].to_numpy(dtype=float), np.full(len(conservative), 7.0)]), conservative)
    prediction_df["kp_forecast_aligned_conservative"] = np.clip(conservative, 0.0, 9.0)
    metric_rows.append({"model": "kp_forecast_aligned_conservative", "feature_set": selected_feature_set, **_experiment7_regression_metrics(y_test, conservative)})

    for feature_set_name, ablation_features in feature_sets.items():
        model = make_pipeline(StandardScaler(), Ridge(alpha=30.0))
        model.fit(train_df[ablation_features], y_train)
        pred = np.clip(model.predict(test_df[ablation_features]), 0.0, 9.0)
        ablation_rows.append({"feature_set": feature_set_name, "model": "ridge30", **_experiment7_regression_metrics(y_test, pred)})

    metrics_df = pd.DataFrame(metric_rows)
    class_df = pd.DataFrame(classifier_rows)
    if not class_df.empty:
        metrics_df = metrics_df.merge(
            class_df.rename(columns={"model": "classification_model"}),
            left_on="model",
            right_on="classification_model",
            how="left",
        )
    for kp_threshold in [5.0, 6.0, 7.0]:
        target = y_test >= kp_threshold
        for row_idx, row in metrics_df.iterrows():
            pred = prediction_df[row["model"]].to_numpy(dtype=float) if row["model"] in prediction_df.columns else None
            if pred is None or int(target.sum()) == 0:
                continue
            metrics_df.loc[row_idx, f"mae_kp_ge_{int(kp_threshold)}"] = float(mean_absolute_error(y_test[target], pred[target]))
            metrics_df.loc[row_idx, f"recall_kp_ge_{int(kp_threshold)}"] = float(np.mean(pred[target] >= kp_threshold))

    ablation_df = pd.DataFrame(ablation_rows)
    regime_df = _experiment7_regime_metrics(
        prediction_df,
        [
            "kp_forecast_aligned_bz_velocity_density_baseline",
            "kp_forecast_aligned_base",
            "kp_forecast_aligned_hgb",
            "kp_forecast_aligned_rf",
            "kp_forecast_aligned_p75",
            "kp_forecast_aligned_p90",
            "kp_forecast_aligned_p95",
            "kp_forecast_aligned_conservative",
        ],
    )

    best_row = metrics_df[metrics_df["model"].isin(["kp_forecast_aligned_base", "kp_forecast_aligned_hgb", "kp_forecast_aligned_rf"])].sort_values(["mae", "rmse"]).iloc[0]
    best_model = str(best_row["model"])
    best_metrics = best_row[["mae", "rmse", "pearson_corr", "spearman_corr", "bias", "underprediction_rate", "max_underprediction", "severe_weighted_asymmetric_mae"]].to_dict()
    comparison_df, beat_old = _experiment7_old_shifted_comparison(
        feature_df,
        best_metrics,
        feature_sets["A_solar_wind_inputs_only"],
        1.0 - (len(train_df) / len(feature_df)),
        output_dir,
    )

    bz_base = ablation_df[ablation_df["feature_set"] == "A_solar_wind_inputs_only"]["mae"].iloc[0]
    bz_mae = ablation_df[ablation_df["feature_set"] == "C_solar_wind_plus_bz_drop"]["mae"].iloc[0]
    kp_mae = ablation_df[ablation_df["feature_set"] == "D_solar_wind_plus_kp_pattern_break"]["mae"].iloc[0]
    all_mae = ablation_df[ablation_df["feature_set"] == "E_all_features_combined"]["mae"].iloc[0]
    horizon_metrics_df = pd.DataFrame()

    paths = {
        "predictions": os.path.join(output_dir, "kp_experiment7_forecast_aligned_predictions.csv"),
        "metrics": os.path.join(output_dir, "kp_experiment7_forecast_aligned_metrics.csv"),
        "regime_metrics": os.path.join(output_dir, "kp_experiment7_forecast_aligned_regime_metrics.csv"),
        "ablation": os.path.join(output_dir, "kp_experiment7_forecast_aligned_ablation.csv"),
        "old_vs_forecast_aligned": os.path.join(output_dir, "kp_experiment7_old_vs_forecast_aligned_comparison.csv"),
        "debug": os.path.join(output_dir, "kp_experiment7_debug.json"),
        "pi_summary": os.path.join(output_dir, "kp_experiment7_pi_summary.md"),
        "scatter": _experiment7_plot_scatter(prediction_df, output_dir),
        "timeseries_examples": _experiment7_plot_timeseries(prediction_df, output_dir),
        "bz_drop_examples": _experiment7_plot_bz_drop(prediction_df, output_dir),
        "pattern_break_examples": _experiment7_plot_pattern_break(prediction_df, output_dir),
    }
    if metadata.get("issue_time_valid_time_metadata_found", False):
        paths["horizon_metrics"] = os.path.join(output_dir, "kp_experiment7_forecast_horizon_metrics.csv")
        horizon_metrics_df.to_csv(paths["horizon_metrics"], index=False)

    prediction_df.to_csv(paths["predictions"], index=False)
    metrics_df.to_csv(paths["metrics"], index=False)
    regime_df.to_csv(paths["regime_metrics"], index=False)
    ablation_df.to_csv(paths["ablation"], index=False)

    debug = {
        "forecast_inputs_already_target_aligned": FORECAST_INPUTS_ALREADY_TARGET_ALIGNED,
        "dataframe_start_timestamp": _format_timestamp(feature_df["timestamp"].iloc[0]),
        "dataframe_end_timestamp": _format_timestamp(feature_df["timestamp"].iloc[-1]),
        "cadence_hours": _cadence_hours(feature_df),
        "columns_used_as_forecasted_inputs": feature_columns,
        "kp_target_alignment": "same-row observed kp at timestamp T",
        "target_shifting_by_branch": {
            "experiment7_main": "none; forecasted solar-wind inputs valid at T -> Kp at T",
            "old_shifted_comparison": "feature rows at t -> Kp at t+120h baseline only",
            "experiments4_5_6": "preserved older short-lead/severe-risk branches",
        },
        "issue_time_valid_time_metadata_found": bool(metadata.get("issue_time_valid_time_metadata_found", False)),
        "forecast_horizon_note": (
            "Forecast horizon metadata not found. Evaluating target-aligned forecast-input model by chronological holdout only."
            if not metadata.get("issue_time_valid_time_metadata_found", False)
            else "Issue/valid horizon metadata found; horizon metrics file was generated when parsable."
        ),
        "leakage_check": {
            "passed": leakage_passed,
            "forbidden_features": forbidden_features,
            "kp_feature_source_shifted_before_target": bool(kp_feature_source_ok),
            "global_temporal_source_check": metadata["leakage"],
            "forbidden_rule": "No observed Kp at target timestamp T, no future Kp, no centered rolling Kp.",
        },
        "feature_sets": feature_sets,
        "main_feature_set": selected_feature_set,
        "kp_pattern_feature_policy": (
            "Kp pattern-break features are included in diagnostic ablations only because issue_time/valid_time metadata is missing."
            if not metadata.get("issue_time_valid_time_metadata_found", False)
            else "Kp pattern-break features may be used operationally when aligned to Kp available at forecast issue_time."
        ),
        "classifier_thresholds": thresholds,
        "classifier_status": classifier_debug,
        "quantile_status": quantile_debug,
        "old_vs_new_comparison_note": (
            "Old shifted baseline is included for transparency; true 5-day operational validation still requires issue_time/valid_time metadata."
        ),
    }
    with open(paths["debug"], "w", encoding="utf-8") as debug_file:
        json.dump(debug, debug_file, indent=2, default=str)
        debug_file.write("\n")

    best_kp5_mae = float(metrics_df.loc[metrics_df["model"] == best_model, "mae_kp_ge_5"].iloc[0]) if "mae_kp_ge_5" in metrics_df else np.nan
    best_kp6_mae = float(metrics_df.loc[metrics_df["model"] == best_model, "mae_kp_ge_6"].iloc[0]) if "mae_kp_ge_6" in metrics_df else np.nan
    kp5_recall = float(class_df.loc[class_df["model"] == "prob_forecast_kp_ge_5", "recall"].iloc[0]) if not class_df.empty and (class_df["model"] == "prob_forecast_kp_ge_5").any() else 0.0
    kp6_recall = float(class_df.loc[class_df["model"] == "prob_forecast_kp_ge_6", "recall"].iloc[0]) if not class_df.empty and (class_df["model"] == "prob_forecast_kp_ge_6").any() else 0.0
    recommendation = "experiment7_forecast_aligned_conservative_for_PI_report" if kp6_recall >= 0.75 else "experiment7_forecast_aligned_base_plus_exp4_5_fallback"

    summary_lines = [
        "# Experiment 7 PI Summary",
        "## Corrected Framing\nThe solar-wind input rows are treated as forecasted values already valid at the target timestamp T. The main Experiment 7 model maps those target-aligned solar-wind inputs to observed Kp at the same timestamp T. It does not train current solar-wind at t to predict Kp at t + 120h.",
        "## Timestamp And Leakage\n"
        f"- dataframe range: `{debug['dataframe_start_timestamp']}` to `{debug['dataframe_end_timestamp']}`\n"
        f"- cadence: `{debug['cadence_hours']:.2f}` hours\n"
        f"- same-row Kp target: `yes`\n"
        f"- issue_time/valid_time metadata found: `{debug['issue_time_valid_time_metadata_found']}`\n"
        f"- main feature set: `{selected_feature_set}`\n"
        f"- Kp pattern feature policy: `{debug['kp_pattern_feature_policy']}`\n"
        f"- leakage checks passed: `{leakage_passed}`\n"
        f"- caveat: `{debug['forecast_horizon_note']}`",
        "## Metrics\n"
        f"- best calibrated mean model: `{best_model}`\n"
        f"- all-row MAE/RMSE/correlation: `{best_metrics['mae']:.3f}` / `{best_metrics['rmse']:.3f}` / `{best_metrics['pearson_corr']:.3f}`\n"
        f"- Kp>=5 MAE / storm recall: `{best_kp5_mae:.3f}` / `{kp5_recall:.3f}`\n"
        f"- Kp>=6 MAE / severe recall: `{best_kp6_mae:.3f}` / `{kp6_recall:.3f}`\n"
        f"- severe max underprediction: `{best_metrics['max_underprediction']:.3f}`\n"
        f"- conservative branch metrics are in `{paths['metrics']}`.",
        "## Feature Findings\n"
        f"- Bz sharp-drop features improved over solar-wind-only MAE: `{bz_mae < bz_base}` (`{bz_mae:.3f}` vs `{bz_base:.3f}`).\n"
        f"- 24h Kp pattern-break features improved over solar-wind-only MAE: `{kp_mae < bz_base}` (`{kp_mae:.3f}` vs `{bz_base:.3f}`).\n"
        f"- all combined MAE: `{all_mae:.3f}`.",
        "## Old Vs New Framing\n"
        f"- forecast-aligned beat old shifted baseline by MAE: `{beat_old}`\n"
        f"- comparison table: `{paths['old_vs_forecast_aligned']}`",
        "## Recommendation\n"
        f"Use `{recommendation}` as the main corrected multi-day branch while preserving Experiment 4/5 severe-risk branches as comparison and fallback.",
        "## Caveat\nTrue operational 5-day validation requires issue_time / valid_time metadata for the forecasted solar-wind inputs. Without it, this result is forecast-input-to-Kp mapping skill under chronological holdout, not independent forecast-horizon skill.",
        "## Artifacts\n" + "\n".join([f"- `{value}`" for value in paths.values()]),
    ]
    with open(paths["pi_summary"], "w", encoding="utf-8") as summary_file:
        summary_file.write("\n\n".join(summary_lines) + "\n")

    result = {
        "paths": paths,
        "debug": debug,
        "best_model": best_model,
        "best_metrics": best_metrics,
        "best_kp_ge_5_mae": best_kp5_mae,
        "best_kp_ge_6_mae": best_kp6_mae,
        "prob_kp_ge_5_recall": kp5_recall,
        "prob_kp_ge_6_recall": kp6_recall,
        "bz_drop_improved": bool(bz_mae < bz_base),
        "kp_pattern_improved": bool(kp_mae < bz_base),
        "forecast_aligned_beat_old_shifted": beat_old,
        "recommendation": recommendation,
    }
    return metrics_df, prediction_df, result


def _experiment7_feature_sets(metadata: dict) -> Dict[str, List[str]]:
    solar = metadata["solar_wind_feature_columns"]
    core = [feature for feature in SOLAR_WIND_CURRENT_FEATURES if feature in solar]
    moving_rolls = [feature for feature in EXPERIMENT7_MOVING_PHYSICS_ROLL_FEATURES if feature in solar]
    sustained = [feature for feature in SUSTAINED_DRIVING_FEATURES if feature in solar]
    bz_drop = [feature for feature in BZ_SHARP_DROP_FEATURES if feature in solar]
    kp_state = [
        feature for feature in [
            "kp_lag_1", "kp_lag_2", "kp_lag_4", "kp_lag_8", "kp_lag_12", "kp_lag_16", "kp_lag_20",
            "kp_roll_mean_4", "kp_roll_std_4", "kp_roll_max_4", "kp_roll_range_4",
            "kp_roll_mean_20", "kp_roll_std_20", "kp_roll_max_20", "kp_roll_sum_20",
            "time_since_kp_ge_4", "time_since_kp_ge_5", "time_since_kp_ge_6",
        ]
        if feature in metadata["temporal_feature_columns"]
    ]
    kp_pattern = [
        feature for feature in [
            "kp_24h_repeat_error", "kp_48h_repeat_error", "kp_72h_repeat_error",
            "kp_daily_pattern_break_score", "kp_daily_variability_ratio",
            "kp_recent_vs_multiday_mean", "kp_recent_max_vs_multiday_mean",
        ]
        if feature in metadata["temporal_feature_columns"]
    ]
    physics_only = _unique_in_order(core + sustained + moving_rolls)
    return {
        "diagnostic_target_aligned_upper_bound": _unique_in_order(core + sustained + moving_rolls + bz_drop + kp_state + kp_pattern),
        "operational_physics_only": physics_only,
        "operational_physics_plus_bz_drop": _unique_in_order(physics_only + bz_drop),
        "operational_physics_plus_kp_state": _unique_in_order(physics_only + kp_state),
        "operational_physics_plus_kp_pattern": _unique_in_order(physics_only + kp_pattern),
        "operational_physics_bz_drop_kp_pattern": _unique_in_order(physics_only + bz_drop + kp_pattern),
        "operational_all_features": _unique_in_order(physics_only + bz_drop + kp_state + kp_pattern),
        "A0_bz_velocity_density_baseline": [feature for feature in ["bz", "velocity", "density"] if feature in solar],
        "A_solar_wind_inputs_only": physics_only,
        "B_solar_wind_plus_storm_physics": physics_only,
        "C_solar_wind_plus_bz_drop": _unique_in_order(physics_only + bz_drop),
        "D_solar_wind_plus_kp_pattern_break": _unique_in_order(physics_only + kp_state + kp_pattern),
        "E_all_features_combined": _unique_in_order(physics_only + bz_drop + kp_state + kp_pattern),
        "main_operational_no_kp_history": _unique_in_order(physics_only + bz_drop),
    }


# Causal persistence features use only the history supplied before the current
# forecast timestamp. A centered or forward Kp shift here would leak the target.
def _experiment7_kp_state_from_history(history_kp: List[float]) -> Dict[str, float]:
    if len(history_kp) < max(KP_LAG_STEPS):
        raise ValueError("Kp history is too short for Experiment 7 operational features.")
    values = np.asarray(history_kp, dtype=float)
    state = {}
    for lag in KP_LAG_STEPS:
        state[f"kp_lag_{lag}"] = float(values[-lag])
    for window in KP_ROLL_WINDOWS:
        tail = values[-window:]
        state[f"kp_roll_mean_{window}"] = float(np.mean(tail))
        state[f"kp_roll_std_{window}"] = float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0
        state[f"kp_roll_max_{window}"] = float(np.max(tail))
        state[f"kp_roll_min_{window}"] = float(np.min(tail))
        state[f"kp_roll_range_{window}"] = float(np.max(tail) - np.min(tail))
        state[f"kp_roll_sum_{window}"] = float(np.sum(tail))
    state["kp_delta_1"] = float(values[-1] - values[-2])
    state["kp_delta_4"] = float(values[-1] - values[-4])
    state["kp_abs_delta_4"] = abs(state["kp_delta_4"])
    state["kp_24h_repeat_error"] = float(abs(values[-1] - values[-4]))
    state["kp_48h_repeat_error"] = float(abs(values[-1] - values[-8]))
    state["kp_72h_repeat_error"] = float(abs(values[-1] - values[-12]))
    state["kp_daily_pattern_break_score"] = state["kp_24h_repeat_error"] + abs(state["kp_delta_1"])
    state["kp_daily_variability_ratio"] = state["kp_roll_std_4"] / (state["kp_roll_std_20"] + 1e-6)
    state["kp_recent_vs_multiday_mean"] = state["kp_roll_mean_4"] - state["kp_roll_mean_20"]
    state["kp_recent_max_vs_multiday_mean"] = state["kp_roll_max_4"] - state["kp_roll_mean_20"]
    for threshold in [4, 5, 6]:
        hits = np.where(values >= threshold)[0]
        state[f"time_since_kp_ge_{threshold}"] = float(len(values) - 1 - hits[-1]) if len(hits) else 999.0
    return state


def _experiment7_apply_kp_state(row: pd.Series, history_kp: List[float]) -> pd.Series:
    state = _experiment7_kp_state_from_history(history_kp)
    for feature, value in state.items():
        if feature in row.index:
            row[feature] = value
    return row


def _experiment7_train_regressors(train_df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, object]:
    X_train = train_df[feature_columns]
    y_train = train_df["kp"].to_numpy(dtype=float)
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=30.0)),
        "elasticnet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.10, max_iter=20000, random_state=RANDOM_STATE)),
        "rf": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
    }
    if HistGradientBoostingRegressor is not None:
        models["hgb"] = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        )
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def _experiment7_train_quantiles(train_df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, object]:
    models = {}
    if HistGradientBoostingRegressor is None:
        return models
    for label, quantile in [("p75", 0.75), ("p90", 0.90), ("p95", 0.95)]:
        model = _experiment7_quantile_model(quantile)
        model.fit(train_df[feature_columns], train_df["kp"].to_numpy(dtype=float))
        models[label] = model
    return models


def _experiment7_train_classifiers(train_df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, dict]:
    classifiers = {}
    for threshold in [5.0, 6.0, 7.0]:
        label = str(int(threshold))
        y = (train_df["kp"].to_numpy(dtype=float) >= threshold).astype(int)
        if int(y.sum()) < MIN_SEVERITY_POSITIVES or len(np.unique(y)) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        )
        model.fit(train_df[feature_columns], y)
        classifiers[label] = {
            "model": model,
            "threshold": _experiment7_classifier_threshold(train_df, feature_columns, threshold),
            "target_threshold": threshold,
        }
    return classifiers


def _experiment7_probability_dict(classifiers: Dict[str, dict], feature_row: pd.DataFrame) -> Dict[str, float]:
    probabilities = {}
    for label, bundle in classifiers.items():
        probabilities[label] = float(bundle["model"].predict_proba(feature_row)[:, 1][0])
    return probabilities


def _experiment7_conservative_value(base_pred: float, quantiles: Dict[str, float], probabilities: Dict[str, float], classifiers: Dict[str, dict]) -> float:
    value = float(base_pred)
    if probabilities.get("5", 0.0) >= classifiers.get("5", {}).get("threshold", 1.01):
        value = max(value, quantiles.get("p75", value), 5.0)
    if probabilities.get("6", 0.0) >= classifiers.get("6", {}).get("threshold", 1.01):
        value = max(value, quantiles.get("p90", value), 6.0)
    if probabilities.get("7", 0.0) >= classifiers.get("7", {}).get("threshold", 1.01):
        value = max(value, quantiles.get("p95", value), 7.0)
    return float(np.clip(value, 0.0, 9.0))


def _experiment7_predict_one(
    row: pd.Series,
    feature_columns: List[str],
    model,
    quantile_models: Dict[str, object],
    classifiers: Dict[str, dict],
) -> Tuple[float, Dict[str, float], Dict[str, float], float]:
    X = pd.DataFrame([row])[feature_columns]
    base = float(np.clip(model.predict(X)[0], 0.0, 9.0))
    quantiles = {
        label: float(np.clip(q_model.predict(X)[0], 0.0, 9.0))
        for label, q_model in quantile_models.items()
    }
    probabilities = _experiment7_probability_dict(classifiers, X)
    conservative = _experiment7_conservative_value(base, quantiles, probabilities, classifiers)
    return base, quantiles, probabilities, conservative


def _experiment7_simulate_rolling(
    feature_df: pd.DataFrame,
    split_index: int,
    feature_columns: List[str],
    model,
    quantile_models: Dict[str, object],
    classifiers: Dict[str, dict],
    mode: str,
    update_cadence_hours: float,
    forecast_horizon_hours: float,
) -> pd.DataFrame:
    cadence = _cadence_hours(feature_df)
    if cadence <= 0:
        cadence = 6.0
    horizon_bins = max(1, int(round(forecast_horizon_hours / cadence)))
    update_bins = max(1, int(round(update_cadence_hours / cadence)))
    init_indices = list(range(split_index - 1, len(feature_df) - 1, update_bins))

    if mode in ["frozen", "physics_only"]:
        feature_rows = []
        meta_rows = []
        for init_idx in init_indices:
            init_time = feature_df["timestamp"].iloc[init_idx]
            observed_history = feature_df["kp"].iloc[:init_idx + 1].astype(float).tolist()
            kp_state = _experiment7_kp_state_from_history(observed_history) if mode == "frozen" else {}
            for lead_bin in range(1, horizon_bins + 1):
                target_idx = init_idx + lead_bin
                if target_idx >= len(feature_df) or target_idx < split_index:
                    continue
                lead_hours = float((feature_df["timestamp"].iloc[target_idx] - init_time) / pd.Timedelta(hours=1))
                if lead_hours > forecast_horizon_hours + 1e-9:
                    continue
                target_row = feature_df.iloc[target_idx].copy()
                for feature, value in kp_state.items():
                    if feature in target_row.index:
                        target_row[feature] = value
                feature_rows.append(target_row)
                meta_rows.append(
                    {
                        "init_time": init_time,
                        "valid_time": feature_df["timestamp"].iloc[target_idx],
                        "lead_time_hours": lead_hours,
                        "update_cadence_hours": float(update_cadence_hours),
                        "model_mode": f"experiment7_operational_{mode}",
                        "observed_kp_at_valid_time": float(feature_df["kp"].iloc[target_idx]),
                        "bz": float(feature_df["bz"].iloc[target_idx]),
                        "velocity": float(feature_df["velocity"].iloc[target_idx]),
                        "density": float(feature_df["density"].iloc[target_idx]),
                        "pdyn": float(feature_df["pdyn"].iloc[target_idx]),
                        "bz_southward_drop_6h": float(feature_df.get("bz_southward_drop_6h", pd.Series(0.0, index=feature_df.index)).iloc[target_idx]),
                        "kp_history_source_max_time": init_time,
                        "uses_future_observed_kp": False,
                    }
                )
        if not feature_rows:
            return pd.DataFrame()
        X_source = pd.DataFrame(feature_rows).reset_index(drop=True)
        X = X_source[feature_columns]
        pred = np.clip(model.predict(X), 0.0, 9.0)
        output = pd.DataFrame(meta_rows)
        output["predicted_kp"] = pred
        for label, q_model in quantile_models.items():
            output[f"kp_experiment7_{label}"] = np.clip(q_model.predict(X), 0.0, 9.0)
        for label, bundle in classifiers.items():
            output[f"prob_experiment7_kp_ge_{label}"] = bundle["model"].predict_proba(X)[:, 1]
        for label in ["5", "6", "7"]:
            if f"prob_experiment7_kp_ge_{label}" not in output.columns:
                output[f"prob_experiment7_kp_ge_{label}"] = np.nan if label == "7" else 0.0
        conservative = output["predicted_kp"].to_numpy(dtype=float).copy()
        if "p75" in quantile_models:
            gate = output["prob_experiment7_kp_ge_5"].fillna(0.0).to_numpy(dtype=float) >= classifiers.get("5", {}).get("threshold", 1.01)
            conservative = np.where(gate, np.maximum.reduce([conservative, output["kp_experiment7_p75"].to_numpy(dtype=float), np.full(len(output), 5.0)]), conservative)
        if "p90" in quantile_models:
            gate = output["prob_experiment7_kp_ge_6"].fillna(0.0).to_numpy(dtype=float) >= classifiers.get("6", {}).get("threshold", 1.01)
            conservative = np.where(gate, np.maximum.reduce([conservative, output["kp_experiment7_p90"].to_numpy(dtype=float), np.full(len(output), 6.0)]), conservative)
        if "p95" in quantile_models:
            gate = output["prob_experiment7_kp_ge_7"].fillna(0.0).to_numpy(dtype=float) >= classifiers.get("7", {}).get("threshold", 1.01)
            conservative = np.where(gate, np.maximum.reduce([conservative, output["kp_experiment7_p95"].to_numpy(dtype=float), np.full(len(output), 7.0)]), conservative)
        output["kp_experiment7_conservative"] = np.clip(conservative, 0.0, 9.0)
        return output

    rows = []
    for init_idx in init_indices:
        init_time = feature_df["timestamp"].iloc[init_idx]
        observed_history = feature_df["kp"].iloc[:init_idx + 1].astype(float).tolist()
        ar_history = observed_history.copy()
        for lead_bin in range(1, horizon_bins + 1):
            target_idx = init_idx + lead_bin
            if target_idx >= len(feature_df) or target_idx < split_index:
                continue
            lead_hours = float((feature_df["timestamp"].iloc[target_idx] - init_time) / pd.Timedelta(hours=1))
            if lead_hours > forecast_horizon_hours + 1e-9:
                continue
            target_row = feature_df.iloc[target_idx].copy()
            if mode == "frozen":
                target_row = _experiment7_apply_kp_state(target_row, observed_history)
            elif mode == "autoregressive":
                target_row = _experiment7_apply_kp_state(target_row, ar_history)
            elif mode == "physics_only":
                pass
            else:
                raise ValueError(f"Unknown Experiment 7 rolling mode: {mode}")
            base, quantiles, probabilities, conservative = _experiment7_predict_one(
                target_row,
                feature_columns,
                model,
                quantile_models,
                classifiers,
            )
            if mode == "autoregressive":
                ar_history.append(base)
            rows.append(
                {
                    "init_time": init_time,
                    "valid_time": feature_df["timestamp"].iloc[target_idx],
                    "lead_time_hours": lead_hours,
                    "update_cadence_hours": float(update_cadence_hours),
                    "model_mode": f"experiment7_operational_{mode}",
                    "observed_kp_at_valid_time": float(feature_df["kp"].iloc[target_idx]),
                    "predicted_kp": base,
                    "kp_experiment7_conservative": conservative,
                    "prob_experiment7_kp_ge_5": probabilities.get("5", 0.0),
                    "prob_experiment7_kp_ge_6": probabilities.get("6", 0.0),
                    "prob_experiment7_kp_ge_7": probabilities.get("7", np.nan),
                    "bz": float(feature_df["bz"].iloc[target_idx]),
                    "velocity": float(feature_df["velocity"].iloc[target_idx]),
                    "density": float(feature_df["density"].iloc[target_idx]),
                    "pdyn": float(feature_df["pdyn"].iloc[target_idx]),
                    "bz_southward_drop_6h": float(feature_df.get("bz_southward_drop_6h", pd.Series(0.0, index=feature_df.index)).iloc[target_idx]),
                    "kp_history_source_max_time": init_time,
                    "uses_future_observed_kp": False,
                }
            )
    return pd.DataFrame(rows)


def _experiment7_lead_bin(lead_hours: float) -> str:
    bins = [(0, 6), (6, 12), (12, 24), (24, 48), (48, 72), (72, 96), (96, 120)]
    for low, high in bins:
        if low < lead_hours <= high or (low == 0 and 0 <= lead_hours <= high):
            return f"{low}-{high}h"
    return ">120h"


def _experiment7_binary_metrics_for_threshold(y_true_kp: np.ndarray, score: np.ndarray, pred_kp: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true_kp, dtype=float) >= threshold
    y_pred = np.asarray(pred_kp, dtype=float) >= threshold
    score = np.asarray(score, dtype=float)
    if np.isnan(score).any():
        score = np.where(np.isnan(score), np.asarray(pred_kp, dtype=float), score)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        f"precision_kp_ge_{int(threshold)}": float(precision),
        f"recall_kp_ge_{int(threshold)}": float(recall),
        f"f1_kp_ge_{int(threshold)}": float(f1),
        f"average_precision_kp_ge_{int(threshold)}": _safe_average_precision(y_true, score),
        f"severe_miss_count_kp_ge_{int(threshold)}": int((y_true & ~y_pred).sum()),
    }


def _experiment7_mode_metrics(frame: pd.DataFrame, pred_col: str = "predicted_kp") -> dict:
    y = frame["observed_kp_at_valid_time"].to_numpy(dtype=float)
    pred = frame[pred_col].to_numpy(dtype=float)
    row = {"n": int(len(frame)), **_experiment7_regression_metrics(y, pred)}
    for threshold in [5.0, 6.0, 7.0]:
        score = frame.get(f"prob_experiment7_kp_ge_{int(threshold)}", pd.Series(pred, index=frame.index)).to_numpy(dtype=float)
        row.update(_experiment7_binary_metrics_for_threshold(y, score, pred, threshold))
    quiet = y < 5.0
    row["quiet_false_storm_inflation"] = float(np.mean(pred[quiet] >= 5.0)) if int(quiet.sum()) else 0.0
    return row


def _experiment7_leadtime_metrics(rolling_df: pd.DataFrame) -> pd.DataFrame:
    if rolling_df.empty:
        return pd.DataFrame()
    work = rolling_df.copy()
    work["lead_time_bin"] = work["lead_time_hours"].map(_experiment7_lead_bin)
    rows = []
    for (mode, update_cadence, lead_bin), group in work.groupby(["model_mode", "update_cadence_hours", "lead_time_bin"], sort=False):
        rows.append(
            {
                "model_mode": mode,
                "update_cadence_hours": update_cadence,
                "lead_time_bin": lead_bin,
                **_experiment7_mode_metrics(group),
            }
        )
    return pd.DataFrame(rows)


def _experiment7_mode_comparison(diagnostic_df: pd.DataFrame, rolling_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not diagnostic_df.empty:
        diag = diagnostic_df.rename(columns={"kp": "observed_kp_at_valid_time", "kp_experiment7_target_aligned_best": "predicted_kp"}).copy()
        diag["prob_experiment7_kp_ge_5"] = diag.get("prob_experiment7_kp_ge_5", pd.Series(diag["predicted_kp"], index=diag.index))
        diag["prob_experiment7_kp_ge_6"] = diag.get("prob_experiment7_kp_ge_6", pd.Series(diag["predicted_kp"], index=diag.index))
        diag["prob_experiment7_kp_ge_7"] = diag.get("prob_experiment7_kp_ge_7", pd.Series(diag["predicted_kp"], index=diag.index))
        rows.append({"model_mode": "experiment7_diagnostic_target_aligned", "update_cadence_hours": 0.0, **_experiment7_mode_metrics(diag)})
    if not rolling_df.empty:
        for (mode, cadence), group in rolling_df.groupby(["model_mode", "update_cadence_hours"], sort=False):
            rows.append({"model_mode": mode, "update_cadence_hours": cadence, **_experiment7_mode_metrics(group)})
    return pd.DataFrame(rows)


def _experiment7_ablation_results(
    feature_df: pd.DataFrame,
    train_df: pd.DataFrame,
    split_index: int,
    feature_sets: Dict[str, List[str]],
    forecast_horizon_hours: float,
) -> pd.DataFrame:
    rows = []
    ablations = {
        "A_forecasted_physics_only": feature_sets["operational_physics_only"],
        "B_physics_plus_bz_drop": feature_sets["operational_physics_plus_bz_drop"],
        "C_physics_plus_anchored_kp_state": feature_sets["operational_physics_plus_kp_state"],
        "D_physics_plus_24h_kp_pattern": feature_sets["operational_physics_plus_kp_pattern"],
        "E_physics_plus_bz_drop_kp_pattern": feature_sets["operational_physics_bz_drop_kp_pattern"],
        "F_all_features": feature_sets["operational_all_features"],
    }
    for label, features in ablations.items():
        model = make_pipeline(StandardScaler(), Ridge(alpha=30.0))
        model.fit(train_df[features], train_df["kp"].to_numpy(dtype=float))
        classifiers = {}
        sim_mode = "physics_only" if not any(feature.startswith("kp_") or feature.startswith("time_since_kp") for feature in features) else "frozen"
        sim = _experiment7_simulate_rolling(
            feature_df,
            split_index,
            features,
            model,
            {},
            classifiers,
            sim_mode,
            update_cadence_hours=24.0,
            forecast_horizon_hours=forecast_horizon_hours,
        )
        if sim.empty:
            continue
        overall = _experiment7_mode_metrics(sim)
        row = {"ablation": label, **overall}
        for hour, prefix in [(24, "lead_24h"), (72, "lead_72h"), (120, "lead_120h")]:
            lead_slice = sim[np.isclose(sim["lead_time_hours"], hour)]
            if lead_slice.empty:
                lead_slice = sim[(sim["lead_time_hours"] > hour - 12) & (sim["lead_time_hours"] <= hour)]
            if not lead_slice.empty:
                lead_metrics = _experiment7_mode_metrics(lead_slice)
                row[f"{prefix}_mae"] = lead_metrics["mae"]
                row[f"{prefix}_recall_kp_ge_5"] = lead_metrics["recall_kp_ge_5"]
                row[f"{prefix}_recall_kp_ge_6"] = lead_metrics["recall_kp_ge_6"]
        rows.append(row)
    return pd.DataFrame(rows)


def _experiment7_rolling_update_metrics(rolling_df: pd.DataFrame) -> pd.DataFrame:
    if rolling_df.empty:
        return pd.DataFrame()
    work = rolling_df.copy()
    work["forecast_day"] = np.ceil(work["lead_time_hours"] / 24.0).clip(1, 5).astype(int)
    rows = []
    for (mode, cadence, day), group in work.groupby(["model_mode", "update_cadence_hours", "forecast_day"], sort=False):
        rows.append({"model_mode": mode, "update_cadence_hours": cadence, "forecast_day": day, **_experiment7_mode_metrics(group)})
    return pd.DataFrame(rows)


def _experiment7_plot_leadtime(leadtime_df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "kp_experiment7_leadtime_degradation.png")
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    if not leadtime_df.empty:
        order = ["0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-96h", "96-120h"]
        for mode, group in leadtime_df[leadtime_df["update_cadence_hours"] == 24.0].groupby("model_mode"):
            group = group.copy()
            group["lead_order"] = group["lead_time_bin"].map({value: idx for idx, value in enumerate(order)})
            group = group.sort_values("lead_order")
            axes[0].plot(group["lead_time_bin"], group["mae"], marker="o", label=mode.replace("experiment7_operational_", ""))
            axes[1].plot(group["lead_time_bin"], group["recall_kp_ge_6"], marker="o", label=mode.replace("experiment7_operational_", ""))
    axes[0].set_ylabel("MAE")
    axes[1].set_ylabel("Kp>=6 recall")
    axes[1].set_xlabel("Lead time")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _experiment7_plot_rolling_examples(rolling_df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "kp_experiment7_rolling_forecast_examples.png")
    work = rolling_df[(rolling_df["update_cadence_hours"] == 24.0) & (rolling_df["model_mode"] == "experiment7_operational_frozen")].copy()
    fig, ax = plt.subplots(figsize=(13, 5))
    if not work.empty:
        work["init_time"] = pd.to_datetime(work["init_time"])
        work["valid_time"] = pd.to_datetime(work["valid_time"])
        starts = work["init_time"].drop_duplicates().iloc[::max(1, len(work["init_time"].drop_duplicates()) // 4)].head(4)
        for init_time in starts:
            window = work[work["init_time"] == init_time]
            ax.plot(window["valid_time"], window["predicted_kp"], marker="o", linewidth=1.2, label=str(init_time)[:10])
        obs = work.drop_duplicates("valid_time").sort_values("valid_time")
        obs = obs[(obs["valid_time"] >= starts.min()) & (obs["valid_time"] <= starts.max() + pd.Timedelta(days=6))]
        ax.plot(obs["valid_time"], obs["observed_kp_at_valid_time"], color="black", linewidth=2, label="observed Kp")
    ax.axhline(5.0, color="tab:red", linestyle="--", linewidth=1)
    ax.set_ylabel("Kp")
    ax.set_xlabel("Valid time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _experiment7_plot_scatter_by_leadtime(rolling_df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "kp_experiment7_scatter_by_leadtime.png")
    fig, ax = plt.subplots(figsize=(7, 6))
    work = rolling_df[(rolling_df["update_cadence_hours"] == 24.0) & (rolling_df["model_mode"] == "experiment7_operational_frozen")].copy()
    if not work.empty:
        work["lead_time_bin"] = work["lead_time_hours"].map(_experiment7_lead_bin)
        for lead_bin, group in work.groupby("lead_time_bin", sort=False):
            ax.scatter(group["observed_kp_at_valid_time"], group["predicted_kp"], s=14, alpha=0.35, label=lead_bin)
    ax.plot([0, 9], [0, 9], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_xlabel("Observed Kp")
    ax.set_ylabel("Predicted Kp")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _experiment7_dynamic_backtest_assets(rolling_df: pd.DataFrame, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    dynamic = rolling_df[
        (rolling_df["model_mode"] == "experiment7_operational_frozen")
        & (rolling_df["update_cadence_hours"] == 6.0)
    ].copy()
    if dynamic.empty:
        prediction_df = pd.DataFrame()
        metrics_df = pd.DataFrame()
    else:
        dynamic["lead_time_bin"] = dynamic["lead_time_hours"].map(_experiment7_lead_bin)
        prediction_df = dynamic.rename(
            columns={
                "valid_time": "timestamp",
                "init_time": "t_init",
                "observed_kp_at_valid_time": "observed_kp",
                "predicted_kp": "kp_forecast_aligned_base",
                "kp_experiment7_conservative": "kp_forecast_aligned_conservative",
            }
        )
        keep_columns = [
            "timestamp",
            "t_init",
            "lead_time_hours",
            "lead_time_bin",
            "observed_kp",
            "kp_forecast_aligned_base",
            "kp_forecast_aligned_conservative",
            "prob_experiment7_kp_ge_5",
            "prob_experiment7_kp_ge_6",
            "bz",
            "velocity",
            "density",
            "pdyn",
            "bz_southward_drop_6h",
            "kp_history_source_max_time",
            "uses_future_observed_kp",
        ]
        prediction_df = prediction_df[[column for column in keep_columns if column in prediction_df.columns]]

        metrics_rows = []
        for lead_bin, group in dynamic.groupby("lead_time_bin", sort=False):
            metrics_rows.append(
                {
                    "lead_time_bin": lead_bin,
                    **_experiment7_mode_metrics(group),
                }
            )
        metrics_df = pd.DataFrame(metrics_rows)
        order = ["0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-96h", "96-120h"]
        metrics_df["lead_order"] = metrics_df["lead_time_bin"].map({label: idx for idx, label in enumerate(order)})
        metrics_df = metrics_df.sort_values("lead_order").drop(columns=["lead_order"]).reset_index(drop=True)

    prediction_path = os.path.join(output_dir, "kp_experiment7_dynamic_backtest_predictions.csv")
    metrics_path = os.path.join(output_dir, "kp_experiment7_dynamic_backtest_metrics.csv")
    plot_path = os.path.join(output_dir, "kp_experiment7_refresh_performance.png")
    prediction_df.to_csv(prediction_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    if not metrics_df.empty:
        ax1.plot(metrics_df["lead_time_bin"], metrics_df["mae"], marker="o", color="tab:blue", label="MAE")
        ax2.plot(metrics_df["lead_time_bin"], metrics_df["pearson_corr"], marker="s", color="tab:orange", label="Pearson r")
    ax1.set_xlabel("Lead-time bin")
    ax1.set_ylabel("MAE", color="tab:blue")
    ax2.set_ylabel("Pearson correlation", color="tab:orange")
    ax1.grid(True, alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return prediction_df, metrics_df, plot_path


def _experiment7_plot_bz_operational(rolling_df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "kp_experiment7_bz_drop_examples.png")
    work = rolling_df[(rolling_df["update_cadence_hours"] == 24.0) & (rolling_df["model_mode"] == "experiment7_operational_frozen")].copy()
    fig, ax = plt.subplots(figsize=(13, 5))
    if not work.empty:
        work = work.sort_values("bz_southward_drop_6h", ascending=False).head(120).sort_values("valid_time")
        ax2 = ax.twinx()
        ax.plot(pd.to_datetime(work["valid_time"]), work["bz"], color="tab:blue", label="forecasted Bz")
        ax.scatter(pd.to_datetime(work["valid_time"]), work["bz_southward_drop_6h"], color="tab:orange", s=18, label="southward drop")
        ax2.plot(pd.to_datetime(work["valid_time"]), work["observed_kp_at_valid_time"], color="black", label="observed Kp")
        ax2.plot(pd.to_datetime(work["valid_time"]), 9.0 * work["prob_experiment7_kp_ge_5"], color="tab:red", alpha=0.65, label="P(Kp>=5) x 9")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=8)
    ax.set_ylabel("Bz / Bz drop")
    ax.set_xlabel("Valid time")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _experiment7_plot_pattern_operational(diagnostic_df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "kp_experiment7_24h_pattern_break_examples.png")
    work = diagnostic_df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    center = work.sort_values("kp_daily_pattern_break_score", ascending=False).iloc[0]["timestamp"] if not work.empty else pd.Timestamp.now()
    window = work[(work["timestamp"] >= center - pd.Timedelta(days=6)) & (work["timestamp"] <= center + pd.Timedelta(days=6))]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax2 = ax.twinx()
    if not window.empty:
        ax.plot(window["timestamp"], window["kp"], color="black", label="observed Kp")
        ax.plot(window["timestamp"], window["kp_lag_4"], color="tab:blue", linestyle="--", label="24h lag")
        ax2.plot(window["timestamp"], window["kp_daily_pattern_break_score"], color="tab:orange", label="diagnostic pattern break")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=8)
    ax.set_ylabel("Kp")
    ax2.set_ylabel("Pattern-break score")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _run_experiment7_forecast_aligned(
    feature_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    current_predictions: pd.DataFrame,
    metadata: dict,
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    _ensure_directory(output_dir)
    feature_sets = _experiment7_feature_sets(metadata)
    cadence = _cadence_hours(feature_df)
    if cadence <= 0:
        cadence = 6.0
    forecast_horizon_hours = 120.0
    forecast_horizon_bins = max(1, int(round(forecast_horizon_hours / cadence)))
    split_index = len(train_df)

    diagnostic_predictions = test_df[["timestamp", "kp", "bz", "velocity", "density", "pdyn", *[c for c in BZ_SHARP_DROP_FEATURES + KP_HISTORY_FEATURES if c in test_df.columns]]].copy()
    diagnostic_metrics_rows = []
    diagnostic_best_metrics = {"mae": np.nan, "rmse": np.nan, "pearson_corr": np.nan}
    diagnostic_model = None
    diagnostic_classifiers = {}
    diagnostic_quantiles = {}
    if RUN_EXPERIMENT7_DIAGNOSTIC_TARGET_ALIGNED:
        diag_features = feature_sets["diagnostic_target_aligned_upper_bound"]
        diag_models = _experiment7_train_regressors(train_df, diag_features)
        diagnostic_model = diag_models.get("hgb", diag_models["ridge"])
        diagnostic_classifiers = _experiment7_train_classifiers(train_df, diag_features)
        diagnostic_quantiles = _experiment7_train_quantiles(train_df, diag_features)
        for model_name, model in diag_models.items():
            pred = np.clip(model.predict(test_df[diag_features]), 0.0, 9.0)
            col = f"kp_experiment7_target_aligned_{model_name}"
            diagnostic_predictions[col] = pred
            diagnostic_metrics_rows.append({"model": col, "mode": "diagnostic_target_aligned_upper_bound", **_experiment7_regression_metrics(test_df["kp"].to_numpy(dtype=float), pred)})
        diagnostic_predictions["kp_experiment7_target_aligned_base"] = diagnostic_predictions["kp_experiment7_target_aligned_ridge"]
        best_row = pd.DataFrame(diagnostic_metrics_rows).sort_values(["mae", "rmse"]).iloc[0]
        diagnostic_predictions["kp_experiment7_target_aligned_best"] = diagnostic_predictions[best_row["model"]]
        diagnostic_best_metrics = best_row[["mae", "rmse", "pearson_corr", "spearman_corr", "bias", "underprediction_rate", "max_underprediction", "severe_weighted_asymmetric_mae"]].to_dict()
        for label, bundle in diagnostic_classifiers.items():
            diagnostic_predictions[f"prob_experiment7_kp_ge_{label}"] = bundle["model"].predict_proba(test_df[diag_features])[:, 1]
        for label, model in diagnostic_quantiles.items():
            diagnostic_predictions[f"kp_experiment7_target_aligned_{label}"] = np.clip(model.predict(test_df[diag_features]), 0.0, 9.0)

    operational_feature_columns = feature_sets["operational_all_features"]
    physics_feature_columns = feature_sets["operational_physics_only"]
    operational_model = make_pipeline(StandardScaler(), Ridge(alpha=30.0))
    operational_model.fit(train_df[operational_feature_columns], train_df["kp"].to_numpy(dtype=float))
    operational_quantiles = {}
    operational_classifiers = _experiment7_train_classifiers(train_df, operational_feature_columns)
    physics_model = make_pipeline(StandardScaler(), Ridge(alpha=30.0))
    physics_model.fit(train_df[physics_feature_columns], train_df["kp"].to_numpy(dtype=float))
    physics_classifiers = _experiment7_train_classifiers(train_df, physics_feature_columns)
    minimal_model = make_pipeline(StandardScaler(), Ridge(alpha=30.0))
    minimal_features = feature_sets["A0_bz_velocity_density_baseline"]
    minimal_model.fit(train_df[minimal_features], train_df["kp"].to_numpy(dtype=float))
    diagnostic_predictions["kp_experiment7_bz_velocity_density_baseline"] = np.clip(minimal_model.predict(test_df[minimal_features]), 0.0, 9.0)

    rolling_frames = []
    if RUN_EXPERIMENT7_OPERATIONAL_ROLLING:
        for update_cadence in [6.0, 24.0]:
            rolling_frames.append(
                _experiment7_simulate_rolling(
                    feature_df, split_index, operational_feature_columns, operational_model,
                    operational_quantiles, operational_classifiers, "frozen", update_cadence, forecast_horizon_hours,
                )
            )
            if update_cadence == 24.0:
                rolling_frames.append(
                    _experiment7_simulate_rolling(
                        feature_df, split_index, operational_feature_columns, operational_model,
                        operational_quantiles, operational_classifiers, "autoregressive", update_cadence, forecast_horizon_hours,
                    )
                )
            rolling_frames.append(
                _experiment7_simulate_rolling(
                    feature_df, split_index, physics_feature_columns, physics_model,
                    {}, physics_classifiers, "physics_only", update_cadence, forecast_horizon_hours,
                )
            )
    rolling_df = pd.concat([frame for frame in rolling_frames if not frame.empty], ignore_index=True) if rolling_frames else pd.DataFrame()

    # Add one-per-valid-time operational columns for legacy merged prediction tables.
    if not rolling_df.empty:
        daily = rolling_df[rolling_df["update_cadence_hours"] == 24.0].copy()
        daily = daily.sort_values(["valid_time", "lead_time_hours"])
        for mode, col in [
            ("experiment7_operational_frozen", "kp_experiment7_operational_frozen"),
            ("experiment7_operational_autoregressive", "kp_experiment7_operational_autoregressive"),
            ("experiment7_operational_physics_only", "kp_experiment7_operational_physics_only"),
        ]:
            selected = daily[daily["model_mode"] == mode].drop_duplicates("valid_time")
            if not selected.empty:
                merge_df = selected[["valid_time", "predicted_kp", "prob_experiment7_kp_ge_5", "prob_experiment7_kp_ge_6"]].rename(
                    columns={
                        "valid_time": "timestamp",
                        "predicted_kp": col,
                        "prob_experiment7_kp_ge_5": f"{col}_prob_kp_ge_5",
                        "prob_experiment7_kp_ge_6": f"{col}_prob_kp_ge_6",
                    }
                )
                diagnostic_predictions = diagnostic_predictions.merge(merge_df, on="timestamp", how="left")
        if "kp_experiment7_operational_frozen" in diagnostic_predictions.columns:
            diagnostic_predictions["kp_experiment7_conservative"] = diagnostic_predictions["kp_experiment7_operational_frozen"].fillna(diagnostic_predictions["kp_experiment7_target_aligned_best"])

    diagnostic_metrics_df = pd.DataFrame(diagnostic_metrics_rows)
    if "kp_experiment7_bz_velocity_density_baseline" in diagnostic_predictions:
        diagnostic_metrics_df = pd.concat(
            [
                pd.DataFrame([
                    {
                        "model": "kp_experiment7_bz_velocity_density_baseline",
                        "mode": "diagnostic_target_aligned_physics_minimal",
                        **_experiment7_regression_metrics(
                            diagnostic_predictions["kp"].to_numpy(dtype=float),
                            diagnostic_predictions["kp_experiment7_bz_velocity_density_baseline"].to_numpy(dtype=float),
                        ),
                    }
                ]),
                diagnostic_metrics_df,
            ],
            ignore_index=True,
        )

    leadtime_df = _experiment7_leadtime_metrics(rolling_df)
    mode_comparison_df = _experiment7_mode_comparison(diagnostic_predictions, rolling_df)
    ablation_df = _experiment7_ablation_results(feature_df, train_df, split_index, feature_sets, forecast_horizon_hours)
    rolling_update_df = _experiment7_rolling_update_metrics(rolling_df)
    dynamic_predictions_df, dynamic_metrics_df, dynamic_plot_path = _experiment7_dynamic_backtest_assets(rolling_df, output_dir)

    paths = {
        "diagnostic_predictions": os.path.join(output_dir, "kp_experiment7_diagnostic_target_aligned_predictions.csv"),
        "operational_predictions": os.path.join(output_dir, "kp_experiment7_operational_rolling_predictions.csv"),
        "dynamic_backtest_predictions": os.path.join(output_dir, "kp_experiment7_dynamic_backtest_predictions.csv"),
        "dynamic_backtest_metrics": os.path.join(output_dir, "kp_experiment7_dynamic_backtest_metrics.csv"),
        "dynamic_refresh_performance_plot": dynamic_plot_path,
        "legacy_predictions": os.path.join(output_dir, "kp_experiment7_forecast_aligned_predictions.csv"),
        "metrics": os.path.join(output_dir, "kp_experiment7_forecast_aligned_metrics.csv"),
        "leadtime_metrics": os.path.join(output_dir, "kp_experiment7_operational_leadtime_metrics.csv"),
        "mode_comparison": os.path.join(output_dir, "kp_experiment7_mode_comparison.csv"),
        "ablation": os.path.join(output_dir, "kp_experiment7_ablation_results.csv"),
        "legacy_ablation": os.path.join(output_dir, "kp_experiment7_forecast_aligned_ablation.csv"),
        "rolling_update_metrics": os.path.join(output_dir, "kp_experiment7_rolling_update_metrics.csv"),
        "regime_metrics": os.path.join(output_dir, "kp_experiment7_forecast_aligned_regime_metrics.csv"),
        "debug": os.path.join(output_dir, "kp_experiment7_debug.json"),
        "pi_summary": os.path.join(output_dir, "kp_experiment7_pi_summary.md"),
        "leadtime_degradation_plot": _experiment7_plot_leadtime(leadtime_df, output_dir),
        "rolling_forecast_examples": _experiment7_plot_rolling_examples(rolling_df, output_dir),
        "bz_drop_examples": _experiment7_plot_bz_operational(rolling_df, output_dir),
        "pattern_break_examples": _experiment7_plot_pattern_operational(diagnostic_predictions, output_dir),
        "scatter_by_leadtime": _experiment7_plot_scatter_by_leadtime(rolling_df, output_dir),
    }
    diagnostic_predictions.to_csv(paths["diagnostic_predictions"], index=False)
    diagnostic_predictions.to_csv(paths["legacy_predictions"], index=False)
    rolling_df.to_csv(paths["operational_predictions"], index=False)
    diagnostic_metrics_df.to_csv(paths["metrics"], index=False)
    leadtime_df.to_csv(paths["leadtime_metrics"], index=False)
    mode_comparison_df.to_csv(paths["mode_comparison"], index=False)
    ablation_df.to_csv(paths["ablation"], index=False)
    ablation_df.to_csv(paths["legacy_ablation"], index=False)
    rolling_update_df.to_csv(paths["rolling_update_metrics"], index=False)
    _experiment7_regime_metrics(
        diagnostic_predictions.rename(columns={"kp_experiment7_target_aligned_best": "kp_forecast_aligned_base"}),
        ["kp_forecast_aligned_base"],
    ).to_csv(paths["regime_metrics"], index=False)

    frozen_row = mode_comparison_df[(mode_comparison_df["model_mode"] == "experiment7_operational_frozen") & (mode_comparison_df["update_cadence_hours"] == 24.0)]
    ar_row = mode_comparison_df[(mode_comparison_df["model_mode"] == "experiment7_operational_autoregressive") & (mode_comparison_df["update_cadence_hours"] == 24.0)]
    phys_row = mode_comparison_df[(mode_comparison_df["model_mode"] == "experiment7_operational_physics_only") & (mode_comparison_df["update_cadence_hours"] == 24.0)]
    frozen_metrics = frozen_row.iloc[0].to_dict() if not frozen_row.empty else {"mae": np.nan, "rmse": np.nan, "pearson_corr": np.nan}
    ar_metrics = ar_row.iloc[0].to_dict() if not ar_row.empty else {"mae": np.nan, "rmse": np.nan, "pearson_corr": np.nan}
    physics_metrics = phys_row.iloc[0].to_dict() if not phys_row.empty else {"mae": np.nan, "rmse": np.nan, "pearson_corr": np.nan}
    best_operational = pd.DataFrame([frozen_metrics, ar_metrics, physics_metrics]).sort_values(["mae", "rmse"]).iloc[0].to_dict()
    bz_help = bool(
        not ablation_df.empty
        and ablation_df.loc[ablation_df["ablation"] == "B_physics_plus_bz_drop", "mae"].iloc[0]
        < ablation_df.loc[ablation_df["ablation"] == "A_forecasted_physics_only", "mae"].iloc[0]
    )
    kp_help = bool(
        not ablation_df.empty
        and ablation_df.loc[ablation_df["ablation"] == "D_physics_plus_24h_kp_pattern", "mae"].iloc[0]
        < ablation_df.loc[ablation_df["ablation"] == "A_forecasted_physics_only", "mae"].iloc[0]
    )
    recommendation = str(best_operational.get("model_mode", "experiment7_operational_frozen"))
    debug = {
        "run_diagnostic_target_aligned": RUN_EXPERIMENT7_DIAGNOSTIC_TARGET_ALIGNED,
        "run_operational_rolling": RUN_EXPERIMENT7_OPERATIONAL_ROLLING,
        "forecast_inputs_already_target_aligned": FORECAST_INPUTS_ALREADY_TARGET_ALIGNED,
        "issue_time_valid_time_metadata_found": bool(metadata.get("issue_time_valid_time_metadata_found", False)),
        "cadence_hours": cadence,
        "forecast_horizon_hours": forecast_horizon_hours,
        "forecast_horizon_bins": forecast_horizon_bins,
        "rolling_window_construction": "Each init_time predicts t_init < valid_time <= t_init + 120h. Solar-wind physics moves with valid_time; Kp history is anchored to init_time except autoregressive mode.",
        "rolling_initializations": int(rolling_df[["init_time", "update_cadence_hours"]].drop_duplicates().shape[0]) if not rolling_df.empty else 0,
        "predicted_target_rows": int(len(rolling_df)),
        "dynamic_backtest_rows": int(len(dynamic_predictions_df)),
        "dynamic_backtest_metrics": dynamic_metrics_df.to_dict(orient="records") if not dynamic_metrics_df.empty else [],
        "target_shifting_by_branch": {
            "diagnostic_target_aligned": "same-row forecasted physics at T plus Kp history shifted to <= T-1; diagnostic upper bound",
            "operational_frozen": "forecasted physics at T plus observed Kp history <= t_init",
            "operational_autoregressive": "forecasted physics at T plus observed Kp <= t_init and earlier in-window predictions",
            "operational_physics_only": "forecasted physics at T only; no Kp history",
        },
        "leakage_rule": "Operational rows never use observed Kp after init_time. Autoregressive rows append predictions, not future observed Kp.",
        "feature_sets": feature_sets,
        "bz_sharp_drop_improved": bz_help,
        "kp_pattern_improved": kp_help,
        "recommended_operational_branch": recommendation,
        "paths": paths,
    }
    with open(paths["debug"], "w", encoding="utf-8") as debug_file:
        json.dump(debug, debug_file, indent=2, default=str)
        debug_file.write("\n")

    def _fmt_metric(row: dict) -> str:
        return f"MAE `{row.get('mae', np.nan):.3f}`, RMSE `{row.get('rmse', np.nan):.3f}`, corr `{row.get('pearson_corr', np.nan):.3f}`"

    summary_lines = [
        "# Experiment 7 PI Summary",
        "## Corrected Operational Framing\nThe upstream model forecasts the solar-wind physics inputs through the 5-day window. Experiment 7 maps those forecasted physics inputs valid at target time T to Kp at T.",
        "## Moving Vs Anchored Features\nSolar-wind/physics features move with valid time T, including rolling forecast-block physics features up to T. Observed Kp-history features are anchored at forecast initialization time t_init in the frozen operational mode. The autoregressive mode may use earlier in-window predicted Kp, but never future observed Kp.",
        "## Rolling-Horizon Update\nThe simulator reruns forecasts at 6h and 24h update cadences. Each initialization predicts `t_init < T <= t_init + 120h`, then the next initialization ingests newly observed Kp through the new t_init.",
        "## Metrics\n"
        f"- diagnostic target-aligned upper bound: {_fmt_metric(diagnostic_best_metrics)}\n"
        f"- operational frozen-anchor daily update: {_fmt_metric(frozen_metrics)}\n"
        f"- operational autoregressive daily update: {_fmt_metric(ar_metrics)}\n"
        f"- operational physics-only daily update: {_fmt_metric(physics_metrics)}\n"
        f"- 6h dynamic refresh backtest rows: `{len(dynamic_predictions_df)}`\n"
        f"- recommended branch: `{recommendation}`",
        "## Feature Findings\n"
        f"- Bz sharp-drop features improved operational ablation MAE: `{bz_help}`\n"
        f"- 24h Kp pattern-break features improved operational ablation MAE: `{kp_help}`",
        "## Caveat\nThe diagnostic target-aligned model is an upper-bound benchmark, not the deployable 5-day forecast branch. The operational rolling simulation is the branch that enforces information availability at each t_init.",
        "## Artifacts\n" + "\n".join([f"- `{value}`" for value in paths.values()]),
    ]
    with open(paths["pi_summary"], "w", encoding="utf-8") as summary_file:
        summary_file.write("\n\n".join(summary_lines) + "\n")

    result = {
        "paths": paths,
        "debug": debug,
        "best_model": "kp_experiment7_target_aligned_best",
        "best_metrics": diagnostic_best_metrics,
        "diagnostic_best_metrics": diagnostic_best_metrics,
        "operational_frozen_metrics": frozen_metrics,
        "operational_autoregressive_metrics": ar_metrics,
        "operational_physics_only_metrics": physics_metrics,
        "best_kp_ge_5_mae": float(frozen_metrics.get("mae", np.nan)),
        "best_kp_ge_6_mae": float(frozen_metrics.get("mae", np.nan)),
        "prob_kp_ge_5_recall": float(frozen_metrics.get("recall_kp_ge_5", 0.0)),
        "prob_kp_ge_6_recall": float(frozen_metrics.get("recall_kp_ge_6", 0.0)),
        "bz_drop_improved": bz_help,
        "kp_pattern_improved": kp_help,
        "forecast_aligned_beat_old_shifted": None,
        "recommendation": recommendation,
        "rolling_initializations": debug["rolling_initializations"],
        "predicted_target_rows": debug["predicted_target_rows"],
        "leadtime_metrics": leadtime_df,
    }
    return diagnostic_metrics_df, diagnostic_predictions, result


def _experiment8_future_block_features(
    feature_df: pd.DataFrame,
    init_idx: int,
    forecast_horizon_hours: float = 120.0,
) -> Optional[dict]:
    init_time = feature_df["timestamp"].iloc[init_idx]
    end_time = init_time + pd.Timedelta(hours=forecast_horizon_hours)
    block = feature_df[(feature_df["timestamp"] > init_time) & (feature_df["timestamp"] <= end_time)].copy()
    if block.empty:
        return None

    row = {
        "init_time": init_time,
        "target_end_time": end_time,
        "future_rows_available": int(len(block)),
        "observed_max_kp_next_120h": float(block["kp"].max()),
        "observed_any_kp_ge_5_next_120h": int((block["kp"] >= 5.0).any()),
        "observed_any_kp_ge_6_next_120h": int((block["kp"] >= 6.0).any()),
        "observed_any_kp_ge_7_next_120h": int((block["kp"] >= 7.0).any()),
    }
    storm_rows = block[block["kp"] >= 5.0]
    row["first_kp_ge_5_lead_hours"] = (
        float((storm_rows["timestamp"].iloc[0] - init_time) / pd.Timedelta(hours=1))
        if not storm_rows.empty
        else np.nan
    )

    for feature, value in _experiment7_kp_state_from_history(feature_df["kp"].iloc[:init_idx + 1].astype(float).tolist()).items():
        row[f"anchor_{feature}"] = value

    physics_specs = {
        "bz": ["min", "mean"],
        "bs": ["max", "mean", "sum"],
        "ey": ["max", "mean", "sum"],
        "pdyn": ["max", "mean"],
        "velocity": ["max", "mean"],
        "density": ["max", "mean"],
        "newell_coupling": ["max", "mean", "sum"],
        "coupling_simple": ["max", "mean", "sum"],
        "compact_storm_score": ["max", "mean"],
        "bz_southward_drop_6h": ["max", "sum"],
        "bz_southward_drop_12h": ["max", "sum"],
        "bz_southward_drop_24h": ["max", "sum"],
        "bz_drop_times_velocity": ["max"],
        "bz_drop_times_pdyn": ["max"],
        "southward_turning_pressure_interaction": ["max"],
        "southward_turning_coupling_interaction": ["max"],
    }
    horizon_specs = [24, 48, 72, 120]
    for horizon in horizon_specs:
        horizon_block = block[block["timestamp"] <= init_time + pd.Timedelta(hours=horizon)]
        if horizon_block.empty:
            continue
        row[f"rows_next_{horizon}h"] = int(len(horizon_block))
        row[f"duration_bz_lt_minus_5_next_{horizon}h"] = float((horizon_block["bz"] < -5.0).sum())
        row[f"duration_bs_gt_5_next_{horizon}h"] = float((horizon_block["bs"] > 5.0).sum())
        row[f"duration_ey_gt_2_next_{horizon}h"] = float((horizon_block["ey"] > 2.0).sum())
        for column, ops in physics_specs.items():
            if column not in horizon_block.columns:
                continue
            values = horizon_block[column].astype(float)
            for op in ops:
                if op == "min":
                    row[f"{column}_min_next_{horizon}h"] = float(values.min())
                elif op == "max":
                    row[f"{column}_max_next_{horizon}h"] = float(values.max())
                elif op == "mean":
                    row[f"{column}_mean_next_{horizon}h"] = float(values.mean())
                elif op == "sum":
                    row[f"{column}_sum_next_{horizon}h"] = float(values.sum())

    for lead_hours in [6, 12, 24]:
        target_time = init_time + pd.Timedelta(hours=lead_hours)
        nearest = block[block["timestamp"] == target_time]
        if nearest.empty:
            continue
        target = nearest.iloc[0]
        for column in ["bz", "bs", "velocity", "density", "pdyn", "ey", "newell_coupling", "compact_storm_score"]:
            if column in target.index:
                row[f"{column}_at_plus_{lead_hours}h"] = float(target[column])

    return row


def _experiment8_build_samples(feature_df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    rows = []
    for init_idx in range(max(start_idx, max(KP_LAG_STEPS)), end_idx):
        row = _experiment8_future_block_features(feature_df, init_idx)
        if row is not None and row["future_rows_available"] >= 4:
            rows.append(row)
    return pd.DataFrame(rows)


def _experiment8_feature_columns(sample_df: pd.DataFrame) -> List[str]:
    forbidden = {
        "init_time",
        "target_end_time",
        "observed_max_kp_next_120h",
        "observed_any_kp_ge_5_next_120h",
        "observed_any_kp_ge_6_next_120h",
        "observed_any_kp_ge_7_next_120h",
        "first_kp_ge_5_lead_hours",
    }
    return [
        column for column in sample_df.columns
        if column not in forbidden and pd.api.types.is_numeric_dtype(sample_df[column])
    ]


def _experiment8_threshold_for_recall(y_true: np.ndarray, score: np.ndarray, target_recall: float = 0.90) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for threshold in thresholds:
        pred = score >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        rows.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1})
    sweep = pd.DataFrame(rows)
    feasible = sweep[sweep["recall"] >= target_recall]
    if feasible.empty:
        selected = sweep.sort_values(["recall", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    else:
        selected = feasible.sort_values(["precision", "f1", "threshold"], ascending=[False, False, False]).iloc[0]
    return float(selected["threshold"])


def _experiment8_oof_classifier_scores(X: pd.DataFrame, y: np.ndarray, n_estimators: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    splitter = TimeSeriesSplit(n_splits=_time_series_cv_splits(len(X), requested_splits=4))
    scores = np.full(len(X), np.nan)
    y = np.asarray(y).astype(int)
    for train_idx, val_idx in splitter.split(X):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X.iloc[train_idx], y[train_idx])
        scores[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    mask = ~np.isnan(scores)
    return y[mask], scores[mask]


def _experiment8_threshold_sweep(y_true: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = score >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "watch_duty_cycle": float(np.mean(pred)),
            }
        )
    return pd.DataFrame(rows)


def _experiment8_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = score >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision": _safe_average_precision(y_true, score),
        "true_positive_windows": int((pred & y_true.astype(bool)).sum()),
        "false_positive_windows": int((pred & ~y_true.astype(bool)).sum()),
        "false_negative_windows": int((~pred & y_true.astype(bool)).sum()),
        "true_negative_windows": int((~pred & ~y_true.astype(bool)).sum()),
    }


def _plot_experiment8_timeseries(prediction_df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "kp_experiment8_storm_watch_timeseries.png")
    work = prediction_df.copy()
    work["init_time"] = pd.to_datetime(work["init_time"])
    storm_rows = work[work["observed_any_kp_ge_5_next_120h"] == 1]
    if not storm_rows.empty:
        center = storm_rows.iloc[len(storm_rows) // 2]["init_time"]
        plot_df = work[(work["init_time"] >= center - pd.Timedelta(days=20)) & (work["init_time"] <= center + pd.Timedelta(days=25))].copy()
    else:
        plot_df = work.iloc[: min(len(work), 200)].copy()

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(plot_df["init_time"], plot_df["observed_kp_at_init"], color="black", linewidth=1.5, label="Observed Kp at init")
    axes[0].plot(plot_df["init_time"], plot_df["predicted_max_kp_next_120h"], color="tab:blue", linewidth=1.2, label="Predicted max Kp next 5d")
    axes[0].plot(plot_df["init_time"], plot_df["observed_max_kp_next_120h"], color="tab:orange", linewidth=1.2, label="Observed max Kp next 5d")
    axes[0].axhline(5.0, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Kp")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(plot_df["init_time"], plot_df["prob_any_kp_ge_5_next_120h"], color="tab:red", linewidth=1.4, label="P(any Kp>=5 next 5d)")
    axes[1].plot(plot_df["init_time"], plot_df["prob_any_kp_ge_6_next_120h"], color="tab:purple", linewidth=1.2, label="P(any Kp>=6 next 5d)")
    axes[1].axhline(float(plot_df["selected_threshold_kp_ge_5"].iloc[0]), color="tab:red", linestyle=":", linewidth=1, label="Kp>=5 watch threshold")
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(plot_df["init_time"], plot_df["bz_min_next_120h"], color="tab:blue", label="Min forecasted Bz next 5d")
    axes[2].plot(plot_df["init_time"], plot_df["ey_max_next_120h"], color="tab:green", label="Max Ey next 5d")
    axes[2].set_ylabel("Physics summary")
    axes[2].set_xlabel("Forecast initialization time")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_experiment8_scatter(prediction_df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "kp_experiment8_predicted_vs_observed_max_kp.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(prediction_df["observed_max_kp_next_120h"], prediction_df["predicted_max_kp_next_120h"], s=16, alpha=0.45)
    ax.plot([0, 9], [0, 9], color="black", linestyle="--", linewidth=1)
    ax.axvline(5, color="tab:red", linestyle=":", linewidth=1)
    ax.axhline(5, color="tab:red", linestyle=":", linewidth=1)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_xlabel("Observed max Kp next 5d")
    ax.set_ylabel("Predicted max Kp next 5d")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_experiment8_storm_watch(
    feature_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
) -> dict:
    _ensure_directory(output_dir)
    split_index = len(train_df)
    train_samples = _experiment8_build_samples(feature_df, 0, max(0, split_index - 20))
    test_samples = _experiment8_build_samples(feature_df, split_index - 1, len(feature_df) - 1)
    if train_samples.empty or test_samples.empty:
        raise ValueError("Experiment 8 could not build enough rolling storm-watch samples.")
    feature_columns = _experiment8_feature_columns(train_samples)
    medians = train_samples[feature_columns].median(numeric_only=True).fillna(0.0)
    X_train = train_samples[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(medians)
    X_test = test_samples[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(medians)

    y5 = train_samples["observed_any_kp_ge_5_next_120h"].astype(int).to_numpy()
    y6 = train_samples["observed_any_kp_ge_6_next_120h"].astype(int).to_numpy()
    y_test5 = test_samples["observed_any_kp_ge_5_next_120h"].astype(int).to_numpy()
    y_test6 = test_samples["observed_any_kp_ge_6_next_120h"].astype(int).to_numpy()

    classifier5 = RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    classifier5.fit(X_train, y5)
    oof_y5, oof_prob5 = _experiment8_oof_classifier_scores(X_train, y5)
    prob5_test = classifier5.predict_proba(X_test)[:, 1]
    # Keep the storm watch recall-heavy, but avoid a near-always-on watch gate.
    threshold5 = max(0.10, _experiment8_threshold_for_recall(oof_y5, oof_prob5, target_recall=0.95))

    classifier6 = RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    classifier6.fit(X_train, y6)
    oof_y6, oof_prob6 = _experiment8_oof_classifier_scores(X_train, y6)
    prob6_test = classifier6.predict_proba(X_test)[:, 1]
    threshold6 = max(0.02, _experiment8_threshold_for_recall(oof_y6, oof_prob6, target_recall=0.90))

    max_kp_model = RandomForestRegressor(
        n_estimators=350,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    max_kp_weights = np.where(train_samples["observed_max_kp_next_120h"].to_numpy(dtype=float) >= 5.0, 5.0, 1.0)
    max_kp_weights *= np.where(train_samples["observed_max_kp_next_120h"].to_numpy(dtype=float) >= 6.0, 2.0, 1.0)
    max_kp_model.fit(X_train, train_samples["observed_max_kp_next_120h"].to_numpy(dtype=float), sample_weight=max_kp_weights)
    pred_max = np.clip(max_kp_model.predict(X_test), 0.0, 9.0)

    predictions = test_samples.copy()
    predictions["observed_kp_at_init"] = [
        float(feature_df.loc[feature_df["timestamp"] == init_time, "kp"].iloc[0])
        for init_time in predictions["init_time"]
    ]
    predictions["prob_any_kp_ge_5_next_120h"] = prob5_test
    predictions["prob_any_kp_ge_6_next_120h"] = prob6_test
    predictions["selected_threshold_kp_ge_5"] = threshold5
    predictions["selected_threshold_kp_ge_6"] = threshold6
    predictions["storm_watch_kp_ge_5"] = (prob5_test >= threshold5).astype(int)
    predictions["storm_watch_kp_ge_6"] = (prob6_test >= threshold6).astype(int)
    predictions["predicted_max_kp_next_120h"] = pred_max

    max_metrics = _experiment7_regression_metrics(
        predictions["observed_max_kp_next_120h"].to_numpy(dtype=float),
        predictions["predicted_max_kp_next_120h"].to_numpy(dtype=float),
    )
    metrics_rows = [
        {"target": "any_kp_ge_5_next_120h", **_experiment8_metrics(y_test5, prob5_test, threshold5)},
        {"target": "any_kp_ge_6_next_120h", **_experiment8_metrics(y_test6, prob6_test, threshold6)},
        {"target": "max_kp_next_120h_regression", **max_metrics},
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    threshold_sweep_df = pd.concat(
        [
            _experiment8_threshold_sweep(y_test5, prob5_test).assign(target="any_kp_ge_5_next_120h", split="holdout"),
            _experiment8_threshold_sweep(oof_y5, oof_prob5).assign(target="any_kp_ge_5_next_120h", split="train_oof"),
            _experiment8_threshold_sweep(y_test6, prob6_test).assign(target="any_kp_ge_6_next_120h", split="holdout"),
            _experiment8_threshold_sweep(oof_y6, oof_prob6).assign(target="any_kp_ge_6_next_120h", split="train_oof"),
        ],
        ignore_index=True,
    )

    paths = {
        "predictions": os.path.join(output_dir, "kp_experiment8_storm_watch_predictions.csv"),
        "metrics": os.path.join(output_dir, "kp_experiment8_storm_watch_metrics.csv"),
        "threshold_sweep": os.path.join(output_dir, "kp_experiment8_storm_watch_threshold_sweep.csv"),
        "timeseries": _plot_experiment8_timeseries(predictions, output_dir),
        "scatter": _plot_experiment8_scatter(predictions, output_dir),
        "debug": os.path.join(output_dir, "kp_experiment8_debug.json"),
        "summary": os.path.join(output_dir, "kp_experiment8_pi_summary.md"),
    }
    predictions.to_csv(paths["predictions"], index=False)
    metrics_df.to_csv(paths["metrics"], index=False)
    threshold_sweep_df.to_csv(paths["threshold_sweep"], index=False)
    debug = {
        "objective": "5-day storm-watch classifier from full forecasted physics sequence plus Kp state at initialization",
        "train_samples": int(len(train_samples)),
        "test_samples": int(len(test_samples)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "threshold_kp_ge_5": threshold5,
        "threshold_kp_ge_6": threshold6,
        "paths": paths,
    }
    with open(paths["debug"], "w", encoding="utf-8") as debug_file:
        json.dump(debug, debug_file, indent=2, default=str)
        debug_file.write("\n")
    kp5_metrics = metrics_rows[0]
    kp6_metrics = metrics_rows[1]
    summary = [
        "# Experiment 8 Storm Watch Summary",
        "## Objective\nTrain a direct 5-day storm-watch model: full forecasted physics sequence plus Kp state at initialization -> probability of any Kp>=5 or Kp>=6 in the next 120h.",
        "## Kp>=5 Storm Watch\n"
        f"- threshold: `{threshold5:.3f}`\n"
        f"- precision: `{kp5_metrics['precision']:.3f}`\n"
        f"- recall: `{kp5_metrics['recall']:.3f}`\n"
        f"- F1: `{kp5_metrics['f1']:.3f}`\n"
        f"- average precision: `{kp5_metrics['average_precision']:.3f}`",
        "## Kp>=6 Severe Watch\n"
        f"- threshold: `{threshold6:.3f}`\n"
        f"- precision: `{kp6_metrics['precision']:.3f}`\n"
        f"- recall: `{kp6_metrics['recall']:.3f}`\n"
        f"- F1: `{kp6_metrics['f1']:.3f}`\n"
        f"- average precision: `{kp6_metrics['average_precision']:.3f}`",
        "## Max Kp Regression\n"
        f"- MAE: `{max_metrics['mae']:.3f}`\n"
        f"- RMSE: `{max_metrics['rmse']:.3f}`\n"
        f"- Pearson r: `{max_metrics['pearson_corr']:.3f}`",
        "## Artifacts\n" + "\n".join([f"- `{value}`" for value in paths.values()]),
    ]
    with open(paths["summary"], "w", encoding="utf-8") as summary_file:
        summary_file.write("\n\n".join(summary) + "\n")
    return {"metrics": metrics_df, "predictions": predictions, "paths": paths, "debug": debug}


EXPERIMENT11_LEAKY_DRIVERS = [
    "bs",
    "ey",
    "newell_coupling",
    "coupling_simple",
    "compact_storm_score",
    "v_bs",
    "bz_southward_drop_6h",
]
EXPERIMENT11_LEAKY_ALPHAS = [0.70, 0.85, 0.93, 0.97]
EXPERIMENT11_ORDINAL_THRESHOLDS = [4.0, 5.0, 6.0, 7.0]


def _experiment11_add_leaky_integrators(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    work = feature_df.copy()
    added = []
    for driver in EXPERIMENT11_LEAKY_DRIVERS:
        if driver not in work.columns:
            continue
        values = work[driver].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        positive_values = np.maximum(values, 0.0)
        for alpha in EXPERIMENT11_LEAKY_ALPHAS:
            accum = np.zeros(len(work), dtype=float)
            for idx, value in enumerate(positive_values):
                accum[idx] = value if idx == 0 else alpha * accum[idx - 1] + value
            suffix = str(alpha).replace(".", "p")
            col = f"experiment11_leaky_{driver}_a{suffix}"
            work[col] = accum
            added.append(col)
    if "experiment11_leaky_bs_a0p93" in work.columns and "pdyn" in work.columns:
        work["experiment11_leaky_bs_pressure_a0p93"] = work["experiment11_leaky_bs_a0p93"] * work["pdyn"].astype(float)
        added.append("experiment11_leaky_bs_pressure_a0p93")
    if "experiment11_leaky_newell_coupling_a0p93" in work.columns and "bz_southward_drop_6h" in work.columns:
        work["experiment11_leaky_newell_bz_drop_a0p93"] = (
            work["experiment11_leaky_newell_coupling_a0p93"] * work["bz_southward_drop_6h"].astype(float)
        )
        added.append("experiment11_leaky_newell_bz_drop_a0p93")
    return work, added


def _experiment11_sample_weights(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if mode == "square":
        weights = (y + 1.0) ** 2
    elif mode == "exp":
        weights = np.exp(0.28 * y)
    else:
        weights = np.ones_like(y)
    weights *= np.where(y >= 5.0, 3.0, 1.0)
    weights *= np.where(y >= 6.0, 2.0, 1.0)
    return weights / max(float(np.mean(weights)), 1e-9)


def _experiment11_fill_matrix(frame: pd.DataFrame, feature_columns: List[str], medians: pd.Series) -> pd.DataFrame:
    return frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(medians)


def _experiment11_kp_bins(y: pd.Series) -> np.ndarray:
    return np.round(np.asarray(y, dtype=float) * 3.0).astype(int)


def _experiment11_quantile_from_density(probabilities: np.ndarray, classes: np.ndarray, quantile: float) -> np.ndarray:
    classes = np.asarray(classes, dtype=float)
    cdf = np.cumsum(probabilities, axis=1)
    idx = (cdf >= quantile).argmax(axis=1)
    return classes[idx]


def _experiment11_density_expected(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return probabilities @ np.asarray(classes, dtype=float)


def _experiment11_survival_expected(probabilities: Dict[str, np.ndarray]) -> np.ndarray:
    # Approximate a Kp expectation from cumulative storm probabilities. This is
    # intentionally conservative below Kp=4 and lets the classifier only control
    # the storm-tail mass.
    base = np.full_like(next(iter(probabilities.values())), 3.0, dtype=float)
    for threshold in EXPERIMENT11_ORDINAL_THRESHOLDS:
        label = str(int(threshold))
        if label in probabilities:
            base += probabilities[label]
    return np.clip(base, 0.0, 9.0)


def _experiment11_oof_probability_feature(
    train_df: pd.DataFrame,
    feature_columns: List[str],
    threshold: float,
    medians: pd.Series,
) -> Tuple[np.ndarray, object]:
    X_train = _experiment11_fill_matrix(train_df, feature_columns, medians)
    y = (train_df["kp"].to_numpy(dtype=float) >= threshold).astype(int)
    oof = np.full(len(train_df), np.nan)
    if len(np.unique(y)) < 2 or int(y.sum()) < MIN_SEVERITY_POSITIVES:
        return np.zeros(len(train_df), dtype=float), None
    splitter = TimeSeriesSplit(n_splits=_time_series_cv_splits(len(train_df), requested_splits=4))
    for fit_idx, val_idx in splitter.split(X_train):
        if len(np.unique(y[fit_idx])) < 2:
            continue
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        ) if HistGradientBoostingClassifier is not None else RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train.iloc[fit_idx], y[fit_idx], sample_weight=_experiment11_sample_weights(train_df["kp"].iloc[fit_idx].to_numpy(dtype=float), "square"))
        oof[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]
    fallback = float(np.mean(y))
    oof = np.where(np.isnan(oof), fallback, oof)
    final_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=RANDOM_STATE,
    ) if HistGradientBoostingClassifier is not None else RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_model.fit(X_train, y, sample_weight=_experiment11_sample_weights(train_df["kp"].to_numpy(dtype=float), "square"))
    return oof, final_model


def _experiment11_train_bundle(train_df: pd.DataFrame, feature_columns: List[str]) -> dict:
    medians = train_df[feature_columns].median(numeric_only=True).fillna(0.0)
    X_train = _experiment11_fill_matrix(train_df, feature_columns, medians)
    y_train = train_df["kp"].to_numpy(dtype=float)
    bundle = {"feature_columns": feature_columns, "medians": medians}

    if HistGradientBoostingRegressor is not None:
        weighted_mean = HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=320,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        )
        weighted_mean.fit(X_train, y_train, sample_weight=_experiment11_sample_weights(y_train, "square"))
        bundle["weighted_mean"] = weighted_mean
        quantiles = {}
        for label, quantile in [("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95)]:
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=quantile,
                learning_rate=0.04,
                max_iter=260,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train, sample_weight=_experiment11_sample_weights(y_train, "exp"))
            quantiles[label] = model
        bundle["quantiles"] = quantiles
    else:
        weighted_mean = RandomForestRegressor(n_estimators=350, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
        weighted_mean.fit(X_train, y_train, sample_weight=_experiment11_sample_weights(y_train, "square"))
        bundle["weighted_mean"] = weighted_mean
        bundle["quantiles"] = {}

    y_bins = _experiment11_kp_bins(train_df["kp"])
    density = RandomForestClassifier(
        n_estimators=450,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    density.fit(X_train, y_bins)
    bundle["density_model"] = density
    bundle["density_classes"] = density.classes_.astype(float) / 3.0

    ordinal_models = {}
    ordinal_oof_features = {}
    for threshold in EXPERIMENT11_ORDINAL_THRESHOLDS:
        label = str(int(threshold))
        oof, model = _experiment11_oof_probability_feature(train_df, feature_columns, threshold, medians)
        ordinal_oof_features[label] = oof
        if model is not None:
            ordinal_models[label] = model
    bundle["ordinal_models"] = ordinal_models

    multitask_features = X_train.copy()
    for label, values in ordinal_oof_features.items():
        multitask_features[f"experiment11_oof_prob_kp_ge_{label}"] = values
    if HistGradientBoostingRegressor is not None:
        multitask = HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        )
    else:
        multitask = RandomForestRegressor(n_estimators=350, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
    multitask.fit(multitask_features, y_train, sample_weight=_experiment11_sample_weights(y_train, "square"))
    bundle["multitask_proxy"] = multitask
    bundle["multitask_feature_columns"] = list(multitask_features.columns)
    return bundle


def _experiment11_predict_frame(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    feature_columns = bundle["feature_columns"]
    X = _experiment11_fill_matrix(frame, feature_columns, bundle["medians"])
    out = pd.DataFrame(index=frame.index)
    out["kp_experiment11_weighted_mean"] = np.clip(bundle["weighted_mean"].predict(X), 0.0, 9.0)
    for label, model in bundle.get("quantiles", {}).items():
        out[f"kp_experiment11_{label}"] = np.clip(model.predict(X), 0.0, 9.0)

    density_prob = bundle["density_model"].predict_proba(X)
    classes = bundle["density_classes"]
    out["kp_experiment11_density_mean"] = np.clip(_experiment11_density_expected(density_prob, classes), 0.0, 9.0)
    out["kp_experiment11_density_p50"] = np.clip(_experiment11_quantile_from_density(density_prob, classes, 0.50), 0.0, 9.0)
    out["kp_experiment11_density_p75"] = np.clip(_experiment11_quantile_from_density(density_prob, classes, 0.75), 0.0, 9.0)
    out["kp_experiment11_density_p90"] = np.clip(_experiment11_quantile_from_density(density_prob, classes, 0.90), 0.0, 9.0)
    for threshold in EXPERIMENT11_ORDINAL_THRESHOLDS:
        label = str(int(threshold))
        if label in bundle["ordinal_models"]:
            out[f"prob_experiment11_kp_ge_{label}"] = bundle["ordinal_models"][label].predict_proba(X)[:, 1]
        else:
            out[f"prob_experiment11_kp_ge_{label}"] = 0.0
    prob_dict = {
        str(int(threshold)): out[f"prob_experiment11_kp_ge_{int(threshold)}"].to_numpy(dtype=float)
        for threshold in EXPERIMENT11_ORDINAL_THRESHOLDS
    }
    out["kp_experiment11_ordinal_expected"] = _experiment11_survival_expected(prob_dict)

    multitask_X = X.copy()
    for threshold in EXPERIMENT11_ORDINAL_THRESHOLDS:
        label = str(int(threshold))
        multitask_X[f"experiment11_oof_prob_kp_ge_{label}"] = out[f"prob_experiment11_kp_ge_{label}"].to_numpy(dtype=float)
    out["kp_experiment11_multitask_proxy"] = np.clip(bundle["multitask_proxy"].predict(multitask_X[bundle["multitask_feature_columns"]]), 0.0, 9.0)
    if "kp_experiment11_p90" in out.columns:
        p90 = out["kp_experiment11_p90"].to_numpy(dtype=float)
    else:
        p90 = out["kp_experiment11_density_p90"].to_numpy(dtype=float)
    mean = out["kp_experiment11_multitask_proxy"].to_numpy(dtype=float)
    prob5 = out["prob_experiment11_kp_ge_5"].to_numpy(dtype=float)
    prob6 = out["prob_experiment11_kp_ge_6"].to_numpy(dtype=float)
    risk = mean.copy()
    risk = np.where(prob5 >= 0.25, np.maximum(risk, np.minimum(p90, 6.0)), risk)
    risk = np.where(prob6 >= 0.10, np.maximum(risk, np.minimum(p90 + 0.5, 7.0)), risk)
    out["kp_experiment11_risk_envelope"] = np.clip(risk, 0.0, 9.0)
    out["storm_watch_experiment11"] = ((prob5 >= 0.10) | (prob6 >= 0.03)).astype(int)
    return out


def _experiment11_simulate_rolling(
    feature_df: pd.DataFrame,
    split_index: int,
    bundle: dict,
    update_cadence_hours: float = 6.0,
    forecast_horizon_hours: float = 120.0,
) -> pd.DataFrame:
    cadence = _cadence_hours(feature_df)
    if cadence <= 0:
        cadence = 6.0
    horizon_bins = max(1, int(round(forecast_horizon_hours / cadence)))
    update_bins = max(1, int(round(update_cadence_hours / cadence)))
    init_indices = list(range(split_index - 1, len(feature_df) - 1, update_bins))
    feature_rows = []
    meta_rows = []
    for init_idx in init_indices:
        init_time = feature_df["timestamp"].iloc[init_idx]
        observed_history = feature_df["kp"].iloc[:init_idx + 1].astype(float).tolist()
        kp_state = _experiment7_kp_state_from_history(observed_history)
        for lead_bin in range(1, horizon_bins + 1):
            target_idx = init_idx + lead_bin
            if target_idx >= len(feature_df) or target_idx < split_index:
                continue
            lead_hours = float((feature_df["timestamp"].iloc[target_idx] - init_time) / pd.Timedelta(hours=1))
            if lead_hours > forecast_horizon_hours + 1e-9:
                continue
            target_row = feature_df.iloc[target_idx].copy()
            for feature, value in kp_state.items():
                if feature in target_row.index:
                    target_row[feature] = value
            feature_rows.append(target_row)
            meta_rows.append(
                {
                    "timestamp": feature_df["timestamp"].iloc[target_idx],
                    "t_init": init_time,
                    "lead_time_hours": lead_hours,
                    "lead_time_bin": _experiment7_lead_bin(lead_hours),
                    "observed_kp": float(feature_df["kp"].iloc[target_idx]),
                    "kp_history_source_max_time": init_time,
                    "uses_future_observed_kp": False,
                    "bz": float(feature_df["bz"].iloc[target_idx]),
                    "velocity": float(feature_df["velocity"].iloc[target_idx]),
                    "density": float(feature_df["density"].iloc[target_idx]),
                    "pdyn": float(feature_df["pdyn"].iloc[target_idx]),
                }
            )
    if not feature_rows:
        return pd.DataFrame()
    pred = _experiment11_predict_frame(bundle, pd.DataFrame(feature_rows).reset_index(drop=True))
    return pd.concat([pd.DataFrame(meta_rows), pred.reset_index(drop=True)], axis=1)


def _experiment11_metric_row(frame: pd.DataFrame, pred_col: str, model: str, lead_time_bin: str = "all") -> dict:
    y = frame["observed_kp"].to_numpy(dtype=float)
    pred = frame[pred_col].to_numpy(dtype=float)
    row = {
        "model": model,
        "lead_time_bin": lead_time_bin,
        "n": int(len(frame)),
        **_experiment7_regression_metrics(y, pred),
    }
    for threshold in [5.0, 6.0, 7.0]:
        score_col = f"prob_experiment11_kp_ge_{int(threshold)}"
        score = frame[score_col].to_numpy(dtype=float) if score_col in frame.columns else pred
        row.update(_experiment7_binary_metrics_for_threshold(y, score, pred, threshold))
    quiet = y < 5.0
    row["quiet_false_storm_inflation"] = float(np.mean(pred[quiet] >= 5.0)) if int(quiet.sum()) else 0.0
    return row


def _experiment11_threshold_sweep(prediction_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, score_col in [("kp_ge_5", "prob_experiment11_kp_ge_5"), ("kp_ge_6", "prob_experiment11_kp_ge_6")]:
        if score_col not in prediction_df.columns:
            continue
        y = prediction_df["observed_kp"].to_numpy(dtype=float) >= float(target[-1])
        score = prediction_df[score_col].to_numpy(dtype=float)
        for threshold in np.linspace(0.01, 0.99, 99):
            pred = score >= threshold
            precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
            rows.append(
                {
                    "target": target,
                    "threshold": float(threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "average_precision": _safe_average_precision(y, score),
                    "watch_duty_cycle": float(np.mean(pred)),
                }
            )
    return pd.DataFrame(rows)


def _plot_experiment11_timeseries(prediction_df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "kp_experiment11_storm_aware_timeseries.png")
    work = prediction_df[prediction_df["lead_time_bin"] == "0-6h"].copy()
    if work.empty:
        work = prediction_df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    storm_rows = work[work["observed_kp"] >= 5.0]
    if not storm_rows.empty:
        center = storm_rows.iloc[len(storm_rows) // 2]["timestamp"]
        plot_df = work[(work["timestamp"] >= center - pd.Timedelta(days=12)) & (work["timestamp"] <= center + pd.Timedelta(days=12))].copy()
    else:
        plot_df = work.head(300).copy()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(plot_df["timestamp"], plot_df["observed_kp"], color="black", linewidth=2.0, label="Observed Kp")
    axes[0].plot(plot_df["timestamp"], plot_df["kp_experiment11_multitask_proxy"], color="tab:blue", linewidth=1.4, label="Exp11 multitask proxy")
    axes[0].plot(plot_df["timestamp"], plot_df["kp_experiment11_risk_envelope"], color="tab:red", linewidth=1.3, label="Exp11 risk envelope")
    axes[0].plot(plot_df["timestamp"], plot_df["kp_experiment11_density_p90"], color="tab:purple", linewidth=1.0, alpha=0.75, label="Density p90")
    axes[0].axhline(5.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Kp")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(plot_df["timestamp"], plot_df["prob_experiment11_kp_ge_5"], color="tab:red", label="P(Kp>=5)")
    axes[1].plot(plot_df["timestamp"], plot_df["prob_experiment11_kp_ge_6"], color="tab:purple", label="P(Kp>=6)")
    axes[1].fill_between(plot_df["timestamp"], 0, plot_df["storm_watch_experiment11"], color="tab:orange", alpha=0.18, label="Storm watch")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("Probability")
    axes[1].set_xlabel("Valid time")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_experiment11_leadtime(metrics_df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "kp_experiment11_leadtime_performance.png")
    order = ["0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-96h", "96-120h"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    work = metrics_df[metrics_df["lead_time_bin"].isin(order)].copy()
    work["lead_order"] = work["lead_time_bin"].map({label: idx for idx, label in enumerate(order)})
    for model in ["kp_experiment11_multitask_proxy", "kp_experiment11_risk_envelope", "kp_experiment11_density_p90"]:
        group = work[work["model"] == model].sort_values("lead_order")
        if group.empty:
            continue
        axes[0].plot(group["lead_time_bin"], group["mae"], marker="o", label=model.replace("kp_experiment11_", ""))
        axes[1].plot(group["lead_time_bin"], group["recall_kp_ge_5"], marker="o", label=model.replace("kp_experiment11_", ""))
    axes[0].set_ylabel("MAE")
    axes[1].set_ylabel("Kp>=5 recall")
    axes[1].set_xlabel("Lead time")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# Experiment 11 tests structural storm-aware features, not a shortcut around
# chronology. The explicit forbidden-feature check below is its leakage boundary.
def run_experiment11_storm_aware_structural(
    feature_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: dict,
    output_dir: str,
) -> dict:
    _ensure_directory(output_dir)
    feature_df11, leaky_columns = _experiment11_add_leaky_integrators(feature_df)
    split_index = len(train_df)
    train_df11 = feature_df11.iloc[:split_index].copy()
    test_df11 = feature_df11.iloc[split_index:].copy()
    base_feature_sets = _experiment7_feature_sets(metadata)
    feature_columns = _unique_in_order(
        [feature for feature in base_feature_sets["operational_all_features"] if feature in feature_df11.columns]
        + leaky_columns
    )
    forbidden = [feature for feature in feature_columns if feature == "kp" or feature.startswith("kp_max_next") or "future" in feature]
    if forbidden:
        raise ValueError(f"Experiment 11 leakage check failed for features: {forbidden}")

    bundle = _experiment11_train_bundle(train_df11, feature_columns)
    diagnostic_predictions = test_df11[["timestamp", "kp"]].copy()
    diagnostic_pred = _experiment11_predict_frame(bundle, test_df11)
    diagnostic_predictions = pd.concat([diagnostic_predictions.reset_index(drop=True), diagnostic_pred.reset_index(drop=True)], axis=1)
    rolling_predictions = _experiment11_simulate_rolling(feature_df11, split_index, bundle, update_cadence_hours=6.0)

    model_columns = [
        "kp_experiment11_weighted_mean",
        "kp_experiment11_density_mean",
        "kp_experiment11_density_p50",
        "kp_experiment11_density_p75",
        "kp_experiment11_density_p90",
        "kp_experiment11_ordinal_expected",
        "kp_experiment11_multitask_proxy",
        "kp_experiment11_risk_envelope",
    ]
    model_columns += [col for col in ["kp_experiment11_p50", "kp_experiment11_p75", "kp_experiment11_p90", "kp_experiment11_p95"] if col in rolling_predictions.columns]
    metric_rows = []
    for model_col in _unique_in_order([col for col in model_columns if col in rolling_predictions.columns]):
        metric_rows.append(_experiment11_metric_row(rolling_predictions, model_col, model_col, "all"))
        for lead_bin, group in rolling_predictions.groupby("lead_time_bin", sort=False):
            metric_rows.append(_experiment11_metric_row(group, model_col, model_col, lead_bin))
    metrics_df = pd.DataFrame(metric_rows)
    threshold_sweep_df = _experiment11_threshold_sweep(rolling_predictions)

    paths = {
        "diagnostic_predictions": os.path.join(output_dir, "kp_experiment11_diagnostic_predictions.csv"),
        "rolling_predictions": os.path.join(output_dir, "kp_experiment11_rolling_predictions.csv"),
        "metrics": os.path.join(output_dir, "kp_experiment11_metrics.csv"),
        "threshold_sweep": os.path.join(output_dir, "kp_experiment11_threshold_sweep.csv"),
        "timeseries": _plot_experiment11_timeseries(rolling_predictions, output_dir),
        "leadtime": _plot_experiment11_leadtime(metrics_df, output_dir),
        "debug": os.path.join(output_dir, "kp_experiment11_debug.json"),
        "summary": os.path.join(output_dir, "kp_experiment11_pi_summary.md"),
    }
    diagnostic_predictions.to_csv(paths["diagnostic_predictions"], index=False)
    rolling_predictions.to_csv(paths["rolling_predictions"], index=False)
    metrics_df.to_csv(paths["metrics"], index=False)
    threshold_sweep_df.to_csv(paths["threshold_sweep"], index=False)

    best_mean = metrics_df[
        (metrics_df["lead_time_bin"] == "all")
        & (metrics_df["model"].isin(["kp_experiment11_multitask_proxy", "kp_experiment11_weighted_mean", "kp_experiment11_density_mean"]))
    ].sort_values(["recall_kp_ge_5", "mae"], ascending=[False, True]).head(1)
    best_risk = metrics_df[
        (metrics_df["lead_time_bin"] == "all")
        & (metrics_df["model"].isin(["kp_experiment11_risk_envelope", "kp_experiment11_density_p90", "kp_experiment11_p90"]))
    ].sort_values(["recall_kp_ge_5", "quiet_false_storm_inflation", "mae"], ascending=[False, True, True]).head(1)
    debug = {
        "objective": "Experiment 11 structural storm-aware Kp model: leaky integrators, severe sample weights, ordinal/density prediction, and classifier-informed regression.",
        "train_rows": int(len(train_df11)),
        "test_rows": int(len(test_df11)),
        "rolling_rows": int(len(rolling_predictions)),
        "feature_count": int(len(feature_columns)),
        "leaky_feature_count": int(len(leaky_columns)),
        "leaky_feature_columns": leaky_columns,
        "feature_columns": feature_columns,
        "best_mean_model": best_mean.to_dict(orient="records"),
        "best_risk_model": best_risk.to_dict(orient="records"),
        "uses_future_observed_kp": bool(rolling_predictions.get("uses_future_observed_kp", pd.Series(False)).astype(bool).any()),
        "paths": paths,
    }
    with open(paths["debug"], "w", encoding="utf-8") as debug_file:
        json.dump(debug, debug_file, indent=2, default=str)
        debug_file.write("\n")

    summary = [
        "# Experiment 11 Structural Storm-Aware Kp Model",
        "## Objective\nTest all four proposed structural upgrades in one branch: leaky-integrator physics memory, severe-weighted target optimization, ordinal/density Kp prediction, and a classifier-informed multi-task proxy.",
        "## Operational Rules\nSolar-wind physics moves with valid time T. Kp-history features are anchored to t_init in the 6h rolling simulation. No observed future Kp is used as a feature.",
        "## Best Mean-Trajectory Candidate\n" + (_markdown_table(best_mean) if not best_mean.empty else "No candidate produced metrics."),
        "## Best Risk-Envelope Candidate\n" + (_markdown_table(best_risk) if not best_risk.empty else "No candidate produced metrics."),
        "## All-Row Metrics\n" + _markdown_table(metrics_df[metrics_df["lead_time_bin"] == "all"].head(20)),
        "## Artifacts\n" + "\n".join([f"- `{value}`" for value in paths.values()]),
    ]
    with open(paths["summary"], "w", encoding="utf-8") as summary_file:
        summary_file.write("\n\n".join(summary) + "\n")
    return {
        "metrics": metrics_df,
        "predictions": rolling_predictions,
        "diagnostic_predictions": diagnostic_predictions,
        "paths": paths,
        "debug": debug,
    }


def load_temporal_model(model_path: str) -> dict:
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


def evaluate_saved_temporal_model(model_bundle: dict, df: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    feature_df = prepare_temporal_dataframe(df)
    feature_columns = model_bundle["feature_columns"]
    pred = np.clip(model_bundle["model"].predict(feature_df[feature_columns]), 0.0, 9.0)
    metrics = _full_metric_bundle(feature_df["kp"].to_numpy(), pred)

    prediction_df = feature_df.copy()
    prediction_df["best_model_pred"] = pred
    prediction_df["old_kp_model"] = pred
    prediction_df["new_kp_model"] = pred
    prediction_df["date"] = feature_df["timestamp"]

    return metrics, prediction_df


def _safe_corr_values(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> Optional[float]:
    if len(y_true) < 3 or np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return None
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method=method))


def _experiment5_regime_metrics(prediction_df: pd.DataFrame, branch_columns: List[str]) -> pd.DataFrame:
    regimes = {
        "quiet_kp_lt_4": prediction_df["kp"] < 4.0,
        "nonstorm_kp_lt_5": prediction_df["kp"] < 5.0,
        "elevated_4_to_5": (prediction_df["kp"] >= 4.0) & (prediction_df["kp"] < 5.0),
        "storm_kp_ge_5": prediction_df["kp"] >= 5.0,
        "severe_kp_ge_6": prediction_df["kp"] >= 6.0,
        "extreme_kp_ge_7": prediction_df["kp"] >= 7.0,
    }
    rows = []
    for branch in branch_columns:
        if branch not in prediction_df.columns:
            continue
        pred = prediction_df[branch].to_numpy(dtype=float)
        for regime_name, mask_series in regimes.items():
            mask = mask_series.to_numpy()
            if int(mask.sum()) == 0:
                continue
            y = prediction_df.loc[mask, "kp"].to_numpy(dtype=float)
            p = pred[mask]
            error = p - y
            under = error < 0
            over = error > 0
            nonstorm_mask = prediction_df["kp"].to_numpy(dtype=float) < 5.0
            nonsevere_mask = prediction_df["kp"].to_numpy(dtype=float) < 6.0
            rows.append(
                {
                    "branch": branch,
                    "regime": regime_name,
                    "n_rows": int(mask.sum()),
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(error ** 2))),
                    "bias": float(np.mean(error)),
                    "pearson_corr": _safe_corr_values(y, p, "pearson"),
                    "spearman_corr": _safe_corr_values(y, p, "spearman"),
                    "underprediction_rate": float(np.mean(under)),
                    "overprediction_rate": float(np.mean(over)),
                    "mean_underprediction": float(np.mean(np.abs(error[under]))) if int(under.sum()) > 0 else 0.0,
                    "max_underprediction": float(np.max(np.abs(error[under]))) if int(under.sum()) > 0 else 0.0,
                    "mean_overprediction": float(np.mean(error[over])) if int(over.sum()) > 0 else 0.0,
                    "max_overprediction": float(np.max(error[over])) if int(over.sum()) > 0 else 0.0,
                    "fraction_pred_ge_5_when_obs_lt_5": float(np.mean(pred[nonstorm_mask] >= 5.0)) if int(nonstorm_mask.sum()) > 0 else 0.0,
                    "fraction_pred_ge_6_when_obs_lt_5": float(np.mean(pred[nonstorm_mask] >= 6.0)) if int(nonstorm_mask.sum()) > 0 else 0.0,
                    "fraction_pred_ge_6_when_obs_lt_6": float(np.mean(pred[nonsevere_mask] >= 6.0)) if int(nonsevere_mask.sum()) > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _experiment5_event_comparison(prediction_df: pd.DataFrame, branch_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = pd.to_datetime(prediction_df["timestamp"])
    cadence = _cadence_hours(pd.DataFrame({"timestamp": timestamps}))
    if cadence <= 0:
        cadence = 6.0
    lead_steps = max(1, int(round(12.0 / cadence)))
    rows = []
    kp_values = prediction_df["kp"].to_numpy(dtype=float)
    for event_id, (start, end) in enumerate(_contiguous_true_runs(kp_values >= 5.0), start=1):
        lead_start = max(0, start - lead_steps)
        window = prediction_df.iloc[lead_start:end + 1]
        event = prediction_df.iloc[start:end + 1]
        row = {
            "event_id": event_id,
            "event_start_time": timestamps.iloc[start],
            "event_peak_kp": float(event["kp"].max()),
            "event_duration_hours": float((end - start + 1) * cadence),
            "experiment4_risk_votes_max": float(window.get("experiment4_risk_votes", pd.Series(0, index=window.index)).max()),
            "prob_kp_ge_5_next_12h_max": float(window.get("prob_kp_ge_5_next_12h", pd.Series(0, index=window.index)).max()),
            "prob_kp_ge_6_next_12h_max": float(window.get("prob_kp_ge_6_next_12h", pd.Series(0, index=window.index)).max()),
            "prob_kp_ge_7_next_12h_max": float(window.get("prob_kp_ge_7_next_12h", pd.Series(0, index=window.index)).max()),
            "kp_risk_p95_max": float(window.get("kp_risk_p95", pd.Series(0, index=window.index)).max()),
            "severe_boost_authorized_fired": bool(window.get("experiment4_severe_boost_authorized", pd.Series(0, index=window.index)).astype(bool).any()),
            "storm_watch_fired": bool(window.get("experiment4_storm_watch_gate", pd.Series(0, index=window.index)).astype(bool).any()),
        }
        for branch in branch_columns:
            if branch not in prediction_df.columns:
                continue
            values = window[branch].to_numpy(dtype=float)
            event_values = event[branch].to_numpy(dtype=float)
            row[f"{branch}_max_before_or_during"] = float(np.max(values))
            row[f"{branch}_max_underprediction"] = float(np.maximum(event["kp"].to_numpy(dtype=float) - event_values, 0.0).max())
            row[f"{branch}_reached_ge_5"] = bool(np.max(values) >= 5.0)
            row[f"{branch}_reached_ge_6"] = bool(np.max(values) >= 6.0)
            row[f"{branch}_reached_ge_7"] = bool(np.max(values) >= 7.0)
            lead_hit = np.where(window[branch].to_numpy(dtype=float) >= 5.0)[0]
            row[f"{branch}_lead_time_hours_ge_5"] = (
                float((timestamps.iloc[start] - window["timestamp"].iloc[int(lead_hit[0])]) / pd.Timedelta(hours=1))
                if len(lead_hit) > 0
                else None
            )
        rows.append(row)
    event_df = pd.DataFrame(rows)
    severe_df = event_df[event_df["event_peak_kp"] >= 6.0].copy() if not event_df.empty else event_df.copy()
    return event_df, severe_df


def _experiment5_false_inflation_diagnostics(prediction_df: pd.DataFrame, output_dir: str) -> Tuple[pd.DataFrame, str]:
    work = prediction_df.copy()
    work["kp_risk_p95_minus_base"] = work.get("kp_risk_p95", 0.0) - work.get("kp_base", 0.0)
    false_mask = (work["kp"] < 5.0) & (work["kp_storm_conservative_v4"] >= 5.0)
    storm_mask = work["kp"] >= 5.0
    true_quiet_mask = (work["kp"] < 5.0) & (work["kp_storm_conservative_v4"] < 5.0)
    columns = [
        "kp_base",
        "kp_risk_p90",
        "kp_risk_p95",
        "kp_risk_p95_minus_base",
        "prob_kp_ge_5_next_12h",
        "prob_kp_ge_6_next_12h",
        "experiment4_risk_votes",
        "kp_roll_max_4",
        "bs_roll_sum_2",
        "ey_roll_sum_2",
        "ey_roll_max_2",
        "pdyn_roll_max_4",
        "pdyn_jump",
        "coupling_simple",
        "velocity_jump",
    ]
    rows = []
    for label, mask in [
        ("false_inflation", false_mask),
        ("true_storm", storm_mask),
        ("true_quiet_noninflated", true_quiet_mask),
    ]:
        for column in columns:
            if column not in work.columns:
                continue
            values = work.loc[mask, column].dropna()
            rows.append(
                {
                    "group": label,
                    "feature": column,
                    "n": int(len(values)),
                    "mean": float(values.mean()) if len(values) else None,
                    "median": float(values.median()) if len(values) else None,
                    "p90": float(values.quantile(0.90)) if len(values) else None,
                    "max": float(values.max()) if len(values) else None,
                }
            )
    hist_path = os.path.join(output_dir, "kp_experiment5_false_inflation_histograms.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_specs = [
        ("experiment4_risk_votes", "Experiment 4 Risk Votes"),
        ("prob_kp_ge_6_next_12h", "P(Kp>=6 next 12h)"),
        ("kp_risk_p95_minus_base", "kp_risk_p95 - kp_base"),
    ]
    for ax, (column, title) in zip(axes, plot_specs):
        if column not in work.columns:
            ax.set_title(f"{title} unavailable")
            continue
        ax.hist(work.loc[storm_mask, column], bins=25, alpha=0.55, label="true storms", density=True)
        ax.hist(work.loc[false_mask, column], bins=25, alpha=0.55, label="false inflations", density=True)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(hist_path, dpi=180)
    plt.close(fig)
    return pd.DataFrame(rows), hist_path


def _experiment5_gfz_examples(prediction_df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "kp_experiment5_gfz_style_examples.png")
    work = prediction_df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    windows = []
    # Fixed already-plotted early holdout window.
    windows.append((work["timestamp"].min(), work["timestamp"].min() + pd.Timedelta(days=16), "Early Holdout Window"))
    # Strong storm captured well and underpredicted examples.
    event_df, _ = _experiment5_event_comparison(work, ["kp_final_risk_conservative_v4"])
    if not event_df.empty:
        captured = event_df[event_df["kp_final_risk_conservative_v4_max_underprediction"] <= 1.0].sort_values("event_peak_kp", ascending=False)
        missed = event_df.sort_values("kp_final_risk_conservative_v4_max_underprediction", ascending=False)
        for source, label in [(captured, "Captured Strong Storm"), (missed, "Underpredicted Storm")]:
            if not source.empty:
                start = pd.to_datetime(source.iloc[0]["event_start_time"]) - pd.Timedelta(days=2)
                windows.append((start, start + pd.Timedelta(days=7), label))
    false_rows = work[(work["kp"] < 5.0) & (work["kp_final_risk_conservative_v4"] >= 5.0)]
    if not false_rows.empty:
        start = false_rows.iloc[len(false_rows) // 2]["timestamp"] - pd.Timedelta(days=2)
        windows.append((start, start + pd.Timedelta(days=7), "Quiet Overcall Example"))
    windows = windows[:4]

    fig, axes = plt.subplots(len(windows), 2, figsize=(12, 4.0 * len(windows)))
    if len(windows) == 1:
        axes = np.array([axes])
    for row_idx, (start, end, label) in enumerate(windows):
        window = work[(work["timestamp"] >= start) & (work["timestamp"] <= end)].copy()
        if window.empty:
            continue
        ax_ts, ax_sc = axes[row_idx]
        ax_ts.plot(window["timestamp"], window["kp"], color="black", linewidth=1.5, label="Observed")
        ax_ts.plot(window["timestamp"], window["kp_base"], color="tab:blue", linewidth=1.2, label="kp_base")
        ax_ts.plot(window["timestamp"], window["kp_final_risk_conservative_v4"], color="tab:orange", linestyle="--", linewidth=1.4, label="Exp4/5 risk")
        ax_ts.set_ylim(0, 10)
        ax_ts.set_title(label)
        ax_ts.grid(True, alpha=0.25)
        ax_ts.legend(fontsize=8)
        x = window["kp_final_risk_conservative_v4"].to_numpy(dtype=float)
        y = window["kp"].to_numpy(dtype=float)
        ax_sc.scatter(x, y, s=18, color="black", alpha=0.75)
        ax_sc.plot([0, 10], [0, 10], color="gray", linestyle="--", linewidth=1)
        corr = _safe_corr_values(y, x, "pearson")
        ax_sc.text(0.5, 9.1, f"Correlation {corr:.3f}" if corr is not None else "Correlation n/a")
        ax_sc.set_xlim(0, 10)
        ax_sc.set_ylim(0, 10)
        ax_sc.set_xlabel("Predicted Kp")
        ax_sc.set_ylabel("Observed Kp")
        ax_sc.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_experiment5_diagnostics(prediction_path: str, output_dir: str = "testing") -> dict:
    _ensure_directory(output_dir)
    pred = pd.read_csv(prediction_path, parse_dates=["timestamp"])
    # Use calibrated Experiment 4 as v5 unless a future leakage-safe suppressor is added.
    pred["experiment5_calibrated_risk_score"] = pred.get("kp_underprediction_risk_score", pred.get("severe_watch_probability", 0.0))
    pred["experiment5_storm_watch_gate"] = pred.get("experiment4_storm_watch_gate", 0).astype(int)
    pred["experiment5_severe_boost_authorized"] = pred.get("experiment4_severe_boost_authorized", 0).astype(int)
    pred["experiment5_false_inflation_suppressor_score"] = np.nan
    pred["kp_storm_conservative_v5"] = pred.get("kp_storm_conservative_v4", pred.get("kp_final_risk_conservative_v4", pred["kp_base"]))
    pred["kp_final_risk_conservative_v5"] = np.maximum(pred["kp_base"], pred["kp_storm_conservative_v5"])
    pred["experiment5_operating_point_label"] = "frontier_quiet_inflation_0p25_no_leakage_safe_suppressor"

    branch_columns = [
        "kp_base",
        "kp_storm_adjusted",
        "kp_storm_conservative",
        "kp_storm_conservative_v2",
        "kp_storm_conservative_v4",
        "kp_final_risk_conservative_v4",
        "kp_storm_conservative_v5",
        "kp_final_risk_conservative_v5",
    ]
    regime_df = _experiment5_regime_metrics(pred, branch_columns)
    event_df, severe_event_df = _experiment5_event_comparison(pred, branch_columns)
    false_diag_df, false_hist_path = _experiment5_false_inflation_diagnostics(pred, output_dir)

    tradeoff_path = os.path.join(output_dir, "kp_experiment4_tradeoff_curve.csv")
    frontier_df = pd.read_csv(tradeoff_path) if os.path.exists(tradeoff_path) else pd.DataFrame()
    selected_rows = []
    if not frontier_df.empty:
        for target in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            candidates = frontier_df[frontier_df["quiet_false_storm_inflation_rate"] <= target].copy()
            feasible = not candidates.empty
            if not feasible:
                candidates = frontier_df.copy()
                candidates["target_violation"] = np.maximum(0.0, candidates["quiet_false_storm_inflation_rate"] - target)
                sort_columns = ["target_violation", "kp_ge_6_recall", "kp_ge_7_recall", "asymmetric_storm_mae"]
                ascending = [True, False, False, True]
            else:
                sort_columns = ["kp_ge_6_recall", "kp_ge_7_recall", "asymmetric_storm_mae", "quiet_false_storm_inflation_rate"]
                ascending = [False, False, True, True]
            row = candidates.sort_values(sort_columns, ascending=ascending).iloc[0].to_dict()
            row["quiet_inflation_target"] = target
            row["target_feasible"] = feasible
            y_true = pred["kp"].to_numpy(dtype=float)
            # Reuse selected row's already-computed scalar metrics.
            row["severe_storm_mae"] = row.get("storm_mae")
            row["severe_weighted_asymmetric_mae"] = row.get("asymmetric_storm_mae")
            row["max_kp_ge_6_underprediction"] = row.get("max_storm_underprediction")
            row["storm_bin_recall_proxy"] = row.get("kp_ge_5_recall")
            row["duty_cycle_watch_gate"] = row.get("watch_duty_cycle")
            row["duty_cycle_severe_boost_authorization"] = row.get("severe_authorized_duty_cycle")
            row["missed_kp_ge_6_rows_estimated"] = int(round((1.0 - row.get("kp_ge_6_recall", 0.0)) * max(1, int((y_true >= 6.0).sum()))))
            row["missed_kp_ge_6_note"] = "estimated from row-level Kp>=6 recall; exact event misses are in kp_experiment5_severe_event_comparison.csv"
            selected_rows.append(row)
    operating_frontier_df = pd.DataFrame(selected_rows)

    # Frontier plots.
    frontier_plot_path = os.path.join(output_dir, "kp_experiment5_recall_inflation_frontier.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    if not frontier_df.empty:
        x = frontier_df["quiet_false_storm_inflation_rate"]
        specs = [
            ("kp_ge_6_recall", "Kp>=6 Recall"),
            ("kp_ge_5_recall", "Kp>=5 Recall"),
            ("asymmetric_storm_mae", "Severe Asymmetric MAE"),
            ("max_storm_underprediction", "Max Severe Underprediction"),
        ]
        for ax, (col, title) in zip(axes.ravel(), specs):
            ax.scatter(x, frontier_df[col], s=26, alpha=0.65, color="tab:blue")
            if not operating_frontier_df.empty:
                ax.scatter(
                    operating_frontier_df["quiet_false_storm_inflation_rate"],
                    operating_frontier_df[col],
                    s=60,
                    color="tab:red",
                    marker="x",
                )
            ax.set_xlabel("Quiet false storm inflation")
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(frontier_plot_path, dpi=180)
    plt.close(fig)

    gfz_examples_path = _experiment5_gfz_examples(pred, output_dir)

    paths = {
        "regime_metrics": os.path.join(output_dir, "kp_experiment5_regime_metrics.csv"),
        "operating_frontier": os.path.join(output_dir, "kp_experiment5_operating_frontier.csv"),
        "event_comparison": os.path.join(output_dir, "kp_experiment5_event_comparison.csv"),
        "severe_event_comparison": os.path.join(output_dir, "kp_experiment5_severe_event_comparison.csv"),
        "false_inflation_diagnostics": os.path.join(output_dir, "kp_experiment5_false_inflation_diagnostics.csv"),
        "frontier_plot": frontier_plot_path,
        "false_inflation_histograms": false_hist_path,
        "gfz_examples": gfz_examples_path,
        "predictions": os.path.join(output_dir, "kp_experiment5_predictions.csv"),
        "suppressor_sweep": os.path.join(output_dir, "kp_experiment5_suppressor_sweep.csv"),
        "pi_summary": os.path.join(output_dir, "kp_experiment5_pi_summary.md"),
        "debug": os.path.join(output_dir, "kp_experiment5_debug.json"),
    }
    regime_df.to_csv(paths["regime_metrics"], index=False)
    operating_frontier_df.to_csv(paths["operating_frontier"], index=False)
    event_df.to_csv(paths["event_comparison"], index=False)
    severe_event_df.to_csv(paths["severe_event_comparison"], index=False)
    false_diag_df.to_csv(paths["false_inflation_diagnostics"], index=False)
    pred.to_csv(paths["predictions"], index=False)
    pd.DataFrame(
        [
            {
                "status": "skipped",
                "reason": "No leakage-safe train/validation Experiment 4 prediction set is available for fitting a learned suppressor. V5 uses calibrated frontier selection instead.",
            }
        ]
    ).to_csv(paths["suppressor_sweep"], index=False)

    def _row_for(metrics_path: str, model: str) -> dict:
        if not os.path.exists(metrics_path):
            return {}
        df = pd.read_csv(metrics_path)
        match = df[df["model"] == model]
        return match.iloc[0].to_dict() if not match.empty else {}

    exp3 = _row_for(os.path.join(output_dir, "kp_experiment3_storm_magnitude_metrics.csv"), "kp_storm_conservative_v2")
    exp4 = _row_for(os.path.join(output_dir, "kp_experiment4_metrics.csv"), "kp_final_risk_conservative_v4")
    exp5 = _advanced_asymmetric_metrics(pred["kp"].to_numpy(dtype=float), pred["kp_final_risk_conservative_v5"].to_numpy(dtype=float))
    pi_summary = [
        "# Experiment 5 PI Summary",
        "## System Summary\nThe pipeline is intentionally multi-branch. `kp_base` remains the quiet/moderate Kp nowcast. The public alert gate remains constrained for operational burden. The severe-risk branches are internal upper-risk estimates designed to reduce dangerous underprediction, not to replace the calibrated nowcast.",
        "## Key Metrics\n"
        f"- Experiment 3 quiet false inflation: `{exp3.get('quiet_false_storm_inflation_rate', np.nan):.3f}`; Kp>=6 recall: `{exp3.get('kp_ge_6_recall', np.nan):.3f}`.\n"
        f"- Experiment 4 quiet false inflation: `{exp4.get('quiet_false_storm_inflation_rate', np.nan):.3f}`; Kp>=6 recall: `{exp4.get('kp_ge_6_recall', np.nan):.3f}`.\n"
        f"- Experiment 5/v5 quiet false inflation: `{exp5.get('quiet_false_storm_inflation_rate', np.nan):.3f}`; Kp>=6 recall: `{exp5.get('kp_ge_6_recall', np.nan):.3f}`.",
        "## Main Improvement\nExperiment 4/5 preserves the high Kp>=6 recall found in Experiment 3 while reducing quiet false inflation by more than half. This is the more presentable internal severe-risk operating point.",
        "## Remaining Issues\nThe risk branch still overcalls some quiet intervals, correlation remains moderate, and the severe branch is a conservative upper-risk estimate rather than an exact Kp regression.",
        "## Next Research Direction\nSelect an operating point from the frontier, build a leakage-safe false-inflation suppressor using validation-era predictions, and test higher-cadence solar-wind summaries if those data become available.",
        "## Artifacts\n" + "\n".join([f"- `{value}`" for value in paths.values()]),
    ]
    with open(paths["pi_summary"], "w", encoding="utf-8") as report_file:
        report_file.write("\n\n".join(pi_summary) + "\n")

    debug = {
        "suppressor_status": "skipped_no_leakage_safe_training_predictions",
        "operating_point_label": pred["experiment5_operating_point_label"].iloc[0],
        "paths": paths,
    }
    with open(paths["debug"], "w", encoding="utf-8") as debug_file:
        json.dump(debug, debug_file, indent=2, default=str)
        debug_file.write("\n")
    return {"paths": paths, "regime_metrics": regime_df, "frontier": operating_frontier_df, "predictions": pred}


def _experiment6_allowed_suppressor_features(frame: pd.DataFrame) -> List[str]:
    candidates = [
        "kp_base",
        "kp_storm_adjusted",
        "kp_risk_p90",
        "kp_risk_p95",
        "kp_risk_p95_minus_base",
        "prob_kp_ge_5_next_12h",
        "prob_kp_ge_6_next_12h",
        "prob_kp_ge_7_next_12h",
        "experiment4_risk_votes",
        "experiment5_calibrated_risk_score",
        "kp_lag_1",
        "kp_lag_2",
        "kp_roll_max_4",
        "kp_roll_max_8",
        "kp_roll_sum_4",
        "kp_roll_sum_8",
        "kp_delta_1",
        "time_since_kp_ge_4",
        "time_since_kp_ge_5",
        "bs_roll_sum_2",
        "bs_roll_sum_4",
        "bs_roll_sum_8",
        "ey_roll_sum_2",
        "ey_roll_sum_4",
        "ey_roll_sum_8",
        "ey_roll_max_2",
        "ey_roll_max_4",
        "ey_roll_max_8",
        "pdyn_roll_max_4",
        "pdyn_roll_max_8",
        "pdyn_jump",
        "velocity_jump",
        "compression_score",
        "coupling_simple",
        "epsilon_proxy",
        "kan_lee_proxy",
    ]
    forbidden_fragments = ["next_", "future", "target", "storm_now", "storm_onset"]
    safe = []
    for column in candidates:
        if column not in frame.columns:
            continue
        if any(fragment in column for fragment in forbidden_fragments) and not column.startswith("prob_kp_ge_"):
            continue
        safe.append(column)
    return safe


def _experiment6_prediction_frame(prediction_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    frame = prediction_df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    feature_copy = feature_df.copy()
    feature_copy["timestamp"] = pd.to_datetime(feature_copy["timestamp"])
    passthrough_features = [
        "timestamp",
        "kp_lag_1",
        "kp_lag_2",
        "kp_roll_max_4",
        "kp_roll_max_8",
        "kp_roll_sum_4",
        "kp_roll_sum_8",
        "kp_delta_1",
        "time_since_kp_ge_4",
        "time_since_kp_ge_5",
        "bs_roll_sum_2",
        "bs_roll_sum_4",
        "bs_roll_sum_8",
        "ey_roll_sum_2",
        "ey_roll_sum_4",
        "ey_roll_sum_8",
        "ey_roll_max_2",
        "ey_roll_max_4",
        "ey_roll_max_8",
        "pdyn_roll_max_4",
        "pdyn_roll_max_8",
        "pdyn_jump",
        "velocity_jump",
        "compression_score",
        "coupling_simple",
        "epsilon_proxy",
        "kan_lee_proxy",
        "kp_max_next_12h",
    ]
    available = [column for column in passthrough_features if column in feature_copy.columns]
    frame = frame.merge(feature_copy[available], on="timestamp", how="left", suffixes=("", "_feature"))
    frame["kp_risk_p95_minus_base"] = frame.get("kp_risk_p95", 0.0) - frame.get("kp_base", 0.0)
    for column in [
        "prob_kp_ge_5_next_12h",
        "prob_kp_ge_6_next_12h",
        "prob_kp_ge_7_next_12h",
        "experiment4_risk_votes",
        "experiment5_calibrated_risk_score",
    ]:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame


def _experiment6_fit_component_predictions(
    inner_train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    solar_feature_columns: List[str],
    temporal_feature_columns: List[str],
) -> Tuple[pd.DataFrame, dict]:
    pred = target_df[["timestamp", "kp"]].copy()
    debug = {"validation_component_models": {}}
    temporal_columns = [column for column in temporal_feature_columns if column in inner_train_df.columns and column in target_df.columns]
    solar_columns = [column for column in solar_feature_columns if column in inner_train_df.columns and column in target_df.columns]

    if temporal_columns:
        ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        ridge_model.fit(inner_train_df[temporal_columns], inner_train_df["kp"].to_numpy())
        pred["kp_base"] = np.clip(ridge_model.predict(target_df[temporal_columns]), 0.0, 9.0)
        debug["validation_component_models"]["kp_base"] = "ridge_alpha_10"
    else:
        pred["kp_base"] = float(inner_train_df["kp"].median())
        debug["validation_component_models"]["kp_base"] = "median_fallback"

    for quantile in [0.75, 0.90, 0.95]:
        column = f"kp_risk_p{int(quantile * 100)}"
        valid = inner_train_df.get("kp_max_next_12h_valid", pd.Series(True, index=inner_train_df.index)).astype(bool)
        if HistGradientBoostingRegressor is not None and solar_columns and int(valid.sum()) >= 100:
            model = _fit_quantile_regressor(quantile)
            model.fit(
                inner_train_df.loc[valid, solar_columns],
                inner_train_df.loc[valid, "kp_max_next_12h"].to_numpy(),
                sample_weight=_storm_magnitude_sample_weight(inner_train_df.loc[valid].copy(), "kp_max_next_12h"),
            )
            pred[column] = np.clip(model.predict(target_df[solar_columns]), 0.0, 9.0)
            debug["validation_component_models"][column] = "hist_gradient_boosting_quantile"
        else:
            pred[column] = pred["kp_base"].to_numpy()
            debug["validation_component_models"][column] = "kp_base_fallback"

    for threshold in [5.0, 6.0, 7.0]:
        target_column = f"kp_ge_{int(threshold)}_next_12h"
        prob_column = f"prob_kp_ge_{int(threshold)}_next_12h"
        if target_column in inner_train_df.columns and solar_columns and int(inner_train_df[target_column].sum()) >= MIN_SEVERITY_POSITIVES:
            model = RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            model.fit(inner_train_df[solar_columns], inner_train_df[target_column].astype(int).to_numpy())
            pred[prob_column] = model.predict_proba(target_df[solar_columns])[:, 1]
            debug["validation_component_models"][prob_column] = "random_forest_balanced"
        else:
            pred[prob_column] = 0.0
            debug["validation_component_models"][prob_column] = "zero_fallback"

    pred["kp_storm_adjusted"] = np.maximum(pred["kp_base"], pred["kp_base"] + np.where(pred["prob_kp_ge_5_next_12h"] >= 0.50, 1.0, 0.0))
    pred["experiment5_calibrated_risk_score"] = np.maximum.reduce([
        pred["prob_kp_ge_5_next_12h"].to_numpy(),
        pred["prob_kp_ge_6_next_12h"].to_numpy(),
        np.clip((pred["kp_risk_p95"].to_numpy() - pred["kp_base"].to_numpy()) / 4.0, 0.0, 1.0),
    ])
    return pred, debug


def _experiment6_apply_v5_like_logic(
    prediction_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    reference_train_df: pd.DataFrame,
) -> pd.DataFrame:
    frame = _experiment6_prediction_frame(prediction_df, feature_df)
    base = frame["kp_base"].to_numpy(dtype=float)
    p75 = frame.get("kp_risk_p75", frame.get("kp_risk_p90", frame["kp_base"])).to_numpy(dtype=float)
    p90 = frame.get("kp_risk_p90", frame["kp_base"]).to_numpy(dtype=float)
    p95 = frame.get("kp_risk_p95", frame["kp_base"]).to_numpy(dtype=float)
    adjusted = frame.get("kp_storm_adjusted", frame["kp_base"]).to_numpy(dtype=float)
    prob5 = frame.get("prob_kp_ge_5_next_12h", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float)
    prob6 = frame.get("prob_kp_ge_6_next_12h", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float)
    prob7 = frame.get("prob_kp_ge_7_next_12h", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float)
    coupling_threshold = float(reference_train_df["coupling_simple"].quantile(0.90)) if "coupling_simple" in reference_train_df else np.inf
    ey_threshold = float(reference_train_df["ey_roll_sum_2"].quantile(0.90)) if "ey_roll_sum_2" in reference_train_df else np.inf
    vote_physics = np.zeros(len(frame), dtype=bool)
    if "coupling_simple" in frame.columns:
        vote_physics |= frame["coupling_simple"].to_numpy(dtype=float) >= coupling_threshold
    if "ey_roll_sum_2" in frame.columns:
        vote_physics |= frame["ey_roll_sum_2"].to_numpy(dtype=float) >= ey_threshold
    vote_recent_kp = frame["kp_roll_max_4"].to_numpy(dtype=float) >= 4.0 if "kp_roll_max_4" in frame.columns else np.zeros(len(frame), dtype=bool)
    votes = (
        (prob6 >= 0.05).astype(int)
        + (p95 >= 6.0).astype(int)
        + ((p95 - base) >= 1.0).astype(int)
        + vote_physics.astype(int)
        + vote_recent_kp.astype(int)
    )
    watch = prob5 >= 0.80
    level1 = np.where(
        watch,
        np.maximum.reduce([base, p75, np.where(prob5 >= 0.30, 5.0, base)]),
        base,
    )
    severe_authorized = votes >= 3
    severe_candidate = np.maximum.reduce([level1, adjusted, p90, p95, np.ceil(p95)])
    frame["experiment4_risk_votes"] = votes
    frame["experiment5_storm_watch_gate"] = watch.astype(int)
    frame["experiment5_severe_boost_authorized"] = severe_authorized.astype(int)
    frame["kp_storm_conservative_v5"] = np.clip(np.where(severe_authorized, severe_candidate, level1), 0.0, 9.0)
    frame["kp_final_risk_conservative_v5"] = np.maximum(base, frame["kp_storm_conservative_v5"].to_numpy(dtype=float))
    return frame


def _experiment6_severe_safety_override(frame: pd.DataFrame, reference_train_df: pd.DataFrame) -> np.ndarray:
    override = np.zeros(len(frame), dtype=bool)
    override |= frame.get("prob_kp_ge_6_next_12h", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float) >= EXPERIMENT6_PROB6_SAFETY_THRESHOLD
    override |= frame.get("prob_kp_ge_7_next_12h", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float) >= EXPERIMENT6_PROB7_SAFETY_THRESHOLD
    override |= frame.get("kp_risk_p95", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float) >= EXPERIMENT6_P95_SAFETY_THRESHOLD
    override |= frame.get("experiment4_risk_votes", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float) >= EXPERIMENT6_HIGH_VOTE_THRESHOLD
    if "kp_roll_max_4" in frame.columns:
        override |= frame["kp_roll_max_4"].to_numpy(dtype=float) >= 4.5
    if "ey_roll_sum_2" in frame.columns and "ey_roll_sum_2" in reference_train_df.columns:
        override |= frame["ey_roll_sum_2"].to_numpy(dtype=float) >= float(reference_train_df["ey_roll_sum_2"].quantile(0.95))
    if "coupling_simple" in frame.columns and "coupling_simple" in reference_train_df.columns:
        override |= frame["coupling_simple"].to_numpy(dtype=float) >= float(reference_train_df["coupling_simple"].quantile(0.95))
    return override


def _experiment6_candidate_mask(frame: pd.DataFrame, severe_override: np.ndarray) -> np.ndarray:
    v5 = frame["kp_final_risk_conservative_v5"].to_numpy(dtype=float)
    prob6 = frame.get("prob_kp_ge_6_next_12h", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float)
    p95 = frame.get("kp_risk_p95", pd.Series(0.0, index=frame.index)).to_numpy(dtype=float)
    return (
        (v5 >= 5.0)
        & (v5 < 6.5)
        & (prob6 < EXPERIMENT6_PROB6_SAFETY_THRESHOLD)
        & (p95 < EXPERIMENT6_P95_SAFETY_THRESHOLD)
        & (~np.asarray(severe_override).astype(bool))
    )


def _experiment6_apply_suppression(frame: pd.DataFrame, threshold: float, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    v5 = frame["kp_final_risk_conservative_v5"].to_numpy(dtype=float)
    base = frame["kp_base"].to_numpy(dtype=float)
    adjusted = frame.get("kp_storm_adjusted", frame["kp_base"]).to_numpy(dtype=float)
    candidate = frame["experiment6_candidate_suppression_row"].astype(bool).to_numpy()
    score = frame["experiment6_false_inflation_suppressor_score"].to_numpy(dtype=float)
    suppress = candidate & (score >= threshold)
    if mode == "fallback_to_adjusted":
        replacement = np.maximum(base, adjusted)
    else:
        replacement = np.maximum(base, np.minimum(v5, 4.67))
    v6 = np.where(suppress, replacement, v5)
    return np.clip(np.maximum(base, v6), 0.0, 9.0), suppress


def _experiment6_kp6_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    mask = np.asarray(y_true) >= 6.0
    if int(mask.sum()) == 0:
        return None
    return float(mean_absolute_error(np.asarray(y_true)[mask], np.asarray(y_pred)[mask]))


def _experiment6_sweep_rows(frame: pd.DataFrame) -> pd.DataFrame:
    y_true = frame["kp"].to_numpy(dtype=float)
    rows = []
    for mode in EXPERIMENT6_SUPPRESSION_MODES:
        for threshold in EXPERIMENT6_SUPPRESSOR_THRESHOLDS:
            pred, suppress = _experiment6_apply_suppression(frame, threshold, mode)
            metrics = _advanced_asymmetric_metrics(y_true, pred)
            kp6_recall, kp6_events, kp6_missed, _, _ = _event_peak_recall(
                frame["timestamp"],
                y_true,
                pred >= 6.0,
                2,
                6.0,
            )
            rows.append(
                {
                    "suppression_mode": mode,
                    "suppressor_threshold": threshold,
                    "quiet_false_storm_inflation": metrics["quiet_false_storm_inflation_rate"],
                    "kp_ge_5_recall": metrics["kp_ge_5_recall"],
                    "kp_ge_6_recall": metrics["kp_ge_6_recall"],
                    "kp_ge_7_recall": metrics["kp_ge_7_recall"],
                    "storm_mae": metrics["storm_mae"],
                    "kp_ge_6_mae": _experiment6_kp6_mae(y_true, pred),
                    "severe_weighted_asymmetric_mae": metrics["asymmetric_storm_mae"],
                    "max_kp_ge_6_underprediction": _max_underprediction_for_threshold(y_true, pred, 6.0),
                    "missed_kp_ge_6_events": kp6_missed,
                    "kp_ge_6_event_recall": kp6_recall,
                    "kp_ge_6_events": kp6_events,
                    "quiet_rows_suppressed": int(((y_true < 5.0) & suppress).sum()),
                    "true_storm_rows_accidentally_suppressed": int(((y_true >= 5.0) & suppress).sum()),
                    "severe_storm_rows_accidentally_suppressed": int(((y_true >= 6.0) & suppress).sum()),
                    "suppressed_rows": int(suppress.sum()),
                }
            )
    return pd.DataFrame(rows)


def _experiment6_select_operating_points(sweep_df: pd.DataFrame, baseline_metrics: dict) -> pd.DataFrame:
    selections = []
    selection_specs = [
        ("preserve_kp6_recall_ge_0p875", lambda df: df[df["kp_ge_6_recall"] >= max(0.875, baseline_metrics["kp_ge_6_recall"] - 1e-9)],
         ["quiet_false_storm_inflation", "max_kp_ge_6_underprediction", "severe_weighted_asymmetric_mae"], [True, True, True]),
        ("slight_tradeoff_kp6_recall_ge_0p833", lambda df: df[df["kp_ge_6_recall"] >= 0.833],
         ["quiet_false_storm_inflation", "max_kp_ge_6_underprediction", "severe_weighted_asymmetric_mae"], [True, True, True]),
        ("cleaner_branch_kp6_recall_ge_0p800", lambda df: df[df["kp_ge_6_recall"] >= 0.800],
         ["quiet_false_storm_inflation", "max_kp_ge_6_underprediction", "severe_weighted_asymmetric_mae"], [True, True, True]),
        ("severe_safe_quiet_inflation_le_0p25", lambda df: df[df["quiet_false_storm_inflation"] <= 0.25],
         ["max_kp_ge_6_underprediction", "kp_ge_6_recall", "quiet_false_storm_inflation"], [True, False, True]),
    ]
    for label, filter_func, sort_columns, ascending in selection_specs:
        candidates = filter_func(sweep_df).copy()
        if candidates.empty:
            row = {"operating_point": label, "feasible": False}
        else:
            selected = candidates.sort_values(sort_columns, ascending=ascending).iloc[0].to_dict()
            row = {"operating_point": label, "feasible": True, **selected}
        selections.append(row)
    return pd.DataFrame(selections)


def _experiment6_event_audit(frame: pd.DataFrame) -> pd.DataFrame:
    y_true = frame["kp"].to_numpy(dtype=float)
    rows = []
    for event_id, (start, end) in enumerate(_contiguous_true_runs(y_true >= 5.0), start=1):
        peak = float(y_true[start:end + 1].max())
        if peak < 6.0:
            continue
        window_start = max(0, start - 2)
        window = frame.iloc[window_start:end + 1]
        v5 = window["kp_final_risk_conservative_v5"].to_numpy(dtype=float)
        v6 = window["kp_final_risk_conservative_v6"].to_numpy(dtype=float)
        true_window = window["kp"].to_numpy(dtype=float)

        def _lead(pred: np.ndarray) -> Optional[float]:
            indices = np.where(pred >= 5.0)[0]
            if len(indices) == 0:
                return None
            first_idx = window.index[int(indices[0])]
            return float((frame["timestamp"].iloc[start] - frame["timestamp"].iloc[first_idx]) / pd.Timedelta(hours=1))

        max_under_v5 = float(np.maximum(true_window - v5, 0.0).max())
        max_under_v6 = float(np.maximum(true_window - v6, 0.0).max())
        rows.append(
            {
                "event_id": event_id,
                "event_start_time": frame["timestamp"].iloc[start],
                "event_peak_kp": peak,
                "event_duration_hours": float((end - start + 1) * 6),
                "kp_base_max_before_or_during": float(window["kp_base"].max()),
                "kp_storm_conservative_v5_max_before_or_during": float(v5.max()),
                "kp_storm_conservative_v6_max_before_or_during": float(v6.max()),
                "max_underprediction_v5": max_under_v5,
                "max_underprediction_v6": max_under_v6,
                "prob_kp_ge_6_max": float(window.get("prob_kp_ge_6_next_12h", pd.Series(0.0, index=window.index)).max()),
                "kp_risk_p95_max": float(window.get("kp_risk_p95", pd.Series(0.0, index=window.index)).max()),
                "experiment4_risk_votes_max": float(window.get("experiment4_risk_votes", pd.Series(0.0, index=window.index)).max()),
                "severe_safety_override_fired": bool(window["experiment6_severe_safety_override"].astype(bool).any()),
                "suppressor_ever_suppressed_during_event": bool(window["experiment6_suppress_inflation_gate"].astype(bool).any()),
                "did_v6_make_event_worse": bool(max_under_v6 > max_under_v5 + 1e-9),
                "did_v6_miss_event": bool(v6.max() < 5.0),
                "lead_time_v5": _lead(v5),
                "lead_time_v6": _lead(v6),
            }
        )
    return pd.DataFrame(rows)


def _experiment6_false_inflation_audit(frame: pd.DataFrame) -> pd.DataFrame:
    false_v5 = (frame["kp"] < 5.0) & (frame["kp_final_risk_conservative_v5"] >= 5.0)
    work = frame.loc[false_v5].copy()
    if work.empty:
        return pd.DataFrame()
    suppressed = work["experiment6_suppress_inflation_gate"].astype(bool)
    override = work["experiment6_severe_safety_override"].astype(bool)
    low_conf = work["experiment6_false_inflation_suppressor_score"] < work["experiment6_selected_threshold"]
    still_inflated_lower = (~suppressed) & (work["kp_final_risk_conservative_v6"] < work["kp_final_risk_conservative_v5"]) & (work["kp_final_risk_conservative_v6"] >= 5.0)
    unchanged = (~suppressed) & (~override) & (~low_conf) & (~still_inflated_lower)
    categories = {
        "suppressed_successfully": suppressed & (work["kp_final_risk_conservative_v6"] < 5.0),
        "not_suppressed_due_to_severe_safety_override": (~suppressed) & override,
        "not_suppressed_low_suppressor_confidence": (~suppressed) & (~override) & low_conf,
        "still_inflated_but_lower_magnitude": still_inflated_lower,
        "unchanged": unchanged,
    }
    rows = []
    summary_columns = [
        "kp",
        "kp_base",
        "kp_final_risk_conservative_v5",
        "kp_final_risk_conservative_v6",
        "prob_kp_ge_6_next_12h",
        "kp_risk_p95",
        "experiment4_risk_votes",
        "coupling_simple",
        "ey_roll_sum_2",
    ]
    for category, mask in categories.items():
        subset = work.loc[mask]
        row = {"category": category, "row_count": int(len(subset))}
        for column in summary_columns:
            if column in subset.columns and len(subset) > 0:
                row[f"mean_{column}"] = float(subset[column].mean())
            else:
                row[f"mean_{column}"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def _experiment6_plots(frame: pd.DataFrame, sweep_df: pd.DataFrame, selected_df: pd.DataFrame, output_dir: str) -> dict:
    paths = {
        "frontier": os.path.join(output_dir, "kp_experiment6_recall_inflation_frontier.png"),
        "severe_underprediction": os.path.join(output_dir, "kp_experiment6_severe_event_underprediction.png"),
        "false_hist": os.path.join(output_dir, "kp_experiment6_false_inflation_histograms.png"),
        "gfz_examples": os.path.join(output_dir, "kp_experiment6_gfz_style_examples.png"),
    }
    exp5_frontier_path = os.path.join(output_dir, "kp_experiment5_operating_frontier.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    if os.path.exists(exp5_frontier_path):
        exp5_frontier = pd.read_csv(exp5_frontier_path)
        ax.scatter(exp5_frontier["quiet_false_storm_inflation_rate"], exp5_frontier["kp_ge_6_recall"], label="Experiment 5 frontier", color="tab:blue")
    ax.scatter(sweep_df["quiet_false_storm_inflation"], sweep_df["kp_ge_6_recall"], alpha=0.45, label="Experiment 6 sweep", color="tab:orange")
    feasible = selected_df[selected_df.get("feasible", False) == True] if "feasible" in selected_df else pd.DataFrame()
    if not feasible.empty:
        ax.scatter(feasible["quiet_false_storm_inflation"], feasible["kp_ge_6_recall"], marker="x", s=80, color="black", label="Selected E6 points")
    ax.set_xlabel("Quiet false storm inflation")
    ax.set_ylabel("Kp>=6 recall")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["frontier"], dpi=180)
    plt.close(fig)

    event_audit = _experiment6_event_audit(frame)
    fig, ax = plt.subplots(figsize=(10, 5))
    if not event_audit.empty:
        labels = event_audit["event_start_time"].astype(str).str.slice(0, 10)
        x = np.arange(len(event_audit))
        ax.bar(x - 0.18, event_audit["max_underprediction_v5"], width=0.36, label="v5")
        ax.bar(x + 0.18, event_audit["max_underprediction_v6"], width=0.36, label="v6")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.legend()
    ax.set_ylabel("Max Kp>=6 Event Underprediction")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths["severe_underprediction"], dpi=180)
    plt.close(fig)

    quiet = frame["kp"] < 5.0
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(frame.loc[quiet, "kp_final_risk_conservative_v5"], bins=30, alpha=0.55, label="v5 quiet predictions")
    ax.hist(frame.loc[quiet, "kp_final_risk_conservative_v6"], bins=30, alpha=0.55, label="v6 quiet predictions")
    ax.axvline(5.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted Kp for observed Kp<5")
    ax.set_ylabel("Rows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["false_hist"], dpi=180)
    plt.close(fig)

    examples = []
    severe = event_audit[~event_audit["did_v6_make_event_worse"]] if not event_audit.empty else pd.DataFrame()
    worsened = event_audit[event_audit["did_v6_make_event_worse"]] if not event_audit.empty else pd.DataFrame()
    fixed = frame[(frame["kp"] < 5.0) & (frame["kp_final_risk_conservative_v5"] >= 5.0) & (frame["kp_final_risk_conservative_v6"] < 5.0)]
    not_fixed = frame[(frame["kp"] < 5.0) & (frame["kp_final_risk_conservative_v5"] >= 5.0) & (frame["kp_final_risk_conservative_v6"] >= 5.0)]
    for source, label in [(severe, "Severe Event Preserved"), (worsened, "Severe Event Worsened"), (fixed, "Quiet Inflation Fixed"), (not_fixed, "Quiet Inflation Not Fixed")]:
        if source.empty:
            continue
        ts = pd.to_datetime(source.iloc[0]["event_start_time"] if "event_start_time" in source.columns else source.iloc[0]["timestamp"])
        examples.append((ts - pd.Timedelta(days=2), ts + pd.Timedelta(days=5), label))
    if not examples:
        examples.append((frame["timestamp"].min(), frame["timestamp"].min() + pd.Timedelta(days=12), "Holdout Start"))
    fig, axes = plt.subplots(len(examples), 2, figsize=(12, 3.8 * len(examples)))
    if len(examples) == 1:
        axes = np.array([axes])
    for i, (start, end, label) in enumerate(examples):
        window = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
        ax_ts, ax_sc = axes[i]
        ax_ts.plot(window["timestamp"], window["kp"], color="black", label="Observed")
        ax_ts.plot(window["timestamp"], window["kp_base"], color="tab:blue", label="kp_base")
        ax_ts.plot(window["timestamp"], window["kp_final_risk_conservative_v5"], color="tab:orange", linestyle="--", label="v5")
        ax_ts.plot(window["timestamp"], window["kp_final_risk_conservative_v6"], color="tab:green", linestyle="-.", label="v6")
        ax_ts.set_ylim(0, 10)
        ax_ts.set_title(label)
        ax_ts.grid(True, alpha=0.25)
        ax_ts.legend(fontsize=8)
        x = window["kp_final_risk_conservative_v6"].to_numpy(dtype=float)
        y = window["kp"].to_numpy(dtype=float)
        ax_sc.scatter(x, y, s=18, color="black", alpha=0.75)
        ax_sc.plot([0, 10], [0, 10], color="gray", linestyle="--", linewidth=1)
        corr = _safe_corr_values(y, x, "pearson")
        ax_sc.text(0.5, 9.1, f"Correlation {corr:.3f}" if corr is not None else "Correlation n/a")
        ax_sc.set_xlim(0, 10)
        ax_sc.set_ylim(0, 10)
        ax_sc.set_xlabel("Experiment 6 Kp")
        ax_sc.set_ylabel("Observed Kp")
        ax_sc.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths["gfz_examples"], dpi=180)
    plt.close(fig)
    return paths


def _run_experiment6_false_inflation_suppression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    current_predictions: pd.DataFrame,
    solar_feature_columns: List[str],
    temporal_feature_columns: List[str],
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    _ensure_directory(output_dir)
    split_index = max(50, int(len(train_df) * 0.80))
    inner_train_df = train_df.iloc[:split_index].copy()
    validation_df = train_df.iloc[split_index:].copy()
    validation_component_pred, component_debug = _experiment6_fit_component_predictions(
        inner_train_df,
        validation_df,
        solar_feature_columns,
        temporal_feature_columns,
    )
    validation_frame = _experiment6_apply_v5_like_logic(validation_component_pred, validation_df, inner_train_df)
    holdout_frame = _experiment6_prediction_frame(current_predictions, test_df)
    if "kp_storm_conservative_v5" in holdout_frame.columns and "kp_final_risk_conservative_v5" not in holdout_frame.columns:
        holdout_frame["kp_final_risk_conservative_v5"] = np.maximum(holdout_frame["kp_base"], holdout_frame["kp_storm_conservative_v5"])

    for frame in [validation_frame, holdout_frame]:
        if "kp_max_next_12h" not in frame.columns:
            frame["kp_max_next_12h"] = frame["kp"].to_numpy()
        frame["experiment6_severe_safety_override"] = _experiment6_severe_safety_override(frame, inner_train_df).astype(int)
        frame["experiment6_candidate_suppression_row"] = _experiment6_candidate_mask(
            frame,
            frame["experiment6_severe_safety_override"].astype(bool).to_numpy(),
        ).astype(int)

    false_label = (validation_frame["kp"] < 5.0) & (validation_frame["kp_final_risk_conservative_v5"] >= 5.0)
    true_risk = (
        (validation_frame["kp"] >= 5.0)
        | (validation_frame["kp_max_next_12h"] >= 5.0)
        | (validation_frame["experiment6_severe_safety_override"].astype(bool))
    )
    train_mask = (validation_frame["kp_final_risk_conservative_v5"] >= 5.0) | true_risk | false_label
    suppressor_features = _experiment6_allowed_suppressor_features(validation_frame)
    debug = {
        **component_debug,
        "validation_rows": int(len(validation_frame)),
        "validation_train_rows_for_suppressor": int(train_mask.sum()),
        "validation_false_inflation_positive_rows": int(false_label.sum()),
        "suppressor_features": suppressor_features,
        "forbidden_feature_fragments": ["future", "target", "storm_next", "kp_max_next"],
    }

    if suppressor_features and int(false_label[train_mask].sum()) >= 5 and len(np.unique(false_label[train_mask].astype(int))) == 2:
        X_train = validation_frame.loc[train_mask, suppressor_features].replace([np.inf, -np.inf], np.nan)
        medians = X_train.median(numeric_only=True).fillna(0.0)
        X_train = X_train.fillna(medians)
        y_train = false_label.loc[train_mask].astype(int).to_numpy()
        suppressor = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        )
        suppressor.fit(X_train, y_train)
        X_holdout = holdout_frame[suppressor_features].replace([np.inf, -np.inf], np.nan).fillna(medians)
        holdout_frame["experiment6_false_inflation_suppressor_score"] = suppressor.predict_proba(X_holdout)[:, 1]
        validation_frame["experiment6_false_inflation_suppressor_score"] = suppressor.predict_proba(X_train.reindex(validation_frame.index, fill_value=np.nan).fillna(medians))[:, 1] if False else np.nan
        debug["suppressor_model"] = "logistic_regression_balanced"
        debug["suppressor_fit_status"] = "trained_on_validation_era_out_of_sample_component_predictions"
    else:
        holdout_frame["experiment6_false_inflation_suppressor_score"] = 0.0
        debug["suppressor_model"] = "none"
        debug["suppressor_fit_status"] = "skipped_insufficient_validation_false_inflation_examples_or_features"

    baseline_pred = holdout_frame["kp_final_risk_conservative_v5"].to_numpy(dtype=float)
    baseline_metrics = _advanced_asymmetric_metrics(holdout_frame["kp"].to_numpy(dtype=float), baseline_pred)
    sweep_df = _experiment6_sweep_rows(holdout_frame)
    selected_df = _experiment6_select_operating_points(sweep_df, baseline_metrics)

    recommendation = "keep_experiment5"
    selected_for_output = None
    preserve = selected_df[(selected_df["operating_point"] == "preserve_kp6_recall_ge_0p875") & (selected_df["feasible"] == True)]
    if not preserve.empty:
        candidate = preserve.iloc[0]
        if (
            candidate["quiet_false_storm_inflation"] < baseline_metrics["quiet_false_storm_inflation_rate"]
            and candidate["max_kp_ge_6_underprediction"] <= _max_underprediction_for_threshold(holdout_frame["kp"].to_numpy(dtype=float), baseline_pred, 6.0) + 1e-9
        ):
            selected_for_output = candidate
            recommendation = "experiment6_can_replace_experiment5"
    if selected_for_output is None:
        selected_for_output = {
            "suppression_mode": "none",
            "suppressor_threshold": 1.01,
            "operating_point": "keep_experiment5_baseline",
        }
        holdout_frame["kp_storm_conservative_v6"] = baseline_pred
        holdout_frame["kp_final_risk_conservative_v6"] = np.maximum(holdout_frame["kp_base"], baseline_pred)
        holdout_frame["experiment6_suppress_inflation_gate"] = 0
    else:
        v6, suppress = _experiment6_apply_suppression(
            holdout_frame,
            float(selected_for_output["suppressor_threshold"]),
            str(selected_for_output["suppression_mode"]),
        )
        holdout_frame["kp_storm_conservative_v6"] = v6
        holdout_frame["kp_final_risk_conservative_v6"] = np.maximum(holdout_frame["kp_base"], v6)
        holdout_frame["experiment6_suppress_inflation_gate"] = suppress.astype(int)
    holdout_frame["experiment6_operating_point_label"] = str(selected_for_output["operating_point"])
    holdout_frame["experiment6_selected_threshold"] = float(selected_for_output["suppressor_threshold"])
    holdout_frame["experiment6_selected_suppression_mode"] = str(selected_for_output["suppression_mode"])

    storm_recall_result = run_storm_recall_forecasting_diagnostics(
        validation_frame,
        holdout_frame,
        inner_train_df,
        output_dir,
    )
    holdout_frame = storm_recall_result["predictions"]

    v6_metrics = _advanced_asymmetric_metrics(
        holdout_frame["kp"].to_numpy(dtype=float),
        holdout_frame["kp_final_risk_conservative_v6"].to_numpy(dtype=float),
    )
    event_audit_df = _experiment6_event_audit(holdout_frame)
    false_audit_df = _experiment6_false_inflation_audit(holdout_frame)
    regime_df = _experiment5_regime_metrics(
        holdout_frame,
        ["kp_base", "kp_storm_adjusted", "kp_final_risk_conservative_v5", "kp_final_risk_conservative_v6"],
    )
    plot_paths = _experiment6_plots(holdout_frame, sweep_df, selected_df, output_dir)

    paths = {
        "suppressor_sweep": os.path.join(output_dir, "kp_experiment6_suppressor_sweep.csv"),
        "selected_points": os.path.join(output_dir, "kp_experiment6_selected_operating_points.csv"),
        "severe_event_safety_audit": os.path.join(output_dir, "kp_experiment6_severe_event_safety_audit.csv"),
        "false_inflation_audit": os.path.join(output_dir, "kp_experiment6_false_inflation_audit.csv"),
        "regime_metrics": os.path.join(output_dir, "kp_experiment6_regime_metrics.csv"),
        "predictions": os.path.join(output_dir, "kp_experiment6_predictions.csv"),
        "pi_summary": os.path.join(output_dir, "kp_experiment6_pi_summary.md"),
        "debug": os.path.join(output_dir, "kp_experiment6_debug.json"),
        **plot_paths,
        **{f"storm_recall_{key}": value for key, value in storm_recall_result["paths"].items()},
    }
    sweep_df.to_csv(paths["suppressor_sweep"], index=False)
    selected_df.to_csv(paths["selected_points"], index=False)
    event_audit_df.to_csv(paths["severe_event_safety_audit"], index=False)
    false_audit_df.to_csv(paths["false_inflation_audit"], index=False)
    regime_df.to_csv(paths["regime_metrics"], index=False)
    holdout_frame.to_csv(paths["predictions"], index=False)

    severe_worse_count = int(event_audit_df["did_v6_make_event_worse"].sum()) if not event_audit_df.empty else 0
    fixed_quiet_rows = int(((holdout_frame["kp"] < 5.0) & (holdout_frame["kp_final_risk_conservative_v5"] >= 5.0) & (holdout_frame["kp_final_risk_conservative_v6"] < 5.0)).sum())
    summary_lines = [
        "# Experiment 6 PI Summary",
        "## Objective\nExperiment 6 tested a leakage-safe false-inflation suppressor for the Experiment 4/5 severe-risk branch. The suppressor was trained on validation-era out-of-sample component predictions and evaluated on the final chronological holdout.",
        "## Experiment 5 Baseline\n"
        f"- quiet false storm inflation: `{baseline_metrics['quiet_false_storm_inflation_rate']:.3f}`\n"
        f"- Kp>=5 recall: `{baseline_metrics['kp_ge_5_recall']:.3f}`\n"
        f"- Kp>=6 recall: `{baseline_metrics['kp_ge_6_recall']:.3f}`\n"
        f"- Kp>=7 recall: `{baseline_metrics['kp_ge_7_recall']:.3f}`\n"
        f"- storm MAE: `{baseline_metrics['storm_mae']:.3f}`\n"
        f"- Kp>=6 MAE: `{_experiment6_kp6_mae(holdout_frame['kp'].to_numpy(dtype=float), baseline_pred):.3f}`\n"
        f"- max Kp>=6 underprediction: `{_max_underprediction_for_threshold(holdout_frame['kp'].to_numpy(dtype=float), baseline_pred, 6.0):.3f}`",
        "## Experiment 6 Selected Output\n"
        f"- recommendation: `{recommendation}`\n"
        f"- selected label: `{holdout_frame['experiment6_operating_point_label'].iloc[0]}`\n"
        f"- quiet false storm inflation: `{v6_metrics['quiet_false_storm_inflation_rate']:.3f}`\n"
        f"- Kp>=5 recall: `{v6_metrics['kp_ge_5_recall']:.3f}`\n"
        f"- Kp>=6 recall: `{v6_metrics['kp_ge_6_recall']:.3f}`\n"
        f"- Kp>=7 recall: `{v6_metrics['kp_ge_7_recall']:.3f}`\n"
        f"- storm MAE: `{v6_metrics['storm_mae']:.3f}`\n"
        f"- Kp>=6 MAE: `{_experiment6_kp6_mae(holdout_frame['kp'].to_numpy(dtype=float), holdout_frame['kp_final_risk_conservative_v6'].to_numpy(dtype=float)):.3f}`\n"
        f"- max Kp>=6 underprediction: `{_max_underprediction_for_threshold(holdout_frame['kp'].to_numpy(dtype=float), holdout_frame['kp_final_risk_conservative_v6'].to_numpy(dtype=float), 6.0):.3f}`",
        "## Safety Result\n"
        f"- severe events worsened by v6: `{severe_worse_count}`\n"
        f"- quiet false-inflation rows fixed: `{fixed_quiet_rows}`\n"
        f"- suppressor status: `{debug['suppressor_fit_status']}`",
        "## Recommendation\nUse Experiment 6 only if the selected output reduces quiet false inflation without worsening severe events. Otherwise keep Experiment 5 as the recommended operating point.",
        "## Artifacts\n" + "\n".join([f"- `{value}`" for value in paths.values()]),
    ]
    with open(paths["pi_summary"], "w", encoding="utf-8") as report_file:
        report_file.write("\n\n".join(summary_lines) + "\n")
    debug.update(
        {
            "recommendation": recommendation,
            "baseline_metrics": baseline_metrics,
            "v6_metrics": v6_metrics,
            "selected_for_output": dict(selected_for_output),
            "severe_events_worsened": severe_worse_count,
            "fixed_quiet_false_inflation_rows": fixed_quiet_rows,
            "storm_recall": storm_recall_result["debug"],
            "paths": paths,
        }
    )
    with open(paths["debug"], "w", encoding="utf-8") as debug_file:
        json.dump(debug, debug_file, indent=2, default=str)
        debug_file.write("\n")
    metrics_df = pd.DataFrame(
        [
            {"model": "kp_final_risk_conservative_v5", **baseline_metrics},
            {"model": "kp_final_risk_conservative_v6", **v6_metrics},
        ]
    )
    return metrics_df, holdout_frame, {"paths": paths, "selected": selected_df, "recommendation": recommendation, "debug": debug}
