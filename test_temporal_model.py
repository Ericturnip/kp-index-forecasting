import json
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from temporal_model import (
    BZ_SHARP_DROP_FEATURES,
    FORECAST_INPUTS_ALREADY_TARGET_ALIGNED,
    KP_HISTORY_FEATURES,
    KP_MODEL_FORECAST_SEMANTICS,
    RUN_EXPERIMENT7_DIAGNOSTIC_TARGET_ALIGNED,
    RUN_EXPERIMENT7_OPERATIONAL_ROLLING,
    _experiment7_feature_sets,
    _experiment11_add_leaky_integrators,
    _storm_recall_regime_rows,
    _storm_recovery_metrics,
    _storm_recall_threshold_sweep,
    prepare_temporal_dataframe,
    run_storm_recall_forecasting_diagnostics,
    run_experiment5_diagnostics,
)


warnings.filterwarnings("ignore", category=PerformanceWarning)


class TemporalModelLeakageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw_df = pd.read_csv("testing/2008_2025_df.txt")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PerformanceWarning)
            cls.feature_df, cls.metadata = prepare_temporal_dataframe(raw_df, return_metadata=True)

    def test_temporal_feature_source_times_are_strictly_past(self):
        leakage = self.metadata["leakage"]
        self.assertTrue(leakage["passed"], leakage["violations"])
        self.assertGreater(leakage["checked_features"], 0)

    def test_project_forecast_semantics_metadata_is_explicit(self):
        driver_metadata = self.metadata["driver_input_metadata"]
        self.assertEqual(driver_metadata["semantics"]["core_task"], "same_timestamp_driver_to_kp_calculation")
        self.assertEqual(driver_metadata["semantics"]["production_horizon_source"], "upstream Bz/velocity/density forecasts")
        self.assertTrue(driver_metadata["forecast_inputs_already_target_aligned"])
        self.assertIn("run_kp_model.py", driver_metadata["original_pi_linear_equation"]["source_module"])
        self.assertIn("dphi_dt", driver_metadata["original_pi_linear_equation"]["equation"])
        self.assertEqual(
            KP_MODEL_FORECAST_SEMANTICS["incorrect_framing"],
            "Given current observed drivers, extrapolate future Kp without forecasted driver inputs.",
        )

    def test_kp_history_availability_metadata_is_leak_safe(self):
        availability = self.metadata["kp_history_availability"]
        self.assertTrue(availability["passed"], availability["violations"])
        self.assertEqual(availability["checked_features"], len(KP_HISTORY_FEATURES))
        min_lookbacks = [row["min_lookback_hours"] for row in availability["features"]]
        self.assertTrue(all(value > 0.0 for value in min_lookbacks))

    def test_kp_roll_mean_uses_previous_values(self):
        row_index = 25
        expected = self.feature_df["kp"].iloc[row_index - 4:row_index].mean()
        actual = self.feature_df["kp_roll_mean_4"].iloc[row_index]
        self.assertAlmostEqual(actual, expected, places=10)

    def test_future_magnitude_target_uses_future_kp_only(self):
        row_index = 25
        expected = self.feature_df["kp"].iloc[row_index + 1:row_index + 3].max()
        actual = self.feature_df["kp_max_next_12h"].iloc[row_index]
        self.assertAlmostEqual(actual, expected, places=10)

    def test_severe_classifier_targets_match_future_max(self):
        self.assertTrue(
            (
                self.feature_df["kp_ge_6_next_12h"].astype(bool)
                == (self.feature_df["kp_max_next_12h"] >= 6.0)
            ).all()
        )

    def test_experiment3_outputs_are_finite_when_present(self):
        prediction_path = "testing/kp_nowcast_predictions.csv"
        predictions = pd.read_csv(prediction_path)
        required = [
            "kp_base",
            "kp_risk_p95",
            "kp_storm_conservative_v2",
            "kp_final_risk_conservative",
            "severe_watch_gate",
        ]
        missing = [column for column in required if column not in predictions.columns]
        self.assertEqual(missing, [])
        self.assertTrue(np.isfinite(predictions[["kp_risk_p95", "kp_storm_conservative_v2"]].to_numpy()).all())
        active = predictions["severe_watch_gate"].astype(bool)
        if active.any():
            self.assertTrue(
                (
                    predictions.loc[active, "kp_storm_conservative_v2"]
                    >= predictions.loc[active, "kp_base"]
                ).all()
            )

    def test_experiment3_watch_sweep_exists(self):
        sweep = pd.read_csv("testing/kp_experiment3_severe_watch_sweep.csv")
        self.assertGreater(len(sweep), 0)
        self.assertIn("kp_ge_6_event_recall", sweep.columns)

    def test_experiment4_calibrated_output_is_finite(self):
        predictions = pd.read_csv("testing/kp_nowcast_predictions.csv")
        required = [
            "experiment4_risk_votes",
            "experiment4_severe_boost_authorized",
            "kp_storm_conservative_v4",
            "kp_final_risk_conservative_v4",
        ]
        missing = [column for column in required if column not in predictions.columns]
        self.assertEqual(missing, [])
        self.assertTrue(np.isfinite(predictions[["kp_storm_conservative_v4", "kp_final_risk_conservative_v4"]].to_numpy()).all())
        self.assertTrue((predictions["kp_final_risk_conservative_v4"] >= predictions["kp_base"]).all())

    def test_experiment4_tradeoff_curve_has_quiet_constraint(self):
        tradeoff = pd.read_csv("testing/kp_experiment4_selected_operating_points.csv")
        self.assertIn("quiet_inflation_target", tradeoff.columns)
        self.assertIn("kp_ge_6_recall", tradeoff.columns)
        self.assertGreater(len(tradeoff), 0)

    def test_experiment5_outputs_are_generated(self):
        result = run_experiment5_diagnostics("testing/kp_nowcast_predictions.csv", "testing")
        predictions = result["predictions"]
        required = [
            "experiment5_calibrated_risk_score",
            "experiment5_storm_watch_gate",
            "experiment5_severe_boost_authorized",
            "kp_storm_conservative_v5",
            "kp_final_risk_conservative_v5",
            "experiment5_false_inflation_suppressor_score",
            "experiment5_operating_point_label",
        ]
        missing = [column for column in required if column not in predictions.columns]
        self.assertEqual(missing, [])
        self.assertTrue(np.isfinite(predictions[["kp_storm_conservative_v5", "kp_final_risk_conservative_v5"]].to_numpy()).all())
        self.assertTrue((predictions["kp_final_risk_conservative_v5"] >= predictions["kp_base"]).all())

    def test_experiment5_frontier_contains_requested_inflation_targets(self):
        frontier = pd.read_csv("testing/kp_experiment5_operating_frontier.csv")
        self.assertIn("quiet_inflation_target", frontier.columns)
        self.assertIn("missed_kp_ge_6_rows_estimated", frontier.columns)
        expected = {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50}
        actual = {round(float(value), 2) for value in frontier["quiet_inflation_target"]}
        self.assertEqual(actual, expected)

    def test_experiment6_output_columns_exist_and_stay_above_base(self):
        predictions = pd.read_csv("testing/kp_nowcast_predictions.csv")
        required = [
            "experiment6_false_inflation_suppressor_score",
            "experiment6_suppress_inflation_gate",
            "experiment6_severe_safety_override",
            "kp_storm_conservative_v6",
            "kp_final_risk_conservative_v6",
            "experiment6_operating_point_label",
        ]
        missing = [column for column in required if column not in predictions.columns]
        self.assertEqual(missing, [])
        self.assertTrue(np.isfinite(predictions[["kp_storm_conservative_v6", "kp_final_risk_conservative_v6"]].to_numpy()).all())
        self.assertTrue((predictions["kp_final_risk_conservative_v6"] >= predictions["kp_base"]).all())

    def test_experiment6_suppressor_features_are_prediction_time_safe(self):
        with open("testing/kp_experiment6_debug.json", "r", encoding="utf-8") as debug_file:
            debug = __import__("json").load(debug_file)
        features = debug.get("suppressor_features", [])
        forbidden = ["future", "target", "storm_next", "kp_max_next"]
        unsafe = [feature for feature in features if any(fragment in feature for fragment in forbidden)]
        self.assertEqual(unsafe, [])
        self.assertEqual(debug.get("suppressor_fit_status"), "trained_on_validation_era_out_of_sample_component_predictions")

    def test_experiment6_severe_override_blocks_suppression(self):
        predictions = pd.read_csv("testing/kp_nowcast_predictions.csv")
        override = predictions["experiment6_severe_safety_override"].astype(bool)
        if override.any():
            self.assertFalse(predictions.loc[override, "experiment6_suppress_inflation_gate"].astype(bool).any())

    def test_experiment6_selected_points_and_severe_audit_exist(self):
        selected = pd.read_csv("testing/kp_experiment6_selected_operating_points.csv")
        self.assertIn("operating_point", selected.columns)
        self.assertIn("preserve_kp6_recall_ge_0p875", set(selected["operating_point"]))
        audit = pd.read_csv("testing/kp_experiment6_severe_event_safety_audit.csv")
        self.assertGreater(len(audit), 0)
        self.assertTrue((audit["event_peak_kp"] >= 6.0).all())

    def test_forecast_aligned_model_does_not_use_target_kp_feature(self):
        self.assertTrue(FORECAST_INPUTS_ALREADY_TARGET_ALIGNED)
        feature_sets = _experiment7_feature_sets(self.metadata)
        physics_features = feature_sets["operational_physics_only"]
        operational_features = feature_sets["operational_all_features"]
        self.assertNotIn("kp", physics_features)
        self.assertFalse(any(feature.startswith("kp_") for feature in physics_features))
        self.assertNotIn("kp", operational_features)

    def test_forecast_aligned_kp_rolling_features_are_shifted(self):
        row_index = 25
        expected_mean_20 = self.feature_df["kp"].iloc[row_index - 20:row_index].mean()
        self.assertAlmostEqual(self.feature_df["kp_roll_mean_20"].iloc[row_index], expected_mean_20, places=10)
        self.assertTrue(all(feature in self.metadata["kp_history_feature_columns"] for feature in KP_HISTORY_FEATURES))

    def test_bz_drop_features_use_current_and_past_forecast_inputs(self):
        row_index = 25
        expected_drop = max(0.0, self.feature_df["bz"].iloc[row_index - 1] - self.feature_df["bz"].iloc[row_index])
        self.assertAlmostEqual(self.feature_df["bz_southward_drop_6h"].iloc[row_index], expected_drop, places=10)
        self.assertTrue(all(feature in self.feature_df.columns for feature in BZ_SHARP_DROP_FEATURES))

    def test_24h_pattern_break_features_do_not_include_future_kp(self):
        row_index = 25
        expected_repeat_error = abs(self.feature_df["kp"].iloc[row_index - 1] - self.feature_df["kp"].iloc[row_index - 4])
        expected_score = expected_repeat_error + abs(
            self.feature_df["kp"].iloc[row_index - 1] - self.feature_df["kp"].iloc[row_index - 2]
        )
        self.assertAlmostEqual(self.feature_df["kp_24h_repeat_error"].iloc[row_index], expected_repeat_error, places=10)
        self.assertAlmostEqual(self.feature_df["kp_daily_pattern_break_score"].iloc[row_index], expected_score, places=10)

    def test_experiment7_outputs_exist_and_main_branch_is_not_old_shifted(self):
        self.assertTrue(RUN_EXPERIMENT7_DIAGNOSTIC_TARGET_ALIGNED)
        self.assertTrue(RUN_EXPERIMENT7_OPERATIONAL_ROLLING)
        predictions = pd.read_csv("testing/kp_experiment7_diagnostic_target_aligned_predictions.csv")
        required = [
            "kp_experiment7_bz_velocity_density_baseline",
            "kp_experiment7_target_aligned_base",
            "kp_experiment7_target_aligned_best",
            "kp_experiment7_operational_frozen",
            "kp_experiment7_operational_autoregressive",
            "kp_experiment7_operational_physics_only",
            "kp_experiment7_conservative",
            "prob_experiment7_kp_ge_5",
            "prob_experiment7_kp_ge_6",
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
        missing = [column for column in required if column not in predictions.columns]
        self.assertEqual(missing, [])
        self.assertTrue(np.isfinite(predictions[["kp_experiment7_target_aligned_best", "kp_experiment7_conservative"]].dropna().to_numpy()).all())
        with open("testing/kp_experiment7_debug.json", "r", encoding="utf-8") as debug_file:
            debug = __import__("json").load(debug_file)
        self.assertIn("operational_frozen", debug["target_shifting_by_branch"])
        self.assertIn("diagnostic_target_aligned", debug["target_shifting_by_branch"])
        self.assertGreater(debug["rolling_initializations"], 0)
        self.assertGreater(debug["predicted_target_rows"], 0)

    def test_experiment7_operational_rolling_outputs_are_leak_safe(self):
        rolling = pd.read_csv("testing/kp_experiment7_operational_rolling_predictions.csv", parse_dates=["init_time", "valid_time", "kp_history_source_max_time"])
        self.assertGreater(len(rolling), 0)
        self.assertTrue((rolling["valid_time"] > rolling["init_time"]).all())
        self.assertTrue((rolling["lead_time_hours"] <= 120.0).all())
        self.assertFalse(rolling["uses_future_observed_kp"].astype(bool).any())
        self.assertTrue((rolling["kp_history_source_max_time"] <= rolling["init_time"]).all())
        autoregressive = rolling[rolling["model_mode"] == "experiment7_operational_autoregressive"]
        if not autoregressive.empty:
            self.assertFalse(autoregressive["uses_future_observed_kp"].astype(bool).any())

    def test_experiment7_leadtime_and_mode_metrics_exist(self):
        lead = pd.read_csv("testing/kp_experiment7_operational_leadtime_metrics.csv")
        self.assertGreater(len(lead), 0)
        self.assertIn("lead_time_bin", lead.columns)
        self.assertIn("recall_kp_ge_6", lead.columns)
        comparison = pd.read_csv("testing/kp_experiment7_mode_comparison.csv")
        self.assertIn("experiment7_operational_frozen", set(comparison["model_mode"]))
        self.assertIn("experiment7_operational_physics_only", set(comparison["model_mode"]))

    def test_experiment7_dynamic_refresh_backtest_outputs_exist(self):
        predictions = pd.read_csv("testing/kp_experiment7_dynamic_backtest_predictions.csv", parse_dates=["timestamp", "t_init"])
        required_prediction_columns = [
            "timestamp",
            "t_init",
            "lead_time_bin",
            "observed_kp",
            "kp_forecast_aligned_base",
            "kp_forecast_aligned_conservative",
        ]
        self.assertEqual([column for column in required_prediction_columns if column not in predictions.columns], [])
        self.assertGreater(len(predictions), 0)
        self.assertTrue((predictions["timestamp"] > predictions["t_init"]).all())
        self.assertTrue((predictions["lead_time_hours"] <= 120.0).all())
        metrics = pd.read_csv("testing/kp_experiment7_dynamic_backtest_metrics.csv")
        self.assertIn("0-6h", set(metrics["lead_time_bin"]))
        self.assertIn("96-120h", set(metrics["lead_time_bin"]))
        near = metrics.set_index("lead_time_bin")
        self.assertLess(abs(float(near.loc["0-6h", "mae"]) - 0.738), 0.02)
        self.assertLess(float(near.loc["0-6h", "quiet_false_storm_inflation"]), 0.0003)

    def test_experiment8_storm_watch_outputs_exist(self):
        predictions = pd.read_csv("testing/kp_experiment8_storm_watch_predictions.csv")
        required = [
            "init_time",
            "prob_any_kp_ge_5_next_120h",
            "prob_any_kp_ge_6_next_120h",
            "storm_watch_kp_ge_5",
            "predicted_max_kp_next_120h",
            "observed_max_kp_next_120h",
        ]
        self.assertEqual([column for column in required if column not in predictions.columns], [])
        metrics = pd.read_csv("testing/kp_experiment8_storm_watch_metrics.csv")
        self.assertIn("any_kp_ge_5_next_120h", set(metrics["target"]))
        kp5 = metrics[metrics["target"] == "any_kp_ge_5_next_120h"].iloc[0]
        self.assertGreaterEqual(float(kp5["recall"]), 0.90)
        self.assertGreater(float(kp5["average_precision"]), 0.0)

    def test_experiment11_leaky_integrators_use_current_and_past_physics(self):
        engineered, leaky_columns = _experiment11_add_leaky_integrators(self.feature_df)
        self.assertIn("experiment11_leaky_bs_a0p85", leaky_columns)
        row_index = 25
        values = np.maximum(engineered["bs"].iloc[: row_index + 1].astype(float).to_numpy(), 0.0)
        expected = 0.0
        for idx, value in enumerate(values):
            expected = value if idx == 0 else 0.85 * expected + value
        self.assertAlmostEqual(engineered["experiment11_leaky_bs_a0p85"].iloc[row_index], expected, places=10)

    def test_experiment11_outputs_exist_and_are_leak_safe(self):
        predictions = pd.read_csv(
            "testing/kp_experiment11_rolling_predictions.csv",
            parse_dates=["timestamp", "t_init", "kp_history_source_max_time"],
        )
        required = [
            "kp_experiment11_density_mean",
            "kp_experiment11_density_p90",
            "kp_experiment11_multitask_proxy",
            "kp_experiment11_risk_envelope",
            "prob_experiment11_kp_ge_5",
            "prob_experiment11_kp_ge_6",
            "storm_watch_experiment11",
        ]
        self.assertEqual([column for column in required if column not in predictions.columns], [])
        self.assertGreater(len(predictions), 0)
        self.assertFalse(predictions["uses_future_observed_kp"].astype(bool).any())
        self.assertTrue((predictions["kp_history_source_max_time"] <= predictions["t_init"]).all())
        metrics = pd.read_csv("testing/kp_experiment11_metrics.csv")
        self.assertIn("kp_experiment11_density_p90", set(metrics["model"]))
        self.assertIn("kp_experiment11_risk_envelope", set(metrics["model"]))

    def test_storm_recall_metric_helpers_track_kp5_and_kp6(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=8, freq="6h"),
                "kp": [2.0, 5.2, 6.1, 3.0, 4.0, 6.4, 7.2, 2.0],
                "kp_storm_recall_conservative": [2.0, 5.0, 6.0, 3.0, 4.5, 5.8, 7.0, 2.0],
            }
        )
        metrics = _storm_recall_regime_rows(frame, ["kp_storm_recall_conservative"])
        threshold_rows = metrics[metrics["metric_group"] == "threshold_recall"]
        self.assertIn("kp_ge_5", set(threshold_rows["regime"]))
        self.assertIn("kp_ge_6", set(threshold_rows["regime"]))
        regression_rows = metrics[metrics["metric_group"] == "regression_regime"]
        self.assertIn("kp_4_to_lt_7_primary_warning_regime", set(regression_rows["regime"]))
        self.assertIn("quiet_kp_lt_4", set(regression_rows["regime"]))
        kp6 = threshold_rows[threshold_rows["regime"] == "kp_ge_6"].iloc[0]
        self.assertGreater(float(kp6["recall"]), 0.0)
        sweep = _storm_recall_threshold_sweep(
            frame,
            np.array([0.0, 0.95, 0.90, 0.0, 0.1, 0.80, 0.99, 0.0]),
            6.0,
            np.array([0.5, 0.85]),
            2,
            "validation",
        )
        self.assertIn("event_recall", sweep.columns)
        self.assertGreaterEqual(float(sweep["recall"].max()), 2.0 / 3.0)

    def test_storm_recovery_metrics_track_driver_relaxation(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2022-01-01", periods=8, freq="6h"),
                "kp": [2.0, 5.5, 6.0, 4.8, 4.2, 3.5, 2.5, 2.0],
                "bz": [-1.0, -6.0, -5.5, -0.5, -0.2, 0.1, 0.2, 0.3],
                "bs": [0.0, 6.0, 5.5, 0.5, 0.2, 0.0, 0.0, 0.0],
                "velocity": [380.0, 560.0, 540.0, 390.0, 385.0, 380.0, 375.0, 370.0],
                "density": [4.0, 8.0, 7.5, 4.2, 4.1, 4.0, 3.8, 3.7],
                "kp_base": [2.0, 4.8, 5.0, 3.5, 3.2, 3.0, 2.4, 2.0],
                "kp_storm_recall_conservative": [2.0, 5.5, 6.0, 4.8, 4.5, 3.8, 2.8, 2.1],
            }
        )
        metrics = _storm_recovery_metrics(frame, ["kp_base", "kp_storm_recall_conservative"])
        self.assertIn("storm_recovery_after_driver_relaxation", set(metrics["metric_group"]))
        recovery = metrics[metrics["metric_group"] == "storm_recovery_after_driver_relaxation"]
        self.assertTrue((recovery["row_count"] > 0).all())

    def test_storm_recall_artifacts_are_generated_with_validation_metadata(self):
        timestamps = pd.date_range("2021-01-01", periods=12, freq="6h")
        base = pd.DataFrame(
            {
                "timestamp": timestamps,
                "kp": [2.0, 3.0, 5.1, 5.6, 2.0, 4.0, 6.2, 6.4, 3.0, 7.4, 6.8, 2.0],
                "bz": [-1.0, -2.0, -5.0, -6.0, -1.0, -3.0, -7.0, -8.0, -2.0, -9.0, -7.5, -1.0],
                "velocity": [400.0, 410.0, 520.0, 540.0, 390.0, 420.0, 560.0, 570.0, 430.0, 610.0, 590.0, 410.0],
                "density": [4.0, 4.1, 6.0, 5.8, 3.9, 4.5, 7.0, 7.2, 4.4, 8.0, 7.6, 4.1],
                "pdyn": [1.0, 1.1, 2.4, 2.5, 1.0, 1.4, 3.0, 3.1, 1.3, 3.5, 3.2, 1.1],
                "kp_base": [2.0, 3.0, 4.4, 4.8, 2.1, 4.0, 5.3, 5.6, 3.2, 6.1, 5.9, 2.2],
                "kp_storm_adjusted": [2.0, 3.0, 5.0, 5.1, 2.1, 4.0, 5.9, 6.0, 3.2, 6.8, 6.2, 2.2],
                "kp_final_risk_conservative_v5": [2.0, 3.0, 5.2, 5.5, 2.1, 4.1, 6.1, 6.2, 3.2, 7.0, 6.5, 2.2],
                "kp_final_risk_conservative_v6": [2.0, 3.0, 5.2, 5.5, 2.1, 4.1, 6.2, 6.3, 3.2, 7.1, 6.6, 2.2],
                "prob_kp_ge_5_next_12h": [0.05, 0.2, 0.92, 0.90, 0.1, 0.3, 0.95, 0.96, 0.2, 0.99, 0.97, 0.1],
                "prob_kp_ge_6_next_12h": [0.01, 0.05, 0.2, 0.3, 0.02, 0.1, 0.88, 0.90, 0.1, 0.95, 0.91, 0.05],
                "prob_kp_ge_7_next_12h": [0.0, 0.0, 0.05, 0.08, 0.0, 0.02, 0.10, 0.12, 0.03, 0.4, 0.25, 0.0],
            }
        )
        validation = base.iloc[:8].copy().reset_index(drop=True)
        holdout = base.iloc[4:].copy().reset_index(drop=True)
        inner_train = base.iloc[:4].copy().reset_index(drop=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_storm_recall_forecasting_diagnostics(validation, holdout, inner_train, tmpdir)
            predictions = result["predictions"]
            required = [
                "storm_recall_probability_kp_ge_5",
                "storm_recall_probability_kp_ge_6",
                "storm_recall_internal_watch_kp_ge_5",
                "storm_recall_internal_watch_kp_ge_6",
                "storm_recall_public_alert_reference",
                "storm_recall_public_alert_retuned_reference",
                "kp_storm_recall_conservative",
                "storm_recall_extreme_diagnostic_label",
            ]
            self.assertEqual([column for column in required if column not in predictions.columns], [])
            for path in result["paths"].values():
                self.assertTrue(__import__("os").path.exists(path), path)
            self.assertIn("watch_public_comparison", result["paths"])
            self.assertIn("storm_recovery_metrics", result["paths"])
            self.assertIn("branch_comparison", result["paths"])
            self.assertFalse(result["watch_public_comparison"].empty)
            self.assertFalse(result["branch_comparison"].empty)
            with open(result["paths"]["debug"], "r", encoding="utf-8") as debug_file:
                debug = json.load(debug_file)
            self.assertEqual(debug["threshold_selection_source"], "validation")
            self.assertEqual(debug["final_evaluation_source"], "holdout")
            self.assertEqual(debug["driver_input_metadata"]["semantics"]["core_task"], "same_timestamp_driver_to_kp_calculation")
            self.assertIn("public_alert_retuned_reference", debug)
            unsafe = [
                feature
                for feature in debug["storm_recall_score_columns"]
                if any(fragment in feature for fragment in debug["forbidden_feature_fragments"])
            ]
            self.assertEqual(unsafe, [])


if __name__ == "__main__":
    unittest.main()
