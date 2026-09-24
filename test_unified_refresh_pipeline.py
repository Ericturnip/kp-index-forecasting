import unittest

import numpy as np
import pandas as pd

from unified_refresh_pipeline import (
    FUTURE_STEPS,
    PAST_STEPS,
    _lead_time_bin,
    build_sequence_tensor_dataset,
    flatten_rolling_predictions,
)


def _synthetic_feature_frame(rows: int = 48) -> pd.DataFrame:
    timestamp = pd.date_range("2022-01-01", periods=rows, freq="6h")
    idx = np.arange(rows, dtype=float)
    bz = np.sin(idx / 4.0) * 3.0
    bs = np.maximum(0.0, -bz)
    velocity = 410.0 + 20.0 * np.cos(idx / 5.0)
    density = 4.0 + 0.2 * np.sin(idx / 3.0)
    pdyn = density * velocity * velocity * 1e-6
    ey = velocity * bs * 1e-3
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "bz": bz,
            "bs": bs,
            "velocity": velocity,
            "density": density,
            "pdyn": pdyn,
            "ey": ey,
            "newell_coupling": (velocity ** 1.2) * (bs + 0.1),
            "coupling_simple": velocity * bs,
            "compact_storm_score": bs + ey,
            "bz_southward_drop_6h": np.r_[0.0, np.maximum(0.0, bz[:-1] - bz[1:])],
            "bz_southward_drop_12h": np.r_[0.0, 0.0, np.maximum(0.0, bz[:-2] - bz[2:])],
            "bz_southward_drop_24h": np.r_[0.0, 0.0, 0.0, 0.0, np.maximum(0.0, bz[:-4] - bz[4:])],
            "bz_crossed_southward": np.r_[0, ((bz[:-1] > 0) & (bz[1:] < 0)).astype(int)],
            "strong_southward_turning": np.r_[0, ((np.maximum(0.0, bz[:-1] - bz[1:]) > 5) & (bz[1:] < -5)).astype(int)],
            "kp": np.clip(2.0 + bs * 0.35 + ey * 0.15, 0.0, 9.0),
        }
    )


class UnifiedRefreshPipelineTest(unittest.TestCase):
    def test_tensor_shapes_and_target_window(self):
        frame = _synthetic_feature_frame()
        data = build_sequence_tensor_dataset(frame, PAST_STEPS - 1, len(frame) - FUTURE_STEPS - 1)
        self.assertGreater(len(data.X), 0)
        self.assertEqual(data.y.shape[1], FUTURE_STEPS)
        init_idx = data.init_indices[0]
        expected_target = frame["kp"].iloc[init_idx + 1 : init_idx + FUTURE_STEPS + 1].to_numpy()
        np.testing.assert_allclose(data.y[0], expected_target)

    def test_future_tensor_forbids_kp_columns(self):
        frame = _synthetic_feature_frame()
        data = build_sequence_tensor_dataset(frame, PAST_STEPS - 1, len(frame) - FUTURE_STEPS - 1)
        future_kp_columns = [column for column in data.feature_columns if column.startswith("future_") and column.endswith("_kp")]
        self.assertEqual(future_kp_columns, [])
        self.assertTrue(any(column.endswith("_kp") and column.startswith("past_") for column in data.feature_columns))

    def test_no_assimilation_removes_past_kp(self):
        frame = _synthetic_feature_frame()
        data = build_sequence_tensor_dataset(frame, PAST_STEPS - 1, len(frame) - FUTURE_STEPS - 1, include_past_kp=False)
        self.assertFalse(any(column.endswith("_kp") for column in data.feature_columns))

    def test_flattened_predictions_are_leak_safe(self):
        frame = _synthetic_feature_frame()
        data = build_sequence_tensor_dataset(frame, PAST_STEPS - 1, len(frame) - FUTURE_STEPS - 1)
        zeros = np.zeros_like(data.y)
        flat = flatten_rolling_predictions(
            frame,
            data,
            {
                "kp_seq_mean_forecast": zeros,
                "kp_seq_p90_envelope": zeros + 1.0,
                "kp_seq_p95_envelope": zeros + 2.0,
            },
        )
        self.assertGreater(len(flat), 0)
        self.assertTrue((pd.to_datetime(flat["timestamp"]) > pd.to_datetime(flat["t_init"])).all())
        self.assertFalse(flat["uses_future_observed_kp"].astype(bool).any())
        self.assertTrue((pd.to_datetime(flat["kp_history_source_max_time"]) <= pd.to_datetime(flat["t_init"])).all())
        self.assertEqual(set(flat["lead_time_bin"].unique()).issubset({"0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-96h", "96-120h"}), True)

    def test_lead_time_bins(self):
        self.assertEqual(_lead_time_bin(6.0), "0-6h")
        self.assertEqual(_lead_time_bin(12.0), "6-12h")
        self.assertEqual(_lead_time_bin(120.0), "96-120h")


if __name__ == "__main__":
    unittest.main()
