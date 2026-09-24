import unittest

import numpy as np
import pandas as pd

import experiment17_pytorch_transformer as exp17
from experiment16_transformer_seq2seq import TOKEN_FEATURES, build_transformer_tensor_dataset
from test_unified_refresh_pipeline import _synthetic_feature_frame
from unified_refresh_pipeline import FUTURE_STEPS, PAST_STEPS


@unittest.skipIf(exp17.torch is None, "PyTorch not installed")
class Experiment17PyTorchTransformerTest(unittest.TestCase):
    def test_model_output_shape(self):
        model = exp17.KpTransformer(input_dim=len(TOKEN_FEATURES))
        x = exp17.torch.zeros(2, PAST_STEPS + FUTURE_STEPS, len(TOKEN_FEATURES), dtype=exp17.torch.float32)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, FUTURE_STEPS, 7))

    def test_loss_is_finite(self):
        model = exp17.KpTransformer(input_dim=len(TOKEN_FEATURES))
        x = exp17.torch.zeros(2, PAST_STEPS + FUTURE_STEPS, len(TOKEN_FEATURES), dtype=exp17.torch.float32)
        target = exp17.torch.ones(2, FUTURE_STEPS, dtype=exp17.torch.float32)
        pos_weight = exp17.torch.tensor(exp17.POS_WEIGHTS, dtype=exp17.torch.float32)
        loss = exp17.total_loss(model(x), target, pos_weight)
        self.assertTrue(exp17.torch.isfinite(loss).item())

    def test_prediction_flattening_is_leak_safe(self):
        frame = _synthetic_feature_frame(rows=52)
        data = build_transformer_tensor_dataset(frame, PAST_STEPS - 1, len(frame) - FUTURE_STEPS - 1)
        zeros = np.zeros_like(data.y)
        predictions = {
            "kp_torch_transformer_mean": zeros,
            "kp_torch_transformer_p90": zeros + 1.0,
            "kp_torch_transformer_p95": zeros + 2.0,
            "prob_torch_transformer_kp_ge_5": zeros,
            "prob_torch_transformer_kp_ge_6": zeros,
            "prob_torch_transformer_kp_ge_7": zeros,
            "prob_torch_transformer_kp_ge_8": zeros,
        }
        flat = exp17.flatten_predictions(frame, data, predictions)
        self.assertGreater(len(flat), 0)
        self.assertFalse(flat["uses_future_observed_kp"].astype(bool).any())
        self.assertTrue((pd.to_datetime(flat["timestamp"]) > pd.to_datetime(flat["t_init"])).all())
        self.assertTrue((pd.to_datetime(flat["kp_history_source_max_time"]) <= pd.to_datetime(flat["t_init"])).all())


if __name__ == "__main__":
    unittest.main()
