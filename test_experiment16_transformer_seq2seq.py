import unittest

import numpy as np

from experiment16_transformer_seq2seq import (
    FUTURE_STEPS,
    PAST_STEPS,
    TOKEN_FEATURES,
    attention_step_features,
    build_transformer_tensor_dataset,
    fit_attention_encoder,
    flatten_transformer_predictions,
    transform_tokens,
)
from test_unified_refresh_pipeline import _synthetic_feature_frame


class Experiment16TransformerSeq2SeqTest(unittest.TestCase):
    def test_transformer_tensor_shape_and_no_future_kp(self):
        frame = _synthetic_feature_frame(rows=52)
        data = build_transformer_tensor_dataset(frame, PAST_STEPS - 1, len(frame) - FUTURE_STEPS - 1)
        self.assertGreater(len(data.tokens), 0)
        self.assertEqual(data.tokens.shape[1], PAST_STEPS + FUTURE_STEPS)
        self.assertEqual(data.y.shape[1], FUTURE_STEPS)
        kp_idx = TOKEN_FEATURES.index("kp")
        self.assertTrue((data.tokens[:, PAST_STEPS:, kp_idx] == 0.0).all())
        self.assertTrue((data.tokens[:, :PAST_STEPS, kp_idx] > 0.0).any())

    def test_attention_outputs_future_step_features(self):
        frame = _synthetic_feature_frame(rows=52)
        data = build_transformer_tensor_dataset(frame, PAST_STEPS - 1, len(frame) - FUTURE_STEPS - 1)
        encoder = fit_attention_encoder(data.tokens, data.token_features)
        embeddings, attention = transform_tokens(encoder, data.tokens, return_attention=True)
        self.assertEqual(embeddings.shape[:2], data.tokens.shape[:2])
        self.assertEqual(attention.shape[1:], (PAST_STEPS + FUTURE_STEPS, PAST_STEPS + FUTURE_STEPS))
        features = attention_step_features(embeddings, data.tokens)
        self.assertEqual(len(features), len(data.tokens) * FUTURE_STEPS)

    def test_flattened_transformer_predictions_are_leak_safe(self):
        frame = _synthetic_feature_frame(rows=52)
        data = build_transformer_tensor_dataset(frame, PAST_STEPS - 1, len(frame) - FUTURE_STEPS - 1)
        zeros = np.zeros_like(data.y)
        flat = flatten_transformer_predictions(
            frame,
            data,
            {
                "kp_transformer_mean": zeros,
                "kp_transformer_p90": zeros + 1,
                "kp_transformer_p95": zeros + 2,
                "prob_transformer_kp_ge_5": zeros,
                "prob_transformer_kp_ge_6": zeros,
                "prob_transformer_kp_ge_7": zeros,
                "prob_transformer_kp_ge_8": zeros,
            },
        )
        self.assertGreater(len(flat), 0)
        self.assertTrue((flat["timestamp"] > flat["t_init"]).all())
        self.assertFalse(flat["uses_future_observed_kp"].astype(bool).any())
        self.assertTrue((flat["kp_history_source_max_time"] <= flat["t_init"]).all())


if __name__ == "__main__":
    unittest.main()
