import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PHASE1_DIR, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from ml import predict  # noqa: E402


class TestPredictModule(unittest.TestCase):
    def test_build_inference_frame_uses_defaults(self):
        df = predict.build_inference_frame(["sample"])
        self.assertEqual(df.loc[0, "rating"], 0.0)
        self.assertEqual(df.loc[0, "helpful_vote"], 0.0)
        self.assertEqual(df.loc[0, "verified_purchase"], 0)

    def test_label_from_probability_respects_threshold(self):
        self.assertEqual(predict.label_from_probability(0.9, 0.5), "fake")
        self.assertEqual(predict.label_from_probability(0.1, 0.5), "genuine")

    def test_scale_numeric_aligns_to_metadata_columns(self):
        num_df = pd.DataFrame({"rating": [5.0], "helpful_vote": [1.0], "extra": [99.0]})
        metadata = {
            "numeric_columns": ["rating", "helpful_vote", "verified_purchase"],
            "numeric_mean": {"rating": 3.0, "helpful_vote": 0.0, "verified_purchase": 0.0},
            "numeric_std": {"rating": 2.0, "helpful_vote": 1.0, "verified_purchase": 1.0},
        }
        scaled = predict.scale_numeric(num_df, metadata)
        self.assertEqual(list(scaled.columns), metadata["numeric_columns"])
        self.assertAlmostEqual(float(scaled.loc[0, "rating"]), 1.0)
        self.assertAlmostEqual(float(scaled.loc[0, "verified_purchase"]), 0.0)

    @patch("ml.predict.probability_from_model", return_value=np.array([0.82]))
    @patch("ml.predict.build_feature_matrix", return_value=np.zeros((1, 2)))
    @patch("ml.predict.load_artifacts")
    def test_predict_batch_returns_expected_schema(self, mock_load, _mock_feat, _mock_prob):
        mock_model = object()
        mock_vectorizer = object()
        mock_feature_meta = {
            "numeric_columns": ["rating", "helpful_vote", "verified_purchase"],
            "numeric_mean": {"rating": 0.0, "helpful_vote": 0.0, "verified_purchase": 0.0},
            "numeric_std": {"rating": 1.0, "helpful_vote": 1.0, "verified_purchase": 1.0},
        }
        mock_model_meta = {"threshold": 0.5, "model_version": "phase1-v2"}
        mock_load.return_value = (mock_model, mock_vectorizer, mock_feature_meta, mock_model_meta)

        out = predict.predict_batch(["great product"], models_dir="dummy")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["label"], "fake")
        self.assertEqual(out[0]["model_version"], "phase1-v2")
        self.assertIn("fake_probability", out[0])


if __name__ == "__main__":
    unittest.main()
