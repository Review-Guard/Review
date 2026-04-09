import math
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


class _ProbaModel:
    def predict_proba(self, x):
        rows = x.shape[0]
        return np.array([[0.2, 0.8]] * rows)


class _DecisionModel:
    def decision_function(self, x):
        return np.array([0.0] * x.shape[0])


class TestPredictUtilsExtended(unittest.TestCase):
    def test_punctuation_ratio_handles_empty(self):
        self.assertEqual(predict.punctuation_ratio(""), 0.0)

    def test_punctuation_ratio_computes_density(self):
        val = predict.punctuation_ratio("Wow!!")
        self.assertGreater(val, 0)

    def test_uppercase_ratio_handles_empty(self):
        self.assertEqual(predict.uppercase_ratio(""), 0.0)

    def test_uppercase_ratio_computes_expected_ratio(self):
        val = predict.uppercase_ratio("ABcd")
        self.assertAlmostEqual(val, 0.5)

    def test_build_behavioral_matrix_defaults_and_shape(self):
        df = pd.DataFrame({"text": ["a", "b"], "rating": [5, None]})
        out = predict.build_behavioral_matrix(df)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(float(out[1, 1]), 0.0)

    def test_probability_from_model_uses_predict_proba_when_available(self):
        x = np.zeros((2, 2))
        probs = predict.probability_from_model(_ProbaModel(), x)
        self.assertEqual(probs.shape[0], 2)
        self.assertTrue(np.allclose(probs, np.array([0.8, 0.8])))

    def test_probability_from_model_falls_back_to_sigmoid(self):
        x = np.zeros((3, 2))
        probs = predict.probability_from_model(_DecisionModel(), x)
        self.assertTrue(np.allclose(probs, np.array([0.5, 0.5, 0.5])))

    def test_build_numeric_features_handles_missing_numeric_values(self):
        df = pd.DataFrame(
            {
                "text": ["Great!!", None],
                "rating": [None, 4],
                "helpful_vote": [None, None],
                "verified_purchase": [None, 1],
            }
        )
        out = predict.build_numeric_features(df)
        self.assertFalse(out.isna().any().any())
        self.assertTrue({"char_count", "word_count", "rating"}.issubset(out.columns))

    @patch("ml.predict.predict_batch")
    def test_predict_single_delegates_to_predict_batch(self, mock_predict_batch):
        mock_predict_batch.return_value = [{"label": "fake", "fake_probability": 0.9}]
        out = predict.predict_single("text")
        self.assertEqual(out["label"], "fake")
        mock_predict_batch.assert_called_once()

    @patch("ml.predict.predict_batch_v3")
    def test_predict_single_v3_delegates_to_predict_batch_v3(self, mock_predict_batch_v3):
        mock_predict_batch_v3.return_value = [{"label": "genuine", "fake_probability": 0.2}]
        out = predict.predict_single_v3("text")
        self.assertEqual(out["label"], "genuine")
        mock_predict_batch_v3.assert_called_once()

    def test_clean_text_for_model_removes_url_and_normalizes_spaces(self):
        out = predict.clean_text_for_model("Visit https://x.com NOW!!!")
        self.assertEqual(out, "visit now!!!")

    def test_label_from_probability_boundary_is_fake(self):
        out = predict.label_from_probability(0.5, 0.5)
        self.assertEqual(out, "fake")


if __name__ == "__main__":
    unittest.main()
