import os
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np


PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PHASE1_DIR, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from ml import predict  # noqa: E402


class _Scaler:
    def transform(self, x):
        return x


class TestPredictV3Logic(unittest.TestCase):
    @patch("ml.predict.build_behavioral_matrix", return_value=np.array([[5.0, 1.0, 1.0]]))
    @patch("ml.predict.build_feature_matrix", return_value=np.zeros((1, 3)))
    @patch("ml.predict.probability_from_model", return_value=np.array([0.8]))
    @patch("ml.predict.load_v3_artifacts")
    def test_predict_batch_v3_blends_text_and_meta_probabilities(
        self, mock_load, _mock_prob, _mock_feat, _mock_meta
    ):
        text_model = object()
        vectorizer = object()
        meta_model = Mock()
        meta_model.predict_proba.return_value = np.array([[0.9, 0.1]])
        meta_scaler = _Scaler()
        feature_meta = {
            "numeric_columns": ["rating", "helpful_vote", "verified_purchase"],
            "numeric_mean": {"rating": 0.0, "helpful_vote": 0.0, "verified_purchase": 0.0},
            "numeric_std": {"rating": 1.0, "helpful_vote": 1.0, "verified_purchase": 1.0},
        }
        model_meta = {"threshold": 0.5, "blend_weight_text": 0.75, "model_version": "phase1-v3"}
        mock_load.return_value = (text_model, vectorizer, meta_model, meta_scaler, feature_meta, model_meta)

        out = predict.predict_batch_v3(["sample"], models_dir="dummy")
        self.assertEqual(len(out), 1)
        # blend = 0.75*0.8 + 0.25*0.1 = 0.625
        self.assertAlmostEqual(out[0]["fake_probability"], 0.625, places=6)
        self.assertEqual(out[0]["label"], "fake")
        self.assertEqual(out[0]["model_version"], "phase1-v3")

    @patch("ml.predict.build_behavioral_matrix", return_value=np.array([[1.0, 0.0, 0.0]]))
    @patch("ml.predict.build_feature_matrix", return_value=np.zeros((1, 3)))
    @patch("ml.predict.probability_from_model", return_value=np.array([0.2]))
    @patch("ml.predict.load_v3_artifacts")
    def test_predict_batch_v3_respects_threshold_for_genuine_label(
        self, mock_load, _mock_prob, _mock_feat, _mock_meta
    ):
        text_model = object()
        vectorizer = object()
        meta_model = Mock()
        meta_model.predict_proba.return_value = np.array([[0.8, 0.3]])
        meta_scaler = _Scaler()
        feature_meta = {
            "numeric_columns": ["rating", "helpful_vote", "verified_purchase"],
            "numeric_mean": {"rating": 0.0, "helpful_vote": 0.0, "verified_purchase": 0.0},
            "numeric_std": {"rating": 1.0, "helpful_vote": 1.0, "verified_purchase": 1.0},
        }
        model_meta = {"threshold": 0.5, "blend_weight_text": 0.5, "model_version": "phase1-v3"}
        mock_load.return_value = (text_model, vectorizer, meta_model, meta_scaler, feature_meta, model_meta)

        out = predict.predict_batch_v3(["sample"], models_dir="dummy")
        # blend = 0.5*0.2 + 0.5*0.3 = 0.25 < 0.5
        self.assertEqual(out[0]["label"], "genuine")
        self.assertAlmostEqual(out[0]["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
