import os
import sys
import unittest
from unittest.mock import patch


PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PHASE1_DIR, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backend.app import create_app  # noqa: E402


class TestApiExtended(unittest.TestCase):
    def setUp(self):
        app = create_app()
        app.testing = True
        self.client = app.test_client()

    def test_predict_all_requires_json(self):
        res = self.client.post("/predict_all", data="plain-text")
        self.assertEqual(res.status_code, 400)
        self.assertIn("Request must be JSON", res.get_json()["error"])

    def test_predict_all_missing_text_field(self):
        res = self.client.post("/predict_all", json={"rating": 5})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Missing required field: text", res.get_json()["error"])

    def test_predict_all_oversized_text_returns_413(self):
        res = self.client.post("/predict_all", json={"text": "a" * 10001})
        self.assertEqual(res.status_code, 413)

    def test_predict_text_must_be_string(self):
        res = self.client.post("/predict", json={"text": 1234})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Field text must be a string", res.get_json()["error"])

    def test_predict_invalid_json_body(self):
        res = self.client.post("/predict", data="{", content_type="application/json")
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid JSON body", res.get_json()["error"])

    @patch("backend.app.run_prediction_for_version")
    def test_predict_casts_numeric_fields_from_strings(self, mock_predict):
        mock_predict.return_value = {
            "label": "genuine",
            "fake_probability": 5.0,
            "threshold": 0.5,
            "threshold_percent": 50.0,
            "model_version": "phase1-v3",
        }
        payload = {
            "text": "solid product",
            "rating": "4.5",
            "helpful_vote": "2",
            "verified_purchase": "1",
            "model_version": "v3",
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        kwargs = mock_predict.call_args.kwargs
        self.assertEqual(kwargs["rating"], 4.5)
        self.assertEqual(kwargs["helpful_vote"], 2.0)
        self.assertEqual(kwargs["verified_purchase"], 1)

    @patch("backend.app.run_prediction_for_version")
    def test_predict_invalid_model_version_returns_400(self, mock_predict):
        mock_predict.side_effect = ValueError("model_version must be 'v1', 'v2', or 'v3'")
        payload = {"text": "unknown model", "model_version": "v9"}
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("model_version must be 'v1', 'v2', or 'v3'", res.get_json()["error"])

    @patch("backend.app.run_prediction_for_version")
    def test_predict_all_agreement_sets_majority_recommendation(self, mock_predict):
        mock_predict.side_effect = [
            {"label": "genuine", "fake_probability": 10.0, "threshold": 0.5, "model_version": "phase1-v1"},
            {"label": "genuine", "fake_probability": 15.0, "threshold": 0.5, "model_version": "phase1-v2"},
            {"label": "genuine", "fake_probability": 12.0, "threshold": 0.5, "model_version": "phase1-v3"},
        ]
        res = self.client.post("/predict_all", json={"text": "clean signal"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertFalse(data["disagreement"])
        self.assertEqual(data["majority_label"], "genuine")
        self.assertEqual(data["recommendation"], "genuine")


if __name__ == "__main__":
    unittest.main()
