import os
import sys
import unittest
from unittest.mock import patch


PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PHASE1_DIR, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from app.backend.app import create_app  # noqa: E402


class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        app = create_app()
        app.testing = True
        self.client = app.test_client()

    def test_health_endpoint_returns_ok(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["status"], "ok")

    def test_predict_requires_json(self):
        res = self.client.post("/predict", data="plain-text")
        self.assertEqual(res.status_code, 400)
        self.assertIn("Request must be JSON", res.get_json()["error"])

    def test_predict_missing_text_field(self):
        res = self.client.post("/predict", json={"rating": 5})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Missing required field: text", res.get_json()["error"])

    def test_predict_oversized_text_returns_413(self):
        long_text = "a" * 10001
        res = self.client.post("/predict", json={"text": long_text})
        self.assertEqual(res.status_code, 413)

    @patch("app.backend.app.run_prediction_for_version")
    def test_predict_success_response_contract(self, mock_predict):
        mock_predict.return_value = {
            "label": "genuine",
            "fake_probability": 12.5,
            "threshold": 0.5,
            "threshold_percent": 50.0,
            "model_version": "phase1-v3",
        }
        payload = {
            "text": "item is good",
            "rating": 5,
            "helpful_vote": 2,
            "verified_purchase": 1,
            "model_version": "v3",
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["label"], "genuine")
        self.assertIn("latency_ms", data)

    @patch("app.backend.app.run_prediction_for_version")
    def test_predict_all_disagreement_sets_manual_review(self, mock_predict):
        mock_predict.side_effect = [
            {"label": "fake", "fake_probability": 90.0, "threshold": 0.5, "model_version": "phase1-v1"},
            {"label": "genuine", "fake_probability": 20.0, "threshold": 0.5, "model_version": "phase1-v2"},
            {"label": "fake", "fake_probability": 80.0, "threshold": 0.5, "model_version": "phase1-v3"},
        ]
        res = self.client.post("/predict_all", json={"text": "mixed signal review"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["disagreement"])
        self.assertEqual(data["recommendation"], "manual_review")


if __name__ == "__main__":
    unittest.main()
