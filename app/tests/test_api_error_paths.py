import os
import sys
import unittest
from unittest.mock import patch


PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PHASE1_DIR, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backend.app import create_app  # noqa: E402


class TestApiErrorPaths(unittest.TestCase):
    def setUp(self):
        app = create_app()
        app.testing = True
        self.client = app.test_client()

    def test_home_route_renders(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Fake Review", res.get_data(as_text=True))

    @patch("backend.app.run_prediction_for_version", side_effect=FileNotFoundError())
    def test_predict_returns_500_when_model_artifacts_missing(self, _mock_predict):
        res = self.client.post("/predict", json={"text": "sample"})
        self.assertEqual(res.status_code, 500)
        self.assertIn("Model artifacts not found", res.get_json()["error"])

    @patch("backend.app.run_prediction_for_version", side_effect=RuntimeError("boom"))
    def test_predict_returns_500_on_unexpected_exception(self, _mock_predict):
        res = self.client.post("/predict", json={"text": "sample"})
        self.assertEqual(res.status_code, 500)
        self.assertIn("Prediction failed", res.get_json()["error"])

    @patch("backend.app.run_prediction_for_version", side_effect=FileNotFoundError())
    def test_predict_all_returns_500_when_model_artifacts_missing(self, _mock_predict):
        res = self.client.post("/predict_all", json={"text": "sample"})
        self.assertEqual(res.status_code, 500)
        self.assertIn("Model artifacts not found", res.get_json()["error"])

    @patch("backend.app.run_prediction_for_version", side_effect=RuntimeError("boom"))
    def test_predict_all_returns_500_on_unexpected_exception(self, _mock_predict):
        res = self.client.post("/predict_all", json={"text": "sample"})
        self.assertEqual(res.status_code, 500)
        self.assertIn("Prediction failed", res.get_json()["error"])

    def test_predict_all_invalid_json_body_returns_400(self):
        res = self.client.post("/predict_all", data="{", content_type="application/json")
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid JSON body", res.get_json()["error"])

    def test_predict_all_invalid_numeric_types_return_400(self):
        payload = {"text": "hello", "rating": "not-a-number"}
        res = self.client.post("/predict_all", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("must be numeric", res.get_json()["error"])

    @patch("backend.app.run_prediction_for_version")
    def test_predict_success_includes_latency_field(self, mock_predict):
        mock_predict.return_value = {
            "label": "genuine",
            "fake_probability": 12.0,
            "threshold": 0.5,
            "threshold_percent": 50.0,
            "model_version": "phase1-v3",
        }
        res = self.client.post("/predict", json={"text": "sample"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("latency_ms", data)
        self.assertGreaterEqual(float(data["latency_ms"]), 0.0)


if __name__ == "__main__":
    unittest.main()
