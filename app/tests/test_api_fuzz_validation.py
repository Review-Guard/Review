import os
import sys
import unittest
from unittest.mock import patch


PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PHASE1_DIR, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backend.app import create_app  # noqa: E402


class TestApiFuzzValidation(unittest.TestCase):
    def setUp(self):
        app = create_app()
        app.testing = True
        self.client = app.test_client()

    def test_predict_rejects_various_bad_payload_shapes(self):
        bad_payloads = [
            {},
            {"text": None},
            {"text": []},
            {"text": {}},
            {"text": "ok", "rating": {"bad": "type"}},
            {"text": "ok", "helpful_vote": [1, 2]},
            {"text": "ok", "verified_purchase": "not-int"},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                res = self.client.post("/predict", json=payload)
                self.assertIn(res.status_code, (400, 413))

    def test_predict_all_rejects_various_bad_payload_shapes(self):
        bad_payloads = [
            {},
            {"text": None},
            {"text": []},
            {"text": {}},
            {"text": "ok", "rating": {"bad": "type"}},
            {"text": "ok", "helpful_vote": [1, 2]},
            {"text": "ok", "verified_purchase": "not-int"},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                res = self.client.post("/predict_all", json=payload)
                self.assertIn(res.status_code, (400, 413))

    @patch("backend.app.run_prediction_for_version")
    def test_predict_defaults_numeric_values_when_omitted(self, mock_predict):
        mock_predict.return_value = {
            "label": "genuine",
            "fake_probability": 10.0,
            "threshold": 0.5,
            "threshold_percent": 50.0,
            "model_version": "phase1-v3",
        }
        res = self.client.post("/predict", json={"text": "hello"})
        self.assertEqual(res.status_code, 200)
        kwargs = mock_predict.call_args.kwargs
        self.assertEqual(kwargs["rating"], 0.0)
        self.assertEqual(kwargs["helpful_vote"], 0.0)
        self.assertEqual(kwargs["verified_purchase"], 0)

    @patch("backend.app.run_prediction_for_version")
    def test_predict_accepts_unicode_text_payload(self, mock_predict):
        mock_predict.return_value = {
            "label": "genuine",
            "fake_probability": 10.0,
            "threshold": 0.5,
            "threshold_percent": 50.0,
            "model_version": "phase1-v3",
        }
        res = self.client.post("/predict", json={"text": "Très bien 👍 こんにちは"})
        self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
