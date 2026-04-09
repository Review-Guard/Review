import os
import sys
import unittest
from unittest.mock import patch


PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PHASE1_DIR, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backend import app as backend_app  # noqa: E402


class TestBackendHelpers(unittest.TestCase):
    def test_first_existing_path_returns_first_existing(self):
        with patch("backend.app.os.path.exists", side_effect=lambda p: p.endswith("b")):
            out = backend_app.first_existing_path(["a", "b", "c"])
        self.assertEqual(out, "b")

    def test_first_existing_path_returns_first_when_none_exists(self):
        with patch("backend.app.os.path.exists", return_value=False):
            out = backend_app.first_existing_path(["x", "y"])
        self.assertEqual(out, "x")

    def test_first_existing_path_returns_none_for_empty(self):
        self.assertIsNone(backend_app.first_existing_path([]))

    @patch("backend.app.first_existing_path", return_value="/tmp/models/v1")
    def test_models_dir_from_version_v1_maps_candidates(self, mock_first):
        out = backend_app.models_dir_from_version("v1")
        self.assertEqual(out, "/tmp/models/v1")
        candidates = mock_first.call_args.args[0]
        self.assertTrue(any("artifacts/models/v1" in p for p in candidates))

    @patch("backend.app.first_existing_path", return_value="/tmp/models/v2")
    def test_models_dir_from_version_v2_maps_candidates(self, mock_first):
        out = backend_app.models_dir_from_version("phase1-v2")
        self.assertEqual(out, "/tmp/models/v2")
        candidates = mock_first.call_args.args[0]
        self.assertTrue(any("artifacts/models/v2" in p for p in candidates))

    @patch("backend.app.first_existing_path", return_value="/tmp/models/v3")
    def test_models_dir_from_version_default_maps_to_v3(self, mock_first):
        out = backend_app.models_dir_from_version(None)
        self.assertEqual(out, "/tmp/models/v3")
        candidates = mock_first.call_args.args[0]
        self.assertTrue(any("artifacts/models/v3" in p for p in candidates))

    def test_models_dir_from_version_unknown_returns_none(self):
        out = backend_app.models_dir_from_version("v999")
        self.assertIsNone(out)

    def test_parse_payload_casts_and_defaults(self):
        payload = {"text": "good", "rating": "4.5", "helpful_vote": "2", "verified_purchase": "1"}
        text, rating, helpful_vote, verified_purchase, model_version = backend_app.parse_payload(payload)
        self.assertEqual(text, "good")
        self.assertEqual(rating, 4.5)
        self.assertEqual(helpful_vote, 2.0)
        self.assertEqual(verified_purchase, 1)
        self.assertEqual(model_version, "v3")

    def test_parse_payload_raises_for_none_payload(self):
        with self.assertRaises(ValueError):
            backend_app.parse_payload(None)

    def test_parse_payload_raises_for_missing_text(self):
        with self.assertRaises(ValueError):
            backend_app.parse_payload({"rating": 3})

    def test_parse_payload_raises_for_non_string_text(self):
        with self.assertRaises(TypeError):
            backend_app.parse_payload({"text": 123})

    def test_parse_payload_raises_for_empty_text(self):
        with self.assertRaises(ValueError):
            backend_app.parse_payload({"text": "   "})

    def test_parse_payload_raises_for_oversized_text(self):
        with self.assertRaises(OverflowError):
            backend_app.parse_payload({"text": "a" * 10001})

    def test_parse_payload_raises_for_non_numeric_fields(self):
        with self.assertRaises(TypeError):
            backend_app.parse_payload({"text": "ok", "rating": "abc"})

    @patch("backend.app.models_dir_from_version", return_value="/tmp/v3")
    @patch("backend.app.predict_single_v3")
    def test_run_prediction_for_version_v3_uses_v3_model_and_formats(self, mock_predict_v3, _mock_models):
        mock_predict_v3.return_value = {
            "label": "fake",
            "fake_probability": 0.81234,
            "threshold": 0.5,
            "model_version": "phase1-v3",
        }
        out = backend_app.run_prediction_for_version("v3", "text", 5, 1, 1)
        self.assertEqual(out["label"], "fake")
        self.assertEqual(out["fake_probability"], 81.23)
        self.assertEqual(out["threshold_percent"], 50.0)
        mock_predict_v3.assert_called_once()

    @patch("backend.app.models_dir_from_version", return_value="/tmp/v1")
    @patch("backend.app.predict_single")
    def test_run_prediction_for_version_v1_uses_base_predict(self, mock_predict, _mock_models):
        mock_predict.return_value = {
            "label": "genuine",
            "fake_probability": 0.1,
            "threshold": 0.5,
            "model_version": "phase1-v1",
        }
        out = backend_app.run_prediction_for_version("v1", "text", 5, 1, 1)
        self.assertEqual(out["label"], "genuine")
        self.assertEqual(out["fake_probability"], 10.0)
        mock_predict.assert_called_once()

    @patch("backend.app.models_dir_from_version", return_value=None)
    def test_run_prediction_for_version_invalid_version_raises(self, _mock_models):
        with self.assertRaises(ValueError):
            backend_app.run_prediction_for_version("vx", "text", 5, 1, 1)


if __name__ == "__main__":
    unittest.main()
