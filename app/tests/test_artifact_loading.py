import json
import os
import sys
import tempfile
import unittest

import joblib


PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PHASE1_DIR, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from ml import predict  # noqa: E402


class _DummyMetaModel:
    def predict_proba(self, _x):
        return [[0.5, 0.5]]


class _DummyScaler:
    def transform(self, x):
        return x


class TestArtifactLoading(unittest.TestCase):
    def test_load_artifacts_reads_expected_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            joblib.dump({"model": 1}, os.path.join(tmpdir, "best_model.joblib"))
            joblib.dump({"vec": 1}, os.path.join(tmpdir, "tfidf_vectorizer.joblib"))
            with open(os.path.join(tmpdir, "feature_metadata.json"), "w", encoding="utf-8") as f:
                json.dump({"numeric_columns": []}, f)
            with open(os.path.join(tmpdir, "model_metadata.json"), "w", encoding="utf-8") as f:
                json.dump({"threshold": 0.5}, f)

            model, vectorizer, feature_metadata, model_metadata = predict.load_artifacts(tmpdir)
            self.assertEqual(model["model"], 1)
            self.assertEqual(vectorizer["vec"], 1)
            self.assertIn("numeric_columns", feature_metadata)
            self.assertEqual(model_metadata["threshold"], 0.5)

    def test_load_v3_artifacts_reads_expected_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            joblib.dump({"text_model": 1}, os.path.join(tmpdir, "text_model.joblib"))
            joblib.dump({"vectorizer": 1}, os.path.join(tmpdir, "text_vectorizer.joblib"))
            joblib.dump(_DummyMetaModel(), os.path.join(tmpdir, "meta_model.joblib"))
            joblib.dump(_DummyScaler(), os.path.join(tmpdir, "meta_scaler.joblib"))
            with open(os.path.join(tmpdir, "feature_metadata.json"), "w", encoding="utf-8") as f:
                json.dump({"numeric_columns": []}, f)
            with open(os.path.join(tmpdir, "model_metadata.json"), "w", encoding="utf-8") as f:
                json.dump({"threshold": 0.5, "blend_weight_text": 0.5}, f)

            text_model, vectorizer, meta_model, meta_scaler, feature_metadata, model_metadata = predict.load_v3_artifacts(
                tmpdir
            )
            self.assertEqual(text_model["text_model"], 1)
            self.assertEqual(vectorizer["vectorizer"], 1)
            self.assertTrue(hasattr(meta_model, "predict_proba"))
            self.assertTrue(hasattr(meta_scaler, "transform"))
            self.assertIn("numeric_columns", feature_metadata)
            self.assertIn("blend_weight_text", model_metadata)

    def test_load_artifacts_missing_files_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                predict.load_artifacts(tmpdir)

    def test_load_v3_artifacts_missing_files_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                predict.load_v3_artifacts(tmpdir)


if __name__ == "__main__":
    unittest.main()
