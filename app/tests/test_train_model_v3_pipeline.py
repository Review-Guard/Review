import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PHASE1_DIR, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from ml.training import data_processing as dp_mod  # noqa: E402
from ml.training import evaluate_model as em_mod  # noqa: E402
from ml.training import feature_engineering as fe_mod  # noqa: E402


# train_model.py (imported by train_model_v3.py) uses phase1.* absolute imports.
phase1_pkg = types.ModuleType("phase1")
phase1_ml_pkg = types.ModuleType("phase1.ml")
phase1_training_pkg = types.ModuleType("phase1.ml.training")
phase1_pkg.ml = phase1_ml_pkg
phase1_ml_pkg.training = phase1_training_pkg
phase1_training_pkg.data_processing = dp_mod
phase1_training_pkg.evaluate_model = em_mod
phase1_training_pkg.feature_engineering = fe_mod

sys.modules.setdefault("phase1", phase1_pkg)
sys.modules["phase1.ml"] = phase1_ml_pkg
sys.modules["phase1.ml.training"] = phase1_training_pkg
sys.modules["phase1.ml.training.data_processing"] = dp_mod
sys.modules["phase1.ml.training.evaluate_model"] = em_mod
sys.modules["phase1.ml.training.feature_engineering"] = fe_mod

from ml.training import train_model_v3 as tmv3  # noqa: E402


class _SimpleModel:
    def __init__(self, p=0.7):
        self.p = p

    def predict_proba(self, x):
        return np.array([[1 - self.p, self.p]] * x.shape[0])


class _SimpleScaler:
    def transform(self, x):
        return x


class TestTrainModelV3Pipeline(unittest.TestCase):
    def test_behavioral_matrix_and_eval_helpers(self):
        df = pd.DataFrame({"rating": [5, None], "helpful_vote": [1, None], "verified_purchase": [1, 0]})
        mat = tmv3.behavioral_matrix(df)
        self.assertEqual(mat.shape, (2, 3))

        y_val = np.array([0, 1, 0, 1])
        text_prob = np.array([0.2, 0.8, 0.3, 0.7])
        meta_prob = np.array([0.4, 0.6, 0.45, 0.55])
        best = tmv3.grid_search_alpha_threshold(y_val, text_prob, meta_prob)
        self.assertIn("alpha", best)
        self.assertIn("threshold", best)

        metrics = tmv3.evaluate_probs(y_val, text_prob, threshold=0.5)
        self.assertIn("f1", metrics)
        self.assertIn("roc_auc", metrics)

    def test_train_models(self):
        x_train = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]]))
        x_meta = np.array([[5.0, 1.0, 1.0], [1.0, 0.0, 0.0], [4.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        y = np.array([1, 0, 1, 0])

        text_model = tmv3.train_text_model(x_train, y, random_seed=42)
        self.assertEqual(text_model.predict(x_train).shape[0], x_train.shape[0])

        meta_model, scaler = tmv3.train_metadata_model(x_meta, y, random_seed=42)
        meta_probs = meta_model.predict_proba(scaler.transform(x_meta))[:, 1]
        self.assertEqual(meta_probs.shape[0], x_meta.shape[0])

    def test_save_v3_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            tmv3.save_v3_artifacts(
                phase1_root=td,
                text_model=_SimpleModel(),
                vectorizer={"v": 1},
                feature_artifacts={"vectorizer": {"v": 1}, "numeric_columns": ["x"]},
                meta_model=_SimpleModel(p=0.4),
                meta_scaler=_SimpleScaler(),
                metadata={"model_version": "phase1-v3"},
            )
            models_dir = os.path.join(td, "artifacts", "models", "v3")
            self.assertTrue(os.path.exists(os.path.join(models_dir, "text_model.joblib")))
            self.assertTrue(os.path.exists(os.path.join(models_dir, "text_vectorizer.joblib")))
            self.assertTrue(os.path.exists(os.path.join(models_dir, "meta_model.joblib")))
            self.assertTrue(os.path.exists(os.path.join(models_dir, "meta_scaler.joblib")))
            self.assertTrue(os.path.exists(os.path.join(models_dir, "feature_metadata.json")))
            self.assertTrue(os.path.exists(os.path.join(models_dir, "model_metadata.json")))

    def test_run_v3_training_mocked_flow(self):
        split_df = pd.DataFrame(
            {
                "text": ["a", "b", "c", "d"],
                "label": [0, 1, 0, 1],
                "rating": [1, 5, 2, 4],
                "helpful_vote": [0, 1, 0, 1],
                "verified_purchase": [0, 1, 0, 1],
            }
        )
        splits = {
            "train": split_df.copy(),
            "calibration": split_df.copy(),
            "validation": split_df.copy(),
            "test": split_df.copy(),
        }
        x = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]]))

        with tempfile.TemporaryDirectory() as td:
            with (
                patch.object(tmv3, "load_processed_splits", return_value=splits),
                patch.object(tmv3, "build_feature_matrices", return_value=(x, x, x, x, {"vectorizer": {"v": 1}})),
                patch.object(tmv3, "train_text_model", return_value=_SimpleModel(0.8)),
                patch.object(tmv3, "train_metadata_model", return_value=(_SimpleModel(0.4), _SimpleScaler())),
                patch.object(tmv3, "save_v3_artifacts") as mock_save_artifacts,
            ):
                report = tmv3.run_v3_training(phase1_root=td, random_seed=42)

            self.assertIn("selected_alpha", report)
            self.assertIn("validation_metrics", report)
            self.assertTrue(os.path.exists(os.path.join(td, "artifacts", "reports", "v3", "training_report_v3.json")))
            mock_save_artifacts.assert_called_once()

    def test_parse_args_and_main(self):
        with patch.object(sys, "argv", ["prog", "--random_seed", "99"]):
            args = tmv3.parse_args()
        self.assertEqual(args.random_seed, 99)

        fake_args = types.SimpleNamespace(phase1_root="phase1", random_seed=42)
        fake_report = {
            "selected_alpha": 0.85,
            "validation_metrics": {"f1": 0.9},
            "test_metrics": {"f1": 0.88},
        }
        with patch.object(tmv3, "parse_args", return_value=fake_args), patch.object(
            tmv3, "run_v3_training", return_value=fake_report
        ), patch("builtins.print") as mock_print:
            tmv3.main()
        self.assertEqual(mock_print.call_count, 3)


if __name__ == "__main__":
    unittest.main()
