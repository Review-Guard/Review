import importlib
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


# train_model.py imports from phase1.*; provide aliases so import works in this repo layout.
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

train_model = importlib.import_module("ml.training.train_model")


class _ProbaModel:
    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        return np.array([[0.2, 0.8]] * x.shape[0])


class _DecisionModel:
    def decision_function(self, x):
        return np.array([0.0] * x.shape[0])


class TestTrainModelPipeline(unittest.TestCase):
    def _tiny_xy(self):
        x = csr_matrix(
            np.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.5, 0.5],
                    [0.1, 0.9],
                    [0.9, 0.1],
                    [0.6, 0.4],
                    [0.4, 0.6],
                    [0.2, 0.8],
                    [0.8, 0.2],
                    [0.3, 0.7],
                ]
            )
        )
        y = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
        return x, y

    def test_load_processed_splits_reads_all_files(self):
        with tempfile.TemporaryDirectory() as td:
            for split in ["train", "calibration", "validation", "test"]:
                pd.DataFrame({"text": ["a", "b"], "label": [0, 1]}).to_csv(
                    os.path.join(td, f"{split}.csv"), index=False
                )
            splits = train_model.load_processed_splits(td)
            self.assertEqual(set(splits.keys()), {"train", "calibration", "validation", "test"})

    def test_fit_models_and_probability_helpers(self):
        x, y = self._tiny_xy()
        log_model = train_model.fit_logistic(x, y, random_seed=42)
        sgd_model = train_model.fit_sgd(x, y, random_seed=42)
        self.assertEqual(log_model.predict(x).shape[0], x.shape[0])
        self.assertEqual(sgd_model.predict(x).shape[0], x.shape[0])

        probs1 = train_model.calibrated_probability(_ProbaModel(), x)
        probs2 = train_model.calibrated_probability(_DecisionModel(), x)
        self.assertEqual(probs1.shape[0], x.shape[0])
        self.assertTrue(np.allclose(probs2, np.array([0.5] * x.shape[0])))

        with self.assertRaises(ValueError):
            train_model.calibrated_probability(object(), x)

    def test_fit_linear_svc_with_calibration_and_eval_helpers(self):
        x, y = self._tiny_xy()
        model = train_model.fit_linear_svc_with_calibration(x, y, x, y, random_seed=42)
        self.assertTrue(hasattr(model, "predict_proba"))

        out = train_model.evaluate_candidate("m", model, y, x)
        self.assertIn("metrics", out)
        self.assertIn("threshold", out)

        best, ranked = train_model.pick_best_model(
            [
                {"name": "a", "metrics": {"f1": 0.5}},
                {"name": "b", "metrics": {"f1": 0.8}},
            ]
        )
        self.assertEqual(best["name"], "b")
        self.assertEqual(ranked[0]["name"], "b")

        test_metrics = train_model.evaluate_on_test(model, x, y, threshold=0.5)
        self.assertIn("f1", test_metrics)

    def test_fit_xgboost_optional_not_available_path(self):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "xgboost":
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            model, status = train_model.fit_xgboost_optional(*self._tiny_xy(), random_seed=42)
        self.assertIsNone(model)
        self.assertEqual(status, "xgboost_not_available")

    def test_fit_xgboost_optional_available_path(self):
        fake_module = types.SimpleNamespace()

        class FakeXGBClassifier:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit(self, x, y):
                self.fitted_shape = x.shape
                return self

        fake_module.XGBClassifier = FakeXGBClassifier
        with patch.dict("sys.modules", {"xgboost": fake_module}):
            model, status = train_model.fit_xgboost_optional(*self._tiny_xy(), random_seed=42)
        self.assertIsNotNone(model)
        self.assertEqual(status, "ok")

    def test_save_bundle_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            model = _ProbaModel()
            feature_artifacts = {
                "vectorizer": {"dummy": True},
                "numeric_columns": ["a"],
                "numeric_mean": {"a": 0.0},
                "numeric_std": {"a": 1.0},
                "include_behavioral": False,
            }
            model_path, vec_path = train_model.save_model_bundle(model, feature_artifacts, td)
            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(vec_path))
            self.assertTrue(os.path.exists(os.path.join(td, "feature_metadata.json")))

            meta = train_model.build_model_metadata("best", 0.55, 7, "vX", True)
            self.assertEqual(meta["model_name"], "best")
            self.assertTrue(meta["include_behavioral"])

    def test_run_training_pipeline_mocked_flow(self):
        train_df = pd.DataFrame(
            {
                "text": ["a", "b", "c", "d"],
                "label": [0, 1, 0, 1],
                "text_clean_for_split": ["a", "b", "c", "d"],
            }
        )
        splits = {
            "train": train_df.copy(),
            "calibration": train_df.copy(),
            "validation": train_df.copy(),
            "test": train_df.copy(),
        }

        x = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]]))
        feature_artifacts = {
            "vectorizer": {"dummy": True},
            "numeric_columns": ["a"],
            "numeric_mean": {"a": 0.0},
            "numeric_std": {"a": 1.0},
            "include_behavioral": False,
        }

        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "artifacts", "models", "default"), exist_ok=True)
            with (
                patch.object(train_model, "run_data_pipeline", return_value=train_df),
                patch.object(train_model, "load_processed_splits", return_value=splits),
                patch.object(train_model, "build_feature_matrices", return_value=(x, x, x, x, feature_artifacts)),
                patch.object(train_model, "fit_logistic", return_value="log_model"),
                patch.object(train_model, "fit_sgd", return_value="sgd_model"),
                patch.object(train_model, "fit_linear_svc_with_calibration", return_value="svc_model"),
                patch.object(train_model, "evaluate_candidate", side_effect=[
                    {"name": "logistic_regression", "threshold": 0.5, "metrics": {"f1": 0.71}, "val_prob": [0.5]},
                    {"name": "sgd_log_loss", "threshold": 0.5, "metrics": {"f1": 0.72}, "val_prob": [0.5]},
                    {"name": "linear_svc_calibrated", "threshold": 0.5, "metrics": {"f1": 0.73}, "val_prob": [0.5]},
                    {"name": "xgboost_optional", "threshold": 0.5, "metrics": {"f1": 0.70}, "val_prob": [0.5]},
                ]),
                patch.object(train_model, "fit_xgboost_optional", return_value=("xgb_model", "ok")),
                patch.object(train_model, "pick_best_model", return_value=(
                    {
                        "name": "linear_svc_calibrated",
                        "threshold": 0.55,
                        "metrics": {"f1": 0.73},
                        "model": "svc_model",
                    },
                    [
                        {"name": "linear_svc_calibrated", "threshold": 0.55, "metrics": {"f1": 0.73}},
                        {"name": "sgd_log_loss", "threshold": 0.5, "metrics": {"f1": 0.72}},
                    ],
                )),
                patch.object(train_model, "evaluate_on_test", return_value={"f1": 0.7}),
                patch.object(train_model, "run_near_duplicate_audit", return_value=(pd.DataFrame({"a": [1]}), 0.02)),
                patch.object(train_model, "save_model_bundle", return_value=(
                    os.path.join(td, "model.joblib"),
                    os.path.join(td, "vec.joblib"),
                )),
                patch.object(train_model, "save_json_report") as mock_save_report,
            ):
                report, metadata = train_model.run_training_pipeline(
                    input_csv="dummy.csv",
                    phase1_root=td,
                    random_seed=11,
                    enable_xgboost=True,
                    include_behavioral=False,
                    model_version="phase1-default",
                )

            self.assertEqual(report["selected_model"], "linear_svc_calibrated")
            self.assertEqual(metadata["xgboost_status"], "ok")
            self.assertTrue(os.path.exists(os.path.join(td, "artifacts", "reports", "default", "near_duplicate_audit.csv")))
            self.assertTrue(os.path.exists(os.path.join(td, "artifacts", "models", "default", "model_metadata.json")))
            mock_save_report.assert_called_once()

    def test_parse_args_and_main(self):
        with patch.object(sys, "argv", ["prog", "--random_seed", "123", "--enable_xgboost"]):
            args = train_model.parse_args()
        self.assertEqual(args.random_seed, 123)
        self.assertTrue(args.enable_xgboost)

        fake_args = types.SimpleNamespace(
            input_csv="in.csv",
            phase1_root="phase1",
            random_seed=1,
            enable_xgboost=False,
            include_behavioral=False,
            model_version="m",
            models_subdir="a/b",
            reports_subdir="c/d",
        )
        with patch.object(train_model, "parse_args", return_value=fake_args), patch.object(
            train_model,
            "run_training_pipeline",
            return_value=(
                {
                    "selected_model": "x",
                    "validation_candidates_ranked": [{"metrics": {"f1": 0.9}}],
                    "test_metrics": {"f1": 0.8},
                },
                {"k": "v"},
            ),
        ), patch("builtins.print") as mock_print:
            train_model.main()
        self.assertGreaterEqual(mock_print.call_count, 4)


if __name__ == "__main__":
    unittest.main()
