import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PHASE1_DIR, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from ml.training import data_processing as dp  # noqa: E402
from ml.training import evaluate_model as em  # noqa: E402
from ml.training import feature_engineering as fe  # noqa: E402


class TestTrainingDataEvalFeature(unittest.TestCase):
    def _toy_raw_df(self):
        rows = []
        for i in range(24):
            rows.append(
                {
                    "text": f"Review text {i}",
                    "label": 1 if i % 2 == 0 else 0,
                    "rating": (i % 5) + 1,
                    "helpful_vote": i % 3,
                    "verified_purchase": "TRUE" if i % 2 == 0 else "FALSE",
                }
            )
        return pd.DataFrame(rows)

    def _toy_split_df(self, n=20):
        labels = [0, 1] * (n // 2)
        rows = []
        for i in range(n):
            rows.append(
                {
                    "text": f"row {i}",
                    "label": labels[i],
                    "text_clean_for_split": f"row {i}",
                    "rating": 5,
                    "helpful_vote": i % 2,
                    "verified_purchase": i % 2,
                }
            )
        return pd.DataFrame(rows)

    def test_normalize_label_variants(self):
        self.assertEqual(dp.normalize_label("fake"), 1)
        self.assertEqual(dp.normalize_label("genuine"), 0)
        self.assertEqual(dp.normalize_label("yes"), 1)
        self.assertEqual(dp.normalize_label("no"), 0)
        self.assertTrue(np.isnan(dp.normalize_label(np.nan)))
        self.assertTrue(np.isnan(dp.normalize_label("unknown")))

    def test_basic_text_clean_and_hash_helpers(self):
        self.assertEqual(dp.basic_text_clean(" A   B  "), "a b")
        self.assertEqual(dp.basic_text_clean(np.nan), "")
        h1 = dp.make_text_hash("abc")
        h2 = dp.make_text_hash("abc")
        self.assertEqual(h1, h2)

    def test_build_base_dataframe_missing_required_column_raises(self):
        with self.assertRaises(ValueError):
            dp.build_base_dataframe(pd.DataFrame({"text": ["x"]}))

    def test_build_base_dataframe_and_cleanup(self):
        raw = self._toy_raw_df()
        raw["label"] = raw["label"].astype(object)
        raw.loc[0, "label"] = "fake"
        raw.loc[1, "label"] = "real"
        raw.loc[2, "text"] = np.nan
        base = dp.build_base_dataframe(raw)
        self.assertIn("text_clean_for_split", base.columns)
        cleaned = dp.remove_invalid_rows(base)
        deduped = dp.dedupe_rows(cleaned)
        hashed = dp.add_hash_groups(deduped)
        self.assertIn("text_hash", hashed.columns)

    def test_split_keys_assign_and_summary(self):
        df = self._toy_split_df(20)
        hashed = dp.add_hash_groups(df)
        split_df = dp.build_split_tables(hashed, random_seed=42, test_size=0.2, calib_size=0.2, val_size=0.2)
        self.assertTrue(set(split_df["split"]).issubset({"train", "calibration", "validation", "test"}))

        summary = dp.split_ratio_summary(split_df)
        self.assertIn("fraction", summary.columns)
        self.assertAlmostEqual(float(summary["fraction"].sum()), 1.0, places=5)

    def test_assign_split_raises_when_unassigned(self):
        df = pd.DataFrame(
            {
                "text_hash": ["a", "b"],
                "label": [0, 1],
                "text_clean_for_split": ["a", "b"],
                "text": ["a", "b"],
            }
        )
        with self.assertRaises(ValueError):
            dp.assign_split(df, {"train": {"a"}})

    def test_save_outputs_and_run_pipeline(self):
        raw = self._toy_raw_df()
        with tempfile.TemporaryDirectory() as td:
            input_csv = os.path.join(td, "raw.csv")
            out_dir = os.path.join(td, "processed")
            raw.to_csv(input_csv, index=False)

            split_df = dp.run_data_pipeline(input_csv=input_csv, output_dir=out_dir, random_seed=42)
            self.assertGreater(len(split_df), 0)

            self.assertTrue(os.path.exists(os.path.join(out_dir, "phase1_clean_full.csv")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "train.csv")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "calibration.csv")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "validation.csv")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "test.csv")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "data_metadata.json")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "splits", "split_summary.csv")))

    def test_expected_calibration_error_and_classwise_gap(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
        ece = em.expected_calibration_error(y_true, y_prob, bins=5)
        gaps = em.classwise_reliability_gap(y_true, y_prob, bins=5)
        self.assertGreaterEqual(ece, 0.0)
        self.assertIn("class_1_ece", gaps)
        self.assertIn("class_0_ece", gaps)

    def test_pick_best_threshold_and_compute_metrics(self):
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.4, 0.95, 0.05])
        threshold, best_f1 = em.pick_best_threshold(y_true, y_prob)
        metrics = em.compute_metrics(y_true, y_prob, threshold)
        self.assertGreaterEqual(best_f1, 0.0)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("ece_10", metrics)

    def test_near_duplicate_audit_and_json_report(self):
        split_frames = {
            "train": pd.DataFrame({"text_clean_for_split": ["alpha one", "beta two", "gamma three"]}),
            "validation": pd.DataFrame({"text_clean_for_split": ["alpha one", "delta four", "epsilon five"]}),
            "test": pd.DataFrame({"text_clean_for_split": ["zeta six", "eta seven", "theta eight"]}),
        }
        audit_df, overall = em.run_near_duplicate_audit(split_frames, sample_size=3, similarity_threshold=0.5)
        self.assertIn("near_duplicate_rate", audit_df.columns)
        self.assertGreaterEqual(overall, 0.0)

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "reports", "report.json")
            em.save_json_report({"ok": True, "rate": overall}, out)
            self.assertTrue(os.path.exists(out))
            with open(out, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertTrue(data["ok"])

    def test_feature_scaling_sparse_and_save_metadata(self):
        train_num = pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 4.0]})
        other_num = [pd.DataFrame({"a": [1.0], "b": [3.0]})]
        train_scaled, others, mean, std = fe.scale_numeric(train_num, other_num)
        self.assertEqual(train_scaled.shape, (2, 2))
        self.assertEqual(len(others), 1)
        self.assertTrue("a" in mean.index and "a" in std.index)

        sparse = fe.to_sparse_matrix(train_scaled)
        self.assertEqual(sparse.shape, (2, 2))

        with tempfile.TemporaryDirectory() as td:
            fe.save_feature_metadata({"x": 1}, td)
            self.assertTrue(os.path.exists(os.path.join(td, "feature_metadata.json")))

    def test_feature_text_style_empty_branches(self):
        self.assertEqual(fe.punctuation_ratio(""), 0.0)
        self.assertEqual(fe.uppercase_ratio(""), 0.0)

    def test_build_feature_matrices_behavioral_toggle(self):
        def make_df(n):
            return pd.DataFrame(
                {
                    "text": [f"good product {i}" if i % 2 == 0 else f"bad product {i}" for i in range(n)],
                    "label": [1 if i % 2 == 0 else 0 for i in range(n)],
                    "rating": [5 if i % 2 == 0 else 1 for i in range(n)],
                    "helpful_vote": [i % 3 for i in range(n)],
                    "verified_purchase": [i % 2 for i in range(n)],
                }
            )

        train_df = make_df(12)
        calib_df = make_df(8)
        val_df = make_df(8)
        test_df = make_df(8)

        x_train, x_cal, x_val, x_test, artifacts = fe.build_feature_matrices(
            train_df,
            calib_df,
            val_df,
            test_df,
            max_features=200,
            include_behavioral=False,
        )
        self.assertEqual(x_train.shape[0], len(train_df))
        self.assertEqual(x_cal.shape[0], len(calib_df))
        self.assertEqual(x_val.shape[0], len(val_df))
        self.assertEqual(x_test.shape[0], len(test_df))
        self.assertFalse(artifacts["include_behavioral"])

    def test_data_processing_parse_args_and_main(self):
        with patch.object(
            sys,
            "argv",
            ["prog", "--input_csv", "input.csv", "--output_dir", "out", "--random_seed", "99"],
        ):
            args = dp.parse_args()
        self.assertEqual(args.input_csv, "input.csv")
        self.assertEqual(args.output_dir, "out")
        self.assertEqual(args.random_seed, 99)

        fake_args = type("Args", (), {"input_csv": "in.csv", "output_dir": "out", "random_seed": 1})()
        fake_split = pd.DataFrame(
            {
                "split": ["train", "test"],
                "label": [0, 1],
                "text_clean_for_split": ["a", "b"],
            }
        )
        with patch.object(dp, "parse_args", return_value=fake_args), patch.object(
            dp, "run_data_pipeline", return_value=fake_split
        ), patch.object(dp, "split_ratio_summary", return_value=pd.DataFrame({"split": ["train"], "rows": [1]})), patch(
            "builtins.print"
        ) as mock_print:
            dp.main()
        self.assertGreaterEqual(mock_print.call_count, 1)


if __name__ == "__main__":
    unittest.main()
