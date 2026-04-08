import os
import sys
import unittest

import numpy as np
import pandas as pd


PHASE1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PHASE1_DIR, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from ml.training.feature_engineering import (  # noqa: E402
    build_numeric_features,
    build_text_columns,
    clean_text_for_model,
    fit_vectorizer,
    tokenize_and_stem,
    transform_text,
)


class TestPreprocessing(unittest.TestCase):
    def test_clean_text_for_model_normalizes_text(self):
        raw = "Amazing PRODUCT!!! Visit https://example.com NOW"
        cleaned = clean_text_for_model(raw)
        self.assertEqual(cleaned, "amazing product!!! visit now")

    def test_tokenize_and_stem_returns_stemmed_tokens(self):
        text = "running runs runner"
        stemmed = tokenize_and_stem(text)
        self.assertEqual(stemmed, "run run runner")

    def test_build_text_columns_adds_required_columns(self):
        df = pd.DataFrame({"text": ["Great item", "Bad item"]})
        out = build_text_columns(df)
        self.assertIn("text_model", out.columns)
        self.assertIn("text_stemmed", out.columns)
        self.assertEqual(len(out), 2)

    def test_build_numeric_features_contains_expected_columns(self):
        df = pd.DataFrame(
            {
                "text": ["Awesome!!!", "Not good?"],
                "text_model": ["awesome", "not good"],
                "rating": [5, 1],
                "helpful_vote": [3, 0],
                "verified_purchase": [1, 0],
            }
        )

        num = build_numeric_features(df, include_behavioral=True)
        expected = {
            "rating",
            "helpful_vote",
            "verified_purchase",
            "char_count",
            "word_count",
            "avg_word_len",
            "exclamation_count",
            "question_count",
            "punctuation_ratio",
            "uppercase_ratio",
        }
        self.assertTrue(expected.issubset(set(num.columns)))
        self.assertFalse(num.isna().any().any())

    def test_tfidf_vectorizer_fit_and_transform(self):
        train_text = pd.Series(["good product", "bad product", "good quality"])  # min_df=2 safe
        vectorizer = fit_vectorizer(train_text, max_features=100)
        x = transform_text(vectorizer, pd.Series(["good product"]))
        self.assertEqual(x.shape[0], 1)
        self.assertGreaterEqual(x.shape[1], 1)
        self.assertTrue(np.isfinite(x.data).all())


if __name__ == "__main__":
    unittest.main()
