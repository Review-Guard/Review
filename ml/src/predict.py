"""Compatibility layer for legacy imports.

ML inference code has been moved to `phase1/ml/predict.py`.
This module re-exports the same symbols so existing imports keep working.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from predict import (  # noqa: F401
    build_behavioral_matrix,
    build_feature_matrix,
    build_inference_frame,
    build_numeric_features,
    clean_text_for_model,
    label_from_probability,
    load_artifacts,
    load_v3_artifacts,
    predict_batch,
    predict_batch_v3,
    predict_single,
    predict_single_v3,
    probability_from_model,
    punctuation_ratio,
    scale_numeric,
    uppercase_ratio,
)
