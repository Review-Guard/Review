import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

from phase1.ml.training.data_processing import run_data_pipeline
from phase1.ml.training.evaluate_model import (
    compute_metrics,
    pick_best_threshold,
    run_near_duplicate_audit,
    save_json_report,
)
from phase1.ml.training.feature_engineering import build_feature_matrices, save_feature_metadata


def load_processed_splits(processed_dir):
    # Load the four processed split files.
    split_names = ["train", "calibration", "validation", "test"]
    split_frames = {}
    for split_name in split_names:
        path = os.path.join(processed_dir, f"{split_name}.csv")
        split_frames[split_name] = pd.read_csv(path)
    return split_frames


def fit_logistic(x_train, y_train, random_seed):
    # Fit Logistic Regression baseline model.
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
        random_state=random_seed,
    )
    model.fit(x_train, y_train)
    return model


def fit_sgd(x_train, y_train, random_seed):
    # Fit SGD model with probabilistic loss for sparse text.
    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=2000,
        class_weight="balanced",
        random_state=random_seed,
    )
    model.fit(x_train, y_train)
    return model


def fit_linear_svc_with_calibration(x_train, y_train, x_calib, y_calib, random_seed):
    # Fit LinearSVC then calibrate probabilities on calibration split.
    base = LinearSVC(class_weight="balanced", random_state=random_seed, max_iter=5000)
    base.fit(x_train, y_train)
    frozen_base = FrozenEstimator(base)
    calibrated = CalibratedClassifierCV(frozen_base, method="sigmoid", cv=None)
    calibrated.fit(x_calib, y_calib)
    return calibrated


def fit_xgboost_optional(x_train, y_train, random_seed):
    # Fit XGBoost only if package is available.
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None, "xgboost_not_available"

    model = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_seed,
        n_jobs=4,
    )
    model.fit(x_train, y_train)
    return model, "ok"


def calibrated_probability(model, x_matrix):
    # Return class-1 probabilities from any candidate model.
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_matrix)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_matrix)
        probs = 1.0 / (1.0 + np.exp(-scores))
        return probs
    raise ValueError("Model does not support probability-like output.")


def evaluate_candidate(name, model, y_val, x_val):
    # Evaluate a candidate model on validation data.
    val_prob = calibrated_probability(model, x_val)
    threshold, _ = pick_best_threshold(y_val, val_prob)
    metrics = compute_metrics(y_val, val_prob, threshold)
    return {
        "name": name,
        "threshold": threshold,
        "metrics": metrics,
        "val_prob": val_prob,
    }


def pick_best_model(candidates):
    # Select best model using validation F1 as primary criterion.
    sorted_items = sorted(candidates, key=lambda x: x["metrics"]["f1"], reverse=True)
    return sorted_items[0], sorted_items


def evaluate_on_test(model, x_test, y_test, threshold):
    # Run untouched test evaluation with frozen model and threshold.
    test_prob = calibrated_probability(model, x_test)
    return compute_metrics(y_test, test_prob, threshold)


def save_model_bundle(model, feature_artifacts, output_dir):
    # Persist model and feature artifacts for inference/API.
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "best_model.joblib")
    vect_path = os.path.join(output_dir, "tfidf_vectorizer.joblib")
    joblib.dump(model, model_path)
    joblib.dump(feature_artifacts["vectorizer"], vect_path)
    feature_copy = dict(feature_artifacts)
    feature_copy.pop("vectorizer", None)
    save_feature_metadata(feature_copy, output_dir)
    return model_path, vect_path


def build_model_metadata(best_name, threshold, random_seed, model_version, include_behavioral):
    # Build compact model metadata payload for runtime use.
    return {
        "model_name": best_name,
        "model_version": model_version,
        "threshold": float(threshold),
        "random_seed": int(random_seed),
        "include_behavioral": bool(include_behavioral),
    }


def run_training_pipeline(
    input_csv,
    phase1_root,
    random_seed=42,
    enable_xgboost=False,
    include_behavioral=False,
    model_version="phase1-default",
    models_subdir="artifacts/models/default",
    reports_subdir="artifacts/reports/default",
):
    # Execute full Phase 1 pipeline from processing to trained model.
    processed_dir = os.path.join(phase1_root, "data", "processed")
    models_dir = os.path.join(phase1_root, models_subdir)
    reports_dir = os.path.join(phase1_root, reports_subdir)
    os.makedirs(reports_dir, exist_ok=True)

    run_data_pipeline(input_csv=input_csv, output_dir=processed_dir, random_seed=random_seed)
    splits = load_processed_splits(processed_dir)

    x_train, x_calib, x_val, x_test, feature_artifacts = build_feature_matrices(
        splits["train"],
        splits["calibration"],
        splits["validation"],
        splits["test"],
        max_features=50000,
        include_behavioral=include_behavioral,
    )

    y_train = splits["train"]["label"].astype(int).values
    y_calib = splits["calibration"]["label"].astype(int).values
    y_val = splits["validation"]["label"].astype(int).values
    y_test = splits["test"]["label"].astype(int).values

    start = time.time()
    candidates = []

    model_log = fit_logistic(x_train, y_train, random_seed)
    candidates.append(evaluate_candidate("logistic_regression", model_log, y_val, x_val) | {"model": model_log})

    model_sgd = fit_sgd(x_train, y_train, random_seed)
    candidates.append(evaluate_candidate("sgd_log_loss", model_sgd, y_val, x_val) | {"model": model_sgd})

    model_linsvc = fit_linear_svc_with_calibration(x_train, y_train, x_calib, y_calib, random_seed)
    candidates.append(evaluate_candidate("linear_svc_calibrated", model_linsvc, y_val, x_val) | {"model": model_linsvc})

    xgb_status = "disabled"
    if enable_xgboost:
        model_xgb, xgb_status = fit_xgboost_optional(x_train, y_train, random_seed)
        if model_xgb is not None:
            candidates.append(
                evaluate_candidate("xgboost_optional", model_xgb, y_val, x_val) | {"model": model_xgb}
            )

    best, ranked = pick_best_model(candidates)
    best_model = best["model"]
    best_threshold = best["threshold"]
    test_metrics = evaluate_on_test(best_model, x_test, y_test, best_threshold)

    # Run near-duplicate heuristic audit and keep residual risk visible.
    near_dup_df, near_dup_rate = run_near_duplicate_audit(
        {
            "train": splits["train"],
            "validation": splits["validation"],
            "test": splits["test"],
        },
        sample_size=300,
        similarity_threshold=0.9,
        random_seed=random_seed,
    )
    near_dup_df.to_csv(os.path.join(reports_dir, "near_duplicate_audit.csv"), index=False)

    model_path, vect_path = save_model_bundle(best_model, feature_artifacts, models_dir)
    metadata = build_model_metadata(
        best["name"],
        best_threshold,
        random_seed,
        model_version=model_version,
        include_behavioral=include_behavioral,
    )
    metadata["xgboost_status"] = xgb_status
    metadata["training_seconds"] = float(time.time() - start)
    metadata["near_duplicate_rate_mean"] = float(near_dup_rate)
    metadata["paths"] = {"model": model_path, "vectorizer": vect_path}

    with open(os.path.join(models_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    report = {
        "validation_candidates_ranked": [
            {
                "name": item["name"],
                "threshold": item["threshold"],
                "metrics": item["metrics"],
            }
            for item in ranked
        ],
        "selected_model": best["name"],
        "selected_threshold": float(best_threshold),
        "test_metrics": test_metrics,
        "residual_risk_note": (
            "Near-duplicate audit is heuristic partial coverage; "
            "semantic paraphrase leakage may still exist."
        ),
        "calibration_contract": {
            "validation_brier_target": 0.20,
            "validation_ece_target": 0.08,
        },
    }
    save_json_report(report, os.path.join(reports_dir, "training_report.json"))
    return report, metadata


def parse_args():
    # Parse command line options for the training pipeline.
    parser = argparse.ArgumentParser(description="Train Phase 1 fake review detector")
    parser.add_argument(
        "--input_csv",
        default="dataset/amazon_labeled_fake_reviews/final_labeled_fake_reviews.csv",
        help="Input dataset path",
    )
    parser.add_argument("--phase1_root", default="phase1", help="Phase 1 root folder")
    parser.add_argument("--random_seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "--enable_xgboost",
        action="store_true",
        help="Enable optional XGBoost training",
    )
    parser.add_argument(
        "--include_behavioral",
        action="store_true",
        help="Include rating/helpful/verified behavioral features (default: off for v2)",
    )
    parser.add_argument(
        "--model_version",
        default="phase1-default",
        help="Model version written to metadata and API output",
    )
    parser.add_argument(
        "--models_subdir",
        default="artifacts/models/default",
        help="Subdirectory under phase root for model artifacts",
    )
    parser.add_argument(
        "--reports_subdir",
        default="artifacts/reports/default",
        help="Subdirectory under phase root for report artifacts",
    )
    return parser.parse_args()


def main():
    # Execute training and print selected model summary.
    args = parse_args()
    report, metadata = run_training_pipeline(
        input_csv=args.input_csv,
        phase1_root=args.phase1_root,
        random_seed=args.random_seed,
        enable_xgboost=args.enable_xgboost,
        include_behavioral=args.include_behavioral,
        model_version=args.model_version,
        models_subdir=args.models_subdir,
        reports_subdir=args.reports_subdir,
    )
    print("Selected model:", report["selected_model"])
    print("Validation F1:", report["validation_candidates_ranked"][0]["metrics"]["f1"])
    print("Test F1:", report["test_metrics"]["f1"])
    print("Model metadata:", metadata)


if __name__ == "__main__":  # pragma: no cover
    main()
