import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from app.ml.training.feature_engineering import build_feature_matrices, save_feature_metadata
from app.ml.training.train_model import load_processed_splits


def behavioral_matrix(df):
    # Build behavioral-only matrix for metadata model.
    out = pd.DataFrame(index=df.index)
    out["rating"] = pd.to_numeric(df.get("rating"), errors="coerce").fillna(0.0)
    out["helpful_vote"] = pd.to_numeric(df.get("helpful_vote"), errors="coerce").fillna(0.0)
    out["verified_purchase"] = pd.to_numeric(df.get("verified_purchase"), errors="coerce").fillna(0.0)
    return out.values


def train_text_model(x_train, y_train, random_seed):
    # Train text-focused logistic model for the v3 blend.
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_seed,
    )
    model.fit(x_train, y_train)
    return model


def train_metadata_model(x_train_meta, y_train, random_seed):
    # Train metadata-only logistic model for the v3 blend.
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train_meta)
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_seed,
    )
    model.fit(x_scaled, y_train)
    return model, scaler


def grid_search_alpha_threshold(y_val, text_prob, meta_prob):
    # Search blend weights and threshold by validation F1.
    best = {"alpha": 0.85, "threshold": 0.5, "f1": -1.0}
    # Favor text-heavy blends to reduce metadata over-influence.
    for alpha in np.linspace(0.7, 0.95, 11):
        blend_prob = alpha * text_prob + (1.0 - alpha) * meta_prob
        for threshold in np.linspace(0.2, 0.8, 25):
            pred = (blend_prob >= threshold).astype(int)
            curr_f1 = f1_score(y_val, pred, zero_division=0)
            if curr_f1 > best["f1"]:
                best = {"alpha": float(alpha), "threshold": float(threshold), "f1": float(curr_f1)}
    return best


def evaluate_probs(y_true, probs, threshold):
    # Compute compact evaluation metrics for blended probabilities.
    pred = (probs >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "threshold": float(threshold),
    }


def save_v3_artifacts(
    phase1_root,
    text_model,
    vectorizer,
    feature_artifacts,
    meta_model,
    meta_scaler,
    metadata,
):
    # Persist all v3 components under artifacts/models/v3 directory.
    models_dir = os.path.join(phase1_root, "artifacts", "models", "v3")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(text_model, os.path.join(models_dir, "text_model.joblib"))
    joblib.dump(vectorizer, os.path.join(models_dir, "text_vectorizer.joblib"))
    joblib.dump(meta_model, os.path.join(models_dir, "meta_model.joblib"))
    joblib.dump(meta_scaler, os.path.join(models_dir, "meta_scaler.joblib"))

    feature_copy = dict(feature_artifacts)
    feature_copy.pop("vectorizer", None)
    save_feature_metadata(feature_copy, models_dir)

    with open(os.path.join(models_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_v3_training(phase1_root, random_seed):
    # Train and save the v3 blended model.
    processed_dir = os.path.join(phase1_root, "data", "processed")
    splits = load_processed_splits(processed_dir)

    # Build text-focused feature matrices (behavioral off here).
    x_train, _, x_val, x_test, feature_artifacts = build_feature_matrices(
        splits["train"],
        splits["calibration"],
        splits["validation"],
        splits["test"],
        max_features=50000,
        include_behavioral=False,
    )

    y_train = splits["train"]["label"].astype(int).values
    y_val = splits["validation"]["label"].astype(int).values
    y_test = splits["test"]["label"].astype(int).values

    text_model = train_text_model(x_train, y_train, random_seed)
    text_prob_val = text_model.predict_proba(x_val)[:, 1]
    text_prob_test = text_model.predict_proba(x_test)[:, 1]

    x_meta_train = behavioral_matrix(splits["train"])
    x_meta_val = behavioral_matrix(splits["validation"])
    x_meta_test = behavioral_matrix(splits["test"])
    meta_model, meta_scaler = train_metadata_model(x_meta_train, y_train, random_seed)
    meta_prob_val = meta_model.predict_proba(meta_scaler.transform(x_meta_val))[:, 1]
    meta_prob_test = meta_model.predict_proba(meta_scaler.transform(x_meta_test))[:, 1]

    best = grid_search_alpha_threshold(y_val, text_prob_val, meta_prob_val)
    alpha = best["alpha"]
    threshold = best["threshold"]
    val_blend = alpha * text_prob_val + (1.0 - alpha) * meta_prob_val
    test_blend = alpha * text_prob_test + (1.0 - alpha) * meta_prob_test

    val_metrics = evaluate_probs(y_val, val_blend, threshold)
    test_metrics = evaluate_probs(y_test, test_blend, threshold)

    metadata = {
        "model_name": "blend_text_and_metadata",
        "model_version": "phase1-v3",
        "blend_weight_text": float(alpha),
        "blend_weight_metadata": float(1.0 - alpha),
        "threshold": float(threshold),
        "random_seed": int(random_seed),
        "include_behavioral": "blended",
    }
    save_v3_artifacts(
        phase1_root,
        text_model,
        feature_artifacts["vectorizer"],
        feature_artifacts,
        meta_model,
        meta_scaler,
        metadata,
    )

    reports_dir = os.path.join(phase1_root, "artifacts", "reports", "v3")
    os.makedirs(reports_dir, exist_ok=True)
    report = {
        "selected_alpha": float(alpha),
        "selected_threshold": float(threshold),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "text_only_note": "v3 blends text model with metadata model by learned alpha.",
    }
    with open(os.path.join(reports_dir, "training_report_v3.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def parse_args():
    # Parse command line arguments for v3 training.
    parser = argparse.ArgumentParser(description="Train phase1-v3 blended model")
    parser.add_argument("--phase1_root", default="phase1", help="Phase 1 root folder")
    parser.add_argument("--random_seed", default=42, type=int, help="Random seed")
    return parser.parse_args()


def main():
    # Run v3 training and print summary.
    args = parse_args()
    report = run_v3_training(args.phase1_root, args.random_seed)
    print("v3 alpha:", report["selected_alpha"])
    print("v3 validation F1:", report["validation_metrics"]["f1"])
    print("v3 test F1:", report["test_metrics"]["f1"])


if __name__ == "__main__":  # pragma: no cover
    main()
