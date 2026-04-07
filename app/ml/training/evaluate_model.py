import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity


def expected_calibration_error(y_true, y_prob, bins=10):
    # Compute ECE using equal-width probability bins.
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)


def classwise_reliability_gap(y_true, y_prob, bins=10):
    # Compute classwise calibration gaps for imbalance-aware checks.
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    pos_ece = expected_calibration_error(y_true, y_prob, bins=bins)
    neg_prob = 1.0 - y_prob
    neg_true = 1 - y_true
    neg_ece = expected_calibration_error(neg_true, neg_prob, bins=bins)
    return {"class_1_ece": float(pos_ece), "class_0_ece": float(neg_ece)}


def pick_best_threshold(y_true, y_prob):
    # Pick threshold by maximizing validation F1 score.
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= threshold).astype(int)
        curr_f1 = f1_score(y_true, y_pred, zero_division=0)
        if curr_f1 > best_f1:
            best_f1 = curr_f1
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def compute_metrics(y_true, y_prob, threshold):
    # Compute core classification and probability-quality metrics.
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece_10": float(expected_calibration_error(y_true, y_prob, bins=10)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    metrics.update(classwise_reliability_gap(y_true, y_prob, bins=10))
    return metrics


def run_near_duplicate_audit(split_frames, sample_size=300, similarity_threshold=0.9, random_seed=42):
    # Run a heuristic cross-split paraphrase-risk audit using char n-gram similarity.
    split_names = list(split_frames.keys())
    rows = []

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            left_name = split_names[i]
            right_name = split_names[j]
            left = split_frames[left_name]
            right = split_frames[right_name]
            left_sample = left.sample(min(sample_size, len(left)), random_state=random_seed)
            right_sample = right.sample(min(sample_size, len(right)), random_state=random_seed)

            corpus = pd.concat(
                [left_sample["text_clean_for_split"], right_sample["text_clean_for_split"]],
                axis=0,
            ).astype(str)
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
            matrix = vectorizer.fit_transform(corpus)

            left_mat = matrix[: len(left_sample)]
            right_mat = matrix[len(left_sample) :]
            sim = cosine_similarity(left_mat, right_mat)
            max_sim_per_left = sim.max(axis=1) if sim.size else np.array([])
            hit_rate = (
                float((max_sim_per_left >= similarity_threshold).mean())
                if len(max_sim_per_left) > 0
                else 0.0
            )

            rows.append(
                {
                    "left_split": left_name,
                    "right_split": right_name,
                    "sample_left": int(len(left_sample)),
                    "sample_right": int(len(right_sample)),
                    "similarity_threshold": float(similarity_threshold),
                    "near_duplicate_rate": hit_rate,
                }
            )

    audit_df = pd.DataFrame(rows)
    overall_rate = float(audit_df["near_duplicate_rate"].mean()) if len(audit_df) else 0.0
    return audit_df, overall_rate


def save_json_report(report_obj, output_path):
    # Save JSON report file with stable indentation.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, indent=2)
