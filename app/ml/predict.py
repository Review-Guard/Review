import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack


def clean_text_for_model(text):
    # Normalize inference text similarly to training flow.
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_inference_frame(texts, ratings=None, helpful_votes=None, verified_purchase=None):
    # Build a DataFrame for single or batch inference.
    n = len(texts)
    ratings = ratings if ratings is not None else [0.0] * n
    helpful_votes = helpful_votes if helpful_votes is not None else [0.0] * n
    verified_purchase = verified_purchase if verified_purchase is not None else [0] * n

    df = pd.DataFrame(
        {
            "text": texts,
            "rating": ratings,
            "helpful_vote": helpful_votes,
            "verified_purchase": verified_purchase,
        }
    )
    return df


def punctuation_ratio(text):
    # Compute punctuation ratio used by numeric features.
    if not text:
        return 0.0
    punct = sum(1 for c in text if c in "!?,.:;-'\"()[]{}")
    return punct / max(len(text), 1)


def uppercase_ratio(text):
    # Compute uppercase ratio used by numeric features.
    if not text:
        return 0.0
    upper = sum(1 for c in text if c.isupper())
    alpha = sum(1 for c in text if c.isalpha())
    return upper / max(alpha, 1)


def build_numeric_features(df):
    # Recreate numeric feature columns used during training.
    raw_text = df["text"].fillna("").astype(str)
    clean_text = df["text"].apply(clean_text_for_model)

    out = pd.DataFrame(index=df.index)
    out["rating"] = pd.to_numeric(df.get("rating"), errors="coerce").fillna(0.0)
    out["helpful_vote"] = pd.to_numeric(df.get("helpful_vote"), errors="coerce").fillna(0.0)
    out["verified_purchase"] = pd.to_numeric(df.get("verified_purchase"), errors="coerce").fillna(0.0)
    out["char_count"] = raw_text.str.len()
    out["word_count"] = clean_text.str.split().str.len().fillna(0)
    out["avg_word_len"] = out["char_count"] / np.maximum(out["word_count"], 1)
    out["exclamation_count"] = raw_text.str.count("!")
    out["question_count"] = raw_text.str.count(r"\?")
    out["punctuation_ratio"] = raw_text.apply(punctuation_ratio)
    out["uppercase_ratio"] = raw_text.apply(uppercase_ratio)
    return out.astype(float)


def load_artifacts(models_dir="phase1/artifacts/models/default"):
    # Load trained model, vectorizer, and metadata from disk.
    model = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    with open(os.path.join(models_dir, "feature_metadata.json"), "r", encoding="utf-8") as f:
        feature_metadata = json.load(f)
    with open(os.path.join(models_dir, "model_metadata.json"), "r", encoding="utf-8") as f:
        model_metadata = json.load(f)
    return model, vectorizer, feature_metadata, model_metadata


def load_v3_artifacts(models_dir="phase1/artifacts/models/v3"):
    # Load v3 blended-model artifacts.
    text_model = joblib.load(os.path.join(models_dir, "text_model.joblib"))
    vectorizer = joblib.load(os.path.join(models_dir, "text_vectorizer.joblib"))
    meta_model = joblib.load(os.path.join(models_dir, "meta_model.joblib"))
    meta_scaler = joblib.load(os.path.join(models_dir, "meta_scaler.joblib"))
    with open(os.path.join(models_dir, "feature_metadata.json"), "r", encoding="utf-8") as f:
        feature_metadata = json.load(f)
    with open(os.path.join(models_dir, "model_metadata.json"), "r", encoding="utf-8") as f:
        model_metadata = json.load(f)
    return text_model, vectorizer, meta_model, meta_scaler, feature_metadata, model_metadata


def scale_numeric(numeric_df, feature_metadata):
    # Scale numeric features using train-time mean/std statistics.
    mean_map = feature_metadata["numeric_mean"]
    std_map = feature_metadata["numeric_std"]
    ordered_cols = feature_metadata["numeric_columns"]

    aligned = numeric_df.reindex(columns=ordered_cols, fill_value=0.0).copy()
    for col in ordered_cols:
        col_mean = float(mean_map[col])
        col_std = float(std_map[col]) if float(std_map[col]) != 0 else 1.0
        aligned[col] = (aligned[col] - col_mean) / col_std
    return aligned


def build_feature_matrix(df, vectorizer, feature_metadata):
    # Build combined sparse feature matrix for inference.
    text_stemmed = df["text"].fillna("").astype(str).apply(clean_text_for_model)
    x_text = vectorizer.transform(text_stemmed)
    numeric = build_numeric_features(df)
    numeric_scaled = scale_numeric(numeric, feature_metadata)
    x_num = csr_matrix(numeric_scaled.values)
    return hstack([x_text, x_num], format="csr")


def build_behavioral_matrix(df):
    # Build behavioral-only matrix for v3 metadata model.
    out = pd.DataFrame(index=df.index)
    rating = df["rating"] if "rating" in df.columns else pd.Series(0.0, index=df.index)
    helpful_vote = df["helpful_vote"] if "helpful_vote" in df.columns else pd.Series(0.0, index=df.index)
    verified_purchase = (
        df["verified_purchase"] if "verified_purchase" in df.columns else pd.Series(0.0, index=df.index)
    )

    out["rating"] = pd.to_numeric(rating, errors="coerce").fillna(0.0)
    out["helpful_vote"] = pd.to_numeric(helpful_vote, errors="coerce").fillna(0.0)
    out["verified_purchase"] = pd.to_numeric(verified_purchase, errors="coerce").fillna(0.0)
    return out.values


def probability_from_model(model, x_matrix):
    # Get class-1 probabilities in a model-agnostic way.
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_matrix)[:, 1]
    scores = model.decision_function(x_matrix)
    return 1.0 / (1.0 + np.exp(-scores))


def label_from_probability(prob, threshold):
    # Map probability to fake/genuine label.
    return "fake" if prob >= threshold else "genuine"


def predict_batch(
    texts,
    ratings=None,
    helpful_votes=None,
    verified_purchase=None,
    models_dir="phase1/artifacts/models/default",
):
    # Run batch predictions and return probability-rich responses.
    model, vectorizer, feature_metadata, model_metadata = load_artifacts(models_dir)
    threshold = float(model_metadata["threshold"])

    infer_df = build_inference_frame(
        texts=texts,
        ratings=ratings,
        helpful_votes=helpful_votes,
        verified_purchase=verified_purchase,
    )
    x_matrix = build_feature_matrix(infer_df, vectorizer, feature_metadata)
    probs = probability_from_model(model, x_matrix)

    results = []
    for text_value, prob in zip(texts, probs):
        prob = float(prob)
        results.append(
            {
                "text": text_value,
                "label": label_from_probability(prob, threshold),
                "fake_probability": prob,
                "threshold": threshold,
                "model_version": model_metadata.get("model_version", "phase1-v1"),
            }
        )
    return results


def predict_single(
    text,
    rating=0.0,
    helpful_vote=0.0,
    verified_purchase=0,
    models_dir="phase1/artifacts/models/default",
):
    # Run a single-text prediction with fake probability output.
    result = predict_batch(
        texts=[text],
        ratings=[rating],
        helpful_votes=[helpful_vote],
        verified_purchase=[verified_purchase],
        models_dir=models_dir,
    )[0]
    return result


def predict_batch_v3(
    texts,
    ratings=None,
    helpful_votes=None,
    verified_purchase=None,
    models_dir="phase1/artifacts/models/v3",
):
    # Run v3 blended predictions combining text and metadata probabilities.
    text_model, vectorizer, meta_model, meta_scaler, feature_metadata, model_metadata = load_v3_artifacts(models_dir)
    threshold = float(model_metadata["threshold"])
    alpha = float(model_metadata["blend_weight_text"])

    infer_df = build_inference_frame(
        texts=texts,
        ratings=ratings,
        helpful_votes=helpful_votes,
        verified_purchase=verified_purchase,
    )
    x_text = build_feature_matrix(infer_df, vectorizer, feature_metadata)
    x_meta = build_behavioral_matrix(infer_df)
    text_probs = probability_from_model(text_model, x_text)
    meta_probs = meta_model.predict_proba(meta_scaler.transform(x_meta))[:, 1]
    blend_probs = alpha * text_probs + (1.0 - alpha) * meta_probs

    results = []
    for text_value, prob in zip(texts, blend_probs):
        prob = float(prob)
        results.append(
            {
                "text": text_value,
                "label": label_from_probability(prob, threshold),
                "fake_probability": prob,
                "threshold": threshold,
                "model_version": model_metadata.get("model_version", "phase1-v3"),
            }
        )
    return results


def predict_single_v3(
    text,
    rating=0.0,
    helpful_vote=0.0,
    verified_purchase=0,
    models_dir="phase1/artifacts/models/v3",
):
    # Run a single-text v3 blended prediction.
    result = predict_batch_v3(
        texts=[text],
        ratings=[rating],
        helpful_votes=[helpful_vote],
        verified_purchase=[verified_purchase],
        models_dir=models_dir,
    )[0]
    return result
