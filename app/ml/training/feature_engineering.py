import json
import os
import re
import string

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer


STEMMER = PorterStemmer()


def clean_text_for_model(text):
    # Clean text for modeling while preserving useful lexical signals.
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_stem(text):
    # Build a lightweight stemmed text variant with regex tokenization.
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    stems = [STEMMER.stem(t) for t in tokens]
    return " ".join(stems)


def build_text_columns(df):
    # Create clean and stemmed text columns for vectorization.
    out = df.copy()
    out["text_model"] = out["text"].apply(clean_text_for_model)
    out["text_stemmed"] = out["text_model"].apply(tokenize_and_stem)
    return out


def punctuation_ratio(text):
    # Measure punctuation density as a simple writing-style feature.
    if not text:
        return 0.0
    punct_count = sum(1 for c in text if c in string.punctuation)
    return punct_count / max(len(text), 1)


def uppercase_ratio(text):
    # Measure uppercase usage as a writing-style feature.
    if not text:
        return 0.0
    upper = sum(1 for c in text if c.isupper())
    alpha = sum(1 for c in text if c.isalpha())
    return upper / max(alpha, 1)


def build_numeric_features(df, include_behavioral=True):
    # Build numeric features from text statistics and optional behavioral fields.
    out = pd.DataFrame(index=df.index)
    raw_text = df["text"].fillna("").astype(str)
    clean_text = df["text_model"].fillna("").astype(str)

    if include_behavioral:
        out["rating"] = pd.to_numeric(df.get("rating"), errors="coerce").fillna(0.0)
        out["helpful_vote"] = pd.to_numeric(df.get("helpful_vote"), errors="coerce").fillna(0.0)
        out["verified_purchase"] = pd.to_numeric(df.get("verified_purchase"), errors="coerce").fillna(0.0)

    # Add compact text statistics for hybrid modeling.
    out["char_count"] = raw_text.str.len()
    out["word_count"] = clean_text.str.split().str.len().fillna(0)
    out["avg_word_len"] = out["char_count"] / np.maximum(out["word_count"], 1)
    out["exclamation_count"] = raw_text.str.count("!")
    out["question_count"] = raw_text.str.count(r"\?")
    out["punctuation_ratio"] = raw_text.apply(punctuation_ratio)
    out["uppercase_ratio"] = raw_text.apply(uppercase_ratio)
    return out.astype(float)


def fit_vectorizer(train_text, max_features=50000):
    # Fit TF-IDF vectorizer on training text only.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=max_features,
        sublinear_tf=True,
    )
    vectorizer.fit(train_text)
    return vectorizer


def transform_text(vectorizer, text_series):
    # Transform text using a pre-fitted TF-IDF vectorizer.
    return vectorizer.transform(text_series)


def scale_numeric(train_numeric, other_numeric_list):
    # Scale numeric features with train-only statistics.
    train_mean = train_numeric.mean(axis=0)
    train_std = train_numeric.std(axis=0).replace(0, 1.0)

    train_scaled = (train_numeric - train_mean) / train_std
    scaled_others = [((num - train_mean) / train_std) for num in other_numeric_list]
    return train_scaled, scaled_others, train_mean, train_std


def to_sparse_matrix(df_numeric):
    # Convert numeric DataFrame to sparse matrix for hstack.
    return csr_matrix(df_numeric.values)


def build_feature_matrices(
    train_df,
    calibration_df,
    validation_df,
    test_df,
    max_features=50000,
    include_behavioral=True,
):
    # Build train/calibration/validation/test sparse feature matrices.
    train_df = build_text_columns(train_df)
    calibration_df = build_text_columns(calibration_df)
    validation_df = build_text_columns(validation_df)
    test_df = build_text_columns(test_df)

    vectorizer = fit_vectorizer(train_df["text_stemmed"], max_features=max_features)
    x_train_text = transform_text(vectorizer, train_df["text_stemmed"])
    x_calib_text = transform_text(vectorizer, calibration_df["text_stemmed"])
    x_val_text = transform_text(vectorizer, validation_df["text_stemmed"])
    x_test_text = transform_text(vectorizer, test_df["text_stemmed"])

    train_num = build_numeric_features(train_df, include_behavioral=include_behavioral)
    calib_num = build_numeric_features(calibration_df, include_behavioral=include_behavioral)
    val_num = build_numeric_features(validation_df, include_behavioral=include_behavioral)
    test_num = build_numeric_features(test_df, include_behavioral=include_behavioral)

    train_num_scaled, scaled_others, num_mean, num_std = scale_numeric(
        train_num, [calib_num, val_num, test_num]
    )
    calib_num_scaled, val_num_scaled, test_num_scaled = scaled_others

    x_train = hstack([x_train_text, to_sparse_matrix(train_num_scaled)], format="csr")
    x_calib = hstack([x_calib_text, to_sparse_matrix(calib_num_scaled)], format="csr")
    x_val = hstack([x_val_text, to_sparse_matrix(val_num_scaled)], format="csr")
    x_test = hstack([x_test_text, to_sparse_matrix(test_num_scaled)], format="csr")

    artifacts = {
        "vectorizer": vectorizer,
        "numeric_columns": list(train_num.columns),
        "numeric_mean": num_mean.to_dict(),
        "numeric_std": num_std.to_dict(),
        "include_behavioral": bool(include_behavioral),
    }
    return x_train, x_calib, x_val, x_test, artifacts


def save_feature_metadata(feature_artifacts, output_dir):
    # Persist feature metadata needed at inference time.
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "feature_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(feature_artifacts, f, indent=2)
