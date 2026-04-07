import argparse
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_dataset(csv_path):
    # Load the raw Amazon labeled fake review dataset.
    return pd.read_csv(csv_path)


def normalize_label(value):
    # Normalize labels to integer 0/1 format.
    if pd.isna(value):
        return np.nan
    try:
        return int(value)
    except Exception:
        value_str = str(value).strip().lower()
        if value_str in {"fake", "1", "true", "yes", "y"}:
            return 1
        if value_str in {"genuine", "real", "0", "false", "no", "n"}:
            return 0
        return np.nan


def basic_text_clean(text):
    # Clean text for stable dedup/split hashing.
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_base_dataframe(df):
    # Select and normalize the canonical Phase 1 columns.
    required_cols = ["text", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    base = pd.DataFrame()
    base["text"] = df["text"].astype(str)
    base["label"] = df["label"].apply(normalize_label)

    # Keep useful behavioral metadata from the same dataset.
    base["rating"] = pd.to_numeric(df.get("rating"), errors="coerce")
    base["helpful_vote"] = pd.to_numeric(df.get("helpful_vote"), errors="coerce").fillna(0)
    base["verified_purchase"] = (
        df.get("verified_purchase", pd.Series(["FALSE"] * len(df)))
        .astype(str)
        .str.upper()
        .map({"TRUE": 1, "FALSE": 0})
        .fillna(0)
        .astype(int)
    )

    # Create normalized text used for dedupe, hashing, and audits.
    base["text_clean_for_split"] = base["text"].apply(basic_text_clean)
    return base


def remove_invalid_rows(df):
    # Remove rows with missing labels or empty text after cleaning.
    out = df.copy()
    out = out[out["label"].isin([0, 1])]
    out = out[out["text_clean_for_split"].str.len() > 0]
    return out.reset_index(drop=True)


def dedupe_rows(df):
    # Remove exact duplicates on normalized text before splitting.
    out = df.drop_duplicates(subset=["text_clean_for_split"]).copy()
    return out.reset_index(drop=True)


def make_text_hash(text_value):
    # Build deterministic hash key for leakage-safe grouping.
    return hashlib.md5(text_value.encode("utf-8")).hexdigest()


def add_hash_groups(df):
    # Add text hash groups for group-aware data splitting.
    out = df.copy()
    out["text_hash"] = out["text_clean_for_split"].apply(make_text_hash)
    return out


def split_group_keys(group_df, random_seed, test_size, calib_size, val_size):
    # Split unique hash groups into train/calib/val/test with stratification.
    keys = group_df["text_hash"].values
    y = group_df["label"].values

    train_calib_val_keys, test_keys, train_calib_val_y, _ = train_test_split(
        keys,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_seed,
    )

    remaining_fraction = 1.0 - test_size
    calib_rel = calib_size / remaining_fraction
    train_val_keys, calib_keys, train_val_y, _ = train_test_split(
        train_calib_val_keys,
        train_calib_val_y,
        test_size=calib_rel,
        stratify=train_calib_val_y,
        random_state=random_seed,
    )

    train_rel = (1.0 - test_size - calib_size)
    val_rel = val_size / train_rel
    train_keys, val_keys, _, _ = train_test_split(
        train_val_keys,
        train_val_y,
        test_size=val_rel,
        stratify=train_val_y,
        random_state=random_seed,
    )

    return {
        "train": set(train_keys.tolist()),
        "calibration": set(calib_keys.tolist()),
        "validation": set(val_keys.tolist()),
        "test": set(test_keys.tolist()),
    }


def assign_split(df, split_keys):
    # Assign each record to a split using its hash group key.
    out = df.copy()
    out["split"] = "unassigned"
    for split_name, key_set in split_keys.items():
        mask = out["text_hash"].isin(key_set)
        out.loc[mask, "split"] = split_name
    if (out["split"] == "unassigned").any():
        raise ValueError("Some rows are unassigned after group split.")
    return out


def build_split_tables(df, random_seed=42, test_size=0.1, calib_size=0.1, val_size=0.1):
    # Create leakage-safe split assignment table from hash groups.
    group_df = df.groupby("text_hash", as_index=False)["label"].first()
    split_keys = split_group_keys(group_df, random_seed, test_size, calib_size, val_size)
    split_df = assign_split(df, split_keys)
    return split_df


def split_ratio_summary(df):
    # Summarize split size and label balance for quick validation.
    stats = []
    total = len(df)
    for split_name, split_frame in df.groupby("split"):
        row = {
            "split": split_name,
            "rows": int(len(split_frame)),
            "fraction": float(len(split_frame) / total),
            "label_1_rate": float(split_frame["label"].mean()),
        }
        stats.append(row)
    return pd.DataFrame(stats).sort_values("split").reset_index(drop=True)


def save_outputs(df, output_dir):
    # Save cleaned full data, per-split files, and split metadata.
    os.makedirs(output_dir, exist_ok=True)
    splits_dir = os.path.join(output_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    full_path = os.path.join(output_dir, "phase1_clean_full.csv")
    df.to_csv(full_path, index=False)

    for split_name, split_frame in df.groupby("split"):
        split_path = os.path.join(output_dir, f"{split_name}.csv")
        split_frame.to_csv(split_path, index=False)
        keys_path = os.path.join(splits_dir, f"{split_name}_hashes.csv")
        split_frame[["text_hash"]].drop_duplicates().to_csv(keys_path, index=False)

    summary = split_ratio_summary(df)
    summary_path = os.path.join(splits_dir, "split_summary.csv")
    summary.to_csv(summary_path, index=False)

    metadata = {
        "source_dataset": "dataset/amazon_labeled_fake_reviews/final_labeled_fake_reviews.csv",
        "label_direction": "Phase1 assumes label=1 is fake and label=0 is genuine; verify against dataset source notes.",
        "rows_after_cleaning": int(len(df)),
        "split_files": ["train.csv", "calibration.csv", "validation.csv", "test.csv"],
    }
    with open(os.path.join(output_dir, "data_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_data_pipeline(input_csv, output_dir, random_seed):
    # Run the end-to-end Phase 1 cleaning and split pipeline.
    raw = load_raw_dataset(input_csv)
    base = build_base_dataframe(raw)
    valid = remove_invalid_rows(base)
    deduped = dedupe_rows(valid)
    hashed = add_hash_groups(deduped)
    split_df = build_split_tables(hashed, random_seed=random_seed)
    save_outputs(split_df, output_dir)
    return split_df


def parse_args():
    # Parse command line options for data processing.
    parser = argparse.ArgumentParser(description="Phase 1 data processing pipeline")
    parser.add_argument(
        "--input_csv",
        default="dataset/amazon_labeled_fake_reviews/final_labeled_fake_reviews.csv",
        help="Input CSV dataset path",
    )
    parser.add_argument(
        "--output_dir",
        default="phase1/data/processed",
        help="Output directory for processed files",
    )
    parser.add_argument("--random_seed", default=42, type=int, help="Random seed")
    return parser.parse_args()


def main():
    # Execute processing and print split summary.
    args = parse_args()
    split_df = run_data_pipeline(args.input_csv, args.output_dir, args.random_seed)
    print(split_ratio_summary(split_df))


if __name__ == "__main__":
    main()
