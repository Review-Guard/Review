# Phase 1 Engineering Notes

This document follows the required structure for each Phase 1 step.

## 1) Problem Definition

### Goal of the Step
Define a binary classifier that predicts whether a review is fake or genuine and outputs fake probability.

### Possible Approaches
- Rule-based heuristics
- Traditional ML text classification
- Deep learning sequence models

### Comparison
- Rule-based: fast but brittle and hard to generalize.
- Traditional ML: strong baseline, explainable, efficient on medium-sized tabular+text data.
- Deep learning: potentially stronger, but higher complexity and compute cost.

### Decision
Use traditional ML in Phase 1 for speed, reproducibility, and reliable baseline performance.

### Implementation
Implemented in:
- `phase1/src/train_model.py`
- `phase1/src/predict.py`

### Validation
Check model can output:
- class label (`fake`/`genuine`)
- `fake_probability` in [0,1]

### Risks / Limitations
- Dataset label semantics may vary by source.
- Model can learn source-specific patterns.

### Future Improvements
- Multi-source domain adaptation in later phases.
- Robustness tests on out-of-domain datasets.

---

## 2) Dataset Exploration (EDA)

### Goal of the Step
Understand quality, label balance, and distribution before modeling.

### Possible Approaches
- Basic summary stats
- Visual analytics (histograms, label plots)
- Automated profiling tools

### Comparison
- Basic stats are quick and deterministic.
- Visual analytics improve interpretability.
- Auto profiling is convenient but less controlled.

### Decision
Use controlled summary + targeted plots/metrics inside pipeline outputs.

### Implementation
Implemented via:
- `phase1/src/data_processing.py` (split summary)
- `phase1/src/train_model.py` (candidate/model reports)

### Validation
Confirm:
- non-empty dataset after cleaning,
- balanced enough labels per split,
- no unassigned split rows.

### Risks / Limitations
- EDA may miss semantic overlap.

### Future Improvements
- Add richer drift and topic-based EDA.

---

## 3) Data Cleaning

### Goal of the Step
Create clean, consistent rows with valid labels and usable text.

### Possible Approaches
- Minimal cleaning
- Aggressive normalization
- Hybrid cleaning

### Comparison
- Minimal cleaning keeps signals but leaves noise.
- Aggressive cleaning can remove useful cues.
- Hybrid balances noise reduction and signal retention.

### Decision
Use hybrid cleaning with conservative normalization and dedupe-before-split.

### Implementation
Implemented in `phase1/src/data_processing.py`:
- label normalization,
- text normalization,
- invalid row filtering,
- exact deduplication.

### Validation
Verify row counts before/after cleaning and dedupe.

### Risks / Limitations
- Dedup only catches exact/normalized duplicates.

### Future Improvements
- Add semantic dedupe checks with embeddings.

---

## 4) Feature Engineering

### Goal of the Step
Capture deception-related signals from both content and behavior.

### Possible Approaches
- Text-only features
- Metadata-only features
- Hybrid feature set

### Comparison
- Text-only: strongest primary signal, may miss behavior cues.
- Metadata-only: weak alone, but useful context.
- Hybrid: better robustness.

### Decision
Use hybrid features: TF-IDF text + numeric style/behavioral features.

### Implementation
Implemented in `phase1/src/feature_engineering.py`:
- text cleaning/stemming,
- text stats,
- behavioral features (`rating`, `helpful_vote`, `verified_purchase`),
- sparse matrix combination.

### Validation
Confirm feature matrices align in shape across train/calib/val/test.

### Risks / Limitations
- Behavioral features may not generalize across platforms.

### Future Improvements
- Add richer linguistic signals and readability features.

---

## 5) Text Vectorization (TF-IDF)

### Goal of the Step
Convert text to machine-learnable sparse vectors.

### Possible Approaches
- CountVectorizer
- TF-IDF
- Pretrained embeddings

### Comparison
- Count: simple but less informative for weighting.
- TF-IDF: strong baseline for sparse text classification.
- Embeddings: richer semantics but more complexity.

### Decision
Use TF-IDF with 1-2 grams in Phase 1.

### Implementation
Implemented in `phase1/src/feature_engineering.py`.

### Validation
Ensure vectorizer fit is train-only and transformed splits have expected dimensions.

### Risks / Limitations
- TF-IDF can underperform on paraphrase-heavy patterns.

### Future Improvements
- Add transformer embeddings in later phases.

---

## 6) Baseline Model (Logistic Regression)

### Goal of the Step
Establish a robust, interpretable baseline.

### Possible Approaches
- Logistic Regression
- Naive Bayes
- Linear SVM

### Comparison
- Logistic: strong probability baseline with balanced performance.
- Naive Bayes: very fast but can be less calibrated.
- Linear SVM: often strong but no native probability.

### Decision
Use Logistic Regression baseline.

### Implementation
Implemented in `phase1/src/train_model.py`.

### Validation
Evaluate F1/precision/recall/confusion matrix and probability quality.

### Risks / Limitations
- Linear decision boundary may miss complex interactions.

### Future Improvements
- Add calibrated margin-based and boosting models.

---

## 7) Improved Models

### Goal of the Step
Improve baseline quality while preserving calibration contract.

### Possible Approaches
- SGDClassifier (`log_loss`)
- LinearSVC + calibration
- XGBoost (optional)

### Comparison
- SGD: scalable sparse-text probabilistic model.
- LinearSVC: strong margin model, needs calibration.
- XGBoost: can help with hybrid signals but dependency may vary.

### Decision
Primary improved models are SGD and calibrated LinearSVC; XGBoost is optional fallback-aware.

### Implementation
Implemented in `phase1/src/train_model.py`.

### Validation
Rank models by validation F1 with calibrated probabilities and threshold tuning.

### Risks / Limitations
- Optional dependency differences across environments.

### Future Improvements
- Add stacking/ensembling only after strict leakage checks.

---

## 8) Model Evaluation

### Goal of the Step
Measure both classification quality and fake-score reliability.

### Possible Approaches
- Single holdout
- Four-way split with calibration
- Nested CV

### Comparison
- Single holdout is simple but riskier for leakage/tuning.
- Four-way split enforces strict calibration protocol.
- Nested CV gives robust estimates but higher runtime cost.

### Decision
Use four-way split; fallback to nested CV if instability appears.

### Implementation
Implemented in:
- `phase1/src/evaluate_model.py`
- `phase1/src/train_model.py`

### Validation
Track:
- precision, recall, F1, ROC-AUC,
- confusion matrix,
- Brier score and ECE,
- classwise reliability gap.

### Risks / Limitations
- Smaller effective train size in four-way split.

### Future Improvements
- Add bootstrap confidence intervals for key metrics.

---

## 9) Error Analysis

### Goal of the Step
Identify model blind spots and leakage warning signals.

### Possible Approaches
- Manual false positive/negative review
- Similarity audit
- Confidence bucket analysis

### Comparison
- Manual review gives actionable context.
- Similarity audit catches potential leakage patterns.
- Confidence analysis surfaces calibration issues.

### Decision
Include heuristic cross-split near-duplicate audit and residual risk reporting.

### Implementation
Implemented in `phase1/src/evaluate_model.py` and stored under `phase1/reports/`.

### Validation
Check near-duplicate rate against Phase 1 threshold and flag release risk if needed.

### Risks / Limitations
- Audit is partial coverage and may miss semantic paraphrases.

### Future Improvements
- Add embedding-based paraphrase overlap checks.

---

## 10) Fake Score Calculation

### Goal of the Step
Produce a stable fake probability score for downstream use.

### Possible Approaches
- Raw model probabilities
- Calibrated probabilities
- Score ranking without threshold

### Comparison
- Raw probabilities may be miscalibrated.
- Calibrated probabilities are more reliable for decisions.
- Ranking-only avoids thresholds but less actionable.

### Decision
Use calibrated fake probability and validation-selected threshold.

### Implementation
Implemented in `phase1/src/train_model.py` and `phase1/src/predict.py`.

### Validation
Validate score range [0,1] and threshold reproducibility from metadata.

### Risks / Limitations
- Calibration may drift on future domain shifts.

### Future Improvements
- Add periodic recalibration with fresh labeled data.

---

## 11) Prediction Function

### Goal of the Step
Expose reusable inference functions for single and batch use.

### Possible Approaches
- Inline inference in API
- Standalone prediction module

### Comparison
- Inline inference duplicates logic.
- Standalone module improves reuse and testability.

### Decision
Use standalone module for single and batch prediction.

### Implementation
Implemented in `phase1/src/predict.py`.

### Validation
Check deterministic outputs and schema consistency.

### Risks / Limitations
- Metadata mismatch can break inference.

### Future Improvements
- Add artifact compatibility checks.

---

## 12) API Endpoint for Prediction

### Goal of the Step
Provide HTTP interface for model inference.

### Possible Approaches
- Flask
- FastAPI
- Streamlit API wrappers

### Comparison
- Flask is lightweight and stable for Phase 1.
- FastAPI provides richer typing/docs with extra setup.
- Streamlit wrappers are less API-focused.

### Decision
Use Flask with strict request validation and deterministic output schema.

### Implementation
Implemented in `phase1/api/app.py`.

### Validation
Validate:
- `/health` success,
- `/predict` success and 4xx behavior for bad input,
- response includes `label`, `fake_probability`, `threshold`, `model_version`.

### Risks / Limitations
- Local server-only in Phase 1 (no production hardening).

### Future Improvements
- Add auth, rate limits, and production deployment profile.
