# Review Guard — Software Testing README

This file is the finalized testing submission copy derived from `TESTING_REPORT.md`.

---

# Software Testing Report — Review Guard

## Table of Content

1. Introduction
2. Test Strategy
3. Test Cases
4. Test Execution Results
5. Test Metrics and Summary
6. Lessons Learned

---

## 1. Introduction

### 1.1 Purpose

This document defines the test strategy, test cases, and execution results for the **Review Guard** system. It verifies that functional and selected non-functional requirements are implemented and working for local execution and API usage.

### 1.2 Scope

**Covered in this report**

- API functional behavior (`/health`, `/predict`, `/predict_all`)
- Unit testing for preprocessing, prediction logic, and API handlers
- Input validation and security-oriented negative testing
- Lightweight local performance check
- Regression check through repeated execution of automated tests

**Excluded / limited scope**

- High-concurrency distributed load/stress testing at production scale

---

## 2. Test Strategy

| Strategy | Description | When to Use |
| --- | --- | --- |
| Big-Bang | All components integrated and tested together at once. | Small systems with few dependencies. |
| Incremental (Top-Down) | Starts at high-level modules, stubs for lower modules. | Early validation of control flow. |
| Incremental (Bottom-Up) | Starts from lower modules and integrates upward. | Easier isolation of lower-level defects. |

**Chosen approach:** Incremental (Bottom-Up) + API-level integration

Reason:

- Core ML/preprocessing units are independently testable.
- Easier defect localization in prediction and preprocessing utilities.
- API integration checks validate end-to-end behavior after unit confidence.

### 2.1 Testing Types

| Testing Type | Description | Applied? |
| --- | --- | --- |
| Functional Testing | Verifies behavior against expected requirements. | Yes |
| Usability Testing | End-user ease-of-use checks. | Yes (automated UI contract checks) |
| Security Testing | Input validation and malicious input handling. | Yes (validation + attack-pattern suite) |
| Performance Testing | Response time/stability under repeated requests. | Yes (lightweight local) |
| Regression Testing | Re-run tests after changes. | Yes |
| Compatibility Testing | Cross-browser/device/OS validation. | Yes (automated user-agent matrix + static assets) |
| Unit Testing | Isolated function/class behavior. | Yes |
| Integration Testing | Module interaction and API data flow. | Yes |
| System Testing | End-to-end complete system validation. | Yes |
| Acceptance Testing | End-user business scenario validation. | Yes |

### 2.2 Tools and Environment

| Category | Tool / Technology | Purpose |
| --- | --- | --- |
| Test Framework | `pytest` | Automated unit/regression tests |
| Mocking Framework | `unittest.mock` | Mocking inference paths in API tests |
| Performance Testing | Custom Python script (`app/tests/run_additional_checks.py`) | Sequential latency/stability checks |
| Security Testing | Flask test client + attack-pattern suites | Input validation, malicious payload hardening |
| Compatibility/Usability | Flask test client + user-agent matrix | Automated browser-profile and UI-contract checks |
| Bug Tracking | GitHub + Jira | Defect/task tracking |
| IDE / Test Runner | VS Code + terminal | Running tests and collecting outputs |
| Test Environment | Local macOS, Python 3.13 venv | Reproducible local execution |
| Test Management | Markdown report + test files | Traceability and results summary |
| CI/CD | GitHub Actions (`.github/workflows/ci.yml`) | Automated test + coverage run on push and pull request |

### 2.3 Test Subjects

- `app/backend/app.py` (API routes and request validation)
- `app/ml/predict.py` (prediction and feature pipeline for inference)
- `app/ml/training/feature_engineering.py` (preprocessing logic)
- `app/tests/*.py` and `app/tests/run_additional_checks.py`

---

## 3. Test Cases

### 3.1 Test Case Table

| TC # | Test Case Name | Related Req. | Priority | Status |
| --- | --- | --- | --- | --- |
| TC-01 | Health endpoint returns service up | FR-API-01 | High | Pass |
| TC-02 | Predict endpoint valid payload returns label/probability | FR-API-02 | High | Pass |
| TC-03 | Predict rejects non-JSON payload | FR-VAL-01 | High | Pass |
| TC-04 | Predict rejects missing `text` field | FR-VAL-02 | High | Pass |
| TC-05 | Predict rejects empty `text` | FR-VAL-03 | High | Pass |
| TC-06 | Predict rejects oversized text | NFR-ROB-01 | Medium | Pass |
| TC-07 | Predict rejects invalid numeric types | FR-VAL-04 | Medium | Pass |
| TC-08 | Predict_all returns multi-model contract | FR-API-03 | Medium | Pass |
| TC-09 | Preprocessing text normalization behavior | FR-PRE-01 | Medium | Pass |
| TC-10 | Numeric scaling aligns to metadata schema | FR-ML-01 | Medium | Pass |
| TC-11 | Label mapping from probability threshold | FR-ML-02 | Medium | Pass |
| TC-12 | Local sequential latency stability (30 requests) | NFR-PERF-01 | Low | Pass |
| TC-13 | Compatibility matrix across browser/device user agents | NFR-COMP-01 | Low | Pass |
| TC-14 | Automated security attack-pattern suite | NFR-SEC-02 | Medium | Pass |
| TC-15 | Predict_all rejects non-JSON payload | FR-VAL-05 | High | Pass |
| TC-16 | Predict_all rejects missing `text` field | FR-VAL-06 | High | Pass |
| TC-17 | Predict_all rejects oversized text | NFR-ROB-02 | Medium | Pass |
| TC-18 | Predict rejects non-string `text` | FR-VAL-07 | High | Pass |
| TC-19 | Predict rejects malformed JSON body | FR-VAL-08 | Medium | Pass |
| TC-20 | Predict casts numeric string payload fields | FR-API-04 | Medium | Pass |
| TC-21 | Predict invalid model version returns 400 | FR-VAL-09 | Medium | Pass |
| TC-22 | Predict_all agreement recommendation path | FR-API-05 | Medium | Pass |
| TC-23 | first_existing_path selects first valid path | FR-UTIL-01 | Medium | Pass |
| TC-24 | models_dir_from_version maps versions correctly | FR-UTIL-02 | Medium | Pass |
| TC-25 | parse_payload validates and casts all fields | FR-VAL-10 | High | Pass |
| TC-26 | run_prediction_for_version routes v1/v3 and formats output | FR-ML-03 | Medium | Pass |
| TC-27 | probability_from_model supports predict_proba + decision_function | FR-ML-04 | Medium | Pass |
| TC-28 | build_behavioral_matrix handles missing columns safely | NFR-ROB-03 | Medium | Pass |
| TC-29 | Home route renders main UI page | FR-UI-01 | Low | Pass |
| TC-30 | Predict returns 500 when artifacts are missing | NFR-ERR-01 | Medium | Pass |
| TC-31 | Predict returns 500 on unexpected runtime exceptions | NFR-ERR-02 | Medium | Pass |
| TC-32 | Predict_all returns 500 when artifacts are missing | NFR-ERR-03 | Medium | Pass |
| TC-33 | Predict_all returns 500 on unexpected runtime exceptions | NFR-ERR-04 | Medium | Pass |
| TC-34 | Predict_all rejects malformed JSON body | FR-VAL-11 | Medium | Pass |
| TC-35 | Predict_batch_v3 blend formula is correct | FR-ML-05 | Medium | Pass |
| TC-36 | Predict_batch_v3 threshold labeling path works | FR-ML-06 | Medium | Pass |
| TC-37 | Predict rejects diverse malformed payload shapes (fuzz set) | FR-VAL-12 | High | Pass |
| TC-38 | Predict_all rejects diverse malformed payload shapes (fuzz set) | FR-VAL-13 | High | Pass |
| TC-39 | Predict defaults omitted numeric fields correctly | FR-API-06 | Medium | Pass |
| TC-40 | Predict accepts unicode/international text input | FR-I18N-01 | Low | Pass |
| TC-41 | Artifact loader reads v1/v2 model/vectorizer/metadata bundle | FR-DEPLOY-01 | Medium | Pass |
| TC-42 | Artifact loader reads v3 blended-model bundle | FR-DEPLOY-02 | Medium | Pass |
| TC-43 | Artifact loader fails safely on missing files | NFR-ROB-04 | Medium | Pass |
| TC-44 | UI page includes core usability controls and labels | FR-UX-01 | Low | Pass |
| TC-45 | Predict flow remains consistent across user-agent matrix | FR-COMP-01 | Medium | Pass |
| TC-46 | Security payloads do not produce 5xx on `/predict` | NFR-SEC-03 | Medium | Pass |
| TC-47 | Security payloads do not produce 5xx on `/predict_all` | NFR-SEC-04 | Medium | Pass |
| TC-48 | Static assets load successfully across user-agent matrix | FR-COMP-02 | Low | Pass |
| TC-49 | JSON content-type mismatch handled safely | NFR-SEC-05 | Medium | Pass |

### 3.2 Unit Tests

Executed via `pytest` (`app/tests`):

- API endpoint tests (`test_api_endpoints.py`)
- Extended API validation tests (`test_api_extended.py`)
- API error-path tests (`test_api_error_paths.py`)
- API fuzz-validation tests (`test_api_fuzz_validation.py`)
- Usability/compatibility automation tests (`test_usability_compatibility.py`)
- Security attack-pattern automation tests (`test_security_attack_patterns.py`)
- Backend helper tests (`test_backend_helpers.py`)
- Artifact loading tests (`test_artifact_loading.py`)
- Prediction module tests (`test_predict_module.py`)
- Prediction utility extended tests (`test_predict_utils_extended.py`)
- Prediction v3 blend logic tests (`test_predict_v3_logic.py`)
- Preprocessing tests (`test_preprocessing.py`)

Unit test command:

```zsh
PYTHONPATH=app python -m pytest -q app/tests
```

Result: **106 passed**.

### 3.3 Additional Tests

#### 3.3.1 Security Testing

Performed via API negative checks:

- Non-JSON payload
- Missing required field
- Empty text
- Oversized text
- Invalid numeric type injection attempt

Command:

```zsh
PYTHONPATH=app python app/tests/run_additional_checks.py
```

#### 3.3.2 Performance (Lightweight Local)

Performed sequentially for 30 `/predict` requests (local process, Flask test client).

Observed (latest run):

- Mean latency: **102.06 ms**
- p95 latency: **102.98 ms**
- Max latency: **103.77 ms**

#### 3.3.3 Acceptance/System Smoke

Validated end-to-end API behavior through successful prediction payload and contract checks for `/predict` and `/predict_all`.

### 3.4 Requirement Traceability Matrix

(See `TESTING_REPORT.md` section 3.4 for full matrix.)

---

## 4. Test Execution Results

| Metric | Count |
| --- | --- |
| Total Test Cases (executed) | 106 |
| Passed | 106 |
| Failed | 0 |
| Blocked | 0 |
| Not Executed | 0 |
| Pass Rate (executed only) | 100% |

Overall outcome:

- All executed critical tests passed.
- Core functional and validation requirements are verified.
- System is suitable for local delivery/demo.

---

## 5. Test Metrics and Summary

- Requirement coverage (executed subset): high for core API + validation + ML inference path.
-Code coverage (latest measured run): app/backend/app.py ~100%, app/ml/predict.py ~100%, training modules ~100%. High code coverage achieved for core modules (approx. 90–100%).
- Coverage note: training pipeline modules are now included in automated tests.
- Defect density (executed suites): low (no functional failures).
- Automated vs manual: predominantly automated.
- Execution time: short local runs suitable for pre-commit checks.

---

## 6. Lessons Learned

- Keeping import paths consistent across branches is critical for zero-friction test execution.
- Early validation and boundary checks prevented common API misuse cases.
- Compatibility and usability can be automated through user-agent matrix and UI contract checks.
- Future improvement: add CI workflow and distributed load testing.

---

> Run commands from project root directory

## Commands Reference

### Unit + regression tests

```zsh
PYTHONPATH=app python -m pytest -q app/tests
```

### Additional integration + security + perf checks

```zsh
PYTHONPATH=app python app/tests/run_additional_checks.py
```

### Coverage run
> Run from project root directory after activating virtual environment

```zsh
PYTHONPATH=app python -m pytest app/tests --cov=app/backend --cov=app/ml --cov-report=term-missing
```
