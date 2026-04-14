# 🕵️ Review Guard

## 📌 Project Overview

This project was developed by a three-person academic team over 10 weeks on a part-time basis with a $0 budget.

**Review Guard** is a locally runnable web application that uses **Natural Language Processing (NLP)** and **Machine Learning** to classify online reviews as:

* ✅ Genuine
* ❌ Fake

## 🚀 Features

* Single review prediction
* Batch CSV upload and processing
* Fake/Genuine classification
* Fake probability score output
* Download results as CSV
* Multiple model support (`v1`, `v2`, `v3`)
* Fully offline system

## 📂 CSV Format

The CSV file must contain the following columns:

```text
text,rating,helpful_vote,verified_purchase
```

## 🎯 Project Goals

* Achieve **greater than 80% classification accuracy** on unseen test data
* Develop a **fully offline, locally runnable web application**
* Support both **single review prediction and batch CSV processing**
* Provide **fake probability scores for each prediction**
* Follow an **Agile development methodology with iterative improvements**
* Ensure **fast response time and user-friendly interface**

## 🛠️ Technologies Used

* **Backend:** Python, Flask
* **Frontend:** HTML, CSS, JavaScript
* **ML/NLP:** Scikit-learn, NLTK
* **Data Processing:** Pandas, NumPy
* **Model Storage:** Joblib
* **Testing:** Pytest
* **CI/CD:** GitHub Actions

## 📂 Dataset

### Label Definition

* `1` → Fake review
* `0` → Genuine review

The system uses binary classification where `1` represents a fake review and `0` represents a genuine review.

## 📂 Project Structure

```text
Review-1/
│
├── run.py
├── requirements.txt
├── README.md
├── app/
│   ├── backend/
│   │   ├── __init__.py
│   │   └── app.py
│   ├── frontend/
│   │   ├── templates/
│   │   │   └── index.html
│   │   └── static/
│   │       ├── css/
│   │       └── js/
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   └── training/
│   │       ├── data_processing.py
│   │       ├── feature_engineering.py
│   │       ├── evaluate_model.py
│   │       ├── train_model.py
│   │       └── train_model_v3.py
│   ├── tests/
│   │   ├── test_api_endpoints.py
│   │   ├── test_predict_module.py
│   │   └── test_preprocessing.py
│   ├── artifacts/
│   │   ├── models/
│   │   │   ├── default/
│   │   │   ├── v1/
│   │   │   ├── v2/
│   │   │   └── v3/
│   │   └── reports/
│   ├── data/
│   ├── notebooks/
│   └── src/
└── images/
```

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

## ✅ Run Application

```bash
python3 run.py
```

## 🏋️ Train Models

### Default Training (Recommended)

```bash
python app/ml/training/train_model.py \
  --input_csv dataset/amazon_labeled_fake_reviews/final_labeled_fake_reviews.csv \
  --phase1_root app \
  --random_seed 42
```

### Train Model Versions

* **v1 (Hybrid):** Text + metadata
* **v2 (Text-only):** Only text features
* **v3 (Best Model):** Blended model combining v1 + v2

#### v1

```bash
python app/ml/training/train_model.py --include_behavioral --model_version phase1-v1
```

#### v2

```bash
python app/ml/training/train_model.py --model_version phase1-v2
```

#### v3 (Best)

```bash
python app/ml/training/train_model_v3.py
```

### Optional (Advanced)

```bash
python app/ml/training/train_model.py --enable_xgboost
```

## 🧪 Testing

* 106 test cases implemented
* 100% pass rate
* Includes:

  * Unit Testing
  * Integration Testing
  * Security Testing
  * Regression Testing

### Run Tests

```bash
python3 -m pytest -q app/tests
```

## 📊 Model Evaluation

The system is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

## ⚠️ Risk and Validation

To reduce data leakage and improve reliability, the project includes:

* Text normalization and hashing
* Duplicate-safe splitting
* Near-duplicate audit

### Remaining Risk

* Paraphrased reviews may still exist in the dataset, creating residual risk

## ⚙️ CI/CD

* GitHub Actions used
* Automated testing on every push

## 📊 Project Management (Jira)

The project was managed using **Jira** with sprint planning and task tracking.

### 🗂️ Jira Board Overview

![Jira Board](images/jira-board.png)

## 📅 Agile Milestones

* Planning and dataset preparation
* Model development and evaluation
* Web application integration
* Testing and final delivery

## 👥 Team

* Kriti Subedi
* Swapnali Kudale
* Aditi Sharma

## 📜 License

Developed for academic purposes only.
