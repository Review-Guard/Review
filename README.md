# 🕵️ Review Guard

## 📌 Project Overview

The **Review Guard System** is a locally runnable web application that uses **Natural Language Processing (NLP)** and **Machine Learning** to classify online reviews as:

- ✅ Genuine  
- ❌ Fake  

Users can:
- Analyze a single review  
- View prediction with Fake probability confidence score  
- Switch between different ML models  

This project was developed by a three-person academic team over 10 weeks (part-time, $0 budget).

---

## 🎯 Project Goals

- Achieve **>80% classification accuracy**
- Build a **fully offline web application**
- Follow **Agile development methodology**

---

## 🛠️ Key Features

- Single review classification  
- Confidence score output  
- Model selection (v1, v2, v3)  
- Simple web interface (Bootstrap-based UI)  
- Fully offline execution  

---

## 🧠 Technology Stack

- **Backend:** Python (Flask 2.x)  
- **Frontend:** HTML, CSS, JavaScript  
- **ML/NLP:** Scikit-learn, NLTK  
- **Data Processing:** Pandas  
- **Model Serialization:** Joblib  
- **Testing:** Pytest  

---

## 📂 Dataset

### Label Direction

- `label = 1 → fake`  
- `label = 0 → genuine`  

---

## 🧠 Models

- **v1 (Hybrid):** Text + metadata  
- **v2 (Text-only):** Only text features  
- **v3 (Best Model):** Blended model combining v1 + v2  

---

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

⚙️ Installation
pip install -r requirements.txt


```
▶️ Run Application
python3 run.py


Open in browser:
http://127.0.0.1:8000


🏋️ Train Models
🔹 Default Training (Recommended)

python app/ml/training/train_model.py \
--input_csv dataset/amazon_labeled_fake_reviews/final_labeled_fake_reviews.csv \
--phase1_root app \
--random_seed 42


🔹 Train Model Versions
v1 (Hybrid): Text + metadata
v2 (Text-only): Only text features
v3 (Best): Combined model


# v1
python app/ml/training/train_model.py --include_behavioral --model_version phase1-v1

# v2
python app/ml/training/train_model.py --model_version phase1-v2

# v3 (Best)
python app/ml/training/train_model_v3.py

Optional (Advanced)

python app/ml/training/train_model.py --enable_xgboost

Run Tests
python3 -m pytest -q app/tests

📊 Model Evaluation
Accuracy (>80%)
Precision
Recall
F1-Score

⚠️ Risk & Validation
To avoid data leakage:
Text normalization + hashing
Duplicate-safe splitting
Near-duplicate audit
Remaining Risk
Paraphrased reviews may still exist → residual risk

## 📊 Project Management (Jira)

## 📊 Project Management (Jira)

The project was managed using Jira with sprint planning and task tracking.

### 🗂️ Jira Board Overview

![Jira Board](images/jira-board.png)

📅 Agile Milestones
Planning & Dataset Preparation
Model Development & Evaluation
Web Application Integration
Testing & Final Delivery

👥 Team
Kriti Subedi
Swapnali Kudale
Aditi Sharma

📜 License
Developed for academic purposes only.