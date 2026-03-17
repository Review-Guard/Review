# 🕵️Review Guard

## 📌 Project Overview

The **Review Guard System** is a locally runnable web application that uses Natural Language Processing (NLP) and Machine Learning to classify online reviews as **genuine** or **fake**.

Users can:

* Analyze a single review text
* Upload a CSV file for batch analysis
* View predictions with confidence percentages
* Run the system fully offline (no cloud required)

This project was developed by a three-person academic team over 10 weeks (part-time, $0 budget).

---

## 🎯 Project Goals

* **Technical Goal:** Achieve >80% classification accuracy on a held-out test dataset.
* **Process Goal:** Deliver a fully functioning local web application using Agile methodology with four milestones.

---

## 🛠️ Key Features

* Single review classification
* Batch CSV review processing
* Confidence score output
* Offline execution
* Simple web interface (Bootstrap-based UI)

---

## 🧠 Technology Stack

* **Backend:** Python (Flask 2.x)
* **Frontend:** HTML, CSS, JavaScript (Bootstrap 5 optional)
* **ML/NLP:** Scikit-learn, NLTK / SpaCy
* **Data Processing:** Pandas
* **Model Serialization:** Joblib
* **Testing:** Pytest

---

## 📂 Project Structure

```
Review-guard/
│
├── backend/
│   ├── app/
│   │   ├── factory.py
│   │   ├── routes.py
│   │   ├── services/
│   │   │   └── inference.py
│   │   ├── templates/
│   │   │   ├── index.html
│   │   │   └── batch.html
│   │   └── static/
│   │       ├── css/
│   │       └── js/
│   │
│   ├── artifacts/
│   │   ├── model.joblib
│   │   ├── vectorizer.joblib
│   │   └── label_map.json
│   │
│   ├── tests/
│   │   └── test_health.py
│   │
│   ├── requirements.txt
│   └── run.py
│
├── ml/
│   ├── src/
│   │   ├── train.py
│   │   ├── preprocess.py
│   │   └── evaluate.py
│   │
│   └── artifacts/
│
├── data/
│   ├── raw/
│   │   └── reviews.csv
│   └── processed/
│
├── scripts/
│
└── README.md
```

---

## 🚀 Installation & Run

# Clone repository
git clone https://github.com/kritisubedi1/review-guard.git
cd review-guard

cd backend

# Install dependencies
pip install -r requirements.txt

# Run application
python run.py

Open: http://localhost:5000

## 📊 Model Evaluation
Accuracy (>80% target)
Precision, Recall, F1-Score

## 📊 Project Management (Jira)

The project was managed using Jira with sprint planning and task tracking.

### 🗂️ Jira Board Overview

![Jira Board](images/jira-board.png)


## 📅 Agile Milestones
1. Planning & Dataset Preparation
2. Model Development & Evaluation
3. Web Application Integration
4. Testing & Final Delivery

---

## 📜 License

Developed for academic purposes only.
