# src/text_preprocessor.py
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    def __init__(self):
        pass

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """AC1: Text cleaning - lowercase + punctuation removal"""
        if pd.isna(text):
            text = ""
        text = str(text).lower()
        # Remove punctuation and special characters (keep only letters, numbers, spaces)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_stopwords(self, text):
        """AC2: Stop word removal configured"""
        tokens = text.split()  # simple split for now
        tokens = [word for word in tokens if word not in self.stop_words]
        return " ".join(tokens)
    
    def tokenize_text(self, text):
        """AC3: Tokenization function created"""
        # Using simple split first (we'll improve later)
        return text.split()
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None

    def fit_vectorizer(self, texts):
        """AC4: Vectorization (TF-IDF) implemented"""
        # Full preprocessing pipeline
        processed = []
        for text in texts:
            cleaned = self.clean_text(text)
            no_stop = self.remove_stopwords(cleaned)
            processed.append(no_stop)
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True
        )
        self.vectorizer.fit(processed)
        return self.vectorizer

    def transform(self, texts):
        """Transform new text using fitted vectorizer"""
        processed = []
        for text in texts:
            cleaned = self.clean_text(text)
            no_stop = self.remove_stopwords(cleaned)
            processed.append(no_stop)
        return self.vectorizer.transform(processed)