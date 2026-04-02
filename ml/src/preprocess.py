# src/text_preprocessor.py
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

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