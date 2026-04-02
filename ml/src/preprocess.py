# src/text_preprocessor.py
import re
import pandas as pd

class TextPreprocessor:
    def __init__(self):
        pass

    def clean_text(self, text):
        """AC1: Text cleaning - lowercase + punctuation removal"""
        if pd.isna(text):
            text = ""
        text = str(text).lower()
        # Remove punctuation and special characters (keep only letters, numbers, spaces)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text