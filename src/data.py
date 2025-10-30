import pandas as pd
import re, string

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

def load_data(path='../data/train.csv'):
    data = pd.read_csv(path)
    data = data[['question1', 'question2', 'is_duplicate']].dropna()
    data['question1'] = data['question1'].apply(clean_text)
    data['question2'] = data['question2'].apply(clean_text)
    print(f"✅ Loaded dataset: {data.shape[0]} rows")
    return data

