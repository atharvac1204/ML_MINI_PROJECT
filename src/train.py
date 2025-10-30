# src/train.py
from sklearn.model_selection import train_test_split
import pandas as pd
from data import load_data
from features import FeatureExtractor
from model import DuplicateModel
import os

def main():
    # Load & clean
    data = load_data('../data/train.csv')

    # Drop missing values to avoid errors
    data.dropna(subset=['question1', 'question2', 'is_duplicate'], inplace=True)

    data = data.sample(5000, random_state=42).reset_index(drop=True)


    # Initialize feature extractor
    fe = FeatureExtractor()
    fe.fit(pd.concat([data['question1'], data['question2']], axis=0))

    # Compute similarity for each question pair
    data['similarity'] = data.apply(
        lambda r: fe.compute_similarity(r['question1'], r['question2']), axis=1
    )

    # Features and labels
    X = data[['similarity']]
    y = data['is_duplicate']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = DuplicateModel()
    model.train(X_train, y_train)

    # Evaluate
    print("✅ Training accuracy:", model.evaluate(X_train, y_train))
    print("✅ Test accuracy:", model.evaluate(X_test, y_test))

    # Ensure models directory exists
    os.makedirs('../models', exist_ok=True)

    # Save model and vectorizer
    model.save('../models/model.pkl', fe.vectorizer)
    print("✅ Model and vectorizer saved successfully in ../models/")

if __name__ == "__main__":
    main()
