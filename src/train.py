# src/train.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os

def load_data(file_path="data/dataset.csv"):
    """Load dataset"""
    return pd.read_csv(file_path)

def prepare_features_labels(file_path="data/dataset.csv"):
    """Prepare numeric features and target"""
    df = load_data(file_path)

    # Assume last column is target
    y = df.iloc[:, -1]

    # Keep only numeric columns for features
    X = df.iloc[:, :-1].select_dtypes(include=["number"])

    return X, y

def train_model(file_path="data/dataset.csv"):
    """Train model and save it"""
    X, y = prepare_features_labels(file_path)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model

if __name__ == "__main__":
    train_model()
    print("Training completed successfully")
