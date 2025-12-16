# tests/test_train.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from src.train import load_data, train_model, prepare_features_labels

# Make sure you have a dataset at data/dataset.csv for testing
DATA_PATH = "data/dataset.csv"

def test_data_loading():
    """Test if data loads correctly"""
    df = load_data(DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_prepare_features_labels():
    """Test if features and labels are correctly prepared"""
    X, y = prepare_features_labels(DATA_PATH)
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 0

def test_model_training():
    """Test if model trains without errors"""
    model = train_model(DATA_PATH)
    assert model is not None
