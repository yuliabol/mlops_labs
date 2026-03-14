import pandas as pd
import pytest
import os


def test_data_existence():
    assert os.path.exists("data/prepared/train.csv"), "Train data missing"
    assert os.path.exists("data/prepared/test.csv"), "Test data missing"


def test_data_columns():
    train_df = pd.read_csv("data/prepared/train.csv")
    required_columns = [
        "stroke",
        "age",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi",
    ]
    for col in required_columns:
        assert col in train_df.columns, f"Missing column: {col}"


def test_data_non_empty():
    train_df = pd.read_csv("data/prepared/train.csv")
    assert len(train_df) > 0, "Train dataset is empty"
