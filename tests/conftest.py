"""Shared pytest fixtures for Week 4 CI assignment (Iris dataset)."""

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ACTIVE_DATA_PATH = PROJECT_ROOT / "data" / "active_data.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
EVAL_PATH = PROJECT_ROOT / "data" / "processed" / "eval.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "model.joblib"

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_COLUMN = "species"
VALID_SPECIES = {"setosa", "versicolor", "virginica"}

# Reasonable bounds for Iris features.
FEATURE_BOUNDS = {
    "sepal_length": (4.0, 8.0),
    "sepal_width": (2.0, 4.5),
    "petal_length": (1.0, 7.0),
    "petal_width": (0.1, 2.5),
}


@pytest.fixture(scope="session")
def active_df() -> pd.DataFrame:
    assert ACTIVE_DATA_PATH.exists(), (
        f"active_data not found at {ACTIVE_DATA_PATH}. "
    
    )
    return pd.read_csv(ACTIVE_DATA_PATH)


@pytest.fixture(scope="session")
def train_df() -> pd.DataFrame:
    assert TRAIN_PATH.exists(), (
        f"Training data not found at {TRAIN_PATH}. Run `dvc pull` first."
    )
    return pd.read_csv(TRAIN_PATH)


@pytest.fixture(scope="session")
def eval_df() -> pd.DataFrame:
    assert EVAL_PATH.exists(), (
        f"Evaluation data not found at {EVAL_PATH}. Run `dvc pull` first."
    )
    return pd.read_csv(EVAL_PATH)
