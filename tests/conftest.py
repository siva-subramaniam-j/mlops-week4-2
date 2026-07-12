"""Shared pytest fixtures for Week 4 CI assignment (Iris dataset)."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ACTIVE_DATA_PATH = PROJECT_ROOT / "data" / "active_data.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "model.joblib"

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_COLUMN = "species"
VALID_SPECIES = {"setosa", "versicolor", "virginica"}
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Reasonable bounds for Iris features.
FEATURE_BOUNDS = {
    "sepal_length": (4.0, 8.0),
    "sepal_width": (2.0, 4.5),
    "petal_length": (1.0, 7.0),
    "petal_width": (0.1, 2.5),
}


def split_active_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split active_data into train and eval sets (same logic as training script)."""
    train_df, eval_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


@pytest.fixture(scope="session")
def active_df() -> pd.DataFrame:
    assert ACTIVE_DATA_PATH.exists(), (
        f"active_data not found at {ACTIVE_DATA_PATH}. Run `dvc pull` first."
    )
    return pd.read_csv(ACTIVE_DATA_PATH)


@pytest.fixture(scope="session")
def train_df(active_df: pd.DataFrame) -> pd.DataFrame:
    train, _ = split_active_data(active_df)
    return train


@pytest.fixture(scope="session")
def eval_df(active_df: pd.DataFrame) -> pd.DataFrame:
    _, eval_set = split_active_data(active_df)
    return eval_set
