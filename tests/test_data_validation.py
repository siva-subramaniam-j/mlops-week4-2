"""Task 1: Data validation tests for Iris training and evaluation datasets."""

import pandas as pd
import pytest

from conftest import (
    ACTIVE_DATA_PATH,
    EVAL_PATH,
    FEATURE_BOUNDS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRAIN_PATH,
    VALID_SPECIES,
    active_df,
    eval_df,
    train_df,
)


@pytest.mark.parametrize("path", [ACTIVE_DATA_PATH, TRAIN_PATH, EVAL_PATH])
def test_dataset_files_exist(path):
    assert path.exists(), f"Missing dataset file: {path}"


def test_active_data_schema(active_df: pd.DataFrame):
    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    assert list(active_df.columns) == expected_columns


def test_train_schema(train_df: pd.DataFrame):
    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    assert list(train_df.columns) == expected_columns


def test_eval_schema(eval_df: pd.DataFrame):
    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    assert list(eval_df.columns) == expected_columns


@pytest.mark.parametrize("df_fixture", ["active_df", "train_df", "eval_df"])
def test_no_missing_values(df_fixture, request):
    df = request.getfixturevalue(df_fixture)
    missing = df.isnull().sum().sum()
    assert missing == 0, f"Found {missing} missing values in {df_fixture}"


@pytest.mark.parametrize("df_fixture", ["active_df", "train_df", "eval_df"])
def test_feature_types(df_fixture, request):
    df = request.getfixturevalue(df_fixture)
    for column in FEATURE_COLUMNS:
        assert pd.api.types.is_numeric_dtype(df[column]), (
            f"{column} must be numeric in {df_fixture}"
        )
    assert set(df[TARGET_COLUMN].unique()).issubset(VALID_SPECIES), (
        f"{TARGET_COLUMN} must be valid Iris species in {df_fixture}"
    )


@pytest.mark.parametrize("df_fixture", ["active_df", "train_df", "eval_df"])
def test_feature_value_ranges(df_fixture, request):
    df = request.getfixturevalue(df_fixture)
    for column, (lower, upper) in FEATURE_BOUNDS.items():
        assert df[column].min() >= lower, (
            f"{column} min {df[column].min()} below {lower} in {df_fixture}"
        )
        assert df[column].max() <= upper, (
            f"{column} max {df[column].max()} above {upper} in {df_fixture}"
        )


def test_active_data_row_count(active_df: pd.DataFrame):
    # Default Week 4 build uses raw + v1 + v2 = 300 rows.
    assert len(active_df) in {150, 300}, (
        f"Unexpected active_data size: {len(active_df)} rows"
    )


def test_train_has_enough_samples(train_df: pd.DataFrame):
    assert len(train_df) >= 80, "Training set should have at least 80 rows"


def test_eval_has_enough_samples(eval_df: pd.DataFrame):
    assert len(eval_df) >= 20, "Evaluation set should have at least 20 rows"
