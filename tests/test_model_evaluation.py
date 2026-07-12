"""Task 2: Model evaluation tests — inference + quality gate metrics."""

import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from conftest import EVAL_PATH, FEATURE_COLUMNS, MODEL_PATH, TARGET_COLUMN, eval_df

# Quality gates — tests fail if the model degrades below these thresholds.
MIN_ACCURACY = 0.90
MIN_PRECISION = 0.85
MIN_RECALL = 0.85
MIN_F1 = 0.85


@pytest.fixture(scope="session")
def model():
    assert MODEL_PATH.exists(), (
        f"Trained model not found at {MODEL_PATH}. Run `dvc pull` first."
    )
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="session")
def eval_predictions(model, eval_df: pd.DataFrame):
    X = eval_df[FEATURE_COLUMNS]
    y_true = eval_df[TARGET_COLUMN]
    y_pred = model.predict(X)
    return y_true, y_pred


def test_model_loads_and_predicts(model, eval_df: pd.DataFrame):
    predictions = model.predict(eval_df[FEATURE_COLUMNS])
    assert len(predictions) == len(eval_df)
    assert set(predictions).issubset({"setosa", "versicolor", "virginica"})


def test_accuracy_threshold(eval_predictions):
    y_true, y_pred = eval_predictions
    accuracy = accuracy_score(y_true, y_pred)
    assert accuracy >= MIN_ACCURACY, (
        f"Accuracy {accuracy:.4f} below minimum {MIN_ACCURACY}"
    )


def test_precision_threshold(eval_predictions):
    y_true, y_pred = eval_predictions
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    assert precision >= MIN_PRECISION, (
        f"Precision {precision:.4f} below minimum {MIN_PRECISION}"
    )


def test_recall_threshold(eval_predictions):
    y_true, y_pred = eval_predictions
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    assert recall >= MIN_RECALL, (
        f"Recall {recall:.4f} below minimum {MIN_RECALL}"
    )


def test_f1_threshold(eval_predictions):
    y_true, y_pred = eval_predictions
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    assert f1 >= MIN_F1, f"F1 {f1:.4f} below minimum {MIN_F1}"


def test_evaluation_metrics_report(eval_predictions):
    """Write metrics to a file consumed by CML in CI."""
    y_true, y_pred = eval_predictions
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "samples": len(y_true),
    }

    lines = ["## Model Evaluation Metrics", ""]
    for name, value in metrics.items():
        if name == "samples":
            lines.append(f"- **{name}**: {int(value)}")
        else:
            lines.append(f"- **{name}**: {value:.4f}")

    report_path = MODEL_PATH.parents[1] / "metrics_report.txt"
    report_path.write_text("\n".join(lines) + "\n")

    assert metrics["accuracy"] >= MIN_ACCURACY
