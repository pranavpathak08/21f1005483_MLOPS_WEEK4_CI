from pathlib import Path

import joblib
import pandas as pd
import pytest

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "iris.csv"
MODEL_PATH = ROOT_DIR / "models" / "iris_model.joblib"

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET = "species"

# Minimum acceptable model performance.
MIN_ACCURACY = 0.90
MIN_PRECISION = 0.90
MIN_RECALL = 0.90
MIN_F1 = 0.90


@pytest.fixture(scope="module")
def evaluation_results():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    return {
        "y_test": y_test,
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, average="weighted"
        ),
        "recall": recall_score(
            y_test, predictions, average="weighted"
        ),
        "f1": f1_score(
            y_test, predictions, average="weighted"
        ),
    }


def test_model_file_exists():
    assert MODEL_PATH.exists(), "Trained model file is missing"


def test_model_generates_predictions(evaluation_results):
    predictions = evaluation_results["predictions"]

    assert len(predictions) == len(evaluation_results["y_test"])
    assert len(predictions) > 0


def test_model_predicts_valid_classes(evaluation_results):
    expected_classes = {"setosa", "versicolor", "virginica"}

    predicted_classes = set(evaluation_results["predictions"])

    assert predicted_classes.issubset(expected_classes)


def test_model_accuracy(evaluation_results):
    assert evaluation_results["accuracy"] >= MIN_ACCURACY


def test_model_precision(evaluation_results):
    assert evaluation_results["precision"] >= MIN_PRECISION


def test_model_recall(evaluation_results):
    assert evaluation_results["recall"] >= MIN_RECALL


def test_model_f1_score(evaluation_results):
    assert evaluation_results["f1"] >= MIN_F1