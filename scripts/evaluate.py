import json
from pathlib import Path

import joblib
import pandas as pd

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
OUTPUT_PATH = ROOT_DIR / "evaluation_metrics.json"

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET = "species"


def main():
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

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(
            precision_score(y_test, predictions, average="weighted")
        ),
        "recall_weighted": float(
            recall_score(y_test, predictions, average="weighted")
        ),
        "f1_weighted": float(
            f1_score(y_test, predictions, average="weighted")
        ),
    }

    OUTPUT_PATH.write_text(json.dumps(metrics, indent=2))

    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()