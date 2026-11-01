from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "iris.csv"

EXPECTED_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
]

EXPECTED_SPECIES = {"setosa", "versicolor", "virginica"}

# Broad, reasonable bounds for Iris measurements.
FEATURE_RANGES = {
    "sepal_length": (4.0, 8.0),
    "sepal_width": (1.8, 4.6),
    "petal_length": (0.8, 7.0),
    "petal_width": (0.0, 2.6),
}


def load_data():
    return pd.read_csv(DATA_PATH)


def test_dataset_exists():
    assert DATA_PATH.exists(), "Iris dataset file does not exist"


def test_dataset_is_not_empty():
    df = load_data()
    assert not df.empty, "Dataset is empty"


def test_dataset_schema():
    df = load_data()
    assert df.columns.tolist() == EXPECTED_COLUMNS


def test_no_missing_values():
    df = load_data()
    assert not df.isnull().any().any(), "Dataset contains missing values"


def test_feature_columns_are_numeric():
    df = load_data()

    for column in FEATURE_RANGES:
        assert pd.api.types.is_numeric_dtype(df[column]), (
            f"{column} must contain numeric values"
        )


def test_species_is_categorical():
    df = load_data()
    assert df["species"].dtype == "object"


def test_species_labels_are_valid():
    df = load_data()
    actual_species = set(df["species"].unique())

    assert actual_species == EXPECTED_SPECIES


def test_feature_values_are_within_reasonable_ranges():
    df = load_data()

    for column, (minimum, maximum) in FEATURE_RANGES.items():
        assert df[column].between(minimum, maximum).all(), (
            f"{column} contains values outside the expected range"
        )