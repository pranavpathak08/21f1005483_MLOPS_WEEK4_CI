from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("/home/jupyter/21f1005483_MLOPS_WEEKLY_ASSIGNMENT/data/iris.csv")
FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET = "species"


def augment_data(additional_rows: int, seed: int) -> None:
    df = pd.read_csv(DATA_PATH)

    rng = np.random.default_rng(seed)

    # Sample existing rows so all species remain represented.
    sampled_indices = rng.integers(0, len(df), size=additional_rows)
    new_rows = df.iloc[sampled_indices].copy().reset_index(drop=True)

    # Add small feature noise to simulate new observations.
    noise = rng.normal(
        loc=0.0,
        scale=0.08,
        size=(additional_rows, len(FEATURES)),
    )

    new_rows[FEATURES] = (
        new_rows[FEATURES].to_numpy(dtype=float) + noise
    )

    # Avoid physically impossible negative measurements.
    new_rows[FEATURES] = new_rows[FEATURES].clip(lower=0.01)

    updated = pd.concat([df, new_rows], ignore_index=True)
    updated.to_csv(DATA_PATH, index=False)

    print(f"Previous rows: {len(df)}")
    print(f"Added rows: {additional_rows}")
    print(f"Updated rows: {len(updated)}")
    print(f"Saved dataset to: {DATA_PATH}")


if __name__ == "__main__":
    augment_data(additional_rows=30, seed=42)