"""Airline dataset loading and processing."""

from pathlib import Path

import pandas as pd

from src.config import (
    DROP_COLUMNS,
    DROP_NA_COLUMNS,
    TARGET_COLUMN,
    TARGET_POS_LABEL,
)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Loads the airline satisfaction dataset and applies basic cleanup.

    Reads a CSV file, drops redundant identifiers, and removes rows with
    missing arrival delay values.

    Args:
        path: Path to the CSV file. Accepts a string or a pathlib.Path object.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    return (
        pd.read_csv(path, index_col=0)
        .drop(columns=DROP_COLUMNS, errors="ignore")
        .dropna(subset=DROP_NA_COLUMNS)
    )


def get_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separates features and target, and one-hot encodes categorical variables.

    Args:
        df: Input DataFrame containing features and 'satisfaction' target column.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - X (pd.DataFrame): Processed feature matrix with dummy-encoded
                                categorical columns.
            - y (pd.Series): Binary target vector (1 for satisfied, 0 otherwise).

    Notes:
        Why do this is Random Forest algorithm accepts categorical features?
                Scikit-learn's implementation can't yet:
                https://github.com/scikit-learn/scikit-learn/pull/29437
                July 2026 Update: But will soon!
                https://github.com/scikit-learn/scikit-learn/pull/33354
                We will stick to our old implementation since we know it works already.
    """
    y = df[TARGET_COLUMN].eq(TARGET_POS_LABEL).astype(int)
    X = pd.get_dummies(df.drop(columns=[TARGET_COLUMN]), drop_first=True, dtype=int)

    return X, y
