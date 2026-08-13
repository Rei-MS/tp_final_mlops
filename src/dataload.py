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


def get_features_targets(
    df: pd.DataFrame,
    feature_columns: list[str] | pd.Index | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa features y target y realiza one-hot encoding.

    Si se proporcionan feature_columns, alinea las columnas
    resultantes con las columnas utilizadas durante train.
    """

    y = (
        df[TARGET_COLUMN]
        .eq(TARGET_POS_LABEL)
        .astype(int)
    )

    X = pd.get_dummies(
        df.drop(columns=[TARGET_COLUMN]),
        drop_first=True,
        dtype=int,
    )

    if feature_columns is not None:
        X = X.reindex(
            columns=list(feature_columns),
            fill_value=0,
        )

    return X, y
