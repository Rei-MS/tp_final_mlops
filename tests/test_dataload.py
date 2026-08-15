import pandas as pd

from src.dataload import get_features_targets


def test_target_encoding():
    df = pd.DataFrame(
        {
            "Age": [20, 40],
            "satisfaction": ["satisfied", "neutral or dissatisfied"],
        }
    )

    X, y = get_features_targets(df)

    assert list(y) == [1, 0]
    assert "satisfaction" not in X.columns


def test_train_test_columns_are_aligned():
    train_df = pd.DataFrame(
        {
            "Gender": ["Female", "Male", "Female"],
            "Age": [20, 40, 30],
            "satisfaction": ["satisfied", "neutral or dissatisfied", "satisfied"],
        }
    )

    test_df = pd.DataFrame(
        {"Gender": ["Female"], "Age": [25], "satisfaction": ["satisfied"]}
    )

    X_train, _ = get_features_targets(train_df)
    X_test, _ = get_features_targets(test_df, feature_columns=X_train.columns)
    assert list(X_train.columns) == list(X_test.columns)
