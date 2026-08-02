"""MLflow tracking helper functions."""

import os
import json
import tempfile

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow

from src.metrics import (
    get_fig_feature_importance,
    get_fig_confusion_matrix,
    get_fig_roc_curve,
    get_fig_pr_curve,
    get_fig_calibration_curve,
    get_fig_prediction_distribution,
)


def log_features(features: list[str]) -> None:
    """Helper function to log features to MLflow."""
    with tempfile.TemporaryDirectory() as temp_dir:
        features_path = os.path.join(temp_dir, "features.json")
        with open(features_path, "w") as f:
            json.dump(features, f)
        mlflow.log_artifact(features_path)


def log_figs(
    best_model: Any,
    X_train: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> None:
    """Helper function to log figures to MLflow."""
    figs = {
        "plots/feature_importance.png": get_fig_feature_importance(
            features=list(X_train.columns),
            importances=best_model.feature_importances_,
        ),
        "plots/confusion_matrix.png": get_fig_confusion_matrix(y_true, y_pred),
        "plots/roc_curve.png": get_fig_roc_curve(y_true, y_score),
        "plots/precision_recall_curve.png": get_fig_pr_curve(y_true, y_score),
        "plots/calibration_curve.png": get_fig_calibration_curve(y_true, y_score),
        "plots/prediction_distribution.png": get_fig_prediction_distribution(
            y_true, y_score
        ),
    }

    for fig_path, fig in figs.items():
        mlflow.log_figure(fig, fig_path)
        plt.close(fig)
