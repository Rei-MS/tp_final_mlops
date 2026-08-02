"""Metrics and Related Figures."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    auc, 
    roc_curve,
    precision_recall_curve,
)

from src.config import DISPLAY_LABELS


METRICS = {
    "test_precision": precision_score,
    "test_accuracy": accuracy_score,
    "test_recall": recall_score,
    "test_f1": f1_score,
}


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculates training metrics.

        Args:
        y_true: Array of ground truth values.
        y_pred: Array of predictions.

    Returns:
        dict: Dictionary containing training metrics.
    """
    return {name: round(metric(y_true, y_pred), 3) for name, metric in METRICS.items()}


def get_fig_feature_importance(
    features: list[str],
    importances: np.ndarray,
    top_n: int = 10,
) -> plt.Figure:
    """Creates a horizontal bar plot of feature importances using Seaborn.

        Extracts the top `top_n` most important features and returns a horizontal
        bar chart figure of features sorted by importance.

    Args:
        features: List of feature names.
        importances: Array of numerical importance scores.
        top_n: Number of top features to display in the plot. Defaults to 10.

    Returns:
        plt.Figure: Feature importance plot as a Matplotlib figure object.
    """
    sns.set_theme(style="whitegrid")  # Seaborn theme

    importance_df = pd.DataFrame({"feature": features, "value": importances}).nlargest(
        top_n, "value"
    )  # Filter and sort top N features

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=importance_df,
        x="value",
        y="feature",
        color="#2E86AB",
        ax=ax,
    )  # Seaborn auto sorts them.

    ax.set_title("Feature Importance", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    sns.despine(left=True, bottom=True)
    fig.tight_layout()

    return fig


def get_fig_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    display_labels: list[str] | None = None,
) -> plt.Figure:
    """Creates the confusion matrix figure.

    Args:
        y_true: Array of ground truth values.
        y_pred: Array of predictions.
        display_labels: Class names for labels. Defaults to neutral/satisfied.

    Returns:
        plt.Figure: Confusion matrix figure as a Matplotlib figure object.
    """
    if display_labels is None:
        display_labels = DISPLAY_LABELS

    sns.set_theme(style="white")  # Seaborn theme

    cm = confusion_matrix(y_true, y_pred)  # Raw confusion matrix

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.light_palette("#2E86AB", as_cmap=True),
        cbar=False,
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
        annot_kws={"size": 11, "weight": "bold"},
    )

    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Ground Truths")

    sns.despine(left=True, bottom=True)
    fig.tight_layout()

    return fig


def get_fig_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> plt.Figure:
    """Creates the ROC curve figure.

    Args:
        y_true: Array of ground truth values.
        y_score: Predicted probabilities or decision scores for the positive
                 class.

    Returns:
        plt.Figure: ROC curve as a Matplotlib figure object.
    """
    sns.set_theme(style="whitegrid")  # Seaborn theme

    # ROC curve metrics (true positive rate, false positive rate) and AUC score.
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr,
        tpr,
        color="#2E86AB",
        linewidth=2.5,
        label=f"ROC Curve (AUC = {roc_auc:.3f})",
    )
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2E86AB")
    ax.plot(
        [0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5
    )  # Diag baseline

    ax.set_title("ROC Curve", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")

    sns.despine(left=True, bottom=True)
    fig.tight_layout()

    return fig


def get_fig_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> plt.Figure:
    """Creates the Precision-Recall curve figure.

    Args:
        y_true: Array of ground truth values.
        y_score: Predicted probabilities or decision scores for the positive
                 class.

    Returns:
        plt.Figure: Precision-Recall curve as a Matplotlib figure object.
    """
    sns.set_theme(style="whitegrid")  # Seaborn theme

    # PR curve metrics and PR-AUC score
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    baseline = np.mean(y_true)  # Random baseline

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        recall,
        precision,
        color="#2E86AB",
        linewidth=2.5,
        label=f"PR Curve (AUC = {pr_auc:.3f})",
    )
    ax.fill_between(recall, precision, alpha=0.15, color="#2E86AB")
    ax.axhline(
        y=baseline,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline ({baseline:.2f})",
    )  # horizontal baseline (random classifier performance)

    ax.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none")

    sns.despine(left=True, bottom=True)
    fig.tight_layout()

    return fig


def get_fig_calibration_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> plt.Figure:
    """Creates the probability calibration curve figure.

    Args:
        y_true: Array of ground truth values.
        y_score: Predicted probabilities or decision scores for the positive
                 class.
        n_bins: Number of bins in the plot. Defaults to 10.

    Returns:
        plt.Figure: Calibration curve as a Matplotlib figure object.
    """
    sns.set_theme(style="whitegrid")  # Seaborn theme

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        prob_pred, prob_true, marker="o", color="#2E86AB", linewidth=2.5, label="Model"
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Perfect Calibration",
    )

    ax.set_title("Calibration Curve", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")

    sns.despine(left=True, bottom=True)
    fig.tight_layout()

    return fig


def get_fig_prediction_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    display_labels: list[str] | None = None,
) -> plt.Figure:
    """Creates the prediction probability distribution by class figure.

    Args:
        y_true: Array of ground truth values.
        y_score: Predicted probabilities or decision scores for the positive
                 class.
        display_labels: Class names for labels. Defaults to neutral/satisfied.

    Returns:
        plt.Figure: Prediction probability distribution as a Matplotlib figure
                    object.
    """
    sns.set_theme(style="whitegrid")  # Seaborn theme

    if display_labels is None:
        display_labels = DISPLAY_LABELS
    mapped_labels = np.array(display_labels)[y_true]

    df = pd.DataFrame(
        {
            "Class": mapped_labels,
            "Probability": y_score,
        }
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.kdeplot(
        data=df,
        x="Probability",
        hue="Class",
        hue_order=display_labels,
        common_norm=False,
        fill=True,
        palette=["#E71D36", "#2E86AB"],
        alpha=0.3,
        linewidth=2,
        ax=ax,
    )

    ax.set_title(
        "Prediction Probability Distribution", fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")

    sns.despine(left=True, bottom=True)
    fig.tight_layout()

    return fig
    