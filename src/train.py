"""Random Forest training."""

from typing import Any

import pandas as pd

import optuna

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from src.config import (
    RANDOM_STATE,
    MODEL_NAME,
)
from src.metrics import get_metrics
from src.tracking import log_figs, log_features


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_trials: int = 30,
) -> dict[str, Any]:
    """
    Trains a Random Forest classifier using Optuna for hyperparameter optimization.

    Registers results with MLflow, with each Optuna trial as a child run.

    Args:
        X_train: Training feature matrix.
        y_train: Training target labels.
        X_test: Testing feature matrix.
        y_test: Testing target labels.
        n_trials: Number of Optuna optimization trials to execute. Defaults to 30.

    Returns:
        dict: Dictionary containing training run data.
    """
    with mlflow.start_run(
        run_name="RandomForest",
        tags={"model": "random_forest"},
    ) as parent_run:
        parent_id = parent_run.info.run_id

        # -------------------
        # Optuna Optimization
        # -------------------

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=10),
                "criterion": trial.suggest_categorical(
                    "criterion", ["gini", "entropy"]
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical(
                    "max_features",
                    ["sqrt", "log2"],
                ),
            }

            with mlflow.start_run(
                run_name=f"trial_{trial.number}",
                nested=True,
                tags={
                    "trial_number": str(trial.number),
                    "parent_run_id": parent_id,
                },
            ):
                mlflow.log_params(params)
                model = RandomForestClassifier(
                    **params, random_state=RANDOM_STATE, n_jobs=-1
                )
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    scoring="f1",
                    cv=3,
                    n_jobs=1,
                )

                score = float(scores.mean())
                score_std = float(scores.std())

                mlflow.log_metric("f1_cv_mean", score)
                mlflow.log_metric("f1_cv_std", score_std)
            return score

        sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)

        study = optuna.create_study(
            direction="maximize",
            study_name="random_forest_optimization",
            sampler=sampler
        )
        
        study.optimize(
            objective,
            n_trials=n_trials,
            gc_after_trial=True,
            show_progress_bar=False,
        )

        # ---------------------------
        # Evaluation & MLflow Logging
        # ---------------------------

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_f1", float(study.best_value))

        best_model = RandomForestClassifier(
            **study.best_params, random_state=RANDOM_STATE, n_jobs=-1
        )
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        y_score = best_model.predict_proba(X_test)[:, 1]

        metrics = get_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)
        log_figs(best_model, X_train, y_test, y_pred, y_score)

        input_example = X_train.head(3).copy()
        signature = infer_signature(
            model_input=X_train,
            model_output=best_model.predict(X_train),
        )
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=input_example,
            signature=signature,
        )
        log_features(list(X_train.columns))

        return {
            "run_id": parent_id,
            "f1_score": metrics["test_f1"],
        }
