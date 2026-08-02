"""Main training execution pipeline."""

import mlflow

from src.config import (
    TRAIN_PATH,
    TEST_PATH,
    EXPERIMENT_NAME,
    MODEL_NAME,
    TRACKING_URI,
    OPTUNA_TRIALS,
)
from src.dataload import load_dataset, get_features_targets
from src.registry import assign_champion_alias
from src.train import train_random_forest


def main() -> None:
    """Executes the complete training, evaluation, and deployment workflow."""
    print("=" * 60)
    print("Random Forest Training")
    print(f"Train: {TRAIN_PATH}")
    print(f"Test: {TEST_PATH}")
    print(f"MLflow: {TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df_train = load_dataset(TRAIN_PATH)
    df_test = load_dataset(TEST_PATH)

    X_train, y_train = get_features_targets(df_train)
    X_test, y_test = get_features_targets(df_test)

    result = train_random_forest(X_train, y_train, X_test, y_test, OPTUNA_TRIALS)

    version = assign_champion_alias(run_id=result["run_id"])

    print("=" * 60)
    print("Training Successful")
    print(f"Registered Version: {version}")
    print(f"FastAPI URI: models:/{MODEL_NAME}@champion")
    print("=" * 60)


if __name__ == "__main__":
    main()
