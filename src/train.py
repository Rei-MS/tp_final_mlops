import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

DATA_PATH = Path(__file__).resolve().parent.parent / (
    "datasets/aerolineas/train.csv"
)

EXPERIMENT_NAME = "airline-satisfaction"
MODEL_NAME = "airline-satisfaction-model"

FEATURES = [
    "Age",
    "Flight Distance",
    "Inflight wifi service",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

TARGET = "satisfaction"


def train_model() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH)

    # Muestra del dataset
    sample_size = min(5000, len(df))
    df = df.sample(
        n=sample_size,
        random_state=42,
    )

    X = df[FEATURES].copy()

    y = df[TARGET].map(
        {
            "neutral or dissatisfied": 0,
            "satisfied": 1,
        }
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    random_state=42,
                ),
            ),
        ]
    )

    with mlflow.start_run(run_name="logistic-regression") as run:
        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)

        accuracy = accuracy_score(
            y_validation,
            predictions,
        )

        f1 = f1_score(
            y_validation,
            predictions,
        )

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("dataset_rows", sample_size)
        mlflow.log_param("validation_size", 0.20)
        mlflow.log_param("features", len(FEATURES))
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("validation_accuracy", accuracy)
        mlflow.log_metric("validation_f1_score", f1)

        signature = infer_signature(
            X_train,
            model.predict(X_train),
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=X_train.head(3),
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")

    # Versión reciente
    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    latest_version = max(
        versions,
        key=lambda version: int(version.version),
    )

    # Carga del alias champion
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=latest_version.version,
    )

    print(
        f"Modelo registrado como {MODEL_NAME}, "
        f"versión {latest_version.version}"
    )


if __name__ == "__main__":
    train_model()