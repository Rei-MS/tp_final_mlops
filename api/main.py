"""API para realizar predicciones de satisfacción de pasajeros."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# Configuración
# ============================================================


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "airline-satisfaction-best-random-forest",
)

MODEL_ALIAS = os.getenv(
    "MLFLOW_MODEL_ALIAS",
    "champion",
)

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

PROJECT_ROOT = Path(
    os.getenv(
        "PROJECT_ROOT",
        Path(__file__).resolve().parent.parent,
    )
)

TRAIN_PATH = Path(
    os.getenv(
        "TRAIN_PATH",
        PROJECT_ROOT / "datasets" / "aerolineas" / "train.csv",
    )
)

TARGET_COLUMN = "satisfaction"

DROP_COLUMNS = [
    "id",
]


# ============================================================
# Variables globales cargadas al iniciar FastAPI
# ============================================================


model = None

raw_feature_columns: list[str] = []

categorical_columns: list[str] = []

categorical_values: dict[str, list[str]] = {}

model_feature_columns: list[str] = []


# ============================================================
# Modelos de entrada y salida
# ============================================================


class PredictionRequest(BaseModel):
    features: dict[str, Any]


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_satisfied: float
    model_uri: str


# ============================================================
# Carga del esquema de entrenamiento
# ============================================================


def load_training_schema() -> None:
    """
    Obtiene las columnas y categorías usadas durante entrenamiento.
    """

    global raw_feature_columns
    global categorical_columns
    global categorical_values

    df = pd.read_csv(TRAIN_PATH)

    df = df.drop(
        columns=DROP_COLUMNS,
        errors="ignore",
    )

    features = df.drop(
        columns=[TARGET_COLUMN],
    )

    raw_feature_columns = features.columns.tolist()

    categorical_columns = features.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    categorical_values = {
        column: sorted(features[column].dropna().astype(str).unique().tolist())
        for column in categorical_columns
    }


# ============================================================
# Preprocesamiento para inferencia
# ============================================================


def preprocess_features(
    features: dict[str, Any],
) -> pd.DataFrame:
    """
    Convierte las variables originales al mismo formato
    utilizado por el modelo durante entrenamiento.
    """

    if model is None:
        raise RuntimeError("El modelo todavía no fue cargado.")
    missing_columns = [
        column for column in raw_feature_columns if column not in features
    ]

    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Faltan variables requeridas.",
                "missing_columns": missing_columns,
            },
        )
    unknown_columns = [
        column for column in features if column not in raw_feature_columns
    ]

    if unknown_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Se recibieron variables desconocidas.",
                "unknown_columns": unknown_columns,
            },
        )
    # Validamos valores categóricos.

    for column in categorical_columns:

        value = str(features[column])

        allowed_values = categorical_values[column]

        if value not in allowed_values:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (f"Valor inválido para '{column}'."),
                    "received": value,
                    "allowed": allowed_values,
                },
            )
    # Creamos exactamente las columnas que espera
    # el Random Forest.

    row = {column: 0 for column in model_feature_columns}

    # Variables numéricas.

    for column in raw_feature_columns:

        if column not in categorical_columns:

            if column in row:

                value = features[column]

                if value is None:
                    raise HTTPException(
                        status_code=422,
                        detail=(f"'{column}' no puede ser null."),
                    )
                row[column] = value
    # Variables categóricas codificadas mediante
    # la misma convención de pd.get_dummies().

    for column in categorical_columns:

        value = str(features[column])

        prefix = f"{column}_"

        dummy_column = prefix + value

        if dummy_column in row:
            row[dummy_column] = 1
    X = pd.DataFrame(
        [row],
        columns=model_feature_columns,
    )

    return X


# ============================================================
# Inicio de FastAPI
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):

    global model
    global model_feature_columns

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print(f"Cargando modelo desde: {MODEL_URI}")

    model = mlflow.sklearn.load_model(MODEL_URI)

    if not hasattr(
        model,
        "feature_names_in_",
    ):
        raise RuntimeError("El modelo no contiene feature_names_in_.")
    model_feature_columns = list(model.feature_names_in_)

    load_training_schema()

    print("Modelo cargado correctamente.")
    print(f"Features del modelo: " f"{len(model_feature_columns)}")

    yield


app = FastAPI(
    title="Airline Satisfaction API",
    description=(
        "API de predicción utilizando " "el modelo champion registrado en MLflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Endpoints
# ============================================================


@app.get("/")
def root():
    return {"message": ("Airline Satisfaction Prediction API")}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
):
    """
    Realiza una predicción de satisfacción.
    """

    X = preprocess_features(request.features)

    prediction = int(model.predict(X)[0])

    probability = float(model.predict_proba(X)[0][1])

    label = "satisfied" if prediction == 1 else "neutral or dissatisfied"

    return PredictionResponse(
        prediction=prediction,
        label=label,
        probability_satisfied=probability,
        model_uri=MODEL_URI,
    )
