"""
Entrenamiento de Random Forest para predecir la satisfacción
de pasajeros de una aerolínea.

Flujo:
1.Carga el dataset.
2.Selecciona una muestra pequeña.
3.Divide los datos en entrenamiento y prueba.
4.Optimiza hiperparámetros con Optuna.
5.Registra trials y métricas en MLflow.
6.Entrena el mejor modelo.
7.Registra el modelo en MLflow Model Registry.
8.Asigna el alias "champion" a la versión registrada.
"""

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd

from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline


# ============================================================
# Configuración general
# ============================================================

RANDOM_STATE = 42

# Raíz del proyecto:
# tp_final_mlops/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ruta predeterminada:
# tp_final_mlops/datasets/aerolineas/train.csv
DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "aerolineas"
    / "train.csv"
)

DATA_PATH = Path(
    os.getenv(
        "DATA_PATH",
        str(DEFAULT_DATA_PATH),
    )
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "airline-satisfaction",
)

MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "airline-satisfaction-random-forest",
)

# Tamaño de la muestra de la base train
SAMPLE_SIZE = int(
    os.getenv(
        "DATASET_SAMPLE_SIZE",
        "5000",
    )
)

# Cantidad de pruebas de Optuna.
N_TRIALS = int(
    os.getenv(
        "OPTUNA_N_TRIALS",
        "10",
    )
)

TARGET = "satisfaction"

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


# ============================================================
# Carga y validación del dataset
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Carga el CSV y verifica que existan las columnas necesarias.

    Returns
    -------
    pd.DataFrame
        Dataset cargado.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "\nNo se encontró el dataset.\n"
            f"Ruta buscada: {DATA_PATH}\n"
            "Comprueba que exista:\n"
            "datasets/aerolineas/train.csv"
        )

    print("=" * 60)
    print("Cargando dataset")
    print(f"Ruta: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Faltan columnas necesarias en el dataset: "
            f"{missing_columns}"
        )

    print(f"Filas originales: {len(df)}")
    print(f"Columnas originales: {len(df.columns)}")
    print("=" * 60)

    return df


def prepare_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Prepara las variables predictoras y la variable objetivo.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test.
    """

    df = load_dataset()

    # Muestra de la base train
    actual_sample_size = min(
        SAMPLE_SIZE,
        len(df),
    )

    df = df.sample(
        n=actual_sample_size,
        random_state=RANDOM_STATE,
    ).copy()

    X = df[FEATURES].copy()

    y = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .map(
            {
                "neutral or dissatisfied": 0,
                "satisfied": 1,
            }
        )
    )

    if y.isna().any():
        invalid_values = (
            df.loc[y.isna(), TARGET]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "Se encontraron categorías no reconocidas en "
            f"'{TARGET}': {invalid_values}"
        )

    y = y.astype(int)

    print("Distribución de la variable objetivo:")
    print(y.value_counts())
    print()
    print("Porcentajes:")
    print(y.value_counts(normalize=True).round(4))
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Muestra utilizada: {actual_sample_size}")
    print(f"Registros de entrenamiento: {len(X_train)}")
    print(f"Registros de prueba: {len(X_test)}")
    print(f"Cantidad de variables: {len(FEATURES)}")
    print("=" * 60)

    return X_train, X_test, y_train, y_test


# ============================================================
# Creación del modelo : Random Forest
# ============================================================

def build_model(
    params: dict[str, Any],
) -> Pipeline:
    """
    Construye el pipeline de imputación y Random Forest.

    Random Forest no necesita escalado, pero sí es necesario
    tratar los valores nulos de Arrival Delay in Minutes.

    Parameters
    ----------
    params : dict
        Hiperparámetros del Random Forest.

    Returns
    -------
    Pipeline
        Pipeline de Scikit-learn.
    """

    classifier = RandomForestClassifier(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


# ============================================================
# Métricas
# ============================================================

def compute_metrics(
    y_true: pd.Series,
    y_pred: Any,
    y_probability: Any,
) -> dict[str, float]:
    """
    Calcula las métricas finales del modelo.
    """

    return {
        "test_accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "test_f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "test_precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "test_recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "test_roc_auc": float(
            roc_auc_score(
                y_true,
                y_probability,
            )
        ),
    }


# ============================================================
# Gráficos
# ============================================================

def build_feature_importance_figure(
    model: Pipeline,
):
    """
    Construye un gráfico con la importancia de las variables.
    """

    classifier = model.named_steps["classifier"]

    importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": classifier.feature_importances_,
        }
    ).sort_values(
        by="importance",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        importance_df["feature"],
        importance_df["importance"],
    )

    ax.set_title(
        "Random Forest - Importancia de variables"
    )
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Variable")

    fig.tight_layout()

    return fig


def build_confusion_matrix_figure(
    y_true: pd.Series,
    y_pred: Any,
):
    """
    Construye la matriz de confusión.
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=[
            "No satisfecho",
            "Satisfecho",
        ],
        ax=ax,
    )

    ax.set_title(
        "Matriz de confusión - Random Forest"
    )

    fig.tight_layout()

    return fig


def build_roc_curve_figure(
    y_true: pd.Series,
    y_probability: Any,
):
    """
    Construye la curva ROC.
    """

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true,
        y_probability,
    )

    roc_auc = auc(
        false_positive_rate,
        true_positive_rate,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC AUC = {roc_auc:.4f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Clasificador aleatorio",
    )

    ax.set_title("Curva ROC - Random Forest")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.legend()

    fig.tight_layout()

    return fig


def build_precision_recall_figure(
    y_true: pd.Series,
    y_probability: Any,
):
    """
    Construye la curva Precision-Recall.
    """

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_probability,
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        recall,
        precision,
    )

    ax.set_title(
        "Curva Precision-Recall - Random Forest"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    fig.tight_layout()

    return fig


def log_figures(
    model: Pipeline,
    y_test: pd.Series,
    y_pred: Any,
    y_probability: Any,
) -> None:
    """
    Registra todos los gráficos en MLflow.
    """

    confusion_matrix_figure = (
        build_confusion_matrix_figure(
            y_test,
            y_pred,
        )
    )

    mlflow.log_figure(
        confusion_matrix_figure,
        "plots/confusion_matrix.png",
    )

    plt.close(confusion_matrix_figure)

    feature_importance_figure = (
        build_feature_importance_figure(model)
    )

    mlflow.log_figure(
        feature_importance_figure,
        "plots/feature_importance.png",
    )

    plt.close(feature_importance_figure)

    roc_figure = build_roc_curve_figure(
        y_test,
        y_probability,
    )

    mlflow.log_figure(
        roc_figure,
        "plots/roc_curve.png",
    )

    plt.close(roc_figure)

    precision_recall_figure = (
        build_precision_recall_figure(
            y_test,
            y_probability,
        )
    )

    mlflow.log_figure(
        precision_recall_figure,
        "plots/precision_recall_curve.png",
    )

    plt.close(precision_recall_figure)


# ============================================================
# Entrenamiento con Optuna
# ============================================================

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_trials: int,
) -> dict[str, Any]:
    """
    Optimiza los hiperparámetros con Optuna y registra
    los resultados en MLflow.

    Cada trial de Optuna se guarda como un run hijo.
    """

    with mlflow.start_run(
        run_name="RandomForest",
        tags={
            "model": "random_forest",
            "dataset": "airline_passenger_satisfaction",
            "training_type": "optuna",
        },
    ) as parent_run:

        parent_run_id = parent_run.info.run_id

        mlflow.log_params(
            {
                "n_trials": n_trials,
                "random_state": RANDOM_STATE,
                "training_rows": len(X_train),
                "test_rows": len(X_test),
                "number_features": len(FEATURES),
                "sample_size": len(X_train) + len(X_test),
                "cv_folds": 3,
                "scoring": "f1",
            }
        )

        mlflow.log_dict(
            {"features": FEATURES},
            "metadata/features.json",
        )

        cross_validation = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        def objective(
            trial: optuna.Trial,
        ) -> float:
            """
            Función objetivo que Optuna intenta maximizar.
            """

            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    50,
                    250,
                    step=50,
                ),
                "criterion": trial.suggest_categorical(
                    "criterion",
                    ["gini", "entropy"],
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    3,
                    20,
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split",
                    2,
                    15,
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    1,
                    10,
                ),
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
                    "parent_run_id": parent_run_id,
                },
            ):

                mlflow.log_params(params)

                trial_model = build_model(params)

                scores = cross_val_score(
                    estimator=trial_model,
                    X=X_train,
                    y=y_train,
                    scoring="f1",
                    cv=cross_validation,
                    n_jobs=1,
                    error_score="raise",
                )

                f1_mean = float(scores.mean())
                f1_std = float(scores.std())

                mlflow.log_metric(
                    "f1_cv_mean",
                    f1_mean,
                )

                mlflow.log_metric(
                    "f1_cv_std",
                    f1_std,
                )

                for fold_number, fold_score in enumerate(
                    scores,
                    start=1,
                ):
                    mlflow.log_metric(
                        f"f1_fold_{fold_number}",
                        float(fold_score),
                    )

                print(
                    f"Trial {trial.number}: "
                    f"F1 promedio = {f1_mean:.4f}"
                )

            return f1_mean

        sampler = optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        )

        study = optuna.create_study(
            direction="maximize",
            study_name="random_forest_optimization",
            sampler=sampler,
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            gc_after_trial=True,
            show_progress_bar=False,
        )

        best_params = study.best_params

        print("=" * 60)
        print("Mejores hiperparámetros encontrados")
        print(best_params)
        print(
            f"Mejor F1 de validación cruzada: "
            f"{study.best_value:.4f}"
        )
        print("=" * 60)

        mlflow.log_params(
            {
                f"best_{key}": value
                for key, value in best_params.items()
            }
        )

        mlflow.log_metric(
            "best_f1_cv",
            float(study.best_value),
        )

        # Entrenamiento train
        best_model = build_model(best_params)

        best_model.fit(
            X_train,
            y_train,
        )

        y_pred = best_model.predict(X_test)

        y_probability = best_model.predict_proba(
            X_test
        )[:, 1]

        metrics = compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_probability=y_probability,
        )

        mlflow.log_metrics(metrics)

        log_figures(
            model=best_model,
            y_test=y_test,
            y_pred=y_pred,
            y_probability=y_probability,
        )

        input_example = X_train.head(3).copy()

        signature = infer_signature(
            model_input=X_train,
            model_output=best_model.predict(X_train),
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=input_example,
        )

        print("Métricas finales:")
        for metric_name, metric_value in metrics.items():
            print(
                f"  {metric_name}: "
                f"{metric_value:.4f}"
            )

        print(f"Run principal: {parent_run_id}")
        print(f"Modelo registrado: {MODEL_NAME}")
        print(f"URI generada: {model_info.model_uri}")

    return {
        "run_id": parent_run_id,
        "best_params": best_params,
        "best_f1_cv": float(study.best_value),
        "metrics": metrics,
    }


# ============================================================
# Alias del modelo
# ============================================================

def assign_alias_to_run_version(
    run_id: str,
    alias: str = "champion",
) -> str:
    """
    Busca la versión creada por el run actual y le asigna
    el alias indicado.

    Esto es más seguro que seleccionar simplemente la última
    versión existente.
    """

    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI,
    )

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    matching_versions = [
        version
        for version in versions
        if version.run_id == run_id
    ]

    if not matching_versions:
        raise RuntimeError(
            "El modelo se registró, pero no se encontró "
            f"una versión asociada al run {run_id}."
        )

    registered_version = max(
        matching_versions,
        key=lambda version: int(version.version),
    )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=alias,
        version=registered_version.version,
    )

    print(
        f"Alias '{alias}' asignado al modelo "
        f"'{MODEL_NAME}', versión "
        f"{registered_version.version}."
    )

    return str(registered_version.version)


# ============================================================
# Ejecución principal
# ============================================================

def main() -> None:
    """
    Ejecuta el entrenamiento completo.
    """

    print("=" * 60)
    print("Inicio del entrenamiento")
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experimento: {EXPERIMENT_NAME}")
    print(f"Modelo registrado: {MODEL_NAME}")
    print(f"Trials de Optuna: {N_TRIALS}")
    print("=" * 60)

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    X_train, X_test, y_train, y_test = prepare_data()

    result = train_random_forest(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_trials=N_TRIALS,
    )

    version = assign_alias_to_run_version(
        run_id=result["run_id"],
        alias="champion",
    )

    print("=" * 60)
    print("Proceso finalizado correctamente")
    print(f"Versión registrada: {version}")
    print(
        "URI para la API: "
        f"models:/{MODEL_NAME}@champion"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()