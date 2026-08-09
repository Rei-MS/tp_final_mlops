"""DAG para entrenar el modelo de satisfacción."""

import os
from datetime import datetime

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator


PROJECT_ROOT = "/opt/airflow/project"


with DAG(
    dag_id="airline_model_training",
    description="Entrena y registra el modelo de satisfacción",
    start_date=datetime(2026, 8, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=[
        "mlops",
        "random-forest",
        "mlflow",
    ],
) as dag:

    train_model = BashOperator(
        task_id="train_random_forest",

        bash_command="python -m src.main",

        # Muy importante:
        # simula exactamente la ejecución desde el root.
        cwd=PROJECT_ROOT,

        env={
            "MLFLOW_TRACKING_URI": (
                "http://mlflow:5000"
            ),

            "MLFLOW_EXPERIMENT_NAME": (
                "airline-satisfaction-v2"
            ),

            "MLFLOW_MODEL_NAME": (
                "airline-satisfaction-"
                "best-random-forest"
            ),

            "OPTUNA_TRIALS": os.getenv(
                "OPTUNA_TRIALS",
                "10",
            ),
            "PYTHONPATH": PROJECT_ROOT,
        },

        append_env=True,
    )