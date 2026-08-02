"""Configuration Variables."""

import os
from pathlib import Path


RANDOM_STATE = 42


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "aerolineas"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


TARGET_COLUMN = "satisfaction"
TARGET_POS_LABEL = "satisfied"
DISPLAY_LABELS = ["Neutral or Dissatisfied", "Satisfied"]

DROP_COLUMNS = ["id"]
DROP_NA_COLUMNS = ["Arrival Delay in Minutes"]


EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "airline-satisfaction")
MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "airline-satisfaction-best-random-forest",
)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

OPTUNA_TRIALS = 10