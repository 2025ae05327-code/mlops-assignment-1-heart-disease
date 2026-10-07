"""Shared paths and feature definitions."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "processed.cleveland.data"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "heart_disease_cleaned.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EDA_ARTIFACTS_DIR = ARTIFACTS_DIR / "eda"
TRAINING_ARTIFACTS_DIR = ARTIFACTS_DIR / "training"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "heart_disease_pipeline.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

FEATURE_COLUMNS = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
)
TARGET_COLUMN = "target"
RAW_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)

NUMERIC_FEATURES = ("age", "trestbps", "chol", "thalach", "oldpeak")
CATEGORICAL_FEATURES = tuple(
    column for column in FEATURE_COLUMNS if column not in NUMERIC_FEATURES
)

NUMERIC_RANGES = {
    "age": (1, 120),
    "trestbps": (50, 300),
    "chol": (50, 700),
    "thalach": (40, 250),
    "oldpeak": (0, 10),
    "ca": (0, 3),
}
CATEGORY_VALUES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}


def configured_model_path() -> Path:
    """Return the model path, allowing container/runtime override."""

    return Path(os.getenv("MODEL_PATH", str(MODEL_PATH))).expanduser().resolve()


def prediction_threshold() -> float:
    """Return the configurable binary decision threshold."""

    threshold = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
    if not 0 < threshold < 1:
        raise ValueError("PREDICTION_THRESHOLD must be between 0 and 1")
    return threshold
