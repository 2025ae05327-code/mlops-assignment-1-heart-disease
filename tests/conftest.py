from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heart_disease_mlops.features import split_features_target
from heart_disease_mlops.modeling import build_candidates


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 80
    frame = pd.DataFrame(
        {
            "age": rng.integers(35, 78, rows),
            "sex": rng.integers(0, 2, rows),
            "cp": rng.integers(1, 5, rows),
            "trestbps": rng.integers(95, 190, rows),
            "chol": rng.integers(150, 390, rows).astype(float),
            "fbs": rng.integers(0, 2, rows),
            "restecg": rng.integers(0, 3, rows),
            "thalach": rng.integers(85, 200, rows),
            "exang": rng.integers(0, 2, rows),
            "oldpeak": rng.uniform(0, 5, rows).round(1),
            "slope": rng.integers(1, 4, rows),
            "ca": rng.integers(0, 4, rows).astype(float),
            "thal": rng.choice([3.0, 6.0, 7.0], rows),
        }
    )
    risk = (
        (frame["age"] - 50) * 0.03
        + frame["sex"] * 0.4
        + (frame["cp"] == 4) * 0.9
        - (frame["thalach"] - 145) * 0.02
        + frame["exang"] * 0.8
        + frame["oldpeak"] * 0.35
        + frame["ca"] * 0.45
        + (frame["thal"] == 7) * 0.5
    )
    frame["target"] = (risk >= risk.median()).astype(int)
    frame.loc[2, "chol"] = np.nan
    frame.loc[5, "ca"] = np.nan
    frame.loc[8, "thal"] = np.nan
    return frame


@pytest.fixture
def trained_model(synthetic_frame: pd.DataFrame):
    features, target = split_features_target(synthetic_frame)
    model = build_candidates(random_state=7)["logistic_regression"]
    return model.fit(features, target)


@pytest.fixture
def valid_payload() -> dict[str, int | float]:
    return {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }
