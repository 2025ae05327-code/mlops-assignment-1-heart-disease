from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heart_disease_mlops.features import build_preprocessor, split_features_target
from heart_disease_mlops.modeling import build_candidates, classification_metrics, tuning_grids


def test_preprocessor_imputes_and_encodes(synthetic_frame: pd.DataFrame) -> None:
    features, _ = split_features_target(synthetic_frame)
    transformed = build_preprocessor().fit_transform(features)

    assert transformed.shape[0] == len(synthetic_frame)
    assert transformed.shape[1] > features.shape[1]
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest"])
def test_candidate_models_return_probabilities(
    synthetic_frame: pd.DataFrame, model_name: str
) -> None:
    features, target = split_features_target(synthetic_frame)
    estimator = build_candidates()[model_name].fit(features, target)
    probabilities = estimator.predict_proba(features)
    metrics = classification_metrics(estimator, features, target)

    assert probabilities.shape == (len(features), 2)
    assert set(metrics) == {"accuracy", "precision", "recall", "roc_auc"}
    assert all(0 <= score <= 1 for score in metrics.values())


def test_quick_grids_cover_both_models() -> None:
    grids = tuning_grids(quick=True)
    assert set(grids) == {"logistic_regression", "random_forest"}
    assert all(all(len(values) == 1 for values in grid.values()) for grid in grids.values())
