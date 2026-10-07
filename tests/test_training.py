from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from heart_disease_mlops.train import train_and_select


def test_training_tracks_both_models_and_packages_winner(
    synthetic_frame: pd.DataFrame, tmp_path: Path
) -> None:
    data_path = tmp_path / "heart.csv"
    model_path = tmp_path / "models" / "pipeline.joblib"
    metadata_path = tmp_path / "models" / "metadata.json"
    synthetic_frame.to_csv(data_path, index=False)

    metadata = train_and_select(
        data_path=data_path,
        model_output=model_path,
        metadata_output=metadata_path,
        artifact_dir=tmp_path / "artifacts",
        tracking_uri=tmp_path / "mlruns",
        experiment_name="pytest-heart-disease",
        cv_folds=2,
        test_size=0.25,
        n_jobs=1,
        quick=True,
    )

    assert model_path.is_file()
    assert metadata_path.is_file()
    assert {item["model_name"] for item in metadata["candidates"]} == {
        "logistic_regression",
        "random_forest",
    }
    assert metadata["selected_model"] in {"logistic_regression", "random_forest"}
    assert joblib.load(model_path).predict_proba(synthetic_frame.head(1)).shape == (1, 2)
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["selected_run_id"]
