"""Tune two classifiers, track experiments in MLflow, and package the winner."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from heart_disease_mlops.config import (
    FEATURE_COLUMNS,
    MLRUNS_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    TRAINING_ARTIFACTS_DIR,
)
from heart_disease_mlops.features import split_features_target
from heart_disease_mlops.modeling import (
    build_candidates,
    classification_metrics,
    save_diagnostic_plots,
    save_model_comparison,
    tuning_grids,
)

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}


def _tracking_uri(value: str | Path) -> str:
    text = str(value)
    if "://" in text or text.startswith("file:"):
        return text
    path = Path(text).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path.as_uri()


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def train_and_select(
    data_path: Path = PROCESSED_DATA_PATH,
    model_output: Path = MODEL_PATH,
    metadata_output: Path = MODEL_METADATA_PATH,
    artifact_dir: Path = TRAINING_ARTIFACTS_DIR,
    tracking_uri: str | Path = MLRUNS_DIR,
    experiment_name: str = "heart-disease-classification",
    *,
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = -1,
    quick: bool = False,
) -> dict[str, Any]:
    """Train both candidates and select the highest mean CV ROC-AUC model."""

    frame = pd.read_csv(Path(data_path))
    features, target = split_features_target(frame)
    if target.value_counts().min() < cv_folds:
        raise ValueError("Each target class must contain at least cv_folds rows")
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(_tracking_uri(tracking_uri))
    mlflow.set_experiment(experiment_name)
    os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")

    candidates = build_candidates(random_state)
    grids = tuning_grids(quick=quick)
    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}

    for model_name, pipeline in candidates.items():
        started = time.perf_counter()
        search = GridSearchCV(
            pipeline,
            param_grid=grids[model_name],
            scoring=SCORING,
            refit="roc_auc",
            cv=cv,
            n_jobs=n_jobs,
            return_train_score=False,
            error_score="raise",
        )
        with mlflow.start_run(run_name=model_name) as run:
            search.fit(train_features, train_target)
            fit_seconds = time.perf_counter() - started
            best_index = int(search.best_index_)
            cv_metrics = {
                metric: float(search.cv_results_[f"mean_test_{metric}"][best_index])
                for metric in SCORING
            }
            cv_std = {
                metric: float(search.cv_results_[f"std_test_{metric}"][best_index])
                for metric in SCORING
            }
            test_metrics = classification_metrics(
                search.best_estimator_, test_features, test_target
            )
            model_artifact_dir = artifact_dir / model_name
            plots = save_diagnostic_plots(
                search.best_estimator_,
                test_features,
                test_target,
                model_artifact_dir,
                model_name,
            )

            mlflow.log_params(
                {
                    "model_name": model_name,
                    "cv_folds": cv_folds,
                    "test_size": test_size,
                    "random_state": random_state,
                    **{
                        key.replace("classifier__", ""): _json_ready(value)
                        for key, value in search.best_params_.items()
                    },
                }
            )
            mlflow.log_metrics({f"cv_{key}": value for key, value in cv_metrics.items()})
            mlflow.log_metrics({f"cv_std_{key}": value for key, value in cv_std.items()})
            mlflow.log_metrics(
                {f"test_{key}": value for key, value in test_metrics.items()}
            )
            mlflow.log_metric("fit_seconds", fit_seconds)
            for plot in plots:
                mlflow.log_artifact(str(plot), artifact_path="diagnostics")
            input_example = train_features.head(3)
            signature = infer_signature(
                input_example,
                search.best_estimator_.predict(input_example),
            )
            mlflow.sklearn.log_model(
                search.best_estimator_,
                artifact_path="model",
                input_example=input_example,
                signature=signature,
            )

            result = {
                "model_name": model_name,
                "run_id": run.info.run_id,
                "best_params": {
                    key: _json_ready(value) for key, value in search.best_params_.items()
                },
                "cv_metrics": cv_metrics,
                "cv_std": cv_std,
                "test_metrics": test_metrics,
                "fit_seconds": fit_seconds,
            }
            results.append(result)
            fitted_models[model_name] = search.best_estimator_

    selected = max(results, key=lambda result: result["cv_metrics"]["roc_auc"])
    selected_name = selected["model_name"]
    selected_model = fitted_models[selected_name]

    model_output = Path(model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_output.with_suffix(model_output.suffix + ".tmp")
    joblib.dump(selected_model, temporary_model)
    temporary_model.replace(model_output)

    comparison_path = save_model_comparison(results, artifact_dir / "model_comparison.png")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "highest mean cross-validated ROC-AUC on the training split",
        "selected_model": selected_name,
        "selected_run_id": selected["run_id"],
        "decision_threshold": 0.5,
        "features": list(FEATURE_COLUMNS),
        "training_rows": int(len(train_features)),
        "test_rows": int(len(test_features)),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "data_path": str(Path(data_path)),
        "model_path": str(model_output),
        "comparison_plot": str(comparison_path),
        "candidates": results,
    }
    metadata_output = Path(metadata_output)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (artifact_dir / "training_results.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH)
    parser.add_argument("--metadata-output", type=Path, default=MODEL_METADATA_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=TRAINING_ARTIFACTS_DIR)
    parser.add_argument("--tracking-uri", default=str(MLRUNS_DIR))
    parser.add_argument("--experiment-name", default="heart-disease-classification")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--quick", action="store_true", help="Use one setting per model")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = train_and_select(
        data_path=args.data,
        model_output=args.model_output,
        metadata_output=args.metadata_output,
        artifact_dir=args.artifact_dir,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        cv_folds=args.cv_folds,
        test_size=args.test_size,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        quick=args.quick,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
