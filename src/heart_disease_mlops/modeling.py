"""Model definitions, tuning grids, metrics, and diagnostic plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from heart_disease_mlops.features import build_preprocessor

plt.switch_backend("Agg")


def build_candidates(random_state: int = 42) -> dict[str, Pipeline]:
    """Return deterministic candidate pipelines with identical preprocessing."""

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def tuning_grids(*, quick: bool = False) -> dict[str, dict[str, list[Any]]]:
    """Return small CI grids or fuller assignment experiment grids."""

    if quick:
        return {
            "logistic_regression": {
                "classifier__C": [1.0],
                "classifier__solver": ["liblinear"],
                "classifier__penalty": ["l2"],
            },
            "random_forest": {
                "classifier__n_estimators": [100],
                "classifier__max_depth": [8],
                "classifier__min_samples_leaf": [2],
            },
        }
    return {
        "logistic_regression": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__solver": ["liblinear"],
            "classifier__penalty": ["l1", "l2"],
        },
        "random_forest": {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [None, 8, 16],
            "classifier__min_samples_leaf": [1, 3],
        },
    }


def classification_metrics(
    estimator: ClassifierMixin,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    predictions = estimator.predict(features)
    probabilities = estimator.predict_proba(features)[:, 1]
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
    }


def save_diagnostic_plots(
    estimator: ClassifierMixin,
    features: pd.DataFrame,
    target: pd.Series,
    output_dir: Path,
    model_name: str,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = estimator.predict(features)
    probabilities = estimator.predict_proba(features)[:, 1]
    generated: list[Path] = []

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        target,
        predictions,
        display_labels=["No disease", "Disease"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title(f"{model_name.replace('_', ' ').title()} Confusion Matrix")
    path = output_dir / "confusion_matrix.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    generated.append(path)

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    RocCurveDisplay.from_predictions(target, probabilities, name=model_name, ax=axis)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Chance")
    axis.set_title(f"{model_name.replace('_', ' ').title()} ROC Curve")
    axis.legend(loc="lower right")
    path = output_dir / "roc_curve.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    generated.append(path)
    return generated


def save_model_comparison(results: list[dict[str, Any]], output_path: Path) -> Path:
    """Plot cross-validation metrics used for transparent model selection."""

    records = [
        {"model": result["model_name"], "metric": metric, "score": score}
        for result in results
        for metric, score in result["cv_metrics"].items()
    ]
    comparison = pd.DataFrame(records)
    figure, axis = plt.subplots(figsize=(10, 5.8))
    for index, (model_name, group) in enumerate(comparison.groupby("model", sort=False)):
        positions = range(len(group))
        offset = (index - 0.5) * 0.34
        axis.bar(
            [position + offset for position in positions],
            group["score"],
            width=0.34,
            label=model_name.replace("_", " ").title(),
        )
    metric_order = list(results[0]["cv_metrics"])
    metric_labels = [name.replace("_", " ").title() for name in metric_order]
    axis.set_xticks(range(len(metric_order)), metric_labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean cross-validation score")
    axis.set_title("Candidate Model Comparison")
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.25)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path
