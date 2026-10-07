"""Generate reproducible exploratory analysis artifacts from the cleaned dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heart_disease_mlops.config import (
    EDA_ARTIFACTS_DIR,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
)
from heart_disease_mlops.data import dataset_summary

plt.switch_backend("Agg")

COLORS = {0: "#287271", 1: "#d1495b"}


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    required = {*FEATURE_COLUMNS, TARGET_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Processed dataset is missing columns: {sorted(missing)}")
    return frame


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def generate_eda(
    data_path: Path = PROCESSED_DATA_PATH,
    output_dir: Path = EDA_ARTIFACTS_DIR,
) -> list[Path]:
    """Create class, distribution, missingness, and correlation visualizations."""

    frame = load_processed_data(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    generated: list[Path] = []

    counts = frame[TARGET_COLUMN].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    bars = axis.bar(
        ["No disease", "Disease"],
        counts.reindex([0, 1], fill_value=0),
        color=[COLORS[0], COLORS[1]],
        width=0.58,
    )
    axis.set(title="Heart Disease Class Balance", xlabel="Clinical outcome", ylabel="Patients")
    axis.bar_label(
        bars,
        labels=[f"{value} ({value / len(frame):.1%})" for value in bars.datavalues],
        padding=5,
    )
    axis.set_ylim(0, max(counts) * 1.2)
    path = output_dir / "class_balance.png"
    _save_figure(figure, path)
    generated.append(path)

    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for axis, column in zip(axes.flat, NUMERIC_FEATURES, strict=False):
        sns.histplot(
            data=frame,
            x=column,
            hue=TARGET_COLUMN,
            hue_order=[0, 1],
            palette=COLORS,
            bins=18,
            element="step",
            stat="density",
            common_norm=False,
            alpha=0.32,
            ax=axis,
        )
        axis.set_title(column.replace("trestbps", "resting blood pressure").title())
    axes.flat[-1].axis("off")
    figure.suptitle("Numeric Feature Distributions by Outcome", fontsize=16, y=1.01)
    figure.tight_layout()
    path = output_dir / "numeric_distributions.png"
    _save_figure(figure, path)
    generated.append(path)

    correlations = frame.corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        correlations,
        cmap=sns.diverging_palette(210, 18, as_cmap=True),
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.35,
        cbar_kws={"label": "Pearson correlation", "shrink": 0.8},
        ax=axis,
    )
    axis.set_title("Feature and Target Correlation Matrix", pad=14)
    path = output_dir / "correlation_heatmap.png"
    _save_figure(figure, path)
    generated.append(path)

    missing_percent = frame.isna().mean().mul(100).sort_values(ascending=False)
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=missing_percent.index,
        y=missing_percent.values,
        color="#457b9d",
        ax=axis,
    )
    axis.set(
        title="Missing Values After Type Normalization",
        xlabel="Feature",
        ylabel="Missing values (%)",
    )
    axis.tick_params(axis="x", rotation=45)
    axis.set_ylim(0, max(5.0, float(missing_percent.max()) * 1.25))
    figure.tight_layout()
    path = output_dir / "missing_values.png"
    _save_figure(figure, path)
    generated.append(path)

    summary = dataset_summary(frame)
    summary["numeric_statistics"] = {
        column: {
            statistic: round(float(value), 4)
            for statistic, value in frame[column].describe().items()
        }
        for column in NUMERIC_FEATURES
    }
    summary_path = output_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    generated.append(summary_path)
    return generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=EDA_ARTIFACTS_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generated = generate_eda(args.data, args.output_dir)
    print("Generated EDA artifacts:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
