from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from heart_disease_mlops.eda import generate_eda


def test_generate_eda_creates_expected_artifacts(
    synthetic_frame: pd.DataFrame, tmp_path: Path
) -> None:
    data_path = tmp_path / "heart.csv"
    output_dir = tmp_path / "eda"
    synthetic_frame.to_csv(data_path, index=False)

    generated = generate_eda(data_path, output_dir)
    names = {path.name for path in generated}

    assert names == {
        "class_balance.png",
        "numeric_distributions.png",
        "correlation_heatmap.png",
        "missing_values.png",
        "eda_summary.json",
    }
    summary = json.loads((output_dir / "eda_summary.json").read_text(encoding="utf-8"))
    assert summary["rows"] == len(synthetic_frame)
