from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heart_disease_mlops.config import RAW_COLUMNS
from heart_disease_mlops.data import clean_data, dataset_summary


def test_clean_data_normalizes_target_missing_and_duplicates() -> None:
    rows = [
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, "?", 3, 2],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, "?", 3, 2],
    ]
    cleaned = clean_data(pd.DataFrame(rows, columns=RAW_COLUMNS))

    assert len(cleaned) == 2
    assert cleaned["target"].tolist() == [0, 1]
    assert np.isnan(cleaned.loc[1, "ca"])
    assert dataset_summary(cleaned)["duplicate_rows"] == 0


def test_clean_data_maps_out_of_domain_values_to_missing() -> None:
    rows = [
        [200, 9, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
        [55, 0, 4, 130, 240, 0, 1, 165, 0, 0.5, 2, 1, 3, 1],
    ]
    cleaned = clean_data(pd.DataFrame(rows, columns=RAW_COLUMNS))

    assert np.isnan(cleaned.loc[0, "age"])
    assert np.isnan(cleaned.loc[0, "sex"])


def test_clean_data_rejects_missing_schema() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        clean_data(pd.DataFrame({"age": [50], "target": [0]}))
