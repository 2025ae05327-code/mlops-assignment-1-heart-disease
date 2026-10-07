"""Download, validate, and clean the UCI Cleveland heart disease dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from heart_disease_mlops.config import (
    CATEGORY_VALUES,
    DATASET_URL,
    FEATURE_COLUMNS,
    NUMERIC_RANGES,
    PROCESSED_DATA_PATH,
    RAW_COLUMNS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def download_raw_dataset(
    url: str = DATASET_URL,
    destination: Path = RAW_DATA_PATH,
    *,
    force: bool = False,
    attempts: int = 3,
    timeout_seconds: int = 30,
) -> Path:
    """Download the source data atomically and reuse a cached copy by default."""

    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    last_error: requests.RequestException | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            if not response.content.strip():
                raise requests.RequestException("The downloaded dataset is empty")
            temporary_path.write_bytes(response.content)
            temporary_path.replace(destination)
            return destination
        except requests.RequestException as error:
            last_error = error
            if attempt < attempts:
                time.sleep(attempt)

    raise RuntimeError(f"Unable to download dataset from {url}") from last_error


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the headerless UCI file and map question marks to missing values."""

    frame = pd.read_csv(
        Path(path),
        names=list(RAW_COLUMNS),
        header=None,
        na_values=["?", ""],
        skipinitialspace=True,
    )
    if frame.shape[1] != len(RAW_COLUMNS):
        raise ValueError(
            f"Expected {len(RAW_COLUMNS)} columns, received {frame.shape[1]} columns"
        )
    return frame


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and invalid values while keeping imputation in the ML pipeline."""

    missing_columns = set(RAW_COLUMNS).difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    cleaned = frame.loc[:, list(RAW_COLUMNS)].copy()
    for column in RAW_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        invalid = ~cleaned[column].between(minimum, maximum) & cleaned[column].notna()
        cleaned.loc[invalid, column] = np.nan

    for column, allowed_values in CATEGORY_VALUES.items():
        invalid = ~cleaned[column].isin(allowed_values) & cleaned[column].notna()
        cleaned.loc[invalid, column] = np.nan

    cleaned[TARGET_COLUMN] = (cleaned[TARGET_COLUMN].astype(int) > 0).astype(int)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("No valid rows remain after cleaning")
    if cleaned[TARGET_COLUMN].nunique() < 2:
        raise ValueError("The cleaned dataset must contain both target classes")
    return cleaned


def dataset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Build a JSON-serializable data-quality summary."""

    return {
        "rows": int(len(frame)),
        "feature_count": len(FEATURE_COLUMNS),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values": {
            column: int(count) for column, count in frame.isna().sum().items()
        },
        "class_balance": {
            str(int(label)): int(count)
            for label, count in frame[TARGET_COLUMN].value_counts().sort_index().items()
        },
    }


def prepare_dataset(
    *,
    url: str = DATASET_URL,
    raw_path: Path = RAW_DATA_PATH,
    processed_path: Path = PROCESSED_DATA_PATH,
    force_download: bool = False,
) -> pd.DataFrame:
    """Execute acquisition and cleaning, then persist data and lineage metadata."""

    raw_path = download_raw_dataset(url, raw_path, force=force_download)
    cleaned = clean_data(load_raw_data(raw_path))

    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(processed_path, index=False)

    source_metadata = {
        "source_url": url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "raw_file": str(raw_path),
        "processed_file": str(processed_path),
        **dataset_summary(cleaned),
    }
    metadata_path = processed_path.with_name("data_summary.json")
    metadata_path.write_text(json.dumps(source_metadata, indent=2), encoding="utf-8")
    return cleaned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DATASET_URL)
    parser.add_argument("--raw-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--processed-path", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--force", action="store_true", help="Redownload the raw file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cleaned = prepare_dataset(
        url=args.url,
        raw_path=args.raw_path,
        processed_path=args.processed_path,
        force_download=args.force,
    )
    print(json.dumps(dataset_summary(cleaned), indent=2))


if __name__ == "__main__":
    main()
