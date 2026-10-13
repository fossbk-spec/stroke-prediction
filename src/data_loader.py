"""Data loading and schema validation.

Goal: every notebook / script uses a SINGLE data entry point, with integrity
checks that catch structural or dtype errors early.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


class SchemaValidationError(ValueError):
    """The dataset does not match the expected schema."""


@dataclass
class DatasetProfile:
    """Quick descriptive summary of a loaded dataset."""

    n_rows: int
    n_cols: int
    n_positive: int
    n_negative: int
    positive_rate: float
    n_missing_bmi: int
    n_unknown_smoking: int
    n_duplicate_ids: int

    def as_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"Rows={self.n_rows:,} | Cols={self.n_cols} | "
            f"Stroke=1: {self.n_positive:,} ({self.positive_rate:.2%}) | "
            f"Stroke=0: {self.n_negative:,} | "
            f"BMI missing={self.n_missing_bmi:,} | "
            f"smoking_status='Unknown'={self.n_unknown_smoking:,} | "
            f"Duplicate IDs={self.n_duplicate_ids}"
        )


def load_raw_data(
    csv_path: str | Path | None = None, *, validate: bool = True
) -> pd.DataFrame:
    """Load the raw Kaggle Stroke Prediction Dataset CSV.

    Args:
        csv_path: optional path. Defaults to `config.RAW_CSV_PATH`.
        validate: whether to run `validate_schema` right after loading.

    Raises:
        FileNotFoundError: when the raw data has not been placed in `data/raw/`.
    """
    path = Path(csv_path) if csv_path is not None else config.RAW_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at: {path}\n"
            "Download it first:  python scripts/download_data.py\n"
            "(or manually from Kaggle: fedesoriano/stroke-prediction-dataset)"
        )

    df = pd.read_csv(path)
    df = _coerce_dtypes(df)
    if validate:
        validate_schema(df)
    return df


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes: `bmi` may contain the literal string 'N/A' in the raw file."""
    df = df.copy()
    for col in config.NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in config.BINARY_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    if config.TARGET_COLUMN in df.columns:
        df[config.TARGET_COLUMN] = pd.to_numeric(df[config.TARGET_COLUMN], errors="coerce")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Validate the required schema. Raises `SchemaValidationError` on mismatch."""
    missing = [c for c in config.EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"Missing required columns: {missing}")

    if df.empty:
        raise SchemaValidationError("Dataset is empty.")

    labels = pd.to_numeric(df[config.TARGET_COLUMN], errors="coerce")
    if labels.isna().any():
        raise SchemaValidationError(
            f"Target column '{config.TARGET_COLUMN}' must not contain nulls or non-convertible labels."
        )
    target = labels.unique()
    invalid_target = sorted(set(np.asarray(target).tolist()) - {0, 1})
    if invalid_target:
        raise SchemaValidationError(
            f"Target column '{config.TARGET_COLUMN}' contains invalid values: {invalid_target}"
        )

    if df[config.ID_COLUMN].isna().any():
        raise SchemaValidationError(f"Column '{config.ID_COLUMN}' must not contain nulls.")

    for col in config.BINARY_FEATURES:
        extra = sorted(set(df[col].dropna().unique().tolist()) - {0, 1})
        if extra:
            raise SchemaValidationError(f"Column '{col}' must be 0/1 only, found: {extra}")


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Compute quick descriptive statistics used in EDA and reporting."""
    n_rows = len(df)
    n_positive = int((df[config.TARGET_COLUMN] == 1).sum())
    smoking = df.get("smoking_status", pd.Series(dtype=object))
    return DatasetProfile(
        n_rows=n_rows,
        n_cols=df.shape[1],
        n_positive=n_positive,
        n_negative=n_rows - n_positive,
        positive_rate=n_positive / n_rows if n_rows else 0.0,
        n_missing_bmi=int(df["bmi"].isna().sum()) if "bmi" in df else 0,
        n_unknown_smoking=int((smoking == "Unknown").sum()),
        n_duplicate_ids=int(df[config.ID_COLUMN].duplicated().sum()),
    )


def split_features_target(
    df: pd.DataFrame, *, keep_id: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Split into the feature matrix X and the label vector y.

    `id` is excluded from X by default: it is a patient identifier with no
    clinical meaning. Keeping `id` in X would create identifier leakage and
    let the model "memorize" individual patients.
    """
    cols = config.FEATURE_COLUMNS.copy()
    if keep_id:
        cols = [config.ID_COLUMN] + cols
    X = df[cols].copy()
    y = df[config.TARGET_COLUMN].astype(int).copy()
    return X, y


def make_synthetic_dataset(
    n_rows: int = 600, *, positive_rate: float = 0.049, random_state: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """Generate a synthetic dataset matching the Kaggle schema.

    Used for unit tests and CI (environments without the original CSV). This
    data MUST NOT be used to report scientific results.
    """
    rng = np.random.default_rng(random_state)
    n_pos = max(2, int(round(n_rows * positive_rate)))
    y = np.zeros(n_rows, dtype=int)
    y[rng.choice(n_rows, size=n_pos, replace=False)] = 1

    # Positive patients tend to be older / have higher glucose levels.
    age = np.clip(rng.normal(np.where(y == 1, 68, 42), 15), 1, 95).round(1)
    glucose = np.clip(rng.normal(np.where(y == 1, 135, 104), 42), 55, 280).round(2)
    bmi = np.clip(rng.normal(np.where(y == 1, 30.5, 28.2), 7), 12, 65).round(1)

    df = pd.DataFrame(
        {
            config.ID_COLUMN: rng.permutation(np.arange(10_000, 10_000 + n_rows)),
            "gender": rng.choice(["Male", "Female", "Other"], n_rows, p=[0.41, 0.586, 0.004]),
            "age": age,
            "hypertension": rng.binomial(1, np.where(y == 1, 0.27, 0.09)),
            "heart_disease": rng.binomial(1, np.where(y == 1, 0.19, 0.04)),
            "ever_married": rng.choice(["Yes", "No"], n_rows, p=[0.66, 0.34]),
            "work_type": rng.choice(
                ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                n_rows,
                p=[0.57, 0.16, 0.13, 0.13, 0.01],
            ),
            "Residence_type": rng.choice(["Urban", "Rural"], n_rows),
            "avg_glucose_level": glucose,
            "bmi": bmi,
            "smoking_status": rng.choice(
                ["never smoked", "Unknown", "formerly smoked", "smokes"],
                n_rows,
                p=[0.37, 0.30, 0.17, 0.16],
            ),
            config.TARGET_COLUMN: y,
        }
    )

    # Simulate ~3.9% missing BMI, matching the real dataset.
    missing_idx = rng.choice(n_rows, size=max(1, int(n_rows * 0.039)), replace=False)
    df.loc[missing_idx, "bmi"] = np.nan
    return df
