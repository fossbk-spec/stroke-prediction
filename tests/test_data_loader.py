"""Unit tests for data loading and schema validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import (
    SchemaValidationError,
    load_raw_data,
    make_synthetic_dataset,
    profile_dataset,
    validate_schema,
)


class TestSchemaValidation:
    def test_valid_dataset_passes(self, synthetic_df):
        validate_schema(synthetic_df)  # must not raise

    def test_missing_column_is_rejected(self, synthetic_df):
        with pytest.raises(SchemaValidationError, match="Missing required"):
            validate_schema(synthetic_df.drop(columns=["bmi"]))

    def test_invalid_target_value_is_rejected(self, synthetic_df):
        broken = synthetic_df.copy()
        broken.loc[0, config.TARGET_COLUMN] = 2
        with pytest.raises(SchemaValidationError, match="invalid values"):
            validate_schema(broken)

    def test_empty_dataset_is_rejected(self, synthetic_df):
        with pytest.raises(SchemaValidationError, match="empty"):
            validate_schema(synthetic_df.iloc[0:0])

    def test_null_id_is_rejected(self, synthetic_df):
        broken = synthetic_df.copy().astype({config.ID_COLUMN: "float"})
        broken.loc[0, config.ID_COLUMN] = np.nan
        with pytest.raises(SchemaValidationError, match="must not contain nulls"):
            validate_schema(broken)


class TestLoadRawData:
    def test_missing_file_gives_actionable_message(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="download_data.py"):
            load_raw_data(tmp_path / "does-not-exist.csv")

    def test_bmi_na_string_is_coerced_to_nan(self, tmp_path, synthetic_df):
        """The original Kaggle file uses the literal string 'N/A' for BMI -> must become NaN."""
        df = synthetic_df.copy()
        df["bmi"] = df["bmi"].astype(object)
        df.loc[0, "bmi"] = "N/A"
        path = tmp_path / "raw.csv"
        df.to_csv(path, index=False)

        loaded = load_raw_data(path)
        assert pd.isna(loaded.loc[0, "bmi"])
        assert pd.api.types.is_numeric_dtype(loaded["bmi"])


class TestProfile:
    def test_profile_counts_are_consistent(self, synthetic_df):
        p = profile_dataset(synthetic_df)
        assert p.n_positive + p.n_negative == p.n_rows
        assert p.positive_rate == pytest.approx(p.n_positive / p.n_rows)
        assert p.n_duplicate_ids == 0

    def test_synthetic_generator_respects_positive_rate(self):
        df = make_synthetic_dataset(n_rows=1000, positive_rate=0.049, random_state=11)
        assert (df[config.TARGET_COLUMN] == 1).sum() == 49
        assert df["bmi"].isna().sum() > 0
        assert list(df.columns) == config.EXPECTED_COLUMNS
