"""Shared fixtures for the whole test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import make_synthetic_dataset  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_df():
    """An 800-row synthetic dataset matching the Kaggle schema."""
    return make_synthetic_dataset(n_rows=800, random_state=7)


@pytest.fixture(scope="session")
def multi_record_df(synthetic_df):
    """A synthetic dataset where EVERY patient has 2-3 records.

    Used to test the patient-level splitting discipline: with a naive random
    split by record, the same patient would end up in both train and test.
    """
    import pandas as pd

    df = synthetic_df.copy()
    duplicated = pd.concat([df, df.sample(frac=0.5, random_state=3)], ignore_index=True)
    return duplicated.reset_index(drop=True)
