"""Fixture dung chung cho toan bo test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import make_synthetic_dataset  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_df():
    """Bo du lieu gia lap 800 ban ghi, dung schema Kaggle."""
    return make_synthetic_dataset(n_rows=800, random_state=7)


@pytest.fixture(scope="session")
def multi_record_df(synthetic_df):
    """Bo du lieu gia lap trong do MOI benh nhan co 2-3 ban ghi.

    Dung de kiem tra ky luat patient-level splitting: neu chia ngau nhien theo
    ban ghi, cung mot benh nhan se roi vao ca train va test.
    """
    import pandas as pd

    df = synthetic_df.copy()
    duplicated = pd.concat([df, df.sample(frac=0.5, random_state=3)], ignore_index=True)
    return duplicated.reset_index(drop=True)
