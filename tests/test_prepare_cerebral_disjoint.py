"""Small in-memory checks for the same-source disjoint sample."""

import numpy as np
import pytest

from scripts.prepare_cerebral_disjoint import prepare_disjoint
from src.data_loader import validate_schema


def test_disjoint_normalization_counts_and_schema(synthetic_df):
    reference = synthetic_df.iloc[:2].copy()
    source = synthetic_df.iloc[:4].copy()
    source.loc[source.index[0], "smoking_status"] = np.nan
    reference.loc[reference.index[0], "smoking_status"] = "Unknown"
    source.loc[source.index[2], "smoking_status"] = np.nan
    source.loc[:, "stroke"] = [1, 0, 1, 0]
    reference.loc[:, "stroke"] = [1, 0]

    disjoint, provenance = prepare_disjoint(source, reference)

    assert disjoint["id"].tolist() == source.iloc[2:]["id"].tolist()
    assert disjoint.iloc[0]["smoking_status"] == "Unknown"
    validate_schema(disjoint)
    assert provenance == {
        "n_source": 4,
        "n_reference": 2,
        "n_overlap_ids": 2,
        "overlap_rows_identical": True,
        "n_disjoint": 2,
        "n_source_positives": 2,
        "source_positive_rate": 0.5,
        "n_reference_positives": 1,
        "reference_positive_rate": 0.5,
        "n_disjoint_positives": 1,
        "disjoint_positive_rate": 0.5,
        "n_smoking_status_filled": 2,
    }


def test_conflicting_overlap_raises(synthetic_df):
    reference = synthetic_df.iloc[:2].copy()
    source = synthetic_df.iloc[:3].copy()
    source.loc[source.index[0], "age"] += 1

    with pytest.raises(ValueError, match="conflicting"):
        prepare_disjoint(source, reference)
