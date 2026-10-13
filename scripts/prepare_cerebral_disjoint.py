#!/usr/bin/env python3
"""Remove bundled patients from the larger, same-source cerebral stroke CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data_loader import load_raw_data, validate_schema  # noqa: E402


def prepare_disjoint(source: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate and return source rows whose patient IDs are absent from reference."""
    for name, frame in (("source", source), ("reference", reference)):
        if set(frame.columns) != set(config.EXPECTED_COLUMNS) or len(frame.columns) != len(config.EXPECTED_COLUMNS):
            raise ValueError(f"{name} columns do not match the bundled schema")
        validate_schema(frame)
        if frame[config.ID_COLUMN].duplicated().any():
            raise ValueError(f"{name} contains duplicate patient IDs")

    source = source.loc[:, config.EXPECTED_COLUMNS].copy()
    reference = reference.loc[:, config.EXPECTED_COLUMNS].copy()
    n_filled = int(source["smoking_status"].isna().sum())
    source["smoking_status"] = source["smoking_status"].fillna("Unknown")
    reference["smoking_status"] = reference["smoking_status"].fillna("Unknown")

    source_ids = source.set_index(config.ID_COLUMN)
    reference_ids = reference.set_index(config.ID_COLUMN)
    overlap = source_ids.index.intersection(reference_ids.index)
    left = source_ids.loc[overlap]
    right = reference_ids.loc[overlap]
    same = left.eq(right) | (left.isna() & right.isna())
    if not bool(same.to_numpy().all()):
        raise ValueError("Overlapping patient IDs have conflicting row values")

    disjoint = source.loc[~source[config.ID_COLUMN].isin(reference[config.ID_COLUMN])].copy()
    if disjoint[config.ID_COLUMN].isin(reference[config.ID_COLUMN]).any():
        raise AssertionError("Disjoint output still contains reference IDs")
    if not disjoint.empty:
        validate_schema(disjoint)

    def positives(frame: pd.DataFrame) -> int:
        return int((frame[config.TARGET_COLUMN] == 1).sum())

    provenance = {
        "n_source": len(source),
        "n_reference": len(reference),
        "n_overlap_ids": len(overlap),
        "overlap_rows_identical": True,
        "n_disjoint": len(disjoint),
        "n_source_positives": positives(source),
        "source_positive_rate": positives(source) / len(source),
        "n_reference_positives": positives(reference),
        "reference_positive_rate": positives(reference) / len(reference),
        "n_disjoint_positives": positives(disjoint),
        "disjoint_positive_rate": positives(disjoint) / len(disjoint) if len(disjoint) else 0.0,
        "n_smoking_status_filled": n_filled,
    }
    return disjoint, provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=config.RAW_DATA_DIR / "dataset.csv")
    parser.add_argument("--reference", type=Path, default=config.RAW_CSV_PATH)
    parser.add_argument("--output", type=Path, default=config.RAW_DATA_DIR / "cerebral-stroke-disjoint.csv")
    parser.add_argument("--provenance", type=Path, default=config.REPORTS_DIR / "data_provenance.json")
    args = parser.parse_args(argv)

    disjoint, provenance = prepare_disjoint(
        load_raw_data(args.source), load_raw_data(args.reference)
    )
    if disjoint.empty:
        raise ValueError("No disjoint patients remain")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    disjoint.to_csv(args.output, index=False)
    load_raw_data(args.output)
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {provenance['n_disjoint']:,} disjoint rows "
          f"({provenance['n_disjoint_positives']:,} strokes) to {args.output}")
    print(f"Provenance: {args.provenance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
