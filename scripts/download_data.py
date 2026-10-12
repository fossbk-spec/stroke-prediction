#!/usr/bin/env python3
"""Download the Kaggle Stroke Prediction Dataset into `data/raw/`.

Priority order:
    1. `kagglehub` (if installed and ~/.kaggle/kaggle.json is configured)
    2. `kaggle` CLI
    3. Manual download instructions

Run:
    python scripts/download_data.py
    python scripts/download_data.py --verify-only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

KAGGLE_SLUG = "fedesoriano/stroke-prediction-dataset"
CSV_NAME = "healthcare-dataset-stroke-data.csv"

MANUAL_INSTRUCTIONS = f"""
Automatic download failed. Please download manually in 3 steps:

  1. Open: https://www.kaggle.com/datasets/{KAGGLE_SLUG}
  2. Click "Download" (requires a Kaggle login).
  3. Unzip and place `{CSV_NAME}` into:
         {config.RAW_DATA_DIR}

Or configure a Kaggle API token before re-running this script:
  - Kaggle > Settings > API > "Create New Token" -> downloads kaggle.json
  - mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
  - chmod 600 ~/.kaggle/kaggle.json
"""


def _try_kagglehub() -> Path | None:
    try:
        import kagglehub  # type: ignore
    except ImportError:
        return None
    try:
        path = Path(kagglehub.dataset_download(KAGGLE_SLUG))
    except Exception as exc:  # pragma: no cover - network-dependent
        print(f"  kagglehub failed: {exc}")
        return None
    found = next(iter(path.rglob(CSV_NAME)), None)
    return found


def _try_kaggle_cli() -> Path | None:
    if shutil.which("kaggle") is None:
        return None
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
           "-p", str(config.RAW_DATA_DIR), "--unzip"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        print(f"  kaggle CLI failed: {exc}")
        return None
    return next(iter(config.RAW_DATA_DIR.rglob(CSV_NAME)), None)


def verify() -> bool:
    """Check that the file exists and matches the expected size/quality."""
    from src.data_loader import load_raw_data, profile_dataset

    if not config.RAW_CSV_PATH.exists():
        print(f"[MISSING] {config.RAW_CSV_PATH}")
        return False

    profile = profile_dataset(load_raw_data())
    print(f"[OK] {config.RAW_CSV_PATH}")
    print(f"     {profile}")

    checks = {
        "n_rows = 5,110": profile.n_rows == config.EXPECTED_N_ROWS,
        "n_positive = 249": profile.n_positive == config.EXPECTED_N_POSITIVE,
        "n_missing_bmi = 201": profile.n_missing_bmi == config.EXPECTED_N_MISSING_BMI,
        "n_unknown_smoking = 1,544": profile.n_unknown_smoking == config.EXPECTED_N_UNKNOWN_SMOKING,
    }
    for label, passed in checks.items():
        print(f"     [{'OK ' if passed else 'FAIL'}] {label}")
    return all(checks.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Only verify, do not download.")
    args = parser.parse_args(argv)

    if args.verify_only:
        return 0 if verify() else 1

    if config.RAW_CSV_PATH.exists():
        print("Data already exists, skipping download.")
        return 0 if verify() else 1

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Trying to download via kagglehub ...")
    source = _try_kagglehub()
    if source is None:
        print("Trying to download via kaggle CLI ...")
        source = _try_kaggle_cli()

    if source is None:
        print(MANUAL_INSTRUCTIONS)
        return 1

    if source.resolve() != config.RAW_CSV_PATH.resolve():
        shutil.copy2(source, config.RAW_CSV_PATH)
    print(f"Saved: {config.RAW_CSV_PATH}")
    return 0 if verify() else 1


if __name__ == "__main__":
    raise SystemExit(main())
