#!/usr/bin/env python3
"""Tai bo du lieu Kaggle Stroke Prediction Dataset vao `data/raw/`.

Thu tu uu tien:
    1. `kagglehub` (neu da cai va da cau hinh ~/.kaggle/kaggle.json)
    2. `kaggle` CLI
    3. Huong dan tai tay

Chay:
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
Khong tu dong tai duoc du lieu. Hay tai tay theo 3 buoc:

  1. Mo: https://www.kaggle.com/datasets/{KAGGLE_SLUG}
  2. Bam "Download" (can dang nhap Kaggle).
  3. Giai nen va dat file `{CSV_NAME}` vao:
         {config.RAW_DATA_DIR}

Hoac cau hinh Kaggle API token truoc khi chay lai script nay:
  - Kaggle > Settings > API > "Create New Token" -> tai ve kaggle.json
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
    except Exception as exc:  # pragma: no cover - phu thuoc mang
        print(f"  kagglehub that bai: {exc}")
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
        print(f"  kaggle CLI that bai: {exc}")
        return None
    return next(iter(config.RAW_DATA_DIR.rglob(CSV_NAME)), None)


def verify() -> bool:
    """Kiem tra file da ton tai va dung ky vong ve kich thuoc / chat luong."""
    from src.data_loader import load_raw_data, profile_dataset

    if not config.RAW_CSV_PATH.exists():
        print(f"[THIEU] {config.RAW_CSV_PATH}")
        return False

    profile = profile_dataset(load_raw_data())
    print(f"[OK] {config.RAW_CSV_PATH}")
    print(f"     {profile}")

    checks = {
        "so ban ghi = 5,110": profile.n_rows == config.EXPECTED_N_ROWS,
        "so ca dot quy = 249": profile.n_positive == config.EXPECTED_N_POSITIVE,
        "BMI khuyet = 201": profile.n_missing_bmi == config.EXPECTED_N_MISSING_BMI,
        "smoking 'Unknown' = 1,544": profile.n_unknown_smoking == config.EXPECTED_N_UNKNOWN_SMOKING,
    }
    for label, passed in checks.items():
        print(f"     [{'OK ' if passed else 'SAI'}] {label}")
    return all(checks.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Chi kiem tra, khong tai.")
    args = parser.parse_args(argv)

    if args.verify_only:
        return 0 if verify() else 1

    if config.RAW_CSV_PATH.exists():
        print("Du lieu da ton tai, bo qua buoc tai.")
        return 0 if verify() else 1

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Dang thu tai qua kagglehub ...")
    source = _try_kagglehub()
    if source is None:
        print("Dang thu tai qua kaggle CLI ...")
        source = _try_kaggle_cli()

    if source is None:
        print(MANUAL_INSTRUCTIONS)
        return 1

    if source.resolve() != config.RAW_CSV_PATH.resolve():
        shutil.copy2(source, config.RAW_CSV_PATH)
    print(f"Da luu: {config.RAW_CSV_PATH}")
    return 0 if verify() else 1


if __name__ == "__main__":
    raise SystemExit(main())
