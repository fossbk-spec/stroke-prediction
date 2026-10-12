"""Nap du lieu va kiem tra cau truc (schema validation).

Muc tieu: moi notebook / script chi co MOT duong vao du lieu duy nhat, kem
kiem tra tinh toan ven de phat hien som loi cau truc hoac loi kieu du lieu.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


class SchemaValidationError(ValueError):
    """Bo du lieu khong dung schema ky vong."""


@dataclass
class DatasetProfile:
    """Thong tin mo ta nhanh mot bo du lieu da nap."""

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

    def __str__(self) -> str:  # pragma: no cover - chi de in ra man hinh
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
    """Nap file CSV goc cua Kaggle Stroke Prediction Dataset.

    Args:
        csv_path: duong dan tuy chon. Mac dinh dung `config.RAW_CSV_PATH`.
        validate: co chay `validate_schema` ngay sau khi nap hay khong.

    Raises:
        FileNotFoundError: khi chua tai du lieu goc vao `data/raw/`.
    """
    path = Path(csv_path) if csv_path is not None else config.RAW_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Khong tim thay du lieu goc tai: {path}\n"
            "Hay tai bo du lieu truoc:  python scripts/download_data.py\n"
            "(hoac tai tay tu Kaggle: fedesoriano/stroke-prediction-dataset)"
        )

    df = pd.read_csv(path)
    df = _coerce_dtypes(df)
    if validate:
        validate_schema(df)
    return df


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Chuan hoa kieu du lieu: `bmi` co the chua chuoi 'N/A' trong file goc."""
    df = df.copy()
    for col in config.NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in config.BINARY_FEATURES + [config.TARGET_COLUMN]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Kiem tra schema bat buoc. Nem `SchemaValidationError` neu sai cau truc."""
    missing = [c for c in config.EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"Thieu cac cot bat buoc: {missing}")

    if df.empty:
        raise SchemaValidationError("Bo du lieu rong.")

    target = df[config.TARGET_COLUMN].dropna().unique()
    invalid_target = sorted(set(np.asarray(target).tolist()) - {0, 1})
    if invalid_target:
        raise SchemaValidationError(
            f"Cot nhan '{config.TARGET_COLUMN}' chua gia tri khong hop le: {invalid_target}"
        )

    if df[config.ID_COLUMN].isna().any():
        raise SchemaValidationError(f"Cot '{config.ID_COLUMN}' khong duoc chua gia tri rong.")

    for col in config.BINARY_FEATURES:
        extra = sorted(set(df[col].dropna().unique().tolist()) - {0, 1})
        if extra:
            raise SchemaValidationError(f"Cot '{col}' chi nhan 0/1, phat hien: {extra}")


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Tinh cac chi so mo ta nhanh phuc vu EDA va bao cao."""
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
    """Tach ma tran dac trung X va vector nhan y.

    `id` bi loai bo khoi X theo mac dinh: day la ma dinh danh benh nhan, khong
    mang y nghia lam sang. Giu lai `id` trong X se tao ra ro ri dac trung
    (identifier leakage) va lam mo hinh "hoc thuoc long" benh nhan.
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
    """Sinh bo du lieu gia lap dung schema Kaggle.

    Dung cho unit test va CI (moi truong khong co file CSV goc). Du lieu nay
    KHONG dung de bao cao ket qua khoa hoc.
    """
    rng = np.random.default_rng(random_state)
    n_pos = max(2, int(round(n_rows * positive_rate)))
    y = np.zeros(n_rows, dtype=int)
    y[rng.choice(n_rows, size=n_pos, replace=False)] = 1

    # Benh nhan duong tinh co xu huong lon tuoi / duong huyet cao hon.
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

    # Gia lap ~3.9% thieu BMI giong du lieu that.
    missing_idx = rng.choice(n_rows, size=max(1, int(n_rows * 0.039)), replace=False)
    df.loc[missing_idx, "bmi"] = np.nan
    return df
