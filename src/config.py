"""Cau hinh trung tam cho toan bo du an (Single Source of Truth cho tham so)."""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Duong dan
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PDF_REFERENCES_DIR = DATA_DIR / "SSoT_pdf_references"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CSV_PATH = RAW_DATA_DIR / "healthcare-dataset-stroke-data.csv"

# --------------------------------------------------------------------------- #
# Schema bo du lieu Kaggle Stroke Prediction (fedesoriano) - 5,110 x 12
# --------------------------------------------------------------------------- #
ID_COLUMN = "id"
TARGET_COLUMN = "stroke"

NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
BINARY_FEATURES = ["hypertension", "heart_disease"]
ORDINAL_BINARY_FEATURES = ["ever_married"]  # Yes/No -> 1/0
NOMINAL_FEATURES = ["gender", "work_type", "Residence_type", "smoking_status"]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_BINARY_FEATURES + NOMINAL_FEATURES
)

# Thu tu cot DUNG NHU trong file CSV goc cua Kaggle (dung cho schema validation).
EXPECTED_COLUMNS = [
    ID_COLUMN,
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    TARGET_COLUMN,
]

# Ky vong ve kich thuoc / chat luong du lieu goc (dung cho data validation)
EXPECTED_N_ROWS = 5110
EXPECTED_N_POSITIVE = 249
EXPECTED_N_MISSING_BMI = 201
EXPECTED_N_UNKNOWN_SMOKING = 1544

# --------------------------------------------------------------------------- #
# Tham so thuc nghiem
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
N_SPLITS = 5          # So fold cho StratifiedGroupKFold
TEST_SIZE = 0.2       # Ty le tap test (hold-out cap do benh nhan)
VAL_SIZE = 0.2        # Ty le tap validation (tach tu phan con lai)

# Do do uu tien trong bai toan y khoa: FN >> FP  =>  beta = 2
F_BETA = 2.0

# Nguong xac suat mac dinh truoc khi toi uu hoa theo F-beta
DEFAULT_THRESHOLD = 0.5
