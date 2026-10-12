"""Central configuration (single source of truth for project parameters)."""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
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
# Kaggle Stroke Prediction Dataset schema (fedesoriano) - 5,110 x 12
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

# Column order EXACTLY as in the original Kaggle CSV (used for schema validation).
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

# Expected raw-data size / quality (used for data validation)
EXPECTED_N_ROWS = 5110
EXPECTED_N_POSITIVE = 249
EXPECTED_N_MISSING_BMI = 201
EXPECTED_N_UNKNOWN_SMOKING = 1544

# --------------------------------------------------------------------------- #
# Experiment parameters
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
N_SPLITS = 5          # Number of folds for StratifiedGroupKFold
TEST_SIZE = 0.2       # Test set fraction (patient-level hold-out)
VAL_SIZE = 0.2        # Validation set fraction (carved out of the remainder)

# Metric priority for the clinical problem: FN >> FP  =>  beta = 2
F_BETA = 2.0

# Default probability threshold, before F-beta-based tuning
DEFAULT_THRESHOLD = 0.5
