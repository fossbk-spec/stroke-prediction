"""Tien xu ly du lieu theo ky luat ZERO DATA LEAKAGE.

Nguyen tac bat buoc cua module nay:

1. **Patient-level splitting**: moi `patient_id` chi xuat hien trong DUNG MOT
   tap (train / validation / test). Dung `StratifiedGroupKFold` (phan tang
   theo nhan `stroke`, nhom theo `id`).
2. **Fit-on-train-only**: `KNNImputer`, `StandardScaler`, `OneHotEncoder`,
   `SMOTE` chi duoc `fit` tren tap train. Validation/test chi `transform`.
3. **SMOTE nam trong Pipeline**: dung `imblearn.pipeline.Pipeline` nen khi
   cross-validation, SMOTE chi chay tren cac fold train — khong bao gio
   chay tren fold validation. Day chinh la loi ma Dritsas & Trigka (2022) va
   Abed et al. (2022) mac phai khi SMOTE toan bo dataset truoc 10-fold CV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src import config


class DataLeakageError(AssertionError):
    """Phat hien ro ri du lieu giua cac tap."""


# --------------------------------------------------------------------------- #
# 1. Chia du lieu cap do benh nhan
# --------------------------------------------------------------------------- #
@dataclass
class DataSplit:
    """Ket qua chia du lieu cap do benh nhan."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, part in (("train", self.train), ("val", self.val), ("test", self.test)):
            n = len(part)
            pos = int((part[config.TARGET_COLUMN] == 1).sum())
            rows.append(
                {
                    "split": name,
                    "n_samples": n,
                    "n_patients": part[config.ID_COLUMN].nunique(),
                    "n_positive": pos,
                    "positive_rate": pos / n if n else 0.0,
                }
            )
        return pd.DataFrame(rows)


def _group_stratified_holdout(
    df: pd.DataFrame, holdout_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tach mot phan hold-out, phan tang theo nhan va nhom theo patient id.

    `StratifiedGroupKFold` duoc uu tien hon `GroupShuffleSplit` vi no vua giu
    nguyen ven nhom benh nhan, vua co gang bao toan ty le lop duong tinh
    (4.9%) — dieu toi quan trong voi du lieu mat can bang nghiem trong.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    if not 0 < holdout_size < 1:
        raise ValueError("holdout_size phai nam trong khoang (0, 1).")

    n_splits = max(2, int(round(1 / holdout_size)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    keep_idx, holdout_idx = next(
        sgkf.split(df, df[config.TARGET_COLUMN], groups=df[config.ID_COLUMN])
    )
    return df.iloc[keep_idx].copy(), df.iloc[holdout_idx].copy()


def patient_level_split(
    df: pd.DataFrame,
    *,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    random_state: int = config.RANDOM_STATE,
    verify: bool = True,
) -> DataSplit:
    """Chia du lieu thanh train / validation / test o CAP DO BENH NHAN.

    Quy trinh 2 buoc:
        df  ->  (train_val, test)  ->  (train, val)

    Args:
        test_size: ty le tap test tren toan bo du lieu.
        val_size: ty le tap validation tren phan `train_val`.
        verify: tu dong goi `verify_zero_data_leakage` sau khi chia.
    """
    df = df.reset_index(drop=True)
    train_val, test = _group_stratified_holdout(df, test_size, random_state)
    train, val = _group_stratified_holdout(
        train_val.reset_index(drop=True), val_size, random_state
    )
    split = DataSplit(train=train, val=val, test=test)
    if verify:
        verify_zero_data_leakage(split.train, split.val, split.test)
    return split


def verify_zero_data_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    id_column: str = config.ID_COLUMN,
    *,
    verbose: bool = True,
) -> bool:
    """Assertion code kiem chung ky luat Zero Data Leakage cap Patient ID.

    Returns:
        True neu khong co bat ky `patient_id` nao xuat hien o hai tap khac nhau.

    Raises:
        DataLeakageError: neu phat hien giao nhau giua cac tap.
    """
    parts: dict[str, set] = {"Train": set(train_df[id_column])}
    if val_df is not None:
        parts["Validation"] = set(val_df[id_column])
    if test_df is not None:
        parts["Test"] = set(test_df[id_column])

    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = parts[a] & parts[b]
            if overlap:
                raise DataLeakageError(
                    f"RO RI DU LIEU! Co {len(overlap)} patient ID xuat hien o ca "
                    f"{a} va {b}. Vi du: {sorted(overlap)[:5]}"
                )

    if verbose:  # pragma: no cover - chi in log
        sizes = " | ".join(f"{k}={len(v):,} IDs" for k, v in parts.items())
        print(f"[OK] ZERO DATA LEAKAGE da duoc xac nhan ({sizes}).")
    return True


# --------------------------------------------------------------------------- #
# 2. Bo tien xu ly (fit-on-train-only)
# --------------------------------------------------------------------------- #
def make_preprocessor(
    *,
    numeric_imputer: str = "knn",
    knn_neighbors: int = 5,
    scale: bool = True,
) -> ColumnTransformer:
    """Tao `ColumnTransformer` tien xu ly du lieu bang/EHR.

    - `age`, `avg_glucose_level`, `bmi` : KNNImputer (hoac median) + StandardScaler.
    - `hypertension`, `heart_disease`   : da la 0/1 -> giu nguyen (imputer median de an toan).
    - `ever_married`                    : Binary encoding No=0 / Yes=1.
    - `gender`, `work_type`, `Residence_type`, `smoking_status` : One-Hot Encoding.
      Gia tri `Unknown` cua `smoking_status` duoc giu lai nhu MOT HANG MUC RIENG
      (missing-not-at-random: viec khong khai bao hut thuoc ban than da mang
      thong tin), thay vi gan ghep vao 'never smoked'.

    Args:
        numeric_imputer: "knn" hoac "median".
        scale: co ap dung StandardScaler cho bien lien tuc hay khong (tat khi
            dung mo hinh dua tren cay).
    """
    if numeric_imputer == "knn":
        imputer: object = KNNImputer(n_neighbors=knn_neighbors, weights="distance")
    elif numeric_imputer == "median":
        imputer = SimpleImputer(strategy="median")
    else:
        raise ValueError("numeric_imputer phai la 'knn' hoac 'median'.")

    numeric_steps: list[tuple[str, object]] = [("imputer", imputer)]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(numeric_steps)

    binary_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])

    married_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(categories=[["No", "Yes"]])),
        ]
    )

    nominal_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=1, sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, config.NUMERIC_FEATURES),
            ("bin", binary_pipe, config.BINARY_FEATURES),
            ("married", married_pipe, config.ORDINAL_BINARY_FEATURES),
            ("nom", nominal_pipe, config.NOMINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_full_pipeline(
    estimator,
    *,
    use_smote: bool = True,
    smote_strategy: float | str = "auto",
    numeric_imputer: str = "knn",
    scale: bool = True,
    random_state: int = config.RANDOM_STATE,
) -> ImbPipeline:
    """Gop tien xu ly + SMOTE + mo hinh thanh MOT pipeline duy nhat.

    Dat SMOTE BEN TRONG pipeline la diem mau chot chong ro ri du lieu: khi
    pipeline duoc dua vao `cross_val_score` / `GridSearchCV`, `fit_resample`
    chi duoc goi tren phan train cua tung fold.
    """
    steps: list[tuple[str, object]] = [
        ("preprocessor", make_preprocessor(numeric_imputer=numeric_imputer, scale=scale))
    ]
    if use_smote:
        steps.append(
            ("smote", SMOTE(sampling_strategy=smote_strategy, random_state=random_state, k_neighbors=5))
        )
    steps.append(("model", estimator))
    return ImbPipeline(steps)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Lay ten dac trung sau khi bien doi (phuc vu SHAP / feature importance)."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - fallback cho phien ban sklearn cu
        return [f"f{i}" for i in range(preprocessor.transform(pd.DataFrame()).shape[1])]


def cv_splitter(
    n_splits: int = config.N_SPLITS, random_state: int = config.RANDOM_STATE
):
    """Tra ve bo chia cross-validation dung chuan: phan tang + theo nhom."""
    from sklearn.model_selection import StratifiedGroupKFold

    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def assert_no_fit_on_holdout(pipeline: ImbPipeline, X_holdout: pd.DataFrame) -> bool:
    """Kiem tra pipeline da fit KHONG thay doi trang thai khi transform hold-out.

    Chup lai cac tham so da hoc (mean_/scale_ cua scaler, categories_ cua
    encoder) truoc va sau khi goi `transform` tren tap hold-out: neu chung
    thay doi thi da co hanh vi `fit` tren du lieu hold-out.
    """
    pre: ColumnTransformer = pipeline.named_steps["preprocessor"]
    before = _fitted_state(pre)
    pre.transform(X_holdout)
    after = _fitted_state(pre)
    if before != after:
        raise DataLeakageError(
            "RO RI DU LIEU! Trang thai cua preprocessor thay doi sau khi "
            "transform tap hold-out -> co buoc fit tren du lieu hold-out."
        )
    return True


def _fitted_state(preprocessor: ColumnTransformer) -> str:
    """Serialize cac tham so da hoc cua preprocessor thanh chuoi so sanh duoc."""
    chunks: list[str] = []
    for name, transformer, _cols in preprocessor.transformers_:
        if not hasattr(transformer, "named_steps"):
            continue
        for step_name, step in transformer.named_steps.items():
            for attr in ("mean_", "scale_", "var_", "statistics_", "categories_"):
                value = getattr(step, attr, None)
                if value is None:
                    continue
                if isinstance(value, list):
                    flat = np.concatenate([np.asarray(v, dtype=object).ravel() for v in value])
                else:
                    flat = np.asarray(value, dtype=object).ravel()
                chunks.append(f"{name}.{step_name}.{attr}={list(flat)}")
    return "|".join(chunks)
