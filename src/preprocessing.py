"""Data preprocessing under the ZERO DATA LEAKAGE discipline.

Mandatory principles of this module:

1. **Patient-level splitting**: every `patient_id` must appear in EXACTLY ONE
   split (train / validation / test). We use `StratifiedGroupKFold`
   (stratified by the `stroke` label, grouped by `id`).
2. **Fit-on-train-only**: `KNNImputer`, `StandardScaler`, `OneHotEncoder`,
   `SMOTE` must only be `fit` on the train split. Validation/test only ever
   call `.transform()`.
3. **SMOTE lives inside the Pipeline**: using `imblearn.pipeline.Pipeline`
   guarantees that during cross-validation, SMOTE only runs on the train
   folds — never on the validation fold. This is exactly the mistake made
   by Dritsas & Trigka (2022) and Abed et al. (2022), who applied SMOTE to
   the whole dataset before 10-fold CV.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src import config


class DataLeakageError(AssertionError):
    """Data leakage detected between splits."""


# --------------------------------------------------------------------------- #
# 1. Patient-level data splitting
# --------------------------------------------------------------------------- #
@dataclass
class DataSplit:
    """Result of a patient-level data split."""

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
    """Carve out a hold-out split, stratified by label and grouped by patient id.

    Splits the unique patient ids (not records) with `train_test_split`,
    stratified on a patient-level label (1 if any of the patient's records is
    positive). This keeps patient groups intact, preserves the positive-class
    ratio (4.9%) — critical with severe class imbalance — and honours ANY
    `holdout_size`, unlike taking one fold of `StratifiedGroupKFold`, which
    only supports fractions of the form 1/k.
    """
    from sklearn.model_selection import train_test_split

    if not 0 < holdout_size < 1:
        raise ValueError("holdout_size must be in the (0, 1) range.")

    patient_label = df.groupby(config.ID_COLUMN)[config.TARGET_COLUMN].max()
    _keep_ids, holdout_ids = train_test_split(
        patient_label.index,
        test_size=holdout_size,
        stratify=patient_label.to_numpy(),
        random_state=random_state,
    )
    in_holdout = df[config.ID_COLUMN].isin(holdout_ids)
    return df[~in_holdout].copy(), df[in_holdout].copy()


def patient_level_split(
    df: pd.DataFrame,
    *,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    random_state: int = config.RANDOM_STATE,
    verify: bool = True,
) -> DataSplit:
    """Split the data into train / validation / test at the PATIENT LEVEL.

    Two-step procedure:
        df  ->  (train_val, test)  ->  (train, val)

    Args:
        test_size: test set fraction of the whole dataset.
        val_size: validation set fraction of the `train_val` remainder.
        verify: automatically call `verify_zero_data_leakage` after splitting.
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
    """Assertion code verifying the ZERO DATA LEAKAGE discipline at Patient ID level.

    Returns:
        True if no `patient_id` appears in two different splits.

    Raises:
        DataLeakageError: if an overlap between splits is detected.
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
                    f"DATA LEAKAGE DETECTED! {len(overlap)} patient ID(s) appear in both "
                    f"{a} and {b}. Example: {sorted(overlap)[:5]}"
                )

    if verbose:  # pragma: no cover - log only
        sizes = " | ".join(f"{k}={len(v):,} IDs" for k, v in parts.items())
        print(f"[OK] ZERO DATA LEAKAGE confirmed ({sizes}).")
    return True


# --------------------------------------------------------------------------- #
# 2. Preprocessor (fit-on-train-only)
# --------------------------------------------------------------------------- #
def make_preprocessor(
    *,
    numeric_imputer: str = "knn",
    knn_neighbors: int = 5,
    scale: bool = True,
) -> ColumnTransformer:
    """Build a `ColumnTransformer` for tabular/EHR preprocessing.

    - `age`, `avg_glucose_level`, `bmi` : KNNImputer (or median) + StandardScaler.
    - `hypertension`, `heart_disease`   : already 0/1 -> kept as-is (median
      imputer added defensively).
    - `ever_married`                    : binary encoding No=0 / Yes=1.
    - `gender`, `work_type`, `Residence_type`, `smoking_status` : One-Hot Encoding.
      The `Unknown` value of `smoking_status` is kept as ITS OWN CATEGORY
      (missing-not-at-random: not disclosing one's smoking status already
      carries information), instead of being merged into 'never smoked'.

    Args:
        numeric_imputer: "knn" or "median".
        scale: whether to apply StandardScaler to continuous features (turn
            off for tree-based models).
    """
    if numeric_imputer == "knn":
        imputer: object = KNNImputer(n_neighbors=knn_neighbors, weights="distance")
    elif numeric_imputer == "median":
        imputer = SimpleImputer(strategy="median")
    else:
        raise ValueError("numeric_imputer must be 'knn' or 'median'.")

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


class AdaptiveSMOTE(SMOTENC):
    """SMOTENC that caps `k_neighbors` to the minority-class size seen at fit time.

    A plain `SMOTE(k_neighbors=5)` crashes with an opaque scikit-learn error
    when a (CV) training fold has fewer than 6 minority samples. This variant
    lowers `k_neighbors` to `n_minority - 1` with a warning, and raises a clear
    error when fewer than 2 minority samples remain.
    """

    def fit_resample(self, X, y, **params):
        n_minority = int(np.unique(np.asarray(y), return_counts=True)[1].min())
        if n_minority < 2:
            raise ValueError(
                f"SMOTE needs at least 2 minority-class samples, got {n_minority}. "
                "Use fewer CV folds or disable SMOTE (use_smote=False)."
            )
        k_neighbors = min(self.k_neighbors, n_minority - 1)
        if k_neighbors < self.k_neighbors:
            warnings.warn(
                f"Only {n_minority} minority-class samples: lowering SMOTE "
                f"k_neighbors from {self.k_neighbors} to {k_neighbors}.",
                stacklevel=2,
            )
        smote = SMOTENC(**{**self.get_params(), "k_neighbors": k_neighbors})
        return smote.fit_resample(X, y, **params)


def make_full_pipeline(
    estimator,
    *,
    use_smote: bool = True,
    smote_strategy: float | str = "auto",
    smote_k_neighbors: int = 5,
    numeric_imputer: str = "knn",
    knn_neighbors: int = 5,
    scale: bool = True,
    random_state: int = config.RANDOM_STATE,
) -> ImbPipeline:
    """Combine raw imputation + SMOTENC + encoding/scaling + model.

    Placing SMOTE INSIDE the pipeline is the key anti-leakage design choice:
    when the pipeline is passed to `cross_val_score` / `GridSearchCV`,
    `fit_resample` is only ever called on the train portion of each fold.
    """
    steps: list[tuple[str, object]] = [
        ("preprocessor", make_preprocessor(numeric_imputer=numeric_imputer,
                                            knn_neighbors=knn_neighbors, scale=scale))
    ]
    if use_smote:
        # Impute raw categories before resampling; one-hot encoding follows it.
        categorical = (config.BINARY_FEATURES + config.ORDINAL_BINARY_FEATURES
                       + config.NOMINAL_FEATURES)
        preparation = ColumnTransformer(
            [("num", SimpleImputer(strategy="median") if numeric_imputer == "median"
              else KNNImputer(n_neighbors=knn_neighbors, weights="distance"), config.NUMERIC_FEATURES),
             ("cat", SimpleImputer(strategy="most_frequent"), categorical)],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        steps.insert(0, ("prepare", preparation))
        steps.insert(1,
            (
                "smote",
                AdaptiveSMOTE(
                    categorical_features=categorical,
                    sampling_strategy=smote_strategy,
                    random_state=random_state,
                    k_neighbors=smote_k_neighbors,
                ),
            )
        )
    steps.append(("model", estimator))
    return ImbPipeline(steps)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Get feature names after transformation (used for SHAP / feature importance)."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - fallback for older sklearn versions
        return [f"f{i}" for i in range(preprocessor.transform(pd.DataFrame()).shape[1])]


def cv_splitter(
    n_splits: int = config.N_SPLITS, random_state: int = config.RANDOM_STATE
):
    """Return the correct cross-validation splitter: stratified AND grouped."""
    from sklearn.model_selection import StratifiedGroupKFold

    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def assert_no_fit_on_holdout(pipeline: ImbPipeline, X_holdout: pd.DataFrame) -> bool:
    """Verify that a fitted pipeline's state does not change when transforming hold-out data.

    Captures the learned parameters (scaler's mean_/scale_, encoder's
    categories_, imputer's learned data) before and after calling `transform` on the hold-out set: if
    they change, a `fit` was mistakenly performed on hold-out data.
    """
    preprocessors = [pipeline.named_steps[name] for name in ("prepare", "preprocessor")
                     if name in pipeline.named_steps]
    before = [_fitted_state(pre) for pre in preprocessors]
    pipeline[:-1].transform(X_holdout)
    after = [_fitted_state(pre) for pre in preprocessors]
    if before != after:
        raise DataLeakageError(
            "DATA LEAKAGE DETECTED! The preprocessor's state changed after "
            "transforming the hold-out set -> a fit step ran on hold-out data."
        )
    return True


def _fitted_state(preprocessor: ColumnTransformer) -> str:
    """Serialize the preprocessor's learned parameters into a comparable string."""
    chunks: list[str] = []
    for name, transformer, _cols in preprocessor.transformers_:
        if not hasattr(transformer, "named_steps"):
            continue
        for step_name, step in transformer.named_steps.items():
            fit_X = getattr(step, "_fit_X", None)
            if fit_X is not None:
                digest = hashlib.sha256(np.asarray(fit_X).tobytes()).hexdigest()
                chunks.append(f"{name}.{step_name}._fit_X={fit_X.shape}:{digest}")
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
