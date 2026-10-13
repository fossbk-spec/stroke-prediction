"""Model factories: baseline, cost-sensitive, and advanced ensemble models.

Model-selection rationale for this problem:

- **Logistic Regression** (mandatory baseline): linear, coefficients can be
  read directly as odds ratios -> very easy to discuss clinically.
- **Cost-sensitive learning** (`class_weight="balanced"`): an imbalance
  countermeasure that does NOT synthesize fake data, so it carries no
  "SMOTE leakage" risk. This is an important control arm against the
  SMOTE-based branch.
- **Random Forest / Gradient Boosting / XGBoost**: capture nonlinear
  interactions (e.g. age x hypertension) that a linear model misses.
- **Stacking Ensemble**: addresses a gap called out in the literature matrix
  (the N22 Team 2025 baseline had not tried advanced stacking) — base
  learners + a Logistic Regression meta-learner, mirroring the Dritsas &
  Trigka (2022) architecture but with a clean data-splitting protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import config

# --------------------------------------------------------------------------- #
# XGBoost is an OPTIONAL dependency (needs OpenMP/libomp on macOS).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - environment-dependent
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBOOST_AVAILABLE = False


def get_model(name: str, *, class_weight: str | None = None, **overrides: Any):
    """Return an estimator by name.

    Args:
        name: one of the keys in `available_models()`.
        class_weight: "balanced" to enable cost-sensitive learning (if the
            model supports it). Do not combine with SMOTE unless intentional.
        **overrides: override default hyperparameters.
    """
    name = name.lower().replace("-", "_")
    rs = config.RANDOM_STATE
    builders: dict[str, Any] = {
        "logistic_regression": lambda: LogisticRegression(
            **{
                "max_iter": 2000,
                "solver": "liblinear",
                "class_weight": class_weight,
                "random_state": rs,
                **overrides,
            }
        ),
        "decision_tree": lambda: DecisionTreeClassifier(
            **{
                "max_depth": 6,
                "min_samples_leaf": 20,
                "class_weight": class_weight,
                "random_state": rs,
                **overrides,
            }
        ),
        "random_forest": lambda: RandomForestClassifier(
            **{
                "n_estimators": 400,
                "max_depth": None,
                "min_samples_leaf": 3,
                "n_jobs": -1,
                "class_weight": class_weight,
                "random_state": rs,
                **overrides,
            }
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            **{
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 3,
                "random_state": rs,
                **overrides,
            }
        ),
        "svm": lambda: SVC(
            **{
                "kernel": "rbf",
                "C": 1.0,
                "probability": True,
                "class_weight": class_weight,
                "random_state": rs,
                **overrides,
            }
        ),
        "knn": lambda: KNeighborsClassifier(
            **{"n_neighbors": 15, "weights": "distance", **overrides}
        ),
        "naive_bayes": lambda: GaussianNB(**overrides),
        "xgboost": lambda: _build_xgboost(**overrides),
        "stacking": lambda: build_stacking_ensemble(class_weight=class_weight, **overrides),
    }
    if name not in builders:
        raise KeyError(f"Model '{name}' does not exist. Valid options: {sorted(builders)}")
    return builders[name]()


def _build_xgboost(**overrides: Any):
    """Instantiate XGBoost; raise a clear error if libomp is missing."""
    if not XGBOOST_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "Failed to load xgboost (missing OpenMP runtime). "
            "On macOS: `brew install libomp`. Or use 'gradient_boosting' instead."
        )
    params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "aucpr",
        "random_state": config.RANDOM_STATE,
        "n_jobs": -1,
    }
    params.update(overrides)
    return XGBClassifier(**params)


class PatientGroupedStackingClassifier(StackingClassifier):
    """Generate OOF predictions with patient-disjoint internal folds.

    Each cloned base pipeline fits preprocessing and resampling within its
    fold. Groups are mandatory so repeated records cannot silently leak.
    """

    def fit(self, X, y, *, groups=None, **fit_params):
        if groups is None:
            raise ValueError("Patient groups are required for stacking.")
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError("groups must have one patient ID per record.")
        splitter = StratifiedGroupKFold(
            n_splits=self.cv, shuffle=True, random_state=config.RANDOM_STATE
        )
        splits = list(splitter.split(X, y, groups))
        # sklearn's stacking implementation clones each complete base pipeline.
        original_cv = self.cv
        try:
            self.cv = splits
            result = super().fit(X, y, **fit_params)
        finally:
            self.cv = original_cv
        self.inner_splits_ = splits
        return result


def build_stacking_ensemble(
    *,
    class_weight: str | None = None,
    base_wrapper: Callable[[Any, bool], Any] | None = None,
    **overrides: Any,
) -> PatientGroupedStackingClassifier:
    """Stacking Ensemble: diverse base learners + Logistic Regression meta-learner.

    Internal StratifiedGroupKFold generates patient-disjoint OOF meta-features.
    Pass patient IDs via fit(..., groups=patient_ids).

    `base_wrapper(estimator, scale)` wraps each base learner (e.g. into a
    preprocessor + SMOTE pipeline) so that SMOTE runs only on the train part of
    each INTERNAL fold. Resampling before the internal CV would let synthetic
    samples and their source records fall into different folds, making the
    out-of-fold meta-features optimistic.
    """
    base_estimators = [
        ("rf", get_model("random_forest", class_weight=class_weight)),
        ("gb", get_model("gradient_boosting")),
        ("nb", get_model("naive_bayes")),
        ("dt", get_model("decision_tree", class_weight=class_weight)),
    ]
    if base_wrapper is not None:
        # Only distance/likelihood-based learners benefit from scaling.
        base_estimators = [(name, base_wrapper(est, name == "nb")) for name, est in base_estimators]
    params = {
        "estimators": base_estimators,
        "final_estimator": LogisticRegression(max_iter=2000, class_weight=class_weight),
        "cv": 5,
        "stack_method": "predict_proba",
        "n_jobs": None,
    }
    params.update(overrides)
    return PatientGroupedStackingClassifier(**params)


def available_models() -> list[str]:
    """List of models usable in the current environment."""
    names = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
        "svm",
        "knn",
        "naive_bayes",
        "stacking",
    ]
    if XGBOOST_AVAILABLE:
        names.append("xgboost")
    return names
