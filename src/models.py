"""Khoi tao cac mo hinh: baseline, cost-sensitive va ensemble nang cao.

Triet ly chon mo hinh cho bai toan nay:

- **Logistic Regression** (baseline bat buoc): tuyen tinh, he so co the doc
  truc tiep thanh odds ratio -> rat de bien luan lam sang.
- **Cost-sensitive learning** (`class_weight="balanced"`): phuong phap chong
  mat can bang KHONG sinh du lieu gia, nen khong co rui ro "SMOTE leakage".
  Day la doi chung quan trong cho nhanh dung SMOTE.
- **Random Forest / Gradient Boosting / XGBoost**: nam bat tuong tac phi tuyen
  (vi du age x hypertension) ma mo hinh tuyen tinh bo sot.
- **Stacking Ensemble**: khoang trong duoc chi ra trong ma tran y van
  (N22 Team 2025 chua thu nghiem) — base learners + meta-learner Logistic
  Regression, lap lai kien truc cua Dritsas & Trigka (2022) nhung voi giao
  thuc chia du lieu sach.
"""

from __future__ import annotations

from typing import Any

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
# XGBoost la phu thuoc TUY CHON (can OpenMP/libomp tren macOS).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - phu thuoc moi truong
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBOOST_AVAILABLE = False


def get_model(name: str, *, class_weight: str | None = None, **overrides: Any):
    """Tra ve mot estimator theo ten.

    Args:
        name: mot trong cac khoa cua `available_models()`.
        class_weight: "balanced" de bat cost-sensitive learning (neu mo hinh
            ho tro). Khong dung dong thoi voi SMOTE tru khi co chu dich.
        **overrides: ghi de sieu tham so mac dinh.
    """
    name = name.lower().replace("-", "_")
    rs = config.RANDOM_STATE
    builders: dict[str, Any] = {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight=class_weight,
            random_state=rs,
            **overrides,
        ),
        "decision_tree": lambda: DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight=class_weight,
            random_state=rs,
            **overrides,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            n_jobs=-1,
            class_weight=class_weight,
            random_state=rs,
            **overrides,
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=rs, **overrides
        ),
        "svm": lambda: SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight=class_weight,
            random_state=rs,
            **overrides,
        ),
        "knn": lambda: KNeighborsClassifier(n_neighbors=15, weights="distance", **overrides),
        "naive_bayes": lambda: GaussianNB(**overrides),
        "xgboost": lambda: _build_xgboost(**overrides),
        "stacking": lambda: build_stacking_ensemble(class_weight=class_weight),
    }
    if name not in builders:
        raise KeyError(f"Mo hinh '{name}' khong ton tai. Hop le: {sorted(builders)}")
    return builders[name]()


def _build_xgboost(**overrides: Any):
    """Khoi tao XGBoost; nem loi ro rang neu moi truong chua co libomp."""
    if not XGBOOST_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "Khong nap duoc xgboost (thieu OpenMP runtime). "
            "Tren macOS: `brew install libomp`. Hoac dung 'gradient_boosting' thay the."
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


def build_stacking_ensemble(*, class_weight: str | None = None) -> StackingClassifier:
    """Stacking Ensemble: base learners da dang + meta-learner Logistic Regression.

    `cv=5` ben trong `StackingClassifier` dam bao meta-features duoc sinh ra
    bang out-of-fold prediction => khong ro ri nhan tu base sang meta.
    """
    base_estimators = [
        ("rf", get_model("random_forest", class_weight=class_weight)),
        ("gb", get_model("gradient_boosting")),
        ("nb", get_model("naive_bayes")),
        ("dt", get_model("decision_tree", class_weight=class_weight)),
    ]
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=2000, class_weight=class_weight),
        cv=5,
        stack_method="predict_proba",
        n_jobs=None,
    )


def available_models() -> list[str]:
    """Danh sach cac mo hinh su dung duoc trong moi truong hien tai."""
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
