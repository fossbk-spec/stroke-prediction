"""Regression tests for model factory overrides."""

import pytest

from src import models


@pytest.mark.parametrize(
    "name, overrides",
    [
        ("logistic_regression", {"max_iter": 12, "solver": "lbfgs"}),
        ("decision_tree", {"max_depth": 2, "min_samples_leaf": 1}),
        ("random_forest", {"n_estimators": 3, "n_jobs": 1}),
        ("gradient_boosting", {"n_estimators": 3, "learning_rate": 0.1}),
        ("svm", {"C": 2.0, "kernel": "linear"}),
        ("knn", {"n_neighbors": 3, "weights": "uniform"}),
        ("naive_bayes", {"var_smoothing": 1e-8}),
        ("stacking", {"cv": 2, "n_jobs": 1}),
        ("xgboost", {"n_estimators": 3, "max_depth": 2}),
    ],
)
def test_factory_overrides(name, overrides):
    if name == "xgboost" and not models.XGBOOST_AVAILABLE:
        pytest.skip("XGBoost is optional")
    estimator = models.get_model(name, **overrides)
    for key, value in overrides.items():
        assert estimator.get_params()[key] == value
