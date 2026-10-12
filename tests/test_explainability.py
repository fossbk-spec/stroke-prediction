"""Unit tests for the XAI (clinical explainability) module."""

from __future__ import annotations

import numpy as np
import pytest

from src import explainability
from src.data_loader import split_features_target
from src.models import get_model
from src.preprocessing import make_full_pipeline, patient_level_split


@pytest.fixture(scope="module")
def fitted_linear(synthetic_df):
    split = patient_level_split(synthetic_df, verify=False)
    X_train, y_train = split_features_target(split.train)
    pipeline = make_full_pipeline(get_model("logistic_regression"))
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, y_train


@pytest.fixture(scope="module")
def fitted_tree(synthetic_df):
    split = patient_level_split(synthetic_df, verify=False)
    X_train, y_train = split_features_target(split.train)
    pipeline = make_full_pipeline(get_model("random_forest", class_weight="balanced"), scale=False)
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, y_train


class TestOddsRatios:
    def test_odds_ratio_equals_exp_coefficient(self, fitted_linear):
        pipeline, X, _y = fitted_linear
        table = explainability.logistic_odds_ratios(pipeline)
        np.testing.assert_allclose(table["odds_ratio"], np.exp(table["coefficient"]), rtol=1e-9)

    def test_confidence_interval_brackets_point_estimate(self, fitted_linear):
        pipeline, X, _y = fitted_linear
        table = explainability.logistic_odds_ratios(pipeline, X)
        assert (table["or_ci_lower"] <= table["odds_ratio"] + 1e-9).all()
        assert (table["or_ci_upper"] >= table["odds_ratio"] - 1e-9).all()
        assert "significant" in table.columns

    def test_age_increases_risk(self, fitted_linear):
        """Clinical sanity check: age must INCREASE risk (OR > 1)."""
        pipeline, _X, _y = fitted_linear
        table = explainability.logistic_odds_ratios(pipeline).set_index("feature")
        assert table.loc["age", "odds_ratio"] > 1.0

    def test_tree_model_rejected_with_clear_message(self, fitted_tree):
        pipeline, _X, _y = fitted_tree
        with pytest.raises(AttributeError, match="linear model"):
            explainability.logistic_odds_ratios(pipeline)


class TestFeatureImportance:
    def test_tree_importance_sums_to_one(self, fitted_tree):
        pipeline, _X, _y = fitted_tree
        table = explainability.tree_feature_importance(pipeline)
        assert table["importance"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_linear_model_rejected_for_tree_importance(self, fitted_linear):
        pipeline, _X, _y = fitted_linear
        with pytest.raises(AttributeError, match="feature_importances_"):
            explainability.tree_feature_importance(pipeline)

    def test_permutation_importance_on_original_features(self, fitted_linear):
        """Permutation importance must return the ORIGINAL feature names."""
        pipeline, X, y = fitted_linear
        table = explainability.permutation_feature_importance(pipeline, X, y, n_repeats=3)
        assert set(table["feature"]) == set(X.columns)
        assert table["importance_mean"].notna().all()


class TestShap:
    def test_shap_summary_for_tree_model(self, fitted_tree):
        pipeline, X, _y = fitted_tree
        values, importance = explainability.shap_summary(pipeline, X, sample_size=80)
        assert values.shape[0] == min(80, len(X))
        assert importance["mean_abs_shap"].iloc[0] >= importance["mean_abs_shap"].iloc[-1]

    def test_shap_summary_for_linear_model(self, fitted_linear):
        pipeline, X, _y = fitted_linear
        values, importance = explainability.shap_summary(pipeline, X, sample_size=80)
        assert values.shape[1] == len(importance)

    def test_single_patient_explanation(self, fitted_tree):
        pipeline, X, _y = fitted_tree
        table = explainability.explain_single_patient(pipeline, X, index=0, top_k=5)
        assert len(table) == 5
        assert set(table["effect"]) <= {"increases risk", "decreases risk"}
