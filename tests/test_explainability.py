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

    def test_bootstrap_intervals_are_reproducible(self, fitted_linear):
        pipeline, X, y = fitted_linear
        kwargs = dict(groups=X.index, n_boot=8, random_state=12)
        table = explainability.logistic_odds_ratios(pipeline, X, y, **kwargs)
        repeated = explainability.logistic_odds_ratios(pipeline, X, y, **kwargs)
        np.testing.assert_allclose(table["or_ci_lower"], repeated["or_ci_lower"])
        valid = table.dropna(subset=["or_ci_lower", "or_ci_upper"])
        assert len(valid) > 0
        assert (valid["or_ci_lower"] <= valid["or_ci_upper"]).all()
        missing = table["or_ci_lower"].isna()
        assert not table.loc[missing, "significant"].any()
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


def test_cluster_bootstrap_strong_and_null_reference(monkeypatch):
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(25)
    n = 600
    strong = rng.normal(size=n)
    # Matched +/- null covariates cancel exactly within each patient.
    X = pd.DataFrame({"strong": np.repeat(strong, 2),
                      "null": np.tile([-1., 1.], n)})
    y = np.repeat(rng.binomial(1, 1 / (1 + np.exp(-2 * strong))), 2)
    groups = np.repeat(np.arange(n), 2)
    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer(
            [("features", "passthrough", ["strong", "null"])],
            verbose_feature_names_out=False)),
        ("model", get_model("logistic_regression", C=100)),
    ]).fit(X, y)
    original_fit = Pipeline.fit
    bootstrap_rows = []

    def spy_fit(self, X_fit, y_fit, **kwargs):
        bootstrap_rows.append(X_fit.index.to_numpy())
        return original_fit(self, X_fit, y_fit, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", spy_fit)
    table = explainability.logistic_odds_ratios(
        pipeline, X, y, groups=groups, n_boot=30, random_state=7,
    ).set_index("feature")
    assert len(bootstrap_rows) == 30
    for rows in bootstrap_rows:
        counts = np.bincount(rows, minlength=2 * n)
        np.testing.assert_array_equal(counts[::2], counts[1::2])
    assert table.loc["strong", "or_ci_lower"] > 1
    assert table.loc["strong", "significant"]
    assert table.loc["null", "or_ci_lower"] <= 1 <= table.loc["null", "or_ci_upper"]
    assert not table.loc["null", "significant"]
    with pytest.raises(ValueError, match="patient groups"):
        explainability.logistic_odds_ratios(pipeline, X, y)
