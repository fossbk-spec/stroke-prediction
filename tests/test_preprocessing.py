"""Unit tests for the Imputer, Encoder, Scaler, and SMOTE."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, preprocessing
from src.data_loader import split_features_target
from src.models import get_model


class TestImputer:
    def test_no_nan_after_transform(self, synthetic_df):
        """After preprocessing, the feature matrix must contain no NaNs."""
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor()
        matrix = pre.fit_transform(X, y)
        assert not np.isnan(np.asarray(matrix, dtype=float)).any()

    def test_missing_bmi_rows_are_kept(self, synthetic_df):
        """Records with missing BMI must not be dropped (avoids losing rare-class samples)."""
        X, y = split_features_target(synthetic_df)
        assert X["bmi"].isna().sum() > 0, "Fixture must have missing BMI values"
        pre = preprocessing.make_preprocessor()
        matrix = pre.fit_transform(X, y)
        assert matrix.shape[0] == len(X)

    @pytest.mark.parametrize("strategy", ["knn", "median"])
    def test_both_imputer_strategies_work(self, synthetic_df, strategy):
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor(numeric_imputer=strategy)
        assert pre.fit_transform(X, y).shape[0] == len(X)

    def test_invalid_imputer_rejected(self):
        with pytest.raises(ValueError):
            preprocessing.make_preprocessor(numeric_imputer="magic")


class TestEncoder:
    def test_ever_married_binary_encoding(self, synthetic_df):
        """`ever_married`: No -> 0, Yes -> 1 (a single column, not one-hot)."""
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor()
        frame = pd.DataFrame(pre.fit_transform(X, y), columns=pre.get_feature_names_out())
        assert "ever_married" in frame.columns
        assert set(frame["ever_married"].unique()) <= {0.0, 1.0}

    def test_nominal_features_are_one_hot(self, synthetic_df):
        """Nominal features are split into 0/1 columns."""
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor()
        names = list(pre.fit(X, y).get_feature_names_out())
        assert any(n.startswith("gender_") for n in names)
        assert any(n.startswith("work_type_") for n in names)
        assert any(n.startswith("smoking_status_") for n in names)

    def test_unknown_smoking_kept_as_own_category(self, synthetic_df):
        """'Unknown' is its own category, not merged into 'never smoked'."""
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor()
        names = list(pre.fit(X, y).get_feature_names_out())
        assert "smoking_status_Unknown" in names

    def test_unseen_category_at_inference_does_not_crash(self, synthetic_df):
        """A category never seen at train time must not crash inference."""
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor()
        pre.fit(X, y)

        X_new = X.iloc[:5].copy()
        X_new.loc[:, "work_type"] = "Astronaut"  # a brand-new category
        assert pre.transform(X_new).shape[0] == 5


class TestScaler:
    def test_train_is_standardized(self, synthetic_df):
        """Continuous features on the train set have mean ~ 0, std ~ 1."""
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor()
        frame = pd.DataFrame(pre.fit_transform(X, y), columns=pre.get_feature_names_out())
        for col in config.NUMERIC_FEATURES:
            assert abs(frame[col].mean()) < 1e-6
            assert abs(frame[col].std(ddof=0) - 1.0) < 1e-6

    def test_scaling_can_be_disabled_for_tree_models(self, synthetic_df):
        """Tree models don't need scaling -> `scale=False` must drop the scaler."""
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor(scale=False)
        pre.fit(X, y)
        assert "scaler" not in pre.named_transformers_["num"].named_steps

    def test_holdout_is_not_restandardized(self, synthetic_df):
        """The test set does NOT have mean 0 / std 1 — proof it uses train's parameters."""
        split = preprocessing.patient_level_split(synthetic_df)
        X_train, y_train = split_features_target(split.train)
        X_test, _ = split_features_target(split.test)

        pre = preprocessing.make_preprocessor()
        pre.fit(X_train, y_train)
        test_frame = pd.DataFrame(pre.transform(X_test), columns=pre.get_feature_names_out())
        deviations = [abs(test_frame[c].mean()) for c in config.NUMERIC_FEATURES]
        assert max(deviations) > 1e-9


class TestSmote:
    def test_smote_balances_training_classes(self, synthetic_df):
        """SMOTE brings the train class ratio to 50:50."""
        X, y = split_features_target(synthetic_df)
        pipeline = preprocessing.make_full_pipeline(get_model("logistic_regression"))
        prepared = pipeline.named_steps["prepare"].fit_transform(X, y)
        X_res, y_res = pipeline.named_steps["smote"].fit_resample(prepared, y)
        categorical = (config.BINARY_FEATURES + config.ORDINAL_BINARY_FEATURES
                       + config.NOMINAL_FEATURES)
        for column in categorical:
            assert set(X_res[column]) <= set(prepared[column])
        for column in config.BINARY_FEATURES:
            assert set(X_res[column]) <= {0, 1}
        assert (y_res == 1).sum() == (y_res == 0).sum()
        assert len(X_res) > len(X)

    def test_pipeline_with_smote_fits_and_predicts(self, synthetic_df):
        X, y = split_features_target(synthetic_df)
        pipeline = preprocessing.make_full_pipeline(get_model("logistic_regression"))
        pipeline.fit(X, y)
        proba = pipeline.predict_proba(X)[:, 1]
        assert proba.shape == (len(X),)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_smote_can_be_disabled(self, synthetic_df):
        X, y = split_features_target(synthetic_df)
        pipeline = preprocessing.make_full_pipeline(
            get_model("logistic_regression", class_weight="balanced"), use_smote=False
        )
        assert "smote" not in pipeline.named_steps
        pipeline.fit(X, y)
        assert len(pipeline.predict(X)) == len(X)

    def test_smote_lowers_k_neighbors_on_tiny_minority(self, synthetic_df):
        """A fold with only 4 positives must still fit (k_neighbors capped to 3)."""
        positives = synthetic_df[synthetic_df[config.TARGET_COLUMN] == 1].head(4)
        negatives = synthetic_df[synthetic_df[config.TARGET_COLUMN] == 0].head(100)
        X, y = split_features_target(pd.concat([positives, negatives]))

        pipeline = preprocessing.make_full_pipeline(get_model("logistic_regression"))
        with pytest.warns(UserWarning, match="k_neighbors from 5 to 3"):
            pipeline.fit(X, y)
        assert len(pipeline.predict(X)) == len(X)

    def test_smote_raises_clear_error_on_single_minority_sample(self, synthetic_df):
        positives = synthetic_df[synthetic_df[config.TARGET_COLUMN] == 1].head(1)
        negatives = synthetic_df[synthetic_df[config.TARGET_COLUMN] == 0].head(100)
        X, y = split_features_target(pd.concat([positives, negatives]))

        pipeline = preprocessing.make_full_pipeline(get_model("logistic_regression"))
        with pytest.raises(ValueError, match="at least 2 minority-class samples"):
            pipeline.fit(X, y)


class TestPipelineIntegration:
    def test_feature_names_match_matrix_width(self, synthetic_df):
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor()
        matrix = pre.fit_transform(X, y)
        assert len(preprocessing.get_feature_names(pre)) == matrix.shape[1]

    def test_pipeline_is_clonable_for_cross_validation(self, synthetic_df):
        """The pipeline must be clonable, or `cross_val_score` would reuse stale state."""
        from sklearn.base import clone

        pipeline = preprocessing.make_full_pipeline(get_model("random_forest"))
        assert clone(pipeline) is not pipeline
