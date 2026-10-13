"""Unit tests for the ZERO DATA LEAKAGE discipline.

This is the most important test suite in the project: it is the technical
safeguard against the mistake several papers made (applying SMOTE/scaling
before splitting, or splitting randomly at the record level on data where a
single patient can have multiple records).
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config, evaluation, preprocessing
from src.data_loader import split_features_target
from src.models import build_stacking_ensemble, get_model
from src.preprocessing import DataLeakageError, patient_level_split, verify_zero_data_leakage


class TestPatientLevelSplit:
    def test_patient_ids_are_disjoint(self, synthetic_df):
        """The ID sets of train / val / test must be pairwise disjoint."""
        split = patient_level_split(synthetic_df, verify=False)
        train_ids = set(split.train[config.ID_COLUMN])
        val_ids = set(split.val[config.ID_COLUMN])
        test_ids = set(split.test[config.ID_COLUMN])

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_no_record_is_lost_or_duplicated(self, synthetic_df):
        """Total records across the 3 splits = original record count (no loss, no duplication)."""
        split = patient_level_split(synthetic_df)
        total = len(split.train) + len(split.val) + len(split.test)
        assert total == len(synthetic_df)

    def test_multi_record_patients_stay_together(self, multi_record_df):
        """Patients with multiple records are never split across splits."""
        split = patient_level_split(multi_record_df)
        assert verify_zero_data_leakage(split.train, split.val, split.test, verbose=False)

        for part in (split.train, split.val, split.test):
            assert len(part) >= part[config.ID_COLUMN].nunique()

    def test_positive_class_present_in_every_split(self, synthetic_df):
        """Stratification must guarantee every split still contains positive cases."""
        split = patient_level_split(synthetic_df)
        for name, part in (("train", split.train), ("val", split.val), ("test", split.test)):
            assert (part[config.TARGET_COLUMN] == 1).sum() > 0, f"Split '{name}' has no positive case"

    @pytest.mark.parametrize("test_size", [0.15, 0.2, 0.3])
    def test_split_honours_requested_fractions(self, synthetic_df, test_size):
        """Hold-out sizes must match the requested fractions, not just 1/k values."""
        split = patient_level_split(synthetic_df, test_size=test_size, val_size=0.25, verify=False)
        n_patients = synthetic_df[config.ID_COLUMN].nunique()
        n_test = split.test[config.ID_COLUMN].nunique()
        n_val = split.val[config.ID_COLUMN].nunique()
        assert n_test / n_patients == pytest.approx(test_size, abs=0.01)
        assert n_val / (n_patients - n_test) == pytest.approx(0.25, abs=0.01)

    def test_split_is_reproducible(self, synthetic_df):
        """Same random_state -> same split result (reproducible)."""
        a = patient_level_split(synthetic_df, random_state=123, verify=False)
        b = patient_level_split(synthetic_df, random_state=123, verify=False)
        assert list(a.test[config.ID_COLUMN]) == list(b.test[config.ID_COLUMN])


class TestLeakageDetector:
    def test_detector_raises_on_overlap(self, synthetic_df):
        """An ID present in both train and test must raise DataLeakageError."""
        train = synthetic_df.iloc[:500]
        test = synthetic_df.iloc[450:]  # 50 overlapping records
        with pytest.raises(DataLeakageError, match="DATA LEAKAGE"):
            verify_zero_data_leakage(train, None, test, verbose=False)

    def test_detector_passes_on_clean_split(self, synthetic_df):
        train = synthetic_df.iloc[:500]
        test = synthetic_df.iloc[500:]
        assert verify_zero_data_leakage(train, None, test, verbose=False) is True

    def test_random_record_split_is_caught(self, multi_record_df):
        """Demonstrate that a NAIVE record-level random split leaks -> gets caught."""
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(
            multi_record_df, test_size=0.2, random_state=0,
            stratify=multi_record_df[config.TARGET_COLUMN],
        )
        with pytest.raises(DataLeakageError):
            verify_zero_data_leakage(train, None, test, verbose=False)


class TestFitOnTrainOnly:
    def test_holdout_imputer_refit_is_detected(self, synthetic_df, monkeypatch):
        split = patient_level_split(synthetic_df)
        X_train, y_train = split_features_target(split.train)
        X_test, _ = split_features_target(split.test)
        pipeline = preprocessing.make_full_pipeline(
            get_model("logistic_regression"), use_smote=False
        )
        pipeline.fit(X_train, y_train)
        imputer = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps[
            "imputer"
        ]
        original_transform = imputer.transform

        def refit_on_transform(X):
            imputer.fit(X)
            return original_transform(X)

        monkeypatch.setattr(imputer, "transform", refit_on_transform)
        with pytest.raises(DataLeakageError, match="DATA LEAKAGE"):
            preprocessing.assert_no_fit_on_holdout(pipeline, X_test)

    def test_pipeline_fit_and_smote_only_see_train_patients(self, multi_record_df, monkeypatch):
        from sklearn.dummy import DummyClassifier

        split = patient_level_split(multi_record_df)
        X_train, y_train = split_features_target(split.train)
        pipeline = preprocessing.make_full_pipeline(
            DummyClassifier(), numeric_imputer="median"
        )
        pre = pipeline.named_steps["prepare"]
        pre.set_output(transform="pandas")
        smote = pipeline.named_steps["smote"]
        seen = {"fit": [], "fit_resample": []}
        original_fit_transform = pre.fit_transform
        original_fit_resample = smote.fit_resample

        def spy_fit_transform(X, y=None, **params):
            seen["fit"].append(set(split.train.loc[X.index, config.ID_COLUMN]))
            return original_fit_transform(X, y, **params)

        def spy_fit_resample(X, y, **params):
            seen["fit_resample"].append(set(split.train.loc[X.index, config.ID_COLUMN]))
            return original_fit_resample(X, y, **params)

        monkeypatch.setattr(pre, "fit_transform", spy_fit_transform)
        monkeypatch.setattr(smote, "fit_resample", spy_fit_resample)
        pipeline.fit(X_train, y_train)
        for part in (split.val, split.test):
            X_holdout, _ = split_features_target(part)
            pipeline.predict_proba(X_holdout)
        holdout_ids = set(split.val[config.ID_COLUMN]) | set(split.test[config.ID_COLUMN])
        for calls in seen.values():
            assert len(calls) == 1
            assert calls[0] == set(split.train[config.ID_COLUMN])
            assert calls[0].isdisjoint(holdout_ids)

    def test_grouped_cv_fit_excludes_evaluated_rows_and_patients(self, multi_record_df, monkeypatch):
        from sklearn.dummy import DummyClassifier

        X, y = split_features_target(multi_record_df)
        groups = multi_record_df[config.ID_COLUMN]
        pipeline = preprocessing.make_full_pipeline(
            DummyClassifier(), use_smote=False, numeric_imputer="median"
        )
        original_fit = type(pipeline).fit
        original_predict = type(pipeline).predict_proba
        seen = []

        def spy_fit(self, X_fit, y_fit, **params):
            self.fit_indices_seen = set(X_fit.index)
            seen.append(self.fit_indices_seen)
            return original_fit(self, X_fit, y_fit, **params)

        def spy_predict(self, X_eval, **params):
            assert self.fit_indices_seen.isdisjoint(X_eval.index)
            fit_ids = set(groups.loc[list(self.fit_indices_seen)])
            assert fit_ids.isdisjoint(set(groups.loc[X_eval.index]))
            return original_predict(self, X_eval, **params)

        monkeypatch.setattr(type(pipeline), "fit", spy_fit)
        monkeypatch.setattr(type(pipeline), "predict_proba", spy_predict)
        result = evaluation.cross_validate_grouped(
            pipeline, X, y, groups, n_splits=3, inner_splits=2
        )
        assert len(result) == 3
        assert len(seen) == 9  # Two inner fits and one outer fit per fold.
        for calls, (tr, va) in zip(
            (seen[i:i + 3] for i in range(0, len(seen), 3)),
            preprocessing.cv_splitter(n_splits=3).split(X, y, groups),
            strict=True,
        ):
            assert calls[-1] == set(X.iloc[tr].index)
            for fit_indices in calls:
                assert fit_indices.isdisjoint(X.iloc[va].index)
                assert set(groups.loc[list(fit_indices)]).isdisjoint(set(groups.iloc[va]))

    def test_holdout_transform_does_not_refit(self, synthetic_df):
        """Transforming the test set must not change learned parameters."""
        split = patient_level_split(synthetic_df)
        X_train, y_train = split_features_target(split.train)
        X_test, _ = split_features_target(split.test)

        pipeline = preprocessing.make_full_pipeline(get_model("logistic_regression"))
        pipeline.fit(X_train, y_train)
        assert preprocessing.assert_no_fit_on_holdout(pipeline, X_test) is True

    def test_scaler_statistics_come_from_train_only(self, synthetic_df):
        """The scaler's mean_ must match the TRAIN set's mean (after imputation)."""
        split = patient_level_split(synthetic_df)
        X_train, y_train = split_features_target(split.train)

        pipeline = preprocessing.make_full_pipeline(
            get_model("logistic_regression"), use_smote=False, numeric_imputer="median"
        )
        pipeline.fit(X_train, y_train)

        pre = pipeline.named_steps["preprocessor"]
        scaler = pre.named_transformers_["num"].named_steps["scaler"]
        expected = X_train[config.NUMERIC_FEATURES].fillna(
            X_train[config.NUMERIC_FEATURES].median()
        ).mean().to_numpy()
        np.testing.assert_allclose(scaler.mean_, expected, rtol=1e-6)

    def test_smote_does_not_touch_holdout_size(self, synthetic_df):
        """SMOTE lives inside the pipeline -> record count is unchanged at predict time."""
        split = patient_level_split(synthetic_df)
        X_train, y_train = split_features_target(split.train)
        X_test, y_test = split_features_target(split.test)

        pipeline = preprocessing.make_full_pipeline(get_model("logistic_regression"))
        pipeline.fit(X_train, y_train)
        assert len(pipeline.predict(X_test)) == len(y_test)

    def test_stacking_smote_runs_inside_each_base_learner(self, synthetic_df):
        """SMOTE must sit inside each base learner, never before the stacking's internal CV."""
        split = patient_level_split(synthetic_df)
        X_train, y_train = split_features_target(split.train)
        X_test, _ = split_features_target(split.test)

        stacking = build_stacking_ensemble(
            base_wrapper=lambda est, scale: preprocessing.make_full_pipeline(est, scale=scale)
        )
        for _name, base in stacking.estimators:
            assert "smote" in base.named_steps

        stacking.fit(X_train, y_train, groups=split.train[config.ID_COLUMN])
        for fitted in stacking.estimators_:
            assert preprocessing.assert_no_fit_on_holdout(fitted, X_test)

    def test_cv_splitter_respects_groups(self, multi_record_df):
        """StratifiedGroupKFold never splits a patient group across two folds."""
        X, y = split_features_target(multi_record_df)
        groups = multi_record_df[config.ID_COLUMN]
        splitter = preprocessing.cv_splitter(n_splits=5)
        for tr, va in splitter.split(X, y, groups=groups):
            assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[va]))


class TestIdentifierLeakage:
    def test_id_column_excluded_from_features(self, synthetic_df):
        """`id` must never end up in the feature matrix X."""
        X, _y = split_features_target(synthetic_df)
        assert config.ID_COLUMN not in X.columns
        assert config.TARGET_COLUMN not in X.columns

    def test_target_not_in_preprocessor_columns(self):
        """The preprocessor must never reference the target column."""
        pre = preprocessing.make_preprocessor()
        referenced = [c for _n, _t, cols in pre.transformers for c in cols]
        assert config.TARGET_COLUMN not in referenced
        assert config.ID_COLUMN not in referenced


def test_stacking_inner_folds_keep_patients_together(multi_record_df):
    X, y = split_features_target(multi_record_df)
    groups = multi_record_df[config.ID_COLUMN]
    assert groups.duplicated().any()
    stacking = build_stacking_ensemble(
        estimators=[("lr", preprocessing.make_full_pipeline(get_model("logistic_regression")))],
        cv=3,
    )
    stacking.fit(X, y, groups=groups)
    seen = np.zeros(len(X), dtype=int)
    for tr, va in stacking.inner_splits_:
        assert set(groups.iloc[tr]).isdisjoint(groups.iloc[va])
        seen[va] += 1
    assert (seen == 1).all()
    assert stacking.predict_proba(X).shape == (len(X), 2)
    with pytest.raises(ValueError, match="groups"):
        stacking.fit(X, y)
