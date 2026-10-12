"""Unit tests for the ZERO DATA LEAKAGE discipline.

This is the most important test suite in the project: it is the technical
safeguard against the mistake several papers made (applying SMOTE/scaling
before splitting, or splitting randomly at the record level on data where a
single patient can have multiple records).
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config, preprocessing
from src.data_loader import split_features_target
from src.models import get_model
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
