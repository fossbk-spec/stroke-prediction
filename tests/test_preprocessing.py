"""Unit test cho Imputer, Encoder, Scaler va SMOTE."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, preprocessing
from src.data_loader import split_features_target
from src.models import get_model


class TestImputer:
    def test_no_nan_after_transform(self, synthetic_df):
        """Sau tien xu ly, ma tran dac trung khong con gia tri khuyet."""
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor()
        matrix = pre.fit_transform(X, y)
        assert not np.isnan(np.asarray(matrix, dtype=float)).any()

    def test_missing_bmi_rows_are_kept(self, synthetic_df):
        """Khong duoc xoa ban ghi thieu BMI (tranh mat mau lop hiem)."""
        X, y = split_features_target(synthetic_df)
        assert X["bmi"].isna().sum() > 0, "Fixture phai co gia tri BMI khuyet"
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
        """`ever_married`: No -> 0, Yes -> 1 (dung 1 cot, khong one-hot)."""
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor()
        frame = pd.DataFrame(pre.fit_transform(X, y), columns=pre.get_feature_names_out())
        assert "ever_married" in frame.columns
        assert set(frame["ever_married"].unique()) <= {0.0, 1.0}

    def test_nominal_features_are_one_hot(self, synthetic_df):
        """Cac bien danh nghia duoc tach thanh cot 0/1."""
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor()
        names = list(pre.fit(X, y).get_feature_names_out())
        assert any(n.startswith("gender_") for n in names)
        assert any(n.startswith("work_type_") for n in names)
        assert any(n.startswith("smoking_status_") for n in names)

    def test_unknown_smoking_kept_as_own_category(self, synthetic_df):
        """'Unknown' la mot hang muc rieng, khong gop vao 'never smoked'."""
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor()
        names = list(pre.fit(X, y).get_feature_names_out())
        assert "smoking_status_Unknown" in names

    def test_unseen_category_at_inference_does_not_crash(self, synthetic_df):
        """Hang muc chua tung xuat hien o train khong duoc lam sap inference."""
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor()
        pre.fit(X, y)

        X_new = X.iloc[:5].copy()
        X_new.loc[:, "work_type"] = "Astronaut"  # hang muc moi hoan toan
        assert pre.transform(X_new).shape[0] == 5


class TestScaler:
    def test_train_is_standardized(self, synthetic_df):
        """Bien lien tuc tren tap train co mean ~ 0, std ~ 1."""
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor()
        frame = pd.DataFrame(pre.fit_transform(X, y), columns=pre.get_feature_names_out())
        for col in config.NUMERIC_FEATURES:
            assert abs(frame[col].mean()) < 1e-6
            assert abs(frame[col].std(ddof=0) - 1.0) < 1e-6

    def test_scaling_can_be_disabled_for_tree_models(self, synthetic_df):
        """Mo hinh cay khong can chuan hoa -> `scale=False` phai bo scaler."""
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor(scale=False)
        pre.fit(X, y)
        assert "scaler" not in pre.named_transformers_["num"].named_steps

    def test_holdout_is_not_restandardized(self, synthetic_df):
        """Tap test KHONG co mean 0 / std 1 — bang chung la no dung tham so train."""
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
        """SMOTE dua ty le lop ve 50:50 tren du lieu train."""
        from imblearn.over_sampling import SMOTE

        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor()
        matrix = pre.fit_transform(X, y)
        X_res, y_res = SMOTE(random_state=config.RANDOM_STATE).fit_resample(matrix, y)
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


class TestPipelineIntegration:
    def test_feature_names_match_matrix_width(self, synthetic_df):
        X, y = split_features_target(synthetic_df)
        pre = preprocessing.make_preprocessor()
        matrix = pre.fit_transform(X, y)
        assert len(preprocessing.get_feature_names(pre)) == matrix.shape[1]

    def test_pipeline_is_clonable_for_cross_validation(self, synthetic_df):
        """Pipeline phai clone duoc, neu khong `cross_val_score` se dung lai state."""
        from sklearn.base import clone

        pipeline = preprocessing.make_full_pipeline(get_model("random_forest"))
        assert clone(pipeline) is not pipeline
