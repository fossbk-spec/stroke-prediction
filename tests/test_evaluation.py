"""Unit test cho bo do do va co che chon nguong quyet dinh."""

from __future__ import annotations

import numpy as np
import pytest

from src import evaluation


@pytest.fixture
def imbalanced_labels():
    """100 mau, 5 ca duong tinh (giong ty le 4.9% cua bo du lieu that)."""
    y = np.zeros(100, dtype=int)
    y[:5] = 1
    return y


class TestMetrics:
    def test_naive_all_negative_predictor_is_exposed(self, imbalanced_labels):
        """Mo hinh 'luon du doan 0': Accuracy 95% nhung Recall = 0 -> phai thay ro."""
        y_proba = np.zeros(100)
        m = evaluation.compute_metrics(imbalanced_labels, y_proba, threshold=0.5)
        assert m["accuracy"] == pytest.approx(0.95)
        assert m["recall"] == 0.0
        assert m["n_false_negative"] == 5.0
        assert m["accuracy"] == pytest.approx(m["null_accuracy"])

    def test_perfect_predictor(self, imbalanced_labels):
        m = evaluation.compute_metrics(imbalanced_labels, imbalanced_labels.astype(float))
        assert m["recall"] == 1.0
        assert m["precision"] == 1.0
        assert m["roc_auc"] == 1.0
        assert m["n_false_negative"] == 0.0

    def test_f2_weights_recall_above_precision(self):
        """F2 cua (Recall cao, Precision thap) > F2 cua (Recall thap, Precision cao)."""
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        high_recall = np.array([0.6, 0.6, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9])  # R=1.0, P=0.67
        high_precision = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.2, 0.2, 0.2])  # R=0.25, P=1.0
        f2_recall = evaluation.compute_metrics(y, high_recall, threshold=0.5)["f2"]
        f2_precision = evaluation.compute_metrics(y, high_precision, threshold=0.5)["f2"]
        assert f2_recall > f2_precision

    def test_specificity_and_balanced_accuracy(self, imbalanced_labels):
        y_proba = np.full(100, 0.9)  # du doan tat ca la duong tinh
        m = evaluation.compute_metrics(imbalanced_labels, y_proba, threshold=0.5)
        assert m["recall"] == 1.0
        assert m["specificity"] == 0.0
        assert m["balanced_accuracy"] == pytest.approx(0.5)

    def test_metrics_include_mandatory_clinical_set(self, imbalanced_labels):
        m = evaluation.compute_metrics(imbalanced_labels, np.random.default_rng(0).random(100))
        for key in ("recall", "pr_auc", "f2", "roc_auc", "specificity", "n_false_negative"):
            assert key in m


class TestThresholdTuning:
    def test_best_threshold_beats_default_on_fbeta(self):
        rng = np.random.default_rng(1)
        y = rng.binomial(1, 0.05, 2000)
        proba = np.clip(0.05 + 0.5 * y + rng.normal(0, 0.2, 2000), 0, 1)

        best_t, best_f2 = evaluation.find_best_threshold(y, proba, beta=2)
        default_f2 = evaluation.compute_metrics(y, proba, threshold=0.5, beta=2)["f2"]
        assert 0.0 <= best_t <= 1.0
        assert best_f2 >= default_f2

    def test_min_recall_constraint_is_satisfied(self):
        rng = np.random.default_rng(2)
        y = rng.binomial(1, 0.1, 1000)
        proba = np.clip(0.1 + 0.4 * y + rng.normal(0, 0.25, 1000), 0, 1)

        t = evaluation.threshold_for_min_recall(y, proba, min_recall=0.9)
        achieved = evaluation.compute_metrics(y, proba, threshold=t)["recall"]
        assert achieved >= 0.9 - 1e-9

    def test_lower_threshold_never_decreases_recall(self):
        rng = np.random.default_rng(3)
        y = rng.binomial(1, 0.2, 500)
        proba = rng.random(500)
        recalls = [
            evaluation.compute_metrics(y, proba, threshold=t)["recall"]
            for t in (0.9, 0.7, 0.5, 0.3, 0.1)
        ]
        assert recalls == sorted(recalls)


class TestModelEvaluation:
    def test_evaluate_model_end_to_end(self, synthetic_df):
        from src.data_loader import split_features_target
        from src.models import get_model
        from src.preprocessing import make_full_pipeline, patient_level_split

        split = patient_level_split(synthetic_df)
        X_train, y_train = split_features_target(split.train)
        X_test, y_test = split_features_target(split.test)

        pipeline = make_full_pipeline(get_model("logistic_regression"))
        pipeline.fit(X_train, y_train)

        result = evaluation.evaluate_model(pipeline, X_test, y_test, model_name="lr")
        assert result.confusion.shape == (2, 2)
        assert result.confusion.sum() == len(y_test)
        assert 0.0 <= result.metrics["recall"] <= 1.0

    def test_comparison_table_sorted_by_recall(self, synthetic_df):
        results = [
            evaluation.EvaluationResult("a", 0.5, {"recall": 0.2, "pr_auc": 0.3}, np.zeros((2, 2))),
            evaluation.EvaluationResult("b", 0.5, {"recall": 0.8, "pr_auc": 0.4}, np.zeros((2, 2))),
        ]
        table = evaluation.comparison_table(results)
        assert table.iloc[0]["model"] == "b"

    def test_grouped_cv_runs_and_returns_all_folds(self, synthetic_df):
        from src import config
        from src.data_loader import split_features_target
        from src.models import get_model
        from src.preprocessing import make_full_pipeline

        X, y = split_features_target(synthetic_df)
        pipeline = make_full_pipeline(get_model("logistic_regression"))
        cv = evaluation.cross_validate_grouped(
            pipeline, X, y, synthetic_df[config.ID_COLUMN], n_splits=3
        )
        assert len(cv) == 3
        assert "recall" in cv.columns
        assert evaluation.summarize_cv(cv).notna().all()
