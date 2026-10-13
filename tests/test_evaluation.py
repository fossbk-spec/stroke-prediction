"""Unit tests for the metric set and decision threshold tuning."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src import evaluation


@pytest.fixture
def imbalanced_labels():
    """100 samples, 5 positive cases (matching the real dataset's ~4.9% rate)."""
    y = np.zeros(100, dtype=int)
    y[:5] = 1
    return y


class TestMetrics:
    def test_average_precision_and_f2_match_hand_calculation(self):
        y = np.array([1, 0, 1, 0])
        proba = np.array([0.9, 0.8, 0.4, 0.1])
        metrics = evaluation.compute_metrics(y, proba, threshold=0.5, beta=2)
        # Positive ranks 1 and 3: AP = (1 + 2/3) / 2.
        assert metrics["pr_auc"] == pytest.approx(5 / 6)
        # TP=1, FP=1, FN=1: F2 = 5*TP / (5*TP + 4*FN + FP).
        assert metrics["f2"] == pytest.approx(0.5)

    def test_naive_all_negative_predictor_is_exposed(self, imbalanced_labels):
        """An 'always predict 0' model: 95% Accuracy but Recall = 0 -> must be exposed."""
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
        """F2 of (high Recall, low Precision) > F2 of (low Recall, high Precision)."""
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        high_recall = np.array([0.6, 0.6, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9])  # R=1.0, P=0.67
        high_precision = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.2, 0.2, 0.2])  # R=0.25, P=1.0
        f2_recall = evaluation.compute_metrics(y, high_recall, threshold=0.5)["f2"]
        f2_precision = evaluation.compute_metrics(y, high_precision, threshold=0.5)["f2"]
        assert f2_recall > f2_precision

    def test_specificity_and_balanced_accuracy(self, imbalanced_labels):
        y_proba = np.full(100, 0.9)  # predict positive for everyone
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

    def test_grouped_cv_with_decision_function_returns_all_folds(self, synthetic_df):
        from sklearn.svm import LinearSVC

        from src import config
        from src.data_loader import split_features_target
        from src.preprocessing import make_full_pipeline

        X, y = split_features_target(synthetic_df)
        pipeline = make_full_pipeline(LinearSVC())
        cv = evaluation.cross_validate_grouped(
            pipeline, X, y, synthetic_df[config.ID_COLUMN], n_splits=3
        )
        assert len(cv) == 3

    def test_grouped_cv_threshold_not_tuned_on_in_sample_predictions(
        self, synthetic_df, monkeypatch
    ):
        """The per-fold threshold must come from out-of-fold (not in-sample) predictions."""
        from sklearn.tree import DecisionTreeClassifier

        from src import config
        from src.data_loader import split_features_target
        from src.preprocessing import make_full_pipeline

        X, y = split_features_target(synthetic_df)
        pipeline = make_full_pipeline(DecisionTreeClassifier(random_state=0), scale=False)
        seen = []
        original = evaluation.find_best_threshold

        def spy(y_true, y_proba, **kwargs):
            seen.append((np.asarray(y_true), np.asarray(y_proba)))
            return original(y_true, y_proba, **kwargs)

        monkeypatch.setattr(evaluation, "find_best_threshold", spy)
        evaluation.cross_validate_grouped(
            pipeline, X, y, synthetic_df[config.ID_COLUMN], n_splits=3
        )
        assert len(seen) == 3
        # An unpruned tree ranks its own training data ~perfectly; OOF predictions do not.
        for y_true, proba in seen:
            assert roc_auc_score(y_true, proba) < 0.99
