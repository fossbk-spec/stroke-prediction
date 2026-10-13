"""Fast checks for the five protocol implementations."""

import pandas as pd
import pytest

from src import config
from src.leakage_experiment import (
    ALL,
    METRICS,
    _group_folds,
    main,
    run_leakage_experiment,
    summarize_leakage,
)


@pytest.fixture(scope="module")
def results(synthetic_df):
    return run_leakage_experiment(
        synthetic_df, ["logistic_regression"], n_splits=2, n_repeats=1,
    )


def test_all_protocols_and_metrics(results):
    assert len(results) == len(ALL) * 2
    assert set(results.protocol) == set(ALL)
    assert set(METRICS).issubset(results.columns)
    assert results.groupby("protocol").size().eq(2).all()


def test_synthetic_validation_rows(results):
    synthetic = results.set_index("protocol").n_synthetic_in_val
    assert (synthetic.loc["smote_before_split"] > 0).all()
    for protocol in ("correct", "preprocess_before_split", "balanced_test", "no_resampling_baseline"):
        assert (synthetic.loc[protocol] == 0).all()


def test_balanced_validation(results):
    balanced = results[results.protocol == "balanced_test"]
    assert (balanced.n_val == 2 * balanced.n_val_positive).all()


def test_correct_folds_are_real_and_patient_disjoint(multi_record_df):
    correct = run_leakage_experiment(
        multi_record_df, ["logistic_regression"], protocols=["correct"],
        n_splits=2, n_repeats=1,
    ).sort_values("fold")
    assert (correct.n_synthetic_in_val == 0).all()
    X = multi_record_df[config.FEATURE_COLUMNS]
    y = multi_record_df[config.TARGET_COLUMN]
    groups = multi_record_df[config.ID_COLUMN]
    for (_, row), (train_idx, val_idx) in zip(
        correct.iterrows(), _group_folds(X, y, groups, 2, config.RANDOM_STATE), strict=True,
    ):
        assert set(groups.iloc[train_idx]).isdisjoint(groups.iloc[val_idx])
        assert row.n_val == len(val_idx)
        assert row.n_val_positive == int(y.iloc[val_idx].sum())


def test_summary_has_paired_deltas(results):
    summary = summarize_leakage(results)
    assert len(summary) == len(ALL) * len(METRICS)
    assert (summary.loc[summary.protocol == "correct", "delta_vs_p0"] == 0).all()
    assert (summary.loc[summary.protocol == "correct", "delta_ci_lower"] == 0).all()
    expected = (
        results[results.protocol == "balanced_test"].roc_auc.mean()
        - results[results.protocol == "correct"].roc_auc.mean()
    )
    actual = summary.query("protocol == 'balanced_test' and metric == 'roc_auc'").delta_vs_p0.iloc[0]
    assert actual == pytest.approx(expected)
    assert summary.query("protocol == 'correct' and metric == 'recall'").ci_lower.isna().all()


def test_cli_writes_outputs(synthetic_df, tmp_path):
    csv = tmp_path / "input.csv"
    output = tmp_path / "report"
    synthetic_df.drop(columns=[config.ID_COLUMN]).to_csv(csv, index=False)
    main(["--models", "logistic_regression", "--n-repeats", "1", "--n-splits", "2",
          "--data", str(csv), "--output", str(output)])
    assert (output / "leakage_results.csv").is_file()
    assert (output / "leakage_summary.csv").is_file()
    assert len(pd.read_csv(output / "leakage_results.csv")) == len(ALL) * 2
    pytest.importorskip("matplotlib")
    assert (output / "figures" / "leakage_inflation.png").is_file()
