"""Fast checks for the leakage experiment and its fit boundaries."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t
from sklearn.compose import ColumnTransformer

from src import config
from src.leakage_experiment import (
    ALL,
    METRICS,
    _group_folds,
    main,
    run_leakage_experiment,
    summarize_leakage,
)
from src.preprocessing import AdaptiveSMOTE


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


def test_p1a_unmatched_and_p1b_matched(results):
    a = results.query("protocol == 'smote_before_split'")
    b = results.query("protocol == 'smote_before_split_matched'")
    p0 = results.query("protocol == 'correct'")
    assert (a.n_synthetic_in_val > 0).all()
    assert (b.n_synthetic_in_val == 0).all()
    assert (b.n_synthetic_train > 0).all()
    assert (b.n_synthetic_train_val_parent >= 0).all()
    for (_, actual), (_, clean) in zip(
        b.sort_values("fold").iterrows(), p0.sort_values("fold").iterrows(), strict=True
    ):
        assert actual.n_val == clean.n_val
        assert json.loads(actual.val_ids) == json.loads(clean.val_ids)


def test_p3_uses_p0_probabilities_for_selected_rows(results):
    p0 = results.query("protocol == 'correct'").set_index("fold")
    p3 = results.query("protocol == 'balanced_test'").set_index("fold")
    for fold, row in p3.iterrows():
        ids = json.loads(p0.loc[fold].val_ids)
        probabilities = json.loads(p0.loc[fold].val_proba)
        selected = json.loads(row.val_ids)
        expected = [probabilities[ids.index(patient)] for patient in selected]
        assert json.loads(row.val_proba) == pytest.approx(expected)
        assert row.n_val == 2 * row.n_val_positive
        assert row.recall == p0.loc[fold].recall


def test_correct_folds_are_real_and_patient_disjoint(multi_record_df):
    correct = run_leakage_experiment(
        multi_record_df, ["logistic_regression"], protocols=["correct"],
        n_splits=2, n_repeats=1,
    ).sort_values("fold")
    X = multi_record_df[config.FEATURE_COLUMNS]
    y = multi_record_df[config.TARGET_COLUMN]
    groups = multi_record_df[config.ID_COLUMN]
    for (_, row), (train_idx, val_idx) in zip(
        correct.iterrows(), _group_folds(X, y, groups, 2, config.RANDOM_STATE), strict=True,
    ):
        assert set(groups.iloc[train_idx]).isdisjoint(groups.iloc[val_idx])
        assert row.n_val == len(val_idx)
        assert row.n_val_positive == int(y.iloc[val_idx].sum())
        assert (row.n_synthetic_in_val, row.n_synthetic_train_val_parent) == (0, 0)


def _fit_spy(monkeypatch):
    """Collect DataFrame inputs to every column-transformer fit and SMOTENC."""
    calls = {"transformer": [], "fit": [], "fit_transform": [], "smote": []}
    fit = ColumnTransformer.fit
    fit_transform = ColumnTransformer.fit_transform
    resample = AdaptiveSMOTE.fit_resample

    def fit_hook(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame):
            calls["transformer"].append(X.copy())
            calls["fit"].append(X.copy())
        return fit(self, X, *args, **kwargs)

    def fit_transform_hook(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame):
            calls["transformer"].append(X.copy())
            calls["fit_transform"].append(X.copy())
        return fit_transform(self, X, *args, **kwargs)

    def resample_hook(self, X, y, **kwargs):
        calls["smote"].append(X.copy())
        return resample(self, X, y, **kwargs)

    monkeypatch.setattr(ColumnTransformer, "fit", fit_hook)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", fit_transform_hook)
    monkeypatch.setattr(AdaptiveSMOTE, "fit_resample", resample_hook)
    return calls


def _marked_holdout(synthetic_df, monkeypatch):
    """Mark one grouped validation fold with an out-of-range sentinel."""
    from src import leakage_experiment as module

    df = synthetic_df.copy()
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]
    train_idx, val_idx = _group_folds(X, y, df[config.ID_COLUMN], 2, config.RANDOM_STATE)[0]
    marker = 9999.0
    df.loc[val_idx, "avg_glucose_level"] = marker
    original = module._group_folds
    monkeypatch.setattr(module, "_group_folds",
                        lambda X, y, groups, n_splits, seed: [(train_idx, val_idx)])
    return df, marker, original


def test_p0_never_fits_validation_rows(synthetic_df, monkeypatch):
    df, marker, _ = _marked_holdout(synthetic_df, monkeypatch)
    calls = _fit_spy(monkeypatch)
    run_leakage_experiment(df, ["logistic_regression"], protocols=["correct"],
                           n_splits=2, n_repeats=1)
    assert calls["transformer"] and calls["smote"]
    for frame in calls["transformer"] + calls["smote"]:
        assert marker not in frame["avg_glucose_level"].to_numpy()


def test_p0_training_unchanged_by_adversarial_holdout(synthetic_df, monkeypatch):
    from src import leakage_experiment as module

    marked, _, _ = _marked_holdout(synthetic_df, monkeypatch)
    original_fit_score = module._fit_score
    training = []

    def capture(estimator, train_X, train_y, val_X):
        training.append((pd.DataFrame(train_X).copy(), np.asarray(train_y).copy()))
        return original_fit_score(estimator, train_X, train_y, val_X)

    monkeypatch.setattr(module, "_fit_score", capture)
    for frame in (synthetic_df, marked):
        run_leakage_experiment(frame, ["logistic_regression"], protocols=["correct"],
                               n_splits=2, n_repeats=1)
    pd.testing.assert_frame_equal(training[0][0], training[1][0])
    np.testing.assert_array_equal(training[0][1], training[1][1])


def test_p2_validation_in_imputer_and_encoder_but_not_smote(synthetic_df, monkeypatch):
    df, marker, _ = _marked_holdout(synthetic_df, monkeypatch)
    calls = _fit_spy(monkeypatch)
    run_leakage_experiment(df, ["logistic_regression"],
                           protocols=["preprocess_before_split"],
                           n_splits=2, n_repeats=1)
    # The first fit sees all raw records; the final encoder sees resampled
    # training rows followed by the prepared validation rows.
    assert any(len(frame) == len(df) and
               marker in frame["avg_glucose_level"].to_numpy()
               for frame in calls["fit_transform"])  # global imputer
    assert any(marker in frame["avg_glucose_level"].to_numpy()
               for frame in calls["fit"])  # final encoder/scaler
    assert calls["smote"]
    assert all(marker not in frame["avg_glucose_level"].to_numpy()
               for frame in calls["smote"])


def test_p4_never_resamples(synthetic_df, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("P4 called fit_resample")
    monkeypatch.setattr(AdaptiveSMOTE, "fit_resample", forbidden)
    run_leakage_experiment(synthetic_df, ["logistic_regression"],
                           protocols=["no_resampling_baseline"],
                           n_splits=2, n_repeats=1)


def _toy_result(protocol, repeat, fold, value):
    row = {"protocol": protocol, "model": "m", "repeat": repeat, "fold": fold,
           "n_train": 80, "n_val": 20}
    row.update(dict.fromkeys(METRICS, value))
    return row


def test_summary_pairs_folds_drops_nan_and_corrects_interval():
    rows = []
    base = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    deltas = [0.01, 0.04, 0.02, 0.09, 0.03, 0.08]
    for i, (b, d) in enumerate(zip(base, deltas, strict=True)):
        repeat, fold = divmod(i, 3)
        rows.extend((_toy_result("correct", repeat, fold, b),
                     _toy_result("preprocess_before_split", repeat, fold,
                                 np.nan if i == 4 else b + d),
                     _toy_result("smote_before_split", repeat, fold, b + 0.5)))
    summary = summarize_leakage(pd.DataFrame(rows))
    p2 = summary.query("protocol == 'preprocess_before_split' and metric == 'accuracy'").iloc[0]
    valid = np.array([d for i, d in enumerate(deltas) if i != 4])
    assert p2.delta_vs_p0 == pytest.approx(valid.mean())
    assert (p2.n_valid_folds, p2.n_valid_repeats) == (5, 2)
    assert p2["mean"] == pytest.approx(np.mean([
        np.mean([base[i] + deltas[i] for i in range(3)]),
        np.mean([base[i] + deltas[i] for i in (3, 5)]),
    ]))
    naive_half = t.ppf(0.975, 4) * valid.std(ddof=1) / np.sqrt(5)
    assert p2.delta_ci_upper - p2.delta_vs_p0 > naive_half
    assert p2.ci_method == "Nadeau-Bengio corrected resampled t"
    p1a = summary.query("protocol == 'smote_before_split' and metric == 'accuracy'").iloc[0]
    assert p1a.comparison == "unmatched"
    assert np.isnan(p1a.delta_vs_p0) and np.isnan(p1a.delta_ci_lower)
    assert p1a.repeat_min <= p1a["mean"] <= p1a.repeat_max


def test_summary_excludes_wholly_invalid_repeat():
    rows = [
        _toy_result("correct", 0, 0, 0.2),
        _toy_result("preprocess_before_split", 0, 0, 0.3),
        _toy_result("correct", 1, 0, 0.4),
        _toy_result("preprocess_before_split", 1, 0, np.nan),
    ]
    summary = summarize_leakage(pd.DataFrame(rows))
    p2 = summary.query("protocol == 'preprocess_before_split' and metric == 'accuracy'").iloc[0]
    assert p2["mean"] == pytest.approx(0.3)
    assert p2.delta_vs_p0 == pytest.approx(0.1)
    assert (p2.n_valid_folds, p2.n_valid_repeats) == (1, 1)
    assert np.isnan(p2.delta_ci_lower)


def test_cli_protocol_subset_and_benchmark(synthetic_df, tmp_path, capsys):
    csv = tmp_path / "input.csv"
    output = tmp_path / "report"
    synthetic_df.drop(columns=[config.ID_COLUMN]).to_csv(csv, index=False)
    args = ["--models", "logistic_regression", "--n-repeats", "1", "--n-splits", "2",
            "--protocols", "correct", "no_resampling_baseline",
            "--data", str(csv), "--output", str(output)]
    main(args)
    assert set(pd.read_csv(output / "leakage_results.csv").protocol) == {
        "correct", "no_resampling_baseline"}
    assert (output / "leakage_summary.csv").is_file()
    benchmark_output = tmp_path / "benchmark"
    main(["--models", "logistic_regression", "--n-splits", "2",
          "--protocols", "correct", "no_resampling_baseline",
          "--data", str(csv), "--output", str(benchmark_output), "--benchmark"])
    assert "fit" in capsys.readouterr().out
    assert not benchmark_output.exists()
    pytest.importorskip("matplotlib")
    assert (output / "figures" / "leakage_inflation.png").is_file()
