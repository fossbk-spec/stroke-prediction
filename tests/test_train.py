"""Light integration coverage for CLI threshold policy propagation."""

from src import evaluation, train
from src.data_loader import make_synthetic_dataset


def test_cli_min_recall_reaches_cv_policy(monkeypatch, tmp_path):
    df = make_synthetic_dataset(n_rows=120, positive_rate=0.25)
    monkeypatch.setattr(train, "make_synthetic_dataset", lambda **kwargs: df)
    original_cv = evaluation.cross_validate_grouped
    original_policy = evaluation.threshold_for_min_recall
    cv_policies = []
    cv_threshold_calls = []
    in_cv = False

    def policy(*args, **kwargs):
        if in_cv:
            cv_threshold_calls.append(kwargs["min_recall"])
        return original_policy(*args, **kwargs)

    def cross_validate(*args, **kwargs):
        nonlocal in_cv
        cv_policies.append(kwargs["min_recall"])
        in_cv = True
        try:
            return original_cv(*args, **kwargs, n_splits=2, inner_splits=2)
        finally:
            in_cv = False

    monkeypatch.setattr(evaluation, "threshold_for_min_recall", policy)
    monkeypatch.setattr(evaluation, "cross_validate_grouped", cross_validate)
    assert (
        train.main(
            [
                "--synthetic",
                "--models",
                "naive_bayes",
                "--no-smote",
                "--min-recall",
                "0.8",
                "--n-boot",
                "20",
                "--cv-repeats",
                "1",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert cv_policies == [0.8]
    assert cv_threshold_calls == [0.8, 0.8]
    assert (tmp_path / "cv_naive_bayes.csv").exists()
    for relative in (
        "test_bootstrap_ci_naive_bayes.csv", "test_bootstrap_ci.csv",
        "calibration_naive_bayes.csv", "figures/calibration_naive_bayes.png",
        "subgroup_naive_bayes.csv",
    ):
        assert (tmp_path / relative).exists()


def test_stacking_groups_reach_training_and_nested_cv(monkeypatch):
    from src import models, preprocessing

    df = make_synthetic_dataset(n_rows=180, positive_rate=0.3)
    # Repeated records must remain grouped through training and both CV levels.
    import pandas as pd
    df = pd.concat([df, df], ignore_index=True)
    builder = models.build_stacking_ensemble
    original_cv = evaluation.cross_validate_grouped

    def lightweight_stacking(**kwargs):
        return builder(
            estimators=[("lr", preprocessing.make_full_pipeline(
                models.get_model("logistic_regression"), use_smote=False))],
            cv=2,
        )

    def small_cv(*args, **kwargs):
        return original_cv(*args, **kwargs, n_splits=2, inner_splits=2)

    monkeypatch.setattr(models, "build_stacking_ensemble", lightweight_stacking)
    monkeypatch.setattr(evaluation, "cross_validate_grouped", small_cv)
    table, cv = train.run_experiment(df, ["stacking"], use_smote=False)
    assert table["model"].tolist() == ["stacking"]
    assert len(cv["stacking"]) == 2
