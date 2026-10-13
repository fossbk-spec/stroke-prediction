"""Repeated comparisons of clean and deliberately leaky stroke CV protocols.

The leaky protocols are methodological demonstrations, not valid estimates of
deployment performance. Each repeat uses seed ``config.RANDOM_STATE + repeat``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.model_selection import StratifiedKFold

from src import config
from src.data_loader import load_raw_data, split_features_target, validate_schema
from src.evaluation import compute_metrics, predict_proba_positive
from src.models import get_model
from src.preprocessing import cv_splitter, make_full_pipeline

ALL = (
    "correct",
    "smote_before_split",
    "preprocess_before_split",
    "balanced_test",
    "no_resampling_baseline",
)
METRICS = ("accuracy", "recall", "precision", "f2", "specificity", "roc_auc", "pr_auc", "brier")
COUNT_COLUMNS = ("n_val", "n_val_positive", "n_synthetic_in_val")
SCALE_MODELS = frozenset(("logistic_regression", "svm", "knn", "naive_bayes"))


def _estimator(name: str, seed: int):
    estimator = get_model(name)
    if "random_state" in estimator.get_params(deep=False):
        estimator.set_params(random_state=seed)
    return estimator


def _pipeline(name: str, seed: int, *, use_smote: bool = True):
    return make_full_pipeline(
        _estimator(name, seed), use_smote=use_smote,
        scale=name in SCALE_MODELS, random_state=seed,
    )


def _group_folds(X, y, groups, n_splits: int, seed: int):
    return list(cv_splitter(n_splits=n_splits, random_state=seed).split(X, y, groups))


def _score(y, proba, *, protocol, model, repeat, fold, n_synthetic=0):
    scores = compute_metrics(y, proba, threshold=0.5, beta=2)
    return {
        "protocol": protocol, "model": model, "repeat": repeat, "fold": fold,
        **{metric: scores[metric] for metric in METRICS},
        "n_val": len(y), "n_val_positive": int(np.sum(y)),
        "n_synthetic_in_val": int(n_synthetic),
    }


def run_leakage_experiment(
    df: pd.DataFrame, model_names, protocols=ALL, n_splits: int = 5, n_repeats: int = 5,
) -> pd.DataFrame:
    """Return one row per protocol, model, repeat, and fold.

    P0/P2/P3/P4 share grouped folds. P1 splits the globally resampled rows
    with ordinary stratified CV; appended SMOTENC rows are marked synthetic.
    """
    protocols = tuple(dict.fromkeys(protocols))
    model_names = tuple(dict.fromkeys(model_names))
    unknown = set(protocols) - set(ALL)
    if unknown:
        raise ValueError(f"Unknown protocol(s): {sorted(unknown)}")
    if n_splits < 2 or n_repeats < 1:
        raise ValueError("n_splits must be >= 2 and n_repeats must be >= 1")
    if not protocols or not model_names:
        raise ValueError("Specify at least one protocol and model")
    if config.ID_COLUMN not in df:
        df = df.copy()
        df[config.ID_COLUMN] = np.arange(len(df))
    validate_schema(df)
    X, y = split_features_target(df.reset_index(drop=True))
    groups = df[config.ID_COLUMN].reset_index(drop=True)
    rows = []

    for repeat in range(n_repeats):
        seed = config.RANDOM_STATE + repeat
        grouped_folds = _group_folds(X, y, groups, n_splits, seed)
        for name in model_names:
            if "smote_before_split" in protocols:
                pipe = _pipeline(name, seed)
                # Both imputers/encoders see all records. SMOTENC also sees
                # all labels before the split, creating synthetic test rows.
                prepared = pipe.named_steps["prepare"].fit_transform(X)
                resampled, y_resampled = pipe.named_steps["smote"].fit_resample(prepared, y)
                encoded = pipe.named_steps["preprocessor"].fit_transform(resampled)
                synthetic = np.arange(len(y_resampled)) >= len(y)
                p1_folds = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=seed,
                ).split(encoded, y_resampled)
                for fold, (train_idx, val_idx) in enumerate(p1_folds):
                    estimator = _estimator(name, seed)
                    estimator.fit(encoded[train_idx], np.asarray(y_resampled)[train_idx])
                    proba = predict_proba_positive(estimator, encoded[val_idx])
                    rows.append(_score(
                        np.asarray(y_resampled)[val_idx], proba,
                        protocol="smote_before_split", model=name, repeat=repeat,
                        fold=fold, n_synthetic=synthetic[val_idx].sum(),
                    ))

            if "preprocess_before_split" in protocols:
                pipe = _pipeline(name, seed)
                prepared = pipe.named_steps["prepare"].fit_transform(X)
                encoder = pipe.named_steps["preprocessor"].fit(prepared)
                for fold, (train_idx, val_idx) in enumerate(grouped_folds):
                    # SMOTE sees training labels only. The imputation and
                    # encoding parameters were already learned from all rows.
                    smote = _pipeline(name, seed).named_steps["smote"]
                    train_raw, train_y = smote.fit_resample(
                        prepared.iloc[train_idx], y.iloc[train_idx],
                    )
                    estimator = _estimator(name, seed)
                    estimator.fit(encoder.transform(train_raw), train_y)
                    proba = predict_proba_positive(
                        estimator, encoder.transform(prepared.iloc[val_idx]),
                    )
                    rows.append(_score(
                        y.iloc[val_idx], proba, protocol="preprocess_before_split",
                        model=name, repeat=repeat, fold=fold,
                    ))

            for fold, (train_idx, val_idx) in enumerate(grouped_folds):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                if "correct" in protocols or "balanced_test" in protocols:
                    pipe = _pipeline(name, seed)
                    pipe.fit(X_train, y_train)
                    proba = predict_proba_positive(pipe, X_val)
                    if "correct" in protocols:
                        rows.append(_score(
                            y_val, proba, protocol="correct", model=name,
                            repeat=repeat, fold=fold,
                        ))
                    if "balanced_test" in protocols:
                        positive = np.flatnonzero(y_val.to_numpy() == 1)
                        negative = np.flatnonzero(y_val.to_numpy() == 0)
                        if len(positive) == 0 or len(negative) < len(positive):
                            raise ValueError("Balanced validation needs both classes and enough negatives")
                        rng = np.random.default_rng(seed + fold)
                        chosen = np.sort(np.concatenate((
                            positive, rng.choice(negative, size=len(positive), replace=False),
                        )))
                        rows.append(_score(
                            y_val.iloc[chosen], proba[chosen], protocol="balanced_test",
                            model=name, repeat=repeat, fold=fold,
                        ))
                if "no_resampling_baseline" in protocols:
                    pipe = _pipeline(name, seed, use_smote=False)
                    pipe.fit(X_train, y_train)
                    rows.append(_score(
                        y_val, predict_proba_positive(pipe, X_val),
                        protocol="no_resampling_baseline", model=name,
                        repeat=repeat, fold=fold,
                    ))
    return pd.DataFrame(rows, columns=("protocol", "model", "repeat", "fold", *METRICS, *COUNT_COLUMNS))


def _interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) == 1:
        return mean, np.nan, np.nan
    half_width = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half_width, mean + half_width


def summarize_leakage(results: pd.DataFrame) -> pd.DataFrame:
    """Average folds per repeat, then compute repeat-level t CIs and paired P0 deltas."""
    repeats = results.groupby(["protocol", "model", "repeat"], as_index=False)[list(METRICS)].mean()
    long = repeats.melt(id_vars=["protocol", "model", "repeat"], var_name="metric", value_name="value")
    baseline = long.loc[long.protocol == "correct", ["model", "repeat", "metric", "value"]]
    baseline = baseline.rename(columns={"value": "baseline"})
    paired = long.merge(baseline, on=["model", "repeat", "metric"], how="left", validate="many_to_one")
    paired["delta"] = paired.value - paired.baseline
    rows = []
    for (protocol, model, metric), group in paired.groupby(["protocol", "model", "metric"], sort=False):
        mean, lo, hi = _interval(group.value)
        delta, delta_lo, delta_hi = _interval(group.delta)
        if protocol == "correct":
            delta = delta_lo = delta_hi = 0.0
        rows.append({
            "protocol": protocol, "model": model, "metric": metric,
            "mean": mean, "ci_lower": lo, "ci_upper": hi,
            "delta_vs_p0": delta, "delta_ci_lower": delta_lo,
            "delta_ci_upper": delta_hi,
        })
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    selected = summary[summary.metric.isin(("roc_auc", "recall"))]
    models = list(dict.fromkeys(selected.model))
    fig, axes = plt.subplots(1, 2, figsize=(max(11, 2.5 * len(models)), 5), sharey=True)
    x = np.arange(len(ALL))
    width = 0.8 / len(models)
    for ax, metric in zip(axes, ("roc_auc", "recall"), strict=True):
        for i, model in enumerate(models):
            values = selected[(selected.metric == metric) & (selected.model == model)].set_index("protocol").reindex(ALL)
            positions = x + (i - (len(models) - 1) / 2) * width
            lower = (values["mean"] - values["ci_lower"]).fillna(0).to_numpy()
            upper = (values["ci_upper"] - values["mean"]).fillna(0).to_numpy()
            ax.bar(positions, values["mean"], width, yerr=np.stack((lower, upper)), capsize=3, label=model)
        ax.set_title(metric)
        ax.set_xticks(x, ALL, rotation=30, ha="right")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Mean across repeats (95% t CI)")
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "random_forest", "gradient_boosting"])
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/leakage"))
    args = parser.parse_args(argv)
    df = load_raw_data(args.data, validate=False)
    if config.ID_COLUMN not in df:
        df[config.ID_COLUMN] = np.arange(len(df))
    results = run_leakage_experiment(df, args.models, n_splits=args.n_splits, n_repeats=args.n_repeats)
    summary = summarize_leakage(results)
    args.output.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output / "leakage_results.csv", index=False)
    summary.to_csv(args.output / "leakage_summary.csv", index=False)
    _plot(summary, args.output / "figures" / "leakage_inflation.png")


if __name__ == "__main__":
    main()
