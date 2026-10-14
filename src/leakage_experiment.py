"""Repeated clean and deliberately leaky stroke CV comparisons.

P1a is an unmatched reproduction of a bad before-split protocol. P1b, P2,
P3, and P4 use P0's original grouped validation rows. P3 is a controlled
prevalence experiment, not a literal Yang/Vu workflow. It reuses P0
probabilities, so recall equals P0 by construction. P4 estimates removal of
the resampling/prepare step from this pipeline; its scaler fits original rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from src import config
from src.data_loader import load_raw_data, split_features_target, validate_schema
from src.evaluation import compute_metrics, predict_proba_positive
from src.models import get_model
from src.preprocessing import cv_splitter, make_full_pipeline

ALL = (
    "correct", "smote_before_split", "smote_before_split_matched",
    "preprocess_before_split", "balanced_test", "no_resampling_baseline",
)
METRICS = ("accuracy", "recall", "precision", "f2", "specificity", "roc_auc", "pr_auc", "brier")
COUNT_COLUMNS = ("n_train", "n_val", "n_val_positive", "n_synthetic_in_val",
                 "n_synthetic_train", "n_synthetic_train_val_parent")
SCALE_MODELS = frozenset(("logistic_regression", "svm", "knn", "naive_bayes"))


def _estimator(name: str, seed: int, n_jobs: int = 1):
    estimator = get_model(name)
    params = estimator.get_params(deep=False)
    overrides = {}
    if "random_state" in params:
        overrides["random_state"] = seed
    if name == "random_forest":
        overrides["n_jobs"] = n_jobs
    if overrides:
        estimator.set_params(**overrides)
    return estimator


def _pipeline(name: str, seed: int, *, use_smote: bool = True, n_jobs: int = 1):
    return make_full_pipeline(
        _estimator(name, seed, n_jobs), use_smote=use_smote,
        scale=name in SCALE_MODELS, random_state=seed,
    )


def _stages(scale: bool, seed: int, *, use_smote: bool = True):
    # Keep the exact preparation, resampling and encoding factories of P0.
    name = "logistic_regression" if scale else "random_forest"
    return _pipeline(name, seed, use_smote=use_smote).named_steps


def _group_folds(X, y, groups, n_splits: int, seed: int):
    return list(cv_splitter(n_splits=n_splits, random_state=seed).split(X, y, groups))


def _json_array(values):
    return json.dumps(np.asarray(values).tolist())


def _score(y, proba, *, protocol, model, repeat, fold, val_ids, n_train,
           n_synthetic=0, n_synthetic_train=0, n_val_parent=0):
    scores = (compute_metrics(y, proba, threshold=0.5, beta=2) if len(y)
              else dict.fromkeys(METRICS, np.nan))
    return {
        "protocol": protocol, "model": model, "repeat": repeat, "fold": fold,
        **{metric: scores[metric] for metric in METRICS},
        "n_train": int(n_train), "n_val": len(y), "n_val_positive": int(np.sum(y)),
        "n_synthetic_in_val": int(n_synthetic),
        "n_synthetic_train": int(n_synthetic_train),
        "n_synthetic_train_val_parent": int(n_val_parent),
        "val_ids": _json_array(val_ids), "val_proba": _json_array(proba),
    }


def _approximate_synthetic_parent(prepared: pd.DataFrame, synthetic: pd.DataFrame,
                                  y, groups):
    """Return nearest original minority patient in an approximate SMOTENC space.

    Numeric columns are standardized using minority-row standard deviations
    and categorical columns are one-hot encoded. This is approximate parent
    attribution, not provenance: a nearest row can differ from the generating
    neighbours used internally by SMOTENC.
    """
    if len(synthetic) == 0:
        return np.empty(0, dtype=object)
    minority_idx = np.flatnonzero(np.asarray(y) == 1)
    minority = prepared.iloc[minority_idx].reset_index(drop=True)
    combined = pd.concat((minority, synthetic.reset_index(drop=True)), ignore_index=True)
    numeric = config.NUMERIC_FEATURES
    matrix = pd.get_dummies(combined, columns=[c for c in combined if c not in numeric],
                            dtype=float).astype(float)
    std = matrix.iloc[:len(minority)][numeric].std().replace(0, 1).fillna(1)
    matrix.loc[:, numeric] = matrix[numeric] / std
    nn = NearestNeighbors(n_neighbors=1).fit(matrix.iloc[:len(minority)].to_numpy())
    nearest = nn.kneighbors(matrix.iloc[len(minority):].to_numpy(), return_distance=False).ravel()
    return np.asarray(groups)[minority_idx[nearest]]


def _fit_score(estimator, train_X, train_y, val_X):
    start = perf_counter()
    estimator.fit(train_X, train_y)
    return predict_proba_positive(estimator, val_X), perf_counter() - start


def run_leakage_experiment(
    df: pd.DataFrame, model_names, protocols=ALL, n_splits: int = 5, n_repeats: int = 5,
    n_jobs: int = 1, benchmark: bool = False,
) -> pd.DataFrame:
    """Return fold scores, or benchmark one fold per model/protocol.

    P1a evaluates stratified folds of globally resampled rows. P1b globally
    prepares and resamples all rows, trains on original training rows plus all
    synthetic rows, and scores on P0's original validation rows. P2 uses P0's
    stage order, adding validation rows only to imputer and encoder/scaler fits.
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
    df = df.reset_index(drop=True)
    X, y = split_features_target(df)
    groups = df[config.ID_COLUMN]
    rows = []
    scales = {name in SCALE_MODELS for name in model_names}

    for repeat in range(1 if benchmark else n_repeats):
        seed = config.RANDOM_STATE + repeat
        grouped_folds = _group_folds(X, y, groups, n_splits, seed)
        if benchmark:
            grouped_folds = grouped_folds[:1]

        global_data = None
        if {"smote_before_split", "smote_before_split_matched"} & set(protocols):
            stages = _stages(False, seed)
            prepared = stages["prepare"].fit_transform(X)
            global_resampled, global_y = stages["smote"].fit_resample(prepared, y)
            global_y = np.asarray(global_y)
            synthetic = global_resampled.iloc[len(y):].reset_index(drop=True)
            parent_ids = (_approximate_synthetic_parent(prepared, synthetic, y, groups)
                          if "smote_before_split_matched" in protocols else None)
            p1_folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                           random_state=seed).split(global_resampled, global_y))
            if benchmark:
                p1_folds = p1_folds[:1]
            global_data = (global_resampled, global_y, synthetic, parent_ids, p1_folds)

        p2_prepared = None
        if "preprocess_before_split" in protocols:
            p2_prepared = _stages(False, seed)["prepare"].fit_transform(X)

        global_encoded = {}
        if global_data is not None:
            for scale in scales:
                encoder = _stages(scale, seed)["preprocessor"]
                global_encoded[scale] = encoder.fit_transform(global_data[0])

        for fold, (train_idx, val_idx) in enumerate(grouped_folds):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            val_ids = groups.iloc[val_idx]
            clean_resampled = clean_y = clean_val = None
            if {"correct", "balanced_test"} & set(protocols):
                stages = _stages(False, seed)
                clean_prepared = stages["prepare"].fit_transform(X_train)
                clean_val = stages["prepare"].transform(X_val)
                clean_resampled, clean_y = stages["smote"].fit_resample(clean_prepared, y_train)

            p2_resampled = p2_y = None
            if p2_prepared is not None:
                p2_resampled, p2_y = _stages(False, seed)["smote"].fit_resample(
                    p2_prepared.iloc[train_idx], y_train)

            for scale in sorted(scales):
                names = [name for name in model_names if (name in SCALE_MODELS) == scale]
                enc_clean = enc_p2 = enc_p1 = enc_p4 = None
                if clean_resampled is not None:
                    encoder = _stages(scale, seed)["preprocessor"]
                    enc_clean = (encoder.fit_transform(clean_resampled),
                                 encoder.transform(clean_val))
                if p2_resampled is not None:
                    encoder = _stages(scale, seed)["preprocessor"]
                    encoder.fit(pd.concat((p2_resampled, p2_prepared.iloc[val_idx]),
                                          ignore_index=True))
                    enc_p2 = (encoder.transform(p2_resampled),
                              encoder.transform(p2_prepared.iloc[val_idx]))
                if global_data is not None:
                    global_resampled, global_y, synthetic, parent_ids, p1_folds = global_data
                    enc_p1 = global_encoded[scale]
                if "no_resampling_baseline" in protocols:
                    encoder = _stages(scale, seed, use_smote=False)["preprocessor"]
                    enc_p4 = (encoder.fit_transform(X_train), encoder.transform(X_val))

                for name in names:
                    def evaluate(train_X, train_y, val_X, val_y, ids, protocol,
                                 *, synthetic_val=0, synthetic_train=0, val_parent=0,
                                 n_train=len(train_idx), score_fold=fold, record=True,
                                 model_name=name, model_seed=seed, model_repeat=repeat):
                        proba, seconds = _fit_score(_estimator(model_name, model_seed, n_jobs),
                                                    train_X, train_y, val_X)
                        if benchmark:
                            print(f"{protocol} {model_name}: fit {seconds:.3f}s")
                        if record:
                            rows.append(_score(val_y, proba, protocol=protocol, model=model_name,
                                               repeat=model_repeat, fold=score_fold, val_ids=ids,
                                               n_train=n_train, n_synthetic=synthetic_val,
                                               n_synthetic_train=synthetic_train,
                                               n_val_parent=val_parent))
                        return proba

                    if "smote_before_split" in protocols and fold == 0:
                        for p1_fold, (p1_train, p1_val) in enumerate(p1_folds):
                            ids = [str(groups.iloc[i]) if i < len(y) else f"synthetic:{i-len(y)}"
                                   for i in p1_val]
                            evaluate(enc_p1[p1_train], global_y[p1_train], enc_p1[p1_val],
                                     global_y[p1_val], ids, "smote_before_split",
                                     synthetic_val=int(np.sum(p1_val >= len(y))),
                                     synthetic_train=int(np.sum(p1_train >= len(y))),
                                     n_train=len(p1_train), score_fold=p1_fold)
                    if "smote_before_split_matched" in protocols:
                        p1_train = np.concatenate((train_idx, np.arange(len(y), len(global_y))))
                        n_parent = int(np.isin(parent_ids, val_ids).sum())
                        evaluate(enc_p1[p1_train], global_y[p1_train], enc_p1[val_idx],
                                 y_val, val_ids, "smote_before_split_matched",
                                 synthetic_train=len(synthetic), val_parent=n_parent)
                    if "preprocess_before_split" in protocols:
                        evaluate(enc_p2[0], p2_y, enc_p2[1], y_val, val_ids,
                                 "preprocess_before_split",
                                 synthetic_train=len(p2_y) - len(y_train))
                    if clean_resampled is not None:
                        proba = evaluate(enc_clean[0], clean_y, enc_clean[1], y_val,
                                         val_ids, ("correct" if "correct" in protocols
                                                   else "balanced_test"),
                                         synthetic_train=len(clean_y) - len(y_train),
                                         record="correct" in protocols)
                        if "balanced_test" in protocols:
                            positive = np.flatnonzero(y_val.to_numpy() == 1)
                            negative = np.flatnonzero(y_val.to_numpy() == 0)
                            if not len(positive) or len(negative) < len(positive):
                                chosen = np.empty(0, dtype=int)  # invalid prevalence fold
                            else:
                                rng = np.random.default_rng(seed + fold)
                                chosen = np.sort(np.concatenate((positive, rng.choice(
                                    negative, size=len(positive), replace=False))))
                            rows.append(_score(y_val.iloc[chosen], proba[chosen],
                                               protocol="balanced_test", model=name,
                                               repeat=repeat, fold=fold,
                                               val_ids=val_ids.iloc[chosen],
                                               n_train=len(train_idx),
                                               n_synthetic_train=len(clean_y) - len(y_train)))
                    if "no_resampling_baseline" in protocols:
                        evaluate(enc_p4[0], y_train, enc_p4[1], y_val, val_ids,
                                 "no_resampling_baseline")
    return pd.DataFrame(rows, columns=("protocol", "model", "repeat", "fold", *METRICS,
                                       *COUNT_COLUMNS, "val_ids", "val_proba"))


def summarize_leakage(results: pd.DataFrame) -> pd.DataFrame:
    """Repeat ranges and paired Nadeau-Bengio corrected resampled t intervals.

    This is a sensitivity analysis assuming exchangeable paired fold deltas,
    not a population-performance CI. Invalid pairs are removed before computing
    both the mean and the effective number of folds/repeats.
    """
    rows = []
    keys = ["protocol", "model", "repeat", "fold"]
    long = results.melt(id_vars=keys + ["n_train", "n_val"], value_vars=list(METRICS),
                        var_name="metric", value_name="value")
    baseline = long[long.protocol == "correct"].drop(columns="protocol").rename(
        columns={"value": "baseline", "n_train": "baseline_train", "n_val": "baseline_val"})
    for (protocol, model, metric), group in long.groupby(["protocol", "model", "metric"], sort=False):
        repeat_means = group.groupby("repeat").value.mean().dropna()
        mean = float(repeat_means.mean()) if len(repeat_means) else np.nan
        low = float(repeat_means.min()) if len(repeat_means) else np.nan
        high = float(repeat_means.max()) if len(repeat_means) else np.nan
        paired = group.merge(baseline[(baseline.model == model) & (baseline.metric == metric)],
                             on=["model", "repeat", "fold", "metric"], how="inner",
                             validate="one_to_one")
        paired = paired[np.isfinite(paired.value) & np.isfinite(paired.baseline)]
        n = len(paired)
        n_repeats = int(paired.repeat.nunique())
        delta = delta_lo = delta_hi = np.nan
        comparison = ("unmatched" if protocol == "smote_before_split" else
                      "reference" if protocol == "correct" else "matched")
        ci_method = ("Nadeau-Bengio corrected resampled t" if comparison == "matched"
                     else "none")
        if comparison == "reference":
            delta = 0.0
        if comparison == "matched" and n:
            differences = (paired.value - paired.baseline).to_numpy(dtype=float)
            delta = float(differences.mean())
            if n > 1:
                ratio = float((paired.baseline_val / paired.baseline_train).mean())
                half = float(t.ppf(0.975, n - 1) * differences.std(ddof=1)
                             * np.sqrt(1 / n + ratio))
                delta_lo, delta_hi = delta - half, delta + half
        if comparison == "unmatched":
            n = n_repeats = 0
        rows.append({"protocol": protocol, "model": model, "metric": metric,
                     "mean": mean, "repeat_min": low, "repeat_max": high,
                     "delta_vs_p0": delta, "delta_ci_lower": delta_lo,
                     "delta_ci_upper": delta_hi, "n_valid_folds": n,
                     "n_valid_repeats": n_repeats, "comparison": comparison,
                     "ci_method": ci_method})
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
            present = values["mean"].notna().to_numpy()
            ax.bar(positions[present], values["mean"].to_numpy()[present],
                   width, label=model)
        ax.set_title(metric)
        ax.set_xticks(x, ALL, rotation=30, ha="right")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Mean over repeats (range in summary CSV)")
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "random_forest", "gradient_boosting"])
    parser.add_argument("--protocols", nargs="+", choices=ALL, default=list(ALL))
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1, help="Random-forest workers")
    parser.add_argument("--benchmark", action="store_true", help="One fold per model and protocol; no files")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/leakage"))
    args = parser.parse_args(argv)
    df = load_raw_data(args.data, validate=False)
    if config.ID_COLUMN not in df:
        df[config.ID_COLUMN] = np.arange(len(df))
    results = run_leakage_experiment(df, args.models, protocols=args.protocols,
                                     n_splits=args.n_splits, n_repeats=args.n_repeats,
                                     n_jobs=args.n_jobs, benchmark=args.benchmark)
    if args.benchmark:
        return
    summary = summarize_leakage(results)
    args.output.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output / "leakage_results.csv", index=False)
    summary.to_csv(args.output / "leakage_summary.csv", index=False)
    _plot(summary, args.output / "figures" / "leakage_inflation.png")


if __name__ == "__main__":
    main()
