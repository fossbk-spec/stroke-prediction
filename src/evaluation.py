"""Evaluation for a rare-event problem with asymmetric costs.

Principle: NEVER report Accuracy in isolation. With a 4.9% positive rate, an
"always predict no-stroke" model reaches 95.1% accuracy with Recall = 0 —
clinically useless (see Akinwumi et al. 2025, where LR/SVM/RF reach
Recall = 0 despite a ROC-AUC of ~0.84).

Mandatory metric set, in priority order: Recall (Sensitivity) > PR-AUC >
F2 > ROC-AUC > Specificity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config
from src.models import fit_with_groups


# --------------------------------------------------------------------------- #
# 1. Metric set
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationResult:
    """Evaluation result for one model at one specific threshold."""

    model_name: str
    threshold: float
    metrics: dict[str, float]
    confusion: np.ndarray = field(repr=False)

    def as_row(self) -> dict[str, float | str]:
        row: dict[str, float | str] = {"model": self.model_name, "threshold": self.threshold}
        row.update(self.metrics)
        return row

    def __str__(self) -> str:  # pragma: no cover - display only
        tn, fp, fn, tp = self.confusion.ravel()
        return (
            f"{self.model_name} @ threshold={self.threshold:.3f}\n"
            f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}\n"
            + "\n".join(f"  {k:<16}= {v:.4f}" for k, v in self.metrics.items())
        )


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    *,
    threshold: float = config.DEFAULT_THRESHOLD,
    beta: float = config.F_BETA,
) -> dict[str, float]:
    """Compute the metric set from predicted probabilities.

    Note: `roc_auc` and `pr_auc` are threshold-independent; the remaining
    metrics are computed at `threshold`. `pr_auc` is Average Precision
    (sklearn `average_precision_score`), not trapezoidal PR area.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    positive_rate = y_true.mean() if n else 0.0

    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan,
        f"f{beta:g}": fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
        "f1": fbeta_score(y_true, y_pred, beta=1, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "balanced_accuracy": 0.5 * (
            (tp / (tp + fn) if (tp + fn) else 0.0) + (tn / (tn + fp) if (tn + fp) else 0.0)
        ),
        "accuracy": (tp + tn) / n if n else 0.0,
        "null_accuracy": max(positive_rate, 1 - positive_rate),
        "brier": brier_score_loss(y_true, y_proba),
        "n_false_negative": float(fn),
        "n_true_positive": float(tp),
        "n_false_positive": float(fp),
    }


def bootstrap_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    *,
    threshold: float,
    beta: float = config.F_BETA,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Percentile confidence intervals from resampled patient IDs.

    Every selected patient contributes all of their rows, including when the
    patient is drawn more than once. The decision threshold remains fixed.
    """
    y = np.asarray(y_true).astype(int)
    proba = np.asarray(y_proba, dtype=float)
    group_ids = np.asarray(groups)
    if not (len(y) == len(proba) == len(group_ids)) or not len(y):
        raise ValueError("y_true, y_proba and groups must have equal nonzero lengths.")
    if n_boot < 1 or not 0 < ci < 1:
        raise ValueError("n_boot must be positive and ci must be between 0 and 1.")

    codes, unique_groups = pd.factorize(group_ids, sort=False)
    if (codes < 0).any():
        raise ValueError("groups must not contain missing patient IDs.")
    row_indices = [np.flatnonzero(codes == i) for i in range(len(unique_groups))]
    metric_names = ["recall", "precision", "pr_auc", f"f{beta:g}",
                    "roc_auc", "specificity", "brier"]
    point = compute_metrics(y, proba, threshold=threshold, beta=beta)
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    skipped = 0
    rng = np.random.default_rng(random_state)
    for _ in range(n_boot):
        drawn = rng.integers(0, len(row_indices), size=len(row_indices))
        rows = np.concatenate([row_indices[i] for i in drawn])
        if np.unique(y[rows]).size < 2:
            skipped += 1
            continue
        metrics = compute_metrics(y[rows], proba[rows], threshold=threshold, beta=beta)
        for name in metric_names:
            samples[name].append(metrics[name])

    alpha = (1 - ci) / 2
    result = []
    for name in metric_names:
        values = samples[name]
        if values:
            lower, upper = np.quantile(values, [alpha, 1 - alpha])
        else:
            lower = upper = np.nan
        result.append({"metric": name, "point": point[name], "lower": lower,
                       "upper": upper, "n_valid": len(values), "n_skipped": skipped})
    return pd.DataFrame(result)


def calibration_table(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Observed event rate and mean predicted risk in each nonempty bin."""
    y = np.asarray(y_true).astype(int)
    proba = np.asarray(y_proba, dtype=float)
    if len(y) != len(proba) or not len(y):
        raise ValueError("y_true and y_proba must have equal nonzero lengths.")
    if n_bins < 1 or strategy not in {"quantile", "uniform"}:
        raise ValueError("n_bins must be positive; strategy must be quantile or uniform.")
    if strategy == "quantile":
        bins = pd.qcut(proba, q=n_bins, labels=False, duplicates="drop")
        # All identical probabilities have no quantile cuts.
        if pd.isna(bins).all():
            bins = np.zeros(len(proba), dtype=int)
    else:
        bins = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    frame = pd.DataFrame({"bin": bins, "y": y, "proba": proba})
    return (frame.groupby("bin", sort=True)
            .agg(mean_predicted=("proba", "mean"), observed_rate=("y", "mean"),
                 count=("y", "size"))
            .reset_index())


def subgroup_metrics(
    X: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    *,
    threshold: float,
    beta: float = config.F_BETA,
) -> pd.DataFrame:
    """Descriptive test metrics by gender and prespecified age band."""
    y = np.asarray(y_true).astype(int)
    proba = np.asarray(y_proba, dtype=float)
    if len(X) != len(y) or len(y) != len(proba):
        raise ValueError("X, y_true and y_proba must have equal lengths.")
    age = np.asarray(X["age"], dtype=float)
    gender = np.asarray(X["gender"])
    age_band = np.select([age < 45, age < 65, age >= 65],
                         ["<45", "45-64", ">=65"], default="missing")
    rows = []
    for subgroup, levels, values in (
        ("gender", ("Male", "Female", "Other"), gender),
        ("age_band", ("<45", "45-64", ">=65"), age_band),
    ):
        for level in levels:
            mask = values == level
            n = int(mask.sum())
            if not n or (level == "Other" and n < 5):
                continue
            metrics = compute_metrics(y[mask], proba[mask], threshold=threshold, beta=beta)
            rows.append({"subgroup": subgroup, "level": level, "n": n,
                         "n_positive": int(y[mask].sum()), "recall": metrics["recall"],
                         "precision": metrics["precision"], "roc_auc": metrics["roc_auc"],
                         "brier": metrics["brier"]})
    return pd.DataFrame(rows, columns=["subgroup", "level", "n", "n_positive",
                                        "recall", "precision", "roc_auc", "brier"])


def evaluate_model(
    model,
    X,
    y,
    *,
    model_name: str = "model",
    threshold: float = config.DEFAULT_THRESHOLD,
    beta: float = config.F_BETA,
) -> EvaluationResult:
    """Evaluate a fitted model/pipeline on a given dataset."""
    y_proba = predict_proba_positive(model, X)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return EvaluationResult(
        model_name=model_name,
        threshold=threshold,
        metrics=compute_metrics(y, y_proba, threshold=threshold, beta=beta),
        confusion=confusion_matrix(np.asarray(y).astype(int), y_pred, labels=[0, 1]),
    )


def predict_proba_positive(model, X) -> np.ndarray:
    """Get the positive-class probability, supporting estimators with only `decision_function`."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))  # squash to [0,1] with a sigmoid
    raise AttributeError("Estimator supports neither predict_proba nor decision_function.")


# --------------------------------------------------------------------------- #
# 2. Decision threshold tuning
# --------------------------------------------------------------------------- #
def find_best_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    *,
    beta: float = config.F_BETA,
) -> tuple[float, float]:
    """Find the threshold that maximizes F-beta (beta=2 weights Recall 2x over Precision).

    The default 0.5 threshold is the WORST choice for imbalanced data: it
    optimizes Accuracy, not Recall. The threshold must be chosen on the
    VALIDATION set, never on the TEST set.

    Returns:
        (best_threshold, best_fbeta)
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]  # drop the last point (no threshold)

    denominator = (beta**2 * precision) + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = np.where(
            denominator > 0, (1 + beta**2) * precision * recall / denominator, 0.0
        )
    if len(fbeta) == 0:
        return config.DEFAULT_THRESHOLD, 0.0
    best = int(np.nanargmax(fbeta))
    return float(thresholds[best]), float(fbeta[best])


def threshold_for_min_recall(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    *,
    min_recall: float = 0.80,
) -> float:
    """The HIGHEST threshold that still guarantees Recall >= `min_recall`.

    This is how to set a threshold under a clinical constraint: "miss at
    most 20% of true cases", then optimize Precision within that constraint.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    _precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    recall, thresholds = recall[:-1], thresholds

    feasible = np.where(recall >= min_recall)[0]
    if len(feasible) == 0:
        return float(np.min(y_proba))  # unreachable -> lowest possible threshold
    return float(thresholds[feasible[-1]])


# --------------------------------------------------------------------------- #
# 3. Group-aware cross-validation
# --------------------------------------------------------------------------- #
def cross_validate_grouped(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int = config.N_SPLITS,
    beta: float = config.F_BETA,
    threshold: float = config.DEFAULT_THRESHOLD,
    tune_threshold: bool = True,
    min_recall: float | None = None,
    inner_splits: int = 3,
    n_repeats: int = 1,
) -> pd.DataFrame:
    """Cross-validation with `StratifiedGroupKFold` + optional per-fold threshold tuning.

    Per fold: choose the threshold on OUT-OF-FOLD predictions from an inner
    grouped CV over the train fold (mirroring "tune on validation"; in-sample
    predictions are overconfident, especially for trees), fit on the whole
    train fold (SMOTE included inside the pipeline), then compute metrics on
    the validation fold. `min_recall` switches the threshold policy from
    F-beta to `threshold_for_min_recall`, as in `train.py`.
    """
    from sklearn.base import clone

    from src.preprocessing import cv_splitter

    if n_repeats < 1:
        raise ValueError("n_repeats must be positive.")
    rows: list[dict[str, float | int | str]] = []

    for repeat in range(n_repeats):
        seed = config.RANDOM_STATE + repeat
        splitter = cv_splitter(n_splits=n_splits, random_state=seed)
        for fold, (tr, va) in enumerate(splitter.split(X, y, groups=groups), start=1):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y.iloc[tr], y.iloc[va]
            groups_tr = groups.iloc[tr]

            fold_threshold = threshold
            if tune_threshold:
                oof_proba = np.empty(len(y_tr), dtype=float)
                for inner_tr, inner_va in cv_splitter(
                    n_splits=inner_splits, random_state=seed
                ).split(X_tr, y_tr, groups=groups_tr):
                    inner_model = clone(pipeline)
                    fit_with_groups(inner_model, X_tr.iloc[inner_tr], y_tr.iloc[inner_tr],
                                    groups_tr.iloc[inner_tr])
                    oof_proba[inner_va] = predict_proba_positive(inner_model, X_tr.iloc[inner_va])
                if min_recall is not None:
                    fold_threshold = threshold_for_min_recall(y_tr, oof_proba, min_recall=min_recall)
                else:
                    fold_threshold, _ = find_best_threshold(y_tr, oof_proba, beta=beta)

            model = clone(pipeline)
            fit_with_groups(model, X_tr, y_tr, groups_tr)

            metrics = compute_metrics(
                y_va, predict_proba_positive(model, X_va), threshold=fold_threshold, beta=beta
            )
            metrics.update({"fold": f"{repeat + 1}-{fold}", "repeat": repeat + 1,
                            "threshold": fold_threshold})
            rows.append(metrics)

    return pd.DataFrame(rows).set_index("fold")


def summarize_cv(cv_results: pd.DataFrame) -> pd.Series:
    """Collapse CV results into a 'mean ± std' summary for each metric."""
    mean, std = cv_results.mean(numeric_only=True), cv_results.std(numeric_only=True)
    return pd.Series(
        {col: f"{mean[col]:.4f} ± {std[col]:.4f}" for col in mean.index}, name="cv_summary"
    )


def comparison_table(results: list[EvaluationResult]) -> pd.DataFrame:
    """Build a model comparison table, sorted by Recall then PR-AUC."""
    df = pd.DataFrame([r.as_row() for r in results])
    sort_cols = [c for c in ("recall", "pr_auc") if c in df.columns]
    return df.sort_values(sort_cols, ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Plots
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(result: EvaluationResult, ax=None):  # pragma: no cover - plotting
    """Plot the confusion matrix with clear TP/FN/FP/TN labels."""
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(4.2, 3.8))
    cm = result.confusion
    ax.imshow(cm, cmap="Blues")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{labels[i][j]}\n{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11,
            )
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title(f"{result.model_name} (t={result.threshold:.2f})")
    return ax


def plot_pr_curve(y_true, y_proba, *, label: str = "model", ax=None):  # pragma: no cover
    """Plot the Precision-Recall curve (more informative than ROC on imbalanced data)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(5, 4))
    precision, recall, _ = precision_recall_curve(np.asarray(y_true).astype(int), y_proba)
    ap = average_precision_score(np.asarray(y_true).astype(int), y_proba)
    ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
    baseline = np.asarray(y_true).astype(int).mean()
    ax.axhline(baseline, ls="--", c="grey", lw=1, label=f"Random ({baseline:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_calibration(
    y_true, y_proba, *, n_bins: int = 10, strategy: str = "quantile",
    label: str = "model", ax=None,
):  # pragma: no cover - plotting
    """Plot observed event rates against mean predicted probabilities."""
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(5, 4))
    table = calibration_table(y_true, y_proba, n_bins=n_bins, strategy=strategy)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="Perfect calibration")
    ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
           ylabel="Observed event rate", title="Calibration")
    ax.legend()
    return ax
