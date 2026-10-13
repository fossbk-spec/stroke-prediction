"""Clinical explainability (XAI): SHAP, permutation importance, odds ratios.

The literature review surfaced a gap: ensemble models reach high AUC but
remain "black boxes", which makes clinicians reluctant to act on them. This
module provides three layers of explanation:

1. **Global, model-agnostic**: permutation importance on the test set.
2. **Global + local, for tree models**: SHAP values.
3. **Direct clinical interpretation**: odds ratios from Logistic Regression —
   the quantity most familiar to a medical audience.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def _transformed_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the pipeline's preprocessing step and return a named DataFrame."""
    pre = pipeline.named_steps["preprocessor"]
    matrix = pipeline[:-1].transform(X)
    try:
        names = list(pre.get_feature_names_out())
    except Exception:  # pragma: no cover
        names = [f"f{i}" for i in range(matrix.shape[1])]
    return pd.DataFrame(np.asarray(matrix), columns=names, index=X.index)


def permutation_feature_importance(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    scoring: str = "average_precision",
    n_repeats: int = 10,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Permutation importance on the ORIGINAL features (before one-hot encoding).

    Measuring on the original features lets a clinician read the result
    directly ("smoking_status matters"), instead of scattered one-hot columns.

    `scoring="average_precision"` (PR-AUC) is used instead of accuracy
    because accuracy barely moves on imbalanced data.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=1
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def tree_feature_importance(pipeline) -> pd.DataFrame:
    """Intrinsic feature importance of a tree model (Gini / gain)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"{type(model).__name__} has no `feature_importances_`. "
            "Use `permutation_feature_importance` or `logistic_odds_ratios` instead."
        )
    pre = pipeline.named_steps["preprocessor"]
    names = list(pre.get_feature_names_out())
    return (
        pd.DataFrame({"feature": names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def logistic_odds_ratios(
    pipeline, X: pd.DataFrame | None = None, y=None, *, groups=None,
    alpha: float = 0.05, n_boot: int = 200, random_state: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """Convert Logistic Regression coefficients to Odds Ratio = exp(beta).

    Reading the result: OR = 2.5 means that for a 1-unit increase in the
    feature (for a standardized feature: 1 standard deviation), the odds of
    stroke increase 2.5x. OR < 1 is a protective factor.

    For percentile CIs, supply the ORIGINAL training X, y and patient groups.
    Each bootstrap samples patients with replacement (all their records) and
    refits the full pipeline, including intercept, regularization and SMOTENC.
    `significant` means the coefficient CI excludes zero; these are exploratory
    stability intervals for this fitted procedure, not causal effects or
    calibrated hypothesis tests. Scaling and category support can vary by fit.
    A feature absent in any replicate receives NaN CI bounds and significant=False.
    Degenerate resamples that cannot fit the pipeline raise rather than being
    silently discarded.
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise AttributeError("Pipeline does not contain a linear model with `coef_`.")

    pre = pipeline.named_steps["preprocessor"]
    names = list(pre.get_feature_names_out())
    beta = np.ravel(model.coef_)

    out = pd.DataFrame(
        {
            "feature": names,
            "coefficient": beta,
            "odds_ratio": np.exp(beta),
            "direction": np.where(beta > 0, "increases risk", "decreases risk"),
        }
    )

    if X is not None:
        lower, upper = _bootstrap_coefficient_ci(
            pipeline, X, y, groups, names, alpha=alpha,
            n_boot=n_boot, random_state=random_state,
        )
        out["or_ci_lower"] = np.exp(lower)
        out["or_ci_upper"] = np.exp(upper)
        out["significant"] = (lower > 0) | (upper < 0)

    out["abs_effect"] = out["coefficient"].abs()
    return (
        out.sort_values("abs_effect", ascending=False)
        .drop(columns="abs_effect")
        .reset_index(drop=True)
    )


def _bootstrap_coefficient_ci(
    pipeline, X, y, groups, names, *, alpha, n_boot, random_state
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster bootstrap with feature-name alignment when categories disappear."""
    from sklearn.base import clone

    if y is None or groups is None:
        raise ValueError("Bootstrap CIs require training y and patient groups.")
    y, groups = np.asarray(y), np.asarray(groups)
    if len(y) != len(X) or len(groups) != len(X):
        raise ValueError("X, y and groups must have equal lengths.")
    if not 0 < alpha < 1 or n_boot < 2:
        raise ValueError("Require 0 < alpha < 1 and n_boot >= 2.")
    if pd.isna(groups).any():
        raise ValueError("Patient groups cannot contain missing IDs.")
    patients = pd.unique(groups)
    records = [np.flatnonzero(groups == patient) for patient in patients]
    rng = np.random.default_rng(random_state)
    coefficients = []
    for _ in range(n_boot):
        rows = np.concatenate([records[i] for i in rng.integers(len(patients), size=len(patients))])
        fitted = clone(pipeline).fit(X.iloc[rows], y[rows])
        boot_names = fitted.named_steps["preprocessor"].get_feature_names_out()
        values = pd.Series(fitted.named_steps["model"].coef_.ravel(), index=boot_names)
        # An absent category has no estimable coefficient in this replicate.
        coefficients.append(values.reindex(names).to_numpy())
    samples = np.asarray(coefficients)
    # Do not claim intervals for features absent from any bootstrap replicate.
    return tuple(np.quantile(samples, [alpha / 2, 1 - alpha / 2], axis=0))


def shap_summary(
    pipeline,
    X: pd.DataFrame,
    *,
    max_display: int = 15,
    sample_size: int = 500,
    save_path: str | Path | None = None,
    random_state: int = config.RANDOM_STATE,
):
    """Compute SHAP values and (optionally) save a summary plot.

    Returns:
        (shap_values, mean_abs_importance_dataframe)
    """
    import shap

    X_t = _transformed_frame(pipeline, X)
    if len(X_t) > sample_size:
        X_t = X_t.sample(sample_size, random_state=random_state)

    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_t)
        if isinstance(values, list):  # binary classification -> take the positive class
            values = values[1]
        elif getattr(values, "ndim", 2) == 3:
            values = values[:, :, 1]
    else:
        explainer = shap.LinearExplainer(model, X_t)
        values = explainer.shap_values(X_t)

    values = np.asarray(values)
    importance = (
        pd.DataFrame(
            {"feature": X_t.columns, "mean_abs_shap": np.abs(values).mean(axis=0)}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    if save_path is not None:  # pragma: no cover - plotting
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        shap.summary_plot(values, X_t, max_display=max_display, show=False)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return values, importance


def explain_single_patient(
    pipeline, X: pd.DataFrame, index: int, *, top_k: int = 6
) -> pd.DataFrame:
    """Local explanation for ONE patient (useful for bedside / clinic counseling).

    Returns the top-k features that push the predicted probability up/down
    the most for that case.
    """
    import shap

    X_t = _transformed_frame(pipeline, X)
    row = X_t.iloc[[index]]
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(row)
        if isinstance(values, list):
            values = values[1]
        elif getattr(values, "ndim", 2) == 3:
            values = values[:, :, 1]
    else:
        explainer = shap.LinearExplainer(model, X_t)
        values = explainer.shap_values(row)

    contrib = np.ravel(np.asarray(values))
    out = pd.DataFrame(
        {
            "feature": X_t.columns,
            "value": row.iloc[0].to_numpy(),
            "shap_contribution": contrib,
            "effect": np.where(contrib > 0, "increases risk", "decreases risk"),
        }
    )
    out["abs_contribution"] = out["shap_contribution"].abs()
    return (
        out.sort_values("abs_contribution", ascending=False)
        .head(top_k)
        .drop(columns="abs_contribution")
        .reset_index(drop=True)
    )
