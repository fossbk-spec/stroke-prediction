"""Tinh giai thich lam sang (XAI): SHAP, permutation importance, odds ratio.

Khoang trong y van da chi ra: cac mo hinh ensemble dat AUC cao nhung la
"hop den", khien bac si khong tin tuong de ra quyet dinh du phong. Module nay
cung cap 3 lop giai thich:

1. **Toan cuc, mo hinh-bat-ky**: permutation importance tren tap test.
2. **Toan cuc + cuc bo, cho mo hinh cay**: SHAP values.
3. **Lam sang truc tiep**: odds ratio tu Logistic Regression — dai luong ma
   gioi y khoa quen doc nhat.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def _transformed_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Ap dung buoc tien xu ly cua pipeline va tra ve DataFrame co ten cot."""
    pre = pipeline.named_steps["preprocessor"]
    matrix = pre.transform(X)
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
    """Permutation importance tren DAC TRUNG GOC (truoc one-hot).

    Do tren dac trung goc giup bac si doc truc tiep ("smoking_status quan
    trong"), thay vi doc 4 cot one-hot roi rac.

    `scoring="average_precision"` (PR-AUC) duoc chon thay vi accuracy vi
    accuracy gan nhu khong thay doi tren du lieu mat can bang.
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
    """Feature importance noi tai cua mo hinh cay (Gini / gain)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"{type(model).__name__} khong co `feature_importances_`. "
            "Dung `permutation_feature_importance` hoac `logistic_odds_ratios`."
        )
    pre = pipeline.named_steps["preprocessor"]
    names = list(pre.get_feature_names_out())
    return (
        pd.DataFrame({"feature": names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def logistic_odds_ratios(
    pipeline, X: pd.DataFrame | None = None, *, alpha: float = 0.05
) -> pd.DataFrame:
    """Chuyen he so Logistic Regression thanh Odds Ratio = exp(beta).

    Doc ket qua: OR = 2.5 nghia la khi dac trung tang 1 don vi (voi bien da
    chuan hoa: 1 do lech chuan), odds bi dot quy tang 2.5 lan. OR < 1 la yeu
    to bao ve.

    Args:
        X: neu duoc cung cap, tinh them khoang tin cay Wald tu ma tran Hessian
            cua log-likelihood: SE = sqrt(diag((X' W X)^-1)) voi W = diag(p(1-p)).
            Luu y: khi pipeline co SMOTE, CI nay mang tinh tham khao vi he so
            duoc hoc tren du lieu da oversampling.
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise AttributeError("Pipeline khong chua mo hinh tuyen tinh co `coef_`.")

    pre = pipeline.named_steps["preprocessor"]
    names = list(pre.get_feature_names_out())
    beta = np.ravel(model.coef_)

    out = pd.DataFrame(
        {
            "feature": names,
            "coefficient": beta,
            "odds_ratio": np.exp(beta),
            "direction": np.where(beta > 0, "tang nguy co", "giam nguy co"),
        }
    )

    if X is not None:
        lower, upper = _wald_odds_ratio_ci(pipeline, X, beta, alpha=alpha)
        out["or_ci_lower"] = lower
        out["or_ci_upper"] = upper
        out["significant"] = ~((lower < 1) & (upper > 1))

    out["abs_effect"] = out["coefficient"].abs()
    return (
        out.sort_values("abs_effect", ascending=False)
        .drop(columns="abs_effect")
        .reset_index(drop=True)
    )


def _wald_odds_ratio_ci(
    pipeline, X: pd.DataFrame, beta: np.ndarray, *, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Khoang tin cay Wald cho odds ratio tren cohort `X`."""
    from scipy.stats import norm

    design = _transformed_frame(pipeline, X).to_numpy(dtype=float)
    proba = np.asarray(pipeline.predict_proba(X))[:, 1]
    weights = np.clip(proba * (1 - proba), 1e-10, None)

    fisher = design.T @ (design * weights[:, None])
    # Them nhieu Tikhonov nho de ma tran kha nghich khi co dac trung gan cong tuyen.
    ridge = 1e-8 * np.eye(fisher.shape[0])
    standard_errors = np.sqrt(np.diag(np.linalg.pinv(fisher + ridge)))

    z = norm.ppf(1 - alpha / 2)
    # Chan mien gia tri truoc khi exp: dac trung gan cong tuyen (vi du
    # `gender_Other` chi co 1 mau) cho SE rat lon -> exp() tran so.
    lower = np.exp(np.clip(beta - z * standard_errors, -700, 700))
    upper = np.exp(np.clip(beta + z * standard_errors, -700, 700))
    return lower, upper


def shap_summary(
    pipeline,
    X: pd.DataFrame,
    *,
    max_display: int = 15,
    sample_size: int = 500,
    save_path: str | Path | None = None,
    random_state: int = config.RANDOM_STATE,
):
    """Tinh SHAP values va (tuy chon) luu bieu do summary.

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
        if isinstance(values, list):  # nhi phan -> lay lop duong tinh
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

    if save_path is not None:  # pragma: no cover - ve hinh
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
    """Giai thich cuc bo cho MOT benh nhan (dung khi tu van tai phong kham).

    Tra ve top-k dac trung day xac suat len/xuong manh nhat cho ca benh do.
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
            "effect": np.where(contrib > 0, "tang nguy co", "giam nguy co"),
        }
    )
    out["abs_contribution"] = out["shap_contribution"].abs()
    return (
        out.sort_values("abs_contribution", ascending=False)
        .head(top_k)
        .drop(columns="abs_contribution")
        .reset_index(drop=True)
    )
