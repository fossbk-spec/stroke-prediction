"""Reusable EDA helpers (requirement: must run end-to-end without errors).

Every function: (1) takes a DataFrame, (2) returns a DataFrame/Figure,
(3) never mutates the input data (no side effects), (4) works even without a
display backend (uses Agg for plotting).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config


# --------------------------------------------------------------------------- #
# 1. Descriptive tables
# --------------------------------------------------------------------------- #
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report missing values, including the implicit 'Unknown' sentinel value.

    `smoking_status == 'Unknown'` is not technically NaN but is semantically
    missing -> it must be reported separately.
    """
    rows = []
    for col in df.columns:
        series = df[col]
        n_nan = int(series.isna().sum())
        is_text = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        n_unknown = int((series == "Unknown").sum()) if is_text else 0
        rows.append(
            {
                "feature": col,
                "dtype": str(series.dtype),
                "n_missing": n_nan,
                "pct_missing": n_nan / len(df) if len(df) else 0.0,
                "n_unknown_token": n_unknown,
                "pct_unknown_token": n_unknown / len(df) if len(df) else 0.0,
                "n_unique": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False).reset_index(drop=True)


def class_balance_report(df: pd.DataFrame) -> pd.DataFrame:
    """Class imbalance table at BOTH levels: sample (record) and patient.

    This distinction is required by the assignment: sample-level imbalance is
    the label skew across records; patient-level imbalance is the label skew
    across unique patients. For the Kaggle dataset, every `id` has exactly 1
    record so the two numbers coincide — but this check would immediately
    flag a difference on real EHR data with multiple records per patient.
    """
    n = len(df)
    pos = int((df[config.TARGET_COLUMN] == 1).sum())

    patient_level = df.groupby(config.ID_COLUMN)[config.TARGET_COLUMN].max()
    n_pat = len(patient_level)
    pos_pat = int((patient_level == 1).sum())

    return pd.DataFrame(
        [
            {
                "level": "sample-level",
                "n_total": n,
                "n_positive": pos,
                "n_negative": n - pos,
                "positive_rate": pos / n if n else 0.0,
                "imbalance_ratio": (n - pos) / pos if pos else np.inf,
            },
            {
                "level": "patient-level",
                "n_total": n_pat,
                "n_positive": pos_pat,
                "n_negative": n_pat - pos_pat,
                "positive_rate": pos_pat / n_pat if n_pat else 0.0,
                "imbalance_ratio": (n_pat - pos_pat) / pos_pat if pos_pat else np.inf,
            },
        ]
    )


def numeric_summary(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Descriptive statistics for continuous features, split by the `stroke` label."""
    features = features or config.NUMERIC_FEATURES
    frames = []
    for col in features:
        grouped = df.groupby(config.TARGET_COLUMN)[col]
        summary = grouped.agg(["count", "mean", "std", "median", "min", "max"]).round(3)
        summary.insert(0, "feature", col)
        frames.append(summary.reset_index())
    return pd.concat(frames, ignore_index=True)


def stroke_rate_by_category(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Stroke rate broken down by each category of a categorical feature."""
    grouped = df.groupby(column, dropna=False)[config.TARGET_COLUMN]
    out = grouped.agg(n_samples="count", n_stroke="sum")
    out["stroke_rate"] = out["n_stroke"] / out["n_samples"]
    return out.sort_values("stroke_rate", ascending=False).reset_index()


def stroke_rate_by_age_band(
    df: pd.DataFrame, bins: tuple[float, ...] = (0, 20, 40, 50, 60, 70, 80, 120)
) -> pd.DataFrame:
    """Stroke rate by age band — confirms the 60-80 age-group epidemiological hypothesis."""
    tmp = df[["age", config.TARGET_COLUMN]].copy()
    tmp["age_band"] = pd.cut(tmp["age"], bins=list(bins), right=False)
    grouped = tmp.groupby("age_band", observed=True)[config.TARGET_COLUMN]
    out = grouped.agg(n_samples="count", n_stroke="sum")
    out["stroke_rate"] = out["n_stroke"] / out["n_samples"]
    return out.reset_index()


def glucose_risk_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Risk stratification by clinical glucose thresholds (ADA).

    <100: normal | 100-125: prediabetes | 126-199: diabetes
    | >=200: severe hyperglycemia.
    """
    tmp = df[["avg_glucose_level", config.TARGET_COLUMN]].copy()
    tmp["glucose_stratum"] = pd.cut(
        tmp["avg_glucose_level"],
        bins=[0, 100, 126, 200, np.inf],
        labels=["<100 (normal)", "100-125 (prediabetes)", "126-199 (diabetes)", ">=200 (severe)"],
        right=False,
    )
    grouped = tmp.groupby("glucose_stratum", observed=True)[config.TARGET_COLUMN]
    out = grouped.agg(n_samples="count", n_stroke="sum")
    out["stroke_rate"] = out["n_stroke"] / out["n_samples"]
    baseline = out["stroke_rate"].get("<100 (normal)", np.nan)
    out["risk_ratio_vs_normal"] = out["stroke_rate"] / baseline if baseline else np.nan
    return out.reset_index()


def correlation_with_target(df: pd.DataFrame) -> pd.DataFrame:
    """Point-biserial correlation between each feature and the `stroke` label."""
    numeric = config.NUMERIC_FEATURES + config.BINARY_FEATURES
    corr = (
        df[numeric + [config.TARGET_COLUMN]]
        .astype(float)
        .corr(numeric_only=True)[config.TARGET_COLUMN]
        .drop(config.TARGET_COLUMN)
        .sort_values(ascending=False)
    )
    return (
        corr.rename("corr_with_stroke")
        .rename_axis("feature")
        .reset_index()
    )


def demographic_bias_report(df: pd.DataFrame) -> pd.DataFrame:
    """Check demographic bias across the gender / residence / work axes.

    Purpose: detect under-represented groups — the model will perform worse
    on these groups once deployed in practice.
    """
    frames = []
    for col in ["gender", "Residence_type", "work_type", "smoking_status", "ever_married"]:
        part = stroke_rate_by_category(df, col).rename(columns={col: "category"})
        part.insert(0, "attribute", col)
        part["share_of_cohort"] = part["n_samples"] / len(df)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def run_full_eda(df: pd.DataFrame, output_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Run the full EDA end-to-end, returning a dict of result tables.

    Args:
        output_dir: if given, save each table as a CSV for inclusion in the report.
    """
    tables: dict[str, pd.DataFrame] = {
        "missing_values": missing_value_report(df),
        "class_balance": class_balance_report(df),
        "numeric_summary": numeric_summary(df),
        "age_bands": stroke_rate_by_age_band(df),
        "glucose_strata": glucose_risk_strata(df),
        "correlation": correlation_with_target(df),
        "demographic_bias": demographic_bias_report(df),
        "comorbidity": pd.concat(
            [
                stroke_rate_by_category(df, "hypertension").assign(attribute="hypertension").rename(
                    columns={"hypertension": "category"}
                ),
                stroke_rate_by_category(df, "heart_disease").assign(attribute="heart_disease").rename(
                    columns={"heart_disease": "category"}
                ),
            ],
            ignore_index=True,
        ),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(out / f"eda_{name}.csv", index=False)
    return tables


# --------------------------------------------------------------------------- #
# 2. Plots
# --------------------------------------------------------------------------- #
def plot_eda_overview(df: pd.DataFrame, save_path: str | Path | None = None):  # pragma: no cover
    """Plot a 2x2 overview figure: class balance, age, glucose, BMI."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    axes[0, 0].bar(["No stroke (0)", "Stroke (1)"], counts.values, color=["#4C78A8", "#E45756"])
    for i, v in enumerate(counts.values):
        axes[0, 0].text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom", fontsize=9)
    axes[0, 0].set_title("Class imbalance")
    axes[0, 0].set_ylim(0, counts.max() * 1.2)

    for label, color in ((0, "#4C78A8"), (1, "#E45756")):
        subset = df.loc[df[config.TARGET_COLUMN] == label, "age"].dropna()
        axes[0, 1].hist(subset, bins=30, alpha=0.6, label=f"stroke={label}", color=color, density=True)
    axes[0, 1].set_title("Age distribution by label")
    axes[0, 1].set_xlabel("age")
    axes[0, 1].legend(fontsize=8)

    bands = stroke_rate_by_age_band(df)
    axes[1, 0].bar(bands["age_band"].astype(str), bands["stroke_rate"], color="#E45756")
    axes[1, 0].set_title("Stroke rate by age band")
    axes[1, 0].tick_params(axis="x", rotation=45, labelsize=8)

    strata = glucose_risk_strata(df)
    axes[1, 1].bar(strata["glucose_stratum"].astype(str), strata["stroke_rate"], color="#F58518")
    axes[1, 1].set_title("Stroke rate by glucose stratum")
    axes[1, 1].tick_params(axis="x", rotation=30, labelsize=8)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
