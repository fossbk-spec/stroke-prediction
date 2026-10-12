"""Cac ham EDA tai su dung duoc (yeu cau: chay end-to-end khong loi).

Moi ham deu: (1) nhan DataFrame, (2) tra ve DataFrame/Figure, (3) khong sua
du lieu goc (khong co side effect), (4) chay duoc ca khi thieu matplotlib
backend do hinh (dung Agg).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config


# --------------------------------------------------------------------------- #
# 1. Bang mo ta
# --------------------------------------------------------------------------- #
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Bao cao gia tri khuyet thieu, bao gom ca gia tri 'Unknown' ngam dinh.

    `smoking_status == 'Unknown'` khong phai NaN ve mat ky thuat nhung la
    khuyet thieu ve mat ngu nghia -> phai duoc bao cao tach bach.
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
                "dtype": str(df[col].dtype),
                "n_missing": n_nan,
                "pct_missing": n_nan / len(df) if len(df) else 0.0,
                "n_unknown_token": n_unknown,
                "pct_unknown_token": n_unknown / len(df) if len(df) else 0.0,
                "n_unique": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False).reset_index(drop=True)


def class_balance_report(df: pd.DataFrame) -> pd.DataFrame:
    """Bang mat can bang lop o CA HAI cap do: ban ghi (sample) va benh nhan (patient).

    Phan biet nay la bat buoc trong de bai: sample-level imbalance la lech nhan
    giua cac ban ghi; patient-level imbalance la lech nhan giua cac benh nhan
    duy nhat. Voi Kaggle dataset, moi `id` chi co 1 ban ghi nen hai con so
    trung nhau — nhung kiem tra nay se phat hien ngay neu du lieu EHR that
    co nhieu ban ghi/benh nhan.
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
    """Thong ke mo ta bien lien tuc, tach theo nhan `stroke` de thay phan tach."""
    features = features or config.NUMERIC_FEATURES
    frames = []
    for col in features:
        grouped = df.groupby(config.TARGET_COLUMN)[col]
        summary = grouped.agg(["count", "mean", "std", "median", "min", "max"]).round(3)
        summary.insert(0, "feature", col)
        frames.append(summary.reset_index())
    return pd.concat(frames, ignore_index=True)


def stroke_rate_by_category(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Ty le dot quy theo tung hang muc cua mot bien phan loai."""
    grouped = df.groupby(column, dropna=False)[config.TARGET_COLUMN]
    out = grouped.agg(n_samples="count", n_stroke="sum")
    out["stroke_rate"] = out["n_stroke"] / out["n_samples"]
    return out.sort_values("stroke_rate", ascending=False).reset_index()


def stroke_rate_by_age_band(
    df: pd.DataFrame, bins: tuple[float, ...] = (0, 20, 40, 50, 60, 70, 80, 120)
) -> pd.DataFrame:
    """Ty le dot quy theo nhom tuoi — xac nhan gia thuyet dich te 60-80 tuoi."""
    tmp = df[["age", config.TARGET_COLUMN]].copy()
    tmp["age_band"] = pd.cut(tmp["age"], bins=list(bins), right=False)
    grouped = tmp.groupby("age_band", observed=True)[config.TARGET_COLUMN]
    out = grouped.agg(n_samples="count", n_stroke="sum")
    out["stroke_rate"] = out["n_stroke"] / out["n_samples"]
    return out.reset_index()


def glucose_risk_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Phan tang nguy co theo nguong duong huyet lam sang (ADA).

    <100: binh thuong | 100-125: tien dai thao duong | 126-199: dai thao duong
    | >=200: tang duong huyet nang.
    """
    tmp = df[["avg_glucose_level", config.TARGET_COLUMN]].copy()
    tmp["glucose_stratum"] = pd.cut(
        tmp["avg_glucose_level"],
        bins=[0, 100, 126, 200, np.inf],
        labels=["<100 (binh thuong)", "100-125 (tien DTD)", "126-199 (DTD)", ">=200 (nang)"],
        right=False,
    )
    grouped = tmp.groupby("glucose_stratum", observed=True)[config.TARGET_COLUMN]
    out = grouped.agg(n_samples="count", n_stroke="sum")
    out["stroke_rate"] = out["n_stroke"] / out["n_samples"]
    baseline = out["stroke_rate"].iloc[0] if len(out) else np.nan
    out["risk_ratio_vs_normal"] = out["stroke_rate"] / baseline if baseline else np.nan
    return out.reset_index()


def correlation_with_target(df: pd.DataFrame) -> pd.DataFrame:
    """Tuong quan point-biserial giua bien so va nhan `stroke`."""
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
    """Kiem tra thien vi nhan khau hoc tren cac truc gender / residence / work.

    Muc dich: phat hien nhom bi dai dien thieu (under-represented) — mo hinh
    se hoat dong kem tren nhung nhom nay khi trien khai thuc te.
    """
    frames = []
    for col in ["gender", "Residence_type", "work_type", "smoking_status", "ever_married"]:
        part = stroke_rate_by_category(df, col).rename(columns={col: "category"})
        part.insert(0, "attribute", col)
        part["share_of_cohort"] = part["n_samples"] / len(df)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def run_full_eda(df: pd.DataFrame, output_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Chay toan bo EDA end-to-end, tra ve dict cac bang ket qua.

    Args:
        output_dir: neu duoc cung cap, luu tung bang thanh CSV de dua vao bao cao.
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
# 2. Bieu do
# --------------------------------------------------------------------------- #
def plot_eda_overview(df: pd.DataFrame, save_path: str | Path | None = None):  # pragma: no cover
    """Ve bang hinh tong quan 4 o: can bang lop, tuoi, duong huyet, BMI."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    axes[0, 0].bar(["Khong dot quy (0)", "Dot quy (1)"], counts.values, color=["#4C78A8", "#E45756"])
    for i, v in enumerate(counts.values):
        axes[0, 0].text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom", fontsize=9)
    axes[0, 0].set_title("Mat can bang lop")
    axes[0, 0].set_ylim(0, counts.max() * 1.2)

    for label, color in ((0, "#4C78A8"), (1, "#E45756")):
        subset = df.loc[df[config.TARGET_COLUMN] == label, "age"].dropna()
        axes[0, 1].hist(subset, bins=30, alpha=0.6, label=f"stroke={label}", color=color, density=True)
    axes[0, 1].set_title("Phan bo tuoi theo nhan")
    axes[0, 1].set_xlabel("age")
    axes[0, 1].legend(fontsize=8)

    bands = stroke_rate_by_age_band(df)
    axes[1, 0].bar(bands["age_band"].astype(str), bands["stroke_rate"], color="#E45756")
    axes[1, 0].set_title("Ty le dot quy theo nhom tuoi")
    axes[1, 0].tick_params(axis="x", rotation=45, labelsize=8)

    strata = glucose_risk_strata(df)
    axes[1, 1].bar(strata["glucose_stratum"].astype(str), strata["stroke_rate"], color="#F58518")
    axes[1, 1].set_title("Ty le dot quy theo phan tang duong huyet")
    axes[1, 1].tick_params(axis="x", rotation=30, labelsize=8)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
