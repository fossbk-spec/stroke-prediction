"""Script huan luyen end-to-end: nap du lieu -> chia -> huan luyen -> danh gia.

Chay:
    python -m src.train                          # chay toan bo mo hinh mac dinh
    python -m src.train --models logistic_regression random_forest
    python -m src.train --no-smote --class-weight balanced
    python -m src.train --synthetic               # chay voi du lieu gia lap (CI/demo)

Ket qua duoc ghi ra `reports/`:
    - model_comparison.csv : bang so sanh cac mo hinh tren tap TEST
    - cv_<model>.csv       : ket qua cross-validation theo fold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src import config, evaluation, models, preprocessing
from src.data_loader import (
    load_raw_data,
    make_synthetic_dataset,
    profile_dataset,
    split_features_target,
)

DEFAULT_MODELS = ["logistic_regression", "random_forest", "gradient_boosting", "stacking"]


def run_experiment(
    df: pd.DataFrame,
    model_names: list[str],
    *,
    use_smote: bool = True,
    class_weight: str | None = None,
    beta: float = config.F_BETA,
    min_recall: float | None = None,
    run_cv: bool = True,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Huan luyen + danh gia nhieu mo hinh theo giao thuc Zero Data Leakage.

    Giao thuc:
        1. Chia patient-level train/val/test (StratifiedGroupKFold).
        2. Fit pipeline (tien xu ly + SMOTE + model) CHI tren train.
        3. Chon nguong quyet dinh tren VALIDATION.
        4. Bao cao do do cuoi cung MOT LAN tren TEST.
    """
    split = preprocessing.patient_level_split(df)
    print("\n--- Phan bo du lieu sau khi chia cap do benh nhan ---")
    print(split.summary().to_string(index=False))

    X_train, y_train = split_features_target(split.train)
    X_val, y_val = split_features_target(split.val)
    X_test, y_test = split_features_target(split.test)

    results: list[evaluation.EvaluationResult] = []
    cv_tables: dict[str, pd.DataFrame] = {}

    for name in model_names:
        print(f"\n=== {name} (smote={use_smote}, class_weight={class_weight}) ===")
        estimator = models.get_model(name, class_weight=class_weight)
        scale = name not in {"random_forest", "gradient_boosting", "decision_tree", "xgboost"}
        pipeline = preprocessing.make_full_pipeline(
            estimator, use_smote=use_smote, scale=scale
        )

        pipeline.fit(X_train, y_train)

        # Kiem chung: transform tap hold-out khong lam thay doi trang thai da fit.
        preprocessing.assert_no_fit_on_holdout(pipeline, X_test)

        # Chon nguong tren VALIDATION (khong bao gio tren TEST).
        val_proba = evaluation.predict_proba_positive(pipeline, X_val)
        if min_recall is not None:
            threshold = evaluation.threshold_for_min_recall(y_val, val_proba, min_recall=min_recall)
        else:
            threshold, val_fbeta = evaluation.find_best_threshold(y_val, val_proba, beta=beta)
            print(f"  Nguong toi uu tren validation: {threshold:.4f} (F{beta:g}={val_fbeta:.4f})")

        result = evaluation.evaluate_model(
            pipeline, X_test, y_test, model_name=name, threshold=threshold, beta=beta
        )
        print(result)
        results.append(result)

        if run_cv:
            train_val = pd.concat([split.train, split.val], ignore_index=True)
            X_tv, y_tv = split_features_target(train_val)
            cv_tables[name] = evaluation.cross_validate_grouped(
                pipeline, X_tv, y_tv, train_val[config.ID_COLUMN], beta=beta
            )
            print("  CV (train+val):")
            print("   ", evaluation.summarize_cv(cv_tables[name]).to_dict())

    table = evaluation.comparison_table(results)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_dir / "model_comparison.csv", index=False)
        for name, cv in cv_tables.items():
            cv.to_csv(output_dir / f"cv_{name}.csv")
        (output_dir / "split_summary.json").write_text(
            json.dumps(split.summary().to_dict(orient="records"), indent=2), encoding="utf-8"
        )
        print(f"\nDa ghi ket qua vao: {output_dir}")

    return table, cv_tables


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Huan luyen mo hinh du doan dot quy (Project N22).")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Danh sach mo hinh. Hop le: {models.available_models()}")
    parser.add_argument("--no-smote", action="store_true", help="Tat SMOTE (dung cost-sensitive thay the).")
    parser.add_argument("--class-weight", default=None, choices=[None, "balanced"],
                        help="Bat cost-sensitive learning.")
    parser.add_argument("--beta", type=float, default=config.F_BETA, help="Beta cho F-beta score.")
    parser.add_argument("--min-recall", type=float, default=None,
                        help="Dat nguong theo rang buoc Recall toi thieu (vi du 0.8).")
    parser.add_argument("--no-cv", action="store_true", help="Bo qua cross-validation.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Dung du lieu gia lap (khi chua tai CSV goc).")
    parser.add_argument("--data", default=None, help="Duong dan CSV tuy chon.")
    parser.add_argument("--output", default=str(config.REPORTS_DIR), help="Thu muc ghi ket qua.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.synthetic:
        print("[CANH BAO] Dang dung DU LIEU GIA LAP - khong dung de bao cao khoa hoc.")
        df = make_synthetic_dataset(n_rows=2000)
    else:
        df = load_raw_data(args.data)

    print(f"\nProfile du lieu: {profile_dataset(df)}")

    table, _ = run_experiment(
        df,
        args.models,
        use_smote=not args.no_smote,
        class_weight=args.class_weight,
        beta=args.beta,
        min_recall=args.min_recall,
        run_cv=not args.no_cv,
        output_dir=Path(args.output),
    )

    print("\n================ BANG SO SANH TREN TAP TEST ================")
    cols = ["model", "threshold", "recall", "pr_auc", f"f{args.beta:g}", "precision",
            "roc_auc", "specificity", "accuracy", "null_accuracy", "n_false_negative"]
    print(table[[c for c in cols if c in table.columns]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
