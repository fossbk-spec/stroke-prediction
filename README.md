# Project N22 — Stroke Prediction using Machine Learning

Predicting stroke risk from clinical data (Kaggle Stroke Prediction Dataset,
5,110 patients), under a strict **Zero Data Leakage** discipline and
prioritizing **Recall/PR-AUC** over Accuracy for a severely imbalanced
medical classification problem (4.9% positive rate). Full rationale:
[`docs/DETAILED_EXAM_REPORT.md`](docs/DETAILED_EXAM_REPORT.md).

## Run on Google Colab / Kaggle Notebooks (no Kaggle token needed)
The dataset is already bundled under `data/raw/`, so each notebook below
clones this repo and installs dependencies in its first cell — no manual
setup required. Colab/Kaggle need GitHub access to clone and PyPI access
to install dependencies (enable Internet in Kaggle). Local runs inside the repo
skip cloning and installation.

| Notebook | Colab | Kaggle |
|---|---|---|
| 01 - Exploratory Data Analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fossbk-spec/stroke-prediction/blob/main/notebooks/01_exploratory_data_analysis.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/fossbk-spec/stroke-prediction/blob/main/notebooks/01_exploratory_data_analysis.ipynb) |
| 02 - Preprocessing & Imbalance | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fossbk-spec/stroke-prediction/blob/main/notebooks/02_preprocessing_and_imbalance.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/fossbk-spec/stroke-prediction/blob/main/notebooks/02_preprocessing_and_imbalance.ipynb) |
| 03 - Model Training & Evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fossbk-spec/stroke-prediction/blob/main/notebooks/03_model_training_and_evaluation.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/fossbk-spec/stroke-prediction/blob/main/notebooks/03_model_training_and_evaluation.ipynb) |

## Quick start (local)
```bash
make setup      # create .venv + install requirements-dev.txt
make verify      # check data/raw/ matches 5,110 records / 249 positive cases
make test        # 103 tests (must pass, especially tests/test_data_leakage.py)
make eda         # export EDA tables + figures into reports/
make train        # train + compare 4 models, results written to reports/
```

## Repository structure
```
stroke-prediction/
├── data/
│   ├── raw/                     # healthcare-dataset-stroke-data.csv (bundled)
│   ├── processed/                # preprocessed data (generated, not committed)
│   └── SSoT_pdf_references/      # SSoT directory for paper PDFs (see its README)
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing_and_imbalance.ipynb
│   └── 03_model_training_and_evaluation.ipynb
├── src/
│   ├── config.py           # central config/schema constants
│   ├── data_loader.py      # load + validate schema + synthetic data generator
│   ├── eda.py               # reusable EDA helpers
│   ├── preprocessing.py     # patient-level split, leakage-safe Pipeline
│   ├── models.py            # Logistic Regression ... Stacking Ensemble
│   ├── evaluation.py        # Recall/PR-AUC/F2, threshold tuning, grouped CV
│   ├── explainability.py    # SHAP, permutation importance, odds ratios
│   └── train.py             # end-to-end training CLI
├── scripts/
│   ├── download_data.py     # fetch data from Kaggle (kagglehub/CLI/manual)
│   └── run_eda.py
├── tests/                    # 103 tests, including the anti-leakage suite
├── docs/
│   ├── DETAILED_EXAM_REPORT.md   # detailed exam-solution report (source brief)
│   └── literature/                # 6 paper summaries + dataset spec
└── .github/workflows/ci_cd_pipeline.yml
```

## Zero Data Leakage discipline (mandatory)
1. **Patient-level holdout**: stratified `train_test_split` on patient IDs
   creates train/validation/test sets. CV uses `StratifiedGroupKFold(groups=id)`;
   all records of a patient stay in the same split.
2. **Fit-on-train-only**: `KNNImputer`, `StandardScaler`, `OneHotEncoder`,
   `SMOTE` are only `fit` on train; validation/test only call `.transform()`.
3. **SMOTENC lives inside the Pipeline** (`imblearn.pipeline.Pipeline`) —
   avoiding the mistake made by Dritsas & Trigka (2022) and Abed et al.
   (2022), who ran SMOTE before cross-validation.
   SMOTENC resamples imputed raw features before one-hot encoding; categorical
   and binary columns retain valid categories. Stacking uses patient-grouped
   internal OOF folds, with preprocessing/resampling fitted inside each fold.

Odds-ratio CIs use a patient-level bootstrap of the full training pipeline:
`logistic_odds_ratios(pipeline, X_train, y_train, groups=patient_ids,
n_boot=200, random_state=42)`. `significant` means the percentile coefficient
CI excludes zero. These are exploratory stability intervals for the regularized,
resampled procedure, not causal inference or calibrated hypothesis tests.
Category support and scaling may vary between refits; absent categories receive
NaN bounds and `significant=False`. Unfit bootstrap resamples raise an error.

4. Every run can be asserted with `verify_zero_data_leakage()` and
   `assert_no_fit_on_holdout()` — see `tests/test_data_leakage.py`.

## Evaluation metrics
Priority order: **Recall (Sensitivity) > PR-AUC > F2-score > ROC-AUC > Specificity**.
Accuracy alone is never used — see Parts I/III of
[`docs/DETAILED_EXAM_REPORT.md`](docs/DETAILED_EXAM_REPORT.md) for the
asymmetric FN vs. FP risk analysis in medical diagnosis.

## Reference results (Test set, threshold tuned on Validation via F2)
| Model | Recall | PR-AUC | F2 | Precision | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.76 | 0.19 | 0.35 | 0.11 | 0.83 |
| Random Forest | 0.70 | 0.17 | 0.38 | 0.13 | 0.82 |
| Gradient Boosting | 0.50 | 0.17 | 0.30 | 0.12 | 0.79 |
| Stacking Ensemble | 0.88 | 0.17 | 0.36 | 0.11 | 0.83 |

Grouped cross-validation (mean ± sample standard deviation across 5 folds):

| Model | Recall | PR-AUC |
|---|---|---|
| Logistic Regression | 0.74 ± 0.11 | 0.16 ± 0.03 |
| Random Forest | 0.83 ± 0.06 | 0.14 ± 0.04 |
| Gradient Boosting | 0.84 ± 0.11 | 0.15 ± 0.03 |
| Stacking Ensemble | 0.74 ± 0.08 | 0.14 ± 0.03 |

The test set has only ~50 positive cases, so test differences are noisy; CV is more reliable.

Reproduce with: `make train`. Pipeline: imputation -> SMOTENC -> encoding/scaling; patient-grouped stacking. Generated on Python 3.14 (scikit-learn 1.9.1, imbalanced-learn 0.14.2); exact numbers can differ slightly on the pinned CI environment.

## License
Source code: MIT (see `LICENSE`). The Kaggle dataset follows Kaggle's own
license; paper PDFs belong to their respective publishers (see
`data/SSoT_pdf_references/README.md`).
