# Project N22 — Stroke Prediction using Machine Learning

Du doan nguy co dot quy tu du lieu lam sang (Kaggle Stroke Prediction
Dataset, 5,110 benh nhan), tuan thu ky luat **Zero Data Leakage** va uu tien
**Recall/PR-AUC** thay vi Accuracy cho bai toan y khoa mat can bang lop
nghiem trong (4.9% duong tinh). Chi tiet lap luan: [`docs/DETAILED_EXAM_REPORT.md`](docs/DETAILED_EXAM_REPORT.md).

## Chay nhanh (local)
```bash
make setup      # tao .venv + cai dat requirements.txt
make verify      # kiem tra data/raw/ dung 5,110 ban ghi / 249 ca duong tinh
make test        # 72 unit test (bat buoc pass, dac biet tests/test_data_leakage.py)
make eda         # xuat bang + bieu do EDA vao reports/
make train        # huan luyen + so sanh 4 mo hinh, ket qua vao reports/
```

## Chay tren Google Colab (offline — khong can Kaggle token)
Dataset da duoc dong goi san trong `data/raw/`. Mo bat ky notebook trong
`notebooks/` tren Colab, cell dau tien se tu `git clone` repo nay va
`pip install -r requirements.txt`.

## Cau truc repository
```
stroke-prediction/
├── data/
│   ├── raw/                     # healthcare-dataset-stroke-data.csv (dong goi san)
│   ├── processed/                # du lieu da tien xu ly (sinh ra, khong commit)
│   └── SSoT_pdf_references/      # thu muc SSoT cho PDF bai bao (xem README trong thu muc)
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing_and_imbalance.ipynb
│   └── 03_model_training_and_evaluation.ipynb
├── src/
│   ├── config.py           # hang so/schema trung tam
│   ├── data_loader.py      # nap + validate schema + sinh du lieu gia lap
│   ├── eda.py               # cac ham EDA tai su dung
│   ├── preprocessing.py     # patient-level split, Pipeline chong ro ri du lieu
│   ├── models.py            # Logistic Regression ... Stacking Ensemble
│   ├── evaluation.py        # Recall/PR-AUC/F2, threshold tuning, grouped CV
│   ├── explainability.py    # SHAP, permutation importance, odds ratio
│   └── train.py             # CLI huan luyen end-to-end
├── scripts/
│   ├── download_data.py     # tai du lieu tu Kaggle (kagglehub/CLI/huong dan tay)
│   └── run_eda.py
├── tests/                    # 72 unit test, bao gom bo test chong ro ri du lieu
├── docs/
│   ├── DETAILED_EXAM_REPORT.md   # bao cao loi giai chi tiet (nguon de bai)
│   └── literature/                # tom tat 6 bai bao + dataset spec
└── .github/workflows/ci_cd_pipeline.yml
```

## Ky luat Zero Data Leakage (bat buoc)
1. **Patient-level split**: `StratifiedGroupKFold(groups=id, stratify=stroke)` —
   khong bao gio chia ngau nhien cap ban ghi.
2. **Fit-on-train-only**: `KNNImputer`, `StandardScaler`, `OneHotEncoder`, `SMOTE`
   chi `fit` tren train; validation/test chi `transform`.
3. **SMOTE nam trong Pipeline** (`imblearn.pipeline.Pipeline`) — tranh loi ma
   Dritsas & Trigka (2022) va Abed et al. (2022) mac phai (SMOTE truoc CV).
4. Moi lan chay deu co the assert bang `verify_zero_data_leakage()` va
   `assert_no_fit_on_holdout()` — xem `tests/test_data_leakage.py`.

## Bo do do danh gia
Thu tu uu tien: **Recall (Sensitivity) > PR-AUC > F2-score > ROC-AUC > Specificity**.
Khong bao gio dung Accuracy don doc — xem Phan I/III cua
[`docs/DETAILED_EXAM_REPORT.md`](docs/DETAILED_EXAM_REPORT.md) ve rui ro bat
doi xung FN vs FP trong chan doan y khoa.

## Ket qua tham khao (tap Test, tune nguong tren Validation bang F2)
| Model | Recall | PR-AUC | F2 | Precision | ROC-AUC |
|---|---|---|---|---|---|
| Stacking Ensemble | 0.84 | 0.12 | 0.38 | 0.12 | 0.77 |
| Gradient Boosting | 0.74 | 0.14 | 0.41 | 0.15 | 0.82 |
| Random Forest | 0.62 | 0.15 | 0.38 | 0.15 | 0.81 |
| Logistic Regression | 0.42 | 0.18 | 0.34 | 0.20 | 0.86 |

Chay lai de tai tao: `make train` (xem `reports/train_log.txt`).

## Giay phep
Ma nguon: MIT (xem `LICENSE`). Du lieu Kaggle tuan theo giay phep rieng cua
Kaggle; PDF bai bao thuoc ban quyen nha xuat ban (xem `data/SSoT_pdf_references/README.md`).
