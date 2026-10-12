# reports/

Results generated from source code (CSV/JSON/PNG). These are NOT committed —
rerun `make eda` / `make train` to regenerate them.

- `figures/` : EDA charts, SHAP summary, confusion matrix.
- `eda/`     : CSV tables from `src/eda.py` (missing values, class balance, ...).
- `model_comparison.csv`, `cv_<model>.csv`, `split_summary.json` : training results.
