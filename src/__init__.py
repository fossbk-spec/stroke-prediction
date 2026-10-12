"""Project N22 - Stroke Prediction using Machine Learning.

The `src` package holds all reusable project logic:

- `config`         : path constants, column names, default parameters.
- `data_loader`    : data loading + schema validation.
- `eda`            : reusable exploratory data analysis helpers.
- `preprocessing`  : imputer / encoder / scaler / SMOTE under the
  Zero Data Leakage discipline.
- `models`         : baseline and advanced model factories.
- `evaluation`      : Recall, PR-AUC, F-beta, confusion matrix, threshold tuning.
- `explainability`  : SHAP / permutation importance for clinical explainability (XAI).
"""

__version__ = "1.0.0"
