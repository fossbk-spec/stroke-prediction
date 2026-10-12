"""Project N22 - Stroke Prediction using Machine Learning.

Package `src` chua toan bo logic co the tai su dung cua du an:

- `config`         : hang so duong dan, ten cot, tham so mac dinh.
- `data_loader`    : nap du lieu + kiem tra schema.
- `eda`            : cac ham phan tich kham pha du lieu (tai su dung duoc).
- `preprocessing`  : imputer / encoder / scaler / SMOTE theo ky luat Zero Data Leakage.
- `models`         : khoi tao cac mo hinh baseline va nang cao.
- `evaluation`     : Recall, PR-AUC, F-beta, confusion matrix, tim nguong toi uu.
- `explainability` : SHAP / permutation importance cho tinh giai thich lam sang (XAI).
"""

__version__ = "1.0.0"
