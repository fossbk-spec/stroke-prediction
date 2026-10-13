# data/raw/

Contains `healthcare-dataset-stroke-data.csv` (Kaggle Stroke Prediction
Dataset, fedesoriano, 5,110 x 12). This file is BUNDLED in the repo so that
notebooks run offline on Google Colab / Kaggle Notebooks / local — no Kaggle
API token required.

If the file is removed, re-download it with:
```
python scripts/download_data.py
```

## Larger cerebral stroke dataset

Place `dataset.csv` from the [Kaggle Cerebral Stroke Prediction-Imbalanced
Dataset](https://www.kaggle.com/shashwatwork/cerebral-stroke-prediction-imbalaced-dataset)
in this directory. It is not committed. The bundled 5,110-row file is an exact
subset of this larger file: all bundled patient IDs and row values match after
empty `smoking_status` values are normalized to `Unknown`.

Run `python scripts/prepare_cerebral_disjoint.py` to write
`data/raw/cerebral-stroke-disjoint.csv` and `reports/data_provenance.json`.
This is a same-source, patient-disjoint sample for robustness checks, not
external validation.
