# Results

> Skeleton v0.1 (2026-10-14). No numbers until `python -m src.leakage_experiment` has been run (deferred by the maintainer). Fill from `reports/leakage/leakage_summary.csv` and `reports/leakage_disjoint/leakage_summary.csv`. Every number must come from these files.

## 3.1 Evidence synthesis
- PRISMA flow: 1,099 identified, 981 unique, 74 eligible, 19 core studies (from `docs/literature/evisyn/search_log.json`). Figure 1: PRISMA diagram (`prisma.mmd`).
- Table 1: the 19 core studies by protocol type (P1/P2/P3 practice, or leak-aware), with the reported headline metrics, and the retracted or flagged articles.

## 3.2 Dataset provenance
- Subset finding and disjoint-sample characteristics (`reports/data_provenance.json`). Table 2: N, events and prevalence for the primary dataset, the full source and the disjoint sample.

## 3.3 Inflation by protocol (primary dataset)
- Table 3: mean (95% CI) of ROC-AUC, PR-AUC, recall, accuracy and Brier score for each protocol and model.
- Table 4: paired difference vs P0 (95% CI).
- Figure 2: `figures/leakage_inflation.png`.
- Pre-specified questions:
  - RQ1: Does P1 inflate discrimination and recall, and by how much?
  - RQ2: Is preprocessing-only leakage (P2) negligible by comparison?
  - RQ3: How much does a balanced evaluation set (P3) change threshold-based measures (precision, accuracy) while leaving ROC-AUC unchanged?
  - RQ4: Does resampling (P0 vs P4) improve discrimination or only shift the threshold and calibration?

## 3.4 Robustness on the patient-disjoint sample
- Repeat Tables 3–4 on the 38,290-patient sample. Report whether the direction and rough size of each inflation hold.
