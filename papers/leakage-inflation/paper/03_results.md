# Results

> Skeleton v0.1 (2026-10-14). No numbers until `python -m src.leakage_experiment` has been run (deferred by the maintainer). Fill from `reports/leakage/leakage_summary.csv` and `reports/leakage_disjoint/leakage_summary.csv`. Every number must come from these files.

## 3.1 Evidence synthesis
- PRISMA flow: 1,099 identified, 981 unique, 74 eligible, 19 core studies (from `docs/literature/evisyn/search_log.json`). Figure 1: PRISMA diagram (`prisma.mmd`).
- Table 1: the 19 core studies by protocol type (P1/P2/P3 practice, or leak-aware), with the reported headline metrics, and the retracted or flagged articles.

## 3.2 Dataset provenance
- Subset finding and disjoint-sample characteristics (`reports/data_provenance.json`). Table 2: N, events and prevalence for the primary dataset, the full source and the disjoint sample.

## 3.3 Inflation by protocol (primary dataset)
- Table 3: for each protocol and model, the mean and repeat range of ROC-AUC, PR-AUC, recall, accuracy and Brier score. P1a is reported separately as a descriptive reproduction of the published flaw.
- Table 4: matched fold-level differences vs P0 (P1b, P2, P3, P4), with corrected resampled 95% intervals and valid fold counts.
- Figure 2: `figures/leakage_inflation.png`.
- Pre-specified questions:
  - RQ1: How much does oversampling before splitting inflate discrimination and recall when it is scored on the same patients (P1b)? How far is the published-style P1a figure from it?
  - RQ2: Is preprocessing-only leakage (P2) small, null or negative, as in other domains?
  - RQ3: How much does a balanced evaluation set (P3) change precision, PR-AUC and accuracy, with recall and ROC-AUC unchanged by construction?
  - RQ4: Does 1:1 resampling (P0 vs P4) improve discrimination, or does it mainly worsen the Brier score and shift threshold-based measures?

## 3.4 Robustness on the patient-disjoint sample
- Repeat Tables 3–4 on the 38,290-patient sample. Report whether the direction and rough size of each inflation hold.
