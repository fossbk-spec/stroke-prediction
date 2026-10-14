# Paper status

- review_type: RAPID REVIEW (evidence synthesis component); the paper itself is methodological meta-research.
- Gate 1 (PICO/search): approved 2026-10-13.
- Gate 2 (screening): approved 2026-10-13.
- Gate 3 (matrix/outline): approved 2026-10-13.
- **Gate 3.5 (gap confirmation, 2026-10-14):** adversarial search on OpenAlex/Europe PMC for prior work refuting the gaps.
  - GAP_01 **revised**: Wen et al. 2025 (preprint) and Vandewiele 2021 already show the SMOTE-before-split inflation. The contribution changes from "quantify inflation" to "decompose it" (P1a vs P1b vs P3, matched folds, corrected intervals).
  - GAP_02 **kept and strengthened**: Gibson et al. 2026 (BMC Med) flag the missing provenance but do not trace the lineage. Our 43,400 → 5,110 outcome-dependent subsample finding (Liu et al. 2019 counts) is new.
  - GAP_03 **downgraded** to a secondary result (known principle: Saito 2015).
  - GAP_04 **downgraded** to confirmatory (Andersen 2026, Carriero 2025, Piccininni 2024; Ratnaningsih 2026 on the same dataset).
  - Scorecard: `data/gap_scorecard.json`.
- Gate 4 (manuscript sign-off): pending results.
