# Literature check of the Methods section (v0.2)

> 2026-10-14. Each methodological claim in `paper/02_methods.md` was checked for supporting literature.
> - 12 references were added. All DOIs were verified on Crossref and none is retracted.
> - Quotes come from the abstracts and are stored in `references/evidence_matrix.json`.
> - Items cited only as the original source of a method (algorithm, score, guideline) carry no numeric claim.
> - See also `methods_literature_review.md` (design changes P1a/P1b, P2, intervals, P3, P4).

| Methods section | Claim | Support | Status |
|---|---|---|---|
| 2.2 Evidence synthesis | Rapid review rather than systematic review | Cochrane Rapid Reviews Methods Group guidance, 26 recommendations [@Garritty2021RapidReview] | Added. Our shortcuts (single screener, 4 human gates) should be listed against that guidance in a supplement |
| 2.2 | PRISMA 2020 reporting | [@Page2021PRISMA] | Added |
| 2.3 Data | "Unknown" smoking status kept as a category | For prediction, missingness patterns can carry information [@Sperrin2020Missing] | Added |
| 2.4 Models | Random forest, gradient boosting | [@Breiman2001RF; @Friedman2001GB] | Added |
| 2.4 | k-nearest-neighbour imputation, fitted inside the pipeline | [@Troyanskaya2001KNN]. A mismatch between the development and deployment missing-data methods "may lead to overoptimistic assessments" [@Sperrin2020Missing] | Added |
| 2.4 | SMOTENC for mixed data | [@Chawla2002SMOTE] | Already cited |
| 2.5 Protocols | Patient-grouped (subject-wise) folds | Record-wise CV "often massively overestimates" accuracy and was used by almost half of the reviewed studies [@Saeb2017SubjectWise] | Added. The Kaggle data has one row per id, so this matters most for the multi-record cohorts discussed |
| 2.6 Measures | Brier score | [@Brier1950; @VanCalster2019Calibration] | Added |
| 2.7 Statistics | Why not a naive paired *t*-test | Paired *t* on resampled splits has high type I error [@Dietterich1998Tests] | Added |
| 2.7 | Corrected resampled *t* (variance factor 1/*J* + *n*<sub>test</sub>/*n*<sub>train</sub>) | [@Nadeau2003Inference; @Bouckaert2004Corrected]; CV intervals under-cover [@Bates2023CV] | Added. **Open: verify the exact factor for repeated k-fold against the Bouckaert & Frank full text (no open abstract).** |
| 2.8 Reporting | TRIPOD+AI and REFORMS (32 questions) | [@Collins2024TRIPODAI; @Kapoor2024REFORMS] | Added. Fill both checklists as supplements |
| (Discussion) | Sample size: 249 events in the primary dataset | Many models are developed on datasets "too small for the total number of participants or outcome events" [@Riley2020SampleSize] | For the Limitations section. Compute a Riley-style minimum sample size for the main model |

## Not cited (no peer-reviewed source, or not needed)
- The Kaggle datasets: cite them by URL and access date in Data availability (no DOI).
- scikit-learn and imbalanced-learn: cite the software papers (JMLR 2011 / 2017) in Software. These have no Crossref DOI, so add the bib entries by hand from the JMLR pages and verify them.
- F2 score: a standard definition; cite only if the target journal requires it.

## Open items
1. Verify the corrected-variance formula against Bouckaert & Frank 2004 (and Nadeau & Bengio 2003).
2. Supplement: rapid-review shortcuts mapped to Garritty et al.
3. Supplement: completed TRIPOD+AI and REFORMS checklists.
4. Discussion: sample-size limitation [@Riley2020SampleSize].
5. Software citations for scikit-learn and imbalanced-learn (manual bib entries).
