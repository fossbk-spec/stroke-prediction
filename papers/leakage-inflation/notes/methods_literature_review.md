# Targeted literature review: methodological basis for the revised leakage experiment

> v0.1, 2026-10-14. Scope: a targeted (non-systematic) review supporting the design changes proposed by the independent Codex review (`fix/leakage-review`).
> - Sources: OpenAlex relevance search on 7 topic queries; metadata and abstracts from Crossref, OpenAlex and Europe PMC.
> - All DOIs were verified on Crossref and none is retracted.
> - One item, van de Mortel et al. 2025 (Mol Psychiatry), was excluded because it is a letter.
> - Quotes come from the abstracts and are stored in `references/evidence_matrix.json`.

## 1. Oversampling before splitting inflates performance (review finding 1 → P1a/P1b)
- Applying oversampling to the entire dataset before cross-validation is a known flaw that produces "biased models and overly-optimistic estimates" [@Santos2018CVImbalanced].
- Vandewiele et al. reproduced near-perfect published results on a public preterm-birth database. They showed that "applying over-sampling before partitioning the data into mutually exclusive training and testing sets" biases the results [@Vandewiele2021Oversampling]. This is the closest precedent to our study: a re-analysis of a public medical dataset with near-perfect published accuracies.
- **Implication for the design.** The published bad protocol (P1a) is worth reproducing descriptively. Attributing inflation to leakage, however, needs an arm scored on the same real validation patients as the clean protocol (P1b). Otherwise synthetic test rows and a changed prevalence confound the contrast.

## 2. Not all leakage forms inflate equally (review finding 2 → P2 isolation)
- Across more than 400 pipelines and 5 forms of leakage, "leakage via feature selection and repeated subjects drastically inflated prediction performance", whereas "other forms of leakage had only minor effects ... or even decreased prediction performance" [@Rosenblatt2024Leakage].
- Leakage typically results in "overoptimistic performance estimates and failure to generalize to new data" [@Sasse2025LeakageScenarios; @Kapoor2023Leakage].
- **Implication for the design.** Preprocessing-only leakage (P2) may be small, null or even negative, so P2 must differ from P0 *only* in whether validation records enter the fits. The fix keeps P0's order (impute, then SMOTENC, then encode/scale). The "repeated subjects" result supports patient-grouped splitting when patients have several records.

## 3. Uncertainty for cross-validated estimates (review findings 3 and 6 → corrected intervals)
- Standard cross-validation confidence intervals "may have coverage far below the desired level ... the usual estimate of variance is too small", because folds share data [@Bates2023CV].
- The corrected resampled *t*-test [@Nadeau2003Inference] inflates the variance of fold-level differences by (1/(k·r) + n_test/n_train). It is a pragmatic correction for paired, same-fold comparisons.
- **Implication for the design.**
  - Report descriptive repeat ranges for single protocols.
  - Use corrected intervals only for matched, same-fold deltas (P1b, P2, P3 and P4 vs P0), and treat them as approximate.
  - Excluding NaN folds consistently follows from computing paired deltas at the fold level.
  - A nested-CV variance estimator [@Bates2023CV] is a possible sensitivity analysis.

## 4. Evaluation prevalence changes precision-based measures (review finding 5 → P3)
- On imbalanced data the ROC plot "can be deceptive". Precision-recall analysis instead evaluates "the fraction of true positives among positive predictions" [@Saito2015PR]. That fraction depends directly on the evaluation prevalence.
- **Implication for the design.** P3 is a controlled prevalence experiment. Recall and ROC-AUC are prevalence-invariant (recall is identical to P0 by construction). Precision, PR-AUC and accuracy shift by design. The paper should present P3 as quantifying this reporting artefact, not as a reproduction of the cited workflows.

## 5. Resampling and calibration (review finding 8 → P4, and the main pipeline)
- Models developed without imbalance correction "consistently had equal or better calibration", and correction caused "an over-estimation of risk" [@Carriero2025Imbalance; @vandenGoorbergh2022Imbalance].
- Random resampling does "not generally improve discrimination performance (i.e., AUC)", and its use is "hard to justify when aiming at providing calibrated predictions". This was shown with an International Stroke Trial example [@Piccininni2024Resampling].
- Across 10 clinical datasets (more than 600,000 patients), 1:1 resampling "had no positive impact on predictive performance". Brier scores rose by 0.029–0.080, and "gains from imbalance correction can typically be reproduced by threshold adjustment" [@Andersen2026TippingBalance].
- **Implication for the design.** P4 (no resampling) is a substantive comparator, not only a reference. A plausible finding is P4 ≈ P0 in ROC-AUC with better calibration, so P0-with-SMOTENC should not be presented as the "best practice" model. **This also affects the project's main pipeline (`src/train.py` uses SMOTENC by default):**
  - consider making the no-resampling model with a validation-chosen threshold the primary model;
  - report calibration for both.
  This is flagged for a maintainer decision.

## Open items
- Nadeau & Bengio 2003: no open abstract. The corrected-variance formula should be checked against the paper (or Bouckaert & Frank 2004) before the Methods text is finalised.
- Consider adding Riley-style sample-size considerations for the 249-event primary dataset (not searched in this round).
