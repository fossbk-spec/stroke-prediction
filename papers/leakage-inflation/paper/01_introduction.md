# Introduction

> Draft v0.1 (2026-10-14). The contribution statements must be re-checked against the Results once the experiment has been run.

Stroke is a leading cause of death and long-term disability, which makes early identification of high-risk individuals a priority for prevention. Machine learning (ML) on routinely collected tabular data has become a popular route to stroke-risk prediction. A recent systematic review included 58 studies [@Soladoye2025SR], of which 48 used electronic health records. It reported that ensemble methods reach accuracies of 90.4–97.8% for stroke occurrence prediction [@Soladoye2025SR]. Much of this literature relies on a small number of public datasets, most prominently the Kaggle Stroke Prediction Dataset.

Such headline figures are hard to reconcile with what is known about the problem. Stroke is rare in these data, so accuracy is dominated by the majority class. Class-imbalance corrections such as oversampling distort predicted risks and can harm calibration [@vandenGoorbergh2022Imbalance]. More generally, data leakage is a recognised source of irreproducible results across ML-based science [@Kaufman2012Leakage; @Kapoor2023Leakage]. Fitting preprocessing or resampling steps before data are split lets information about the evaluation set leak into training, and the resulting estimates no longer describe performance on new patients.

The stroke-prediction literature shows these problems clearly.
- **Leaky protocols.** In a stacking study on the Kaggle data, SMOTE was applied to the whole dataset before 10-fold cross-validation on 6148 instances. The study reported an AUC of 98.9% and a recall of 97.4% [@Dritsas2022Stacking]. Another study oversampled the data before dividing it into training and test sets, and reported 100% accuracy and recall for its stacking ensemble [@Rehman2023RDET]. Other studies evaluated models on artificially balanced test sets [@Yang2021Hypertensive; @Vu2024Suita].
- **Leak-aware evaluations.** Under nested cross-validation, logistic regression reached an AUC of 83.30 and matched more complex learners [@Kokkotis2022Explainable]. A recent comparison on the Kaggle data found that logistic regression detected none of 50 test strokes despite 95.11% accuracy [@Akinwumi2025Evaluating].
- **Large cohorts and pooled analyses.** In a pooled analysis of 62,482 participants, ML did not improve discrimination over the pooled cohort equations [@Hong2023Race]. A tree-based score fell from a C-statistic of 87.43% in derivation to 75.29% at external validation [@Orfanoudaki2020NonLinear].
- **Reviews.** Reviews of the field flag suspected overfitting, missing calibration and external validation [@HeseltineCarp2025SR], and training data of unclear or unrepresentative provenance [@Soladoye2025SR; @Velagapudi2021Discrepancies].

The flaw itself is not new. Oversampling before splitting is a documented source of over-optimism [@Santos2018CVImbalanced; @Vandewiele2021Oversampling]. Leakage can inflate performance in clinical prediction from tabular data more broadly [@Starcke2025Parkinson]. A recent preprint on a public stroke dataset reported AUC values up to 0.99 with SMOTE applied before splitting, against 0.84 for XGBoost when SMOTE was restricted to the training set [@Wen2025LeakagePreprint]. Such comparisons, however, score the leaky arm on a re-balanced test set that contains synthetic records. They therefore mix three effects: leakage into training, the changed evaluation prevalence, and synthetic test cases. Leakage forms also differ widely in their effect, from drastic inflation to none [@Rosenblatt2024Leakage]. What is missing is a decomposition, on the same data, models and validation patients, of how much each of these mechanisms contributes, with uncertainty that respects repeated cross-validation [@Bates2023CV].

A second gap concerns the data. Gibson et al. showed that the Kaggle stroke dataset lacks verifiable provenance. They found 125 prediction-model studies built on it and its diabetes counterpart, and noted that with a maximum patient identifier of 72,940 but only 5,110 records, 93% of potential observations are unaccounted for [@Gibson2026Provenance]. Where those records went, and how the 5,110 were selected, has not been established. This matters because the dataset's prevalence is the basis for every reported accuracy, recall and precision in that literature.

This study addresses both gaps. Its contributions are:
1. **Decomposition of leakage inflation.** On the Kaggle Stroke Prediction Dataset we compare a leak-free, patient-grouped protocol with several alternatives:
   - oversampling before splitting, both as published (P1a) and matched to the same validation patients (P1b);
   - preprocessing before splitting (P2);
   - a balanced evaluation set (P3);
   - no resampling (P4).

   This separates leakage into training from evaluation-prevalence and synthetic-test effects. For the matched protocols we report fold-level paired differences with corrected resampled intervals, for discrimination, threshold-based measures and the Brier score.
2. **The lineage of the dataset.** We show that all 5,110 records are contained, with identical values and labels, in a 43,400-record dataset with 783 strokes. Those counts match the cerebral stroke dataset of Liu et al. [@Liu2019CerebralStroke]. The subset was sampled depending on the outcome: it retains 249 of 783 strokes but only 4,861 of 42,617 non-strokes, which raises the prevalence from 1.80% to 4.87%. We repeat the experiment on the 38,290 patients who do not overlap with the subset.
3. **An evidence base for the protocols.** We link each protocol to documented practice through a rapid review of 19 core studies with verified citations, which also identified retracted studies in this literature.
4. **Reproducible tooling.** We release open, tested code that reproduces every number in this paper.

Consistent with Gibson et al. [@Gibson2026Provenance], neither dataset should inform clinical decisions. Our analyses concern the methods of the literature built on them, not a deployable model.
