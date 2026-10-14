# Introduction

> Draft v0.1 (2026-10-14). The contribution statements must be re-checked against the Results once the experiment has been run.

Stroke is a leading cause of death and long-term disability, which makes early identification of high-risk individuals a priority for prevention. Machine learning (ML) on routinely collected tabular data has become a popular route to stroke-risk prediction. A recent systematic review included 58 studies [@Soladoye2025SR], of which 48 used electronic health records. It reported that ensemble methods reach accuracies of 90.4–97.8% for stroke occurrence prediction [@Soladoye2025SR]. Much of this literature relies on a small number of public datasets, most prominently the Kaggle Stroke Prediction Dataset.

Such headline figures are hard to reconcile with what is known about the problem. Stroke is rare in these data, so accuracy is dominated by the majority class. Class-imbalance corrections such as oversampling distort predicted risks and can harm calibration [@vandenGoorbergh2022Imbalance]. More generally, data leakage is a recognised source of irreproducible results across ML-based science [@Kaufman2012Leakage; @Kapoor2023Leakage]. Fitting preprocessing or resampling steps before data are split lets information about the evaluation set leak into training, and the resulting estimates no longer describe performance on new patients.

The stroke-prediction literature shows these problems clearly.
- **Leaky protocols.** In a stacking study on the Kaggle data, SMOTE was applied to the whole dataset before 10-fold cross-validation on 6148 instances. The study reported an AUC of 98.9% and a recall of 97.4% [@Dritsas2022Stacking]. Another study oversampled the data before dividing it into training and test sets, and reported 100% accuracy and recall for its stacking ensemble [@Rehman2023RDET]. Other studies evaluated models on artificially balanced test sets [@Yang2021Hypertensive; @Vu2024Suita].
- **Leak-aware evaluations.** Under nested cross-validation, logistic regression reached an AUC of 83.30 and matched more complex learners [@Kokkotis2022Explainable]. A recent comparison on the Kaggle data found that logistic regression detected none of 50 test strokes despite 95.11% accuracy [@Akinwumi2025Evaluating].
- **Large cohorts and pooled analyses.** In a pooled analysis of 62,482 participants, ML did not improve discrimination over the pooled cohort equations [@Hong2023Race]. A tree-based score fell from a C-statistic of 87.43% in derivation to 75.29% at external validation [@Orfanoudaki2020NonLinear].
- **Reviews.** Reviews of the field flag suspected overfitting, missing calibration and external validation [@HeseltineCarp2025SR], and training data of unclear or unrepresentative provenance [@Soladoye2025SR; @Velagapudi2021Discrepancies].

What is missing is a controlled estimate, on the same data, models and folds, of how much each common leaky protocol inflates reported performance. Without it, readers cannot tell how far published results on these datasets reflect method rather than leakage.

This study provides that estimate. Its contributions are:
1. **Controlled quantification of inflation.** On the Kaggle Stroke Prediction Dataset we compare a leak-free, patient-grouped protocol with four alternatives: oversampling before splitting, preprocessing before splitting, a balanced evaluation set, and no resampling. We report paired differences with confidence intervals across repeated cross-validation, for discrimination, threshold-based measures and calibration.
2. **A provenance finding and a robustness check.** We show that the widely used 5,110-record dataset is an exact subset of a larger 43,400-record dataset. We repeat the experiment on the 38,290 patients who do not overlap with it.
3. **An evidence base for the protocols.** We link each protocol to documented practice through a rapid review of 19 core studies with verified citations, which also identified retracted studies in this literature.
4. **Reproducible tooling.** We release open, tested code that reproduces every number in this paper with a single command.
