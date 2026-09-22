# Model card: historical loan approval label

## Intended use

A tabular-classification model for the historical loan approval label. It estimates whether an application resembles cases labeled `Y` (approved) in this Kaggle dataset. It does **not** predict default, repayment, creditworthiness, or the outcome of any current lender's process. Never use it to decide a real person's access to credit.

## Data and method

- Source: [Kaggle Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset/data); the source license is marked `Unknown`. The exact 614-row CSV checksum is in [data/README.md](data/README.md).
- Split: stratified 491 training rows and 123 held-out test rows; random seed 42. No external, temporal, or multi-seed validation.
- Features: income, co-applicant income, loan amount/term, dependents, education, self-employment, credit history, property area. `Loan_ID`, `Gender`, and `Married` are excluded.
- Preprocessing: median numeric imputation and scaling; most-frequent categorical imputation and one-hot encoding, all fitted inside cross-validation folds.
- Selection: five-fold training-set balanced accuracy among fixed logistic-regression, random-forest, and XGBoost candidates. Selected random forest (mean CV balanced accuracy 0.697). Sigmoid probability calibration and a 0.42 threshold were chosen using only training data.
- The dataset is roughly 69% `Y` and 31% `N`; balanced accuracy and both classes' recall matter more than raw accuracy alone.

## Held-out result

| Metric | Value |
| --- | ---: |
| Balanced accuracy | 0.770 |
| Accuracy | 0.854 |
| ROC AUC | 0.780 |
| Approved (`Y`) recall | 0.988 |
| Rejected (`N`) recall | 0.553 |
| Approved (`Y`) precision | 0.832 |
| Rejected (`N`) precision | 0.955 |
| Brier score | 0.136 |

Confusion matrix (`Y` positive): 21 true negatives, 17 false positives, 1 false negative, 84 true positives. A false positive means the model says `Y` when the historical label was `N`; it does not mean the borrower would default. A false negative means the model says `N` when the recorded label was `Y`.

## Risks and gaps

The tiny test set makes metrics uncertain. Missing data and unknown dataset sampling/collection methods limit generalization. Historical decisions may embed bias. Excluding gender and marital status does not remove their possible proxies. The generated report includes counts and group-level error summaries, but many groups are too small to support strong fairness conclusions (only 25 female-labeled and 2 missing-gender rows in the test split). The reported probabilities were calibrated on training folds but have not been independently validated for deployment.

Before any real decision-support use, one would need licensed representative data, clear label semantics, temporal/external validation, robust subgroup and calibration analysis, security/privacy review, monitoring, and qualified human/regulatory oversight. Those are outside this project's scope.
