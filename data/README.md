# Dataset Documentation

## Provenance

- **File:** `train_u6lujuX_CVtuZ9i.csv`
- **Source platform:** Kaggle
- **Dataset:** [Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset/data)
- **Uploader:** Debdatta Chatterjee (`altruistdelhite04`)
- **Kaggle dataset reference:** `altruistdelhite04/loan-prediction-problem-dataset`
- **Kaggle version:** 1, published March 12, 2019
- **Source verified:** September 22, 2026
- **Original lineage:** The filename and schema match the Analytics Vidhya Loan
  Prediction III practice dataset.
- **License:** `Unknown` in Kaggle's dataset metadata.

The exact Kaggle source is confirmed, but it does not declare a reusable
license. Do not redistribute the raw file without permission from the relevant
rights holder. Other Kaggle mirrors may use different license labels; those
labels do not apply automatically to this source.

## Local Snapshot

| Property | Value |
| --- | --- |
| Rows | 614 |
| Columns | 13 |
| Input columns | 12 |
| Target | `Loan_Status` |
| Duplicate rows | 0 |
| Unique `Loan_ID` values | 614 |
| File size | 38,013 bytes |
| SHA-256 | `8C2CC67A13279EC746B15F76F14E8503154274EF7F221D2437033B4407C80CA8` |

The checksum identifies the exact raw snapshot used by this project. Do not
edit the raw CSV in place. Generate cleaned data under `data/processed/`.

## Schema

| Column | Kind | Role |
| --- | --- | --- |
| `Loan_ID` | Identifier | Unique application identifier; exclude from model features |
| `Gender` | Categorical | Applicant gender |
| `Married` | Categorical | Marital status |
| `Dependents` | Categorical | Number of dependents, including the `3+` category |
| `Education` | Categorical | Graduate status |
| `Self_Employed` | Categorical | Self-employment status |
| `ApplicantIncome` | Numeric | Applicant income |
| `CoapplicantIncome` | Numeric | Co-applicant income |
| `LoanAmount` | Numeric | Requested loan amount; source unit requires confirmation |
| `Loan_Amount_Term` | Numeric | Loan term; source unit requires confirmation |
| `Credit_History` | Binary categorical | Whether credit history meets the dataset guideline |
| `Property_Area` | Categorical | Rural, semiurban, or urban property area |
| `Loan_Status` | Binary target | `Y` for historically approved and `N` for historically rejected |

The source data dictionary should be treated as authoritative once the exact
Kaggle listing is confirmed. Units are not inferred solely from column names.

## Target Distribution

| Value | Rows | Share |
| --- | ---: | ---: |
| `Y` | 422 | 68.7% |
| `N` | 192 | 31.3% |

Use stratified splits where practical and report metrics for both classes.

## Missing Values

The snapshot contains 149 missing cells across seven columns.

| Column | Missing values |
| --- | ---: |
| `Gender` | 13 |
| `Married` | 3 |
| `Dependents` | 15 |
| `Self_Employed` | 32 |
| `LoanAmount` | 22 |
| `Loan_Amount_Term` | 14 |
| `Credit_History` | 50 |

Imputation must be learned from each training fold and then applied to its
validation fold to prevent data leakage.

## Obtaining the Data

1. Download the training CSV from the Kaggle listing linked above.
2. Place it at `data/raw/train_u6lujuX_CVtuZ9i.csv`.
3. Verify that its SHA-256 checksum matches the value above.

Files under `data/raw/` and `data/processed/` are ignored by Git. This README is
tracked so the dataset can be reproduced without committing the raw records.
