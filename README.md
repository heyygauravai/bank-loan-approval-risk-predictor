# Loan Approval Prediction

## Project Status

This project is currently in the foundation and repository setup stage.

## Problem Statement

<!-- Explain:
- What will the system predict?
- What information will it use?
- Why could the prediction be useful?
-->

## Target Variable

<!-- Explain:
- The target column name
- What Y and N mean
- What numeric values 1 and 0 will represent
-->

## Intended Use

<!-- Explain:
- Who could use the result?
- Is it educational, advisory, or an automated decision?
- State clearly that this model must not make real lending decisions by itself.
-->

## Out of Scope

<!-- Explain what the project does NOT do.
Important: the dataset contains approval outcomes, not post-loan defaults.
-->

## Error Costs

### False Approval

<!-- What does it mean if the model predicts approval but the historical label is rejection? -->

### False Rejection

<!-- What does it mean if the model predicts rejection but the historical label is approval? -->

### Evaluation Decision

<!-- Do not select a primary metric yet.
State that it will be chosen after comparing the consequences of both errors.
-->

## Dataset

- **Source:** TODO
- **Original URL:** TODO
- **License/usage terms:** TODO
- **Target column:** `Loan_Status`
- **Raw data location:** `data/raw/`

<!-- Add:
- What one row represents
- Approximate number of rows and columns
- Whether the data will be committed to Git
- How another developer can obtain it
-->

## Ethical Considerations

<!-- Discuss:
- Gender and marital status as sensitive attributes
- Potential historical bias
- Subgroup evaluation
- Human review
- Why model output should not be the sole basis for lending decisions
-->

## Local Setup

### Requirements

- `uv`
- Python 3.12, managed through `uv`

### Installation

```powershell
git clone <repository-url>
Set-Location loan-approval-prediction
uv sync --locked