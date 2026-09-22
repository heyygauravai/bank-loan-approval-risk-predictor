# Loan Approval Prediction

[![CI](https://github.com/heyygauravai/bank-loan-approval-risk-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/heyygauravai/bank-loan-approval-risk-predictor/actions/workflows/ci.yml)

A compact, end-to-end MLOps learning project built from the [Kaggle Loan Prediction Problem training dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset/data). It validates data, trains and evaluates a model, tracks experiments, serves predictions through an API, and includes a small Streamlit demo. The original exploratory notebook remains in `notebooks/`.

The target is **historical loan approval** (`Loan_Status`: `Y` or `N`), **not loan default or repayment**. This is a portfolio demonstration, not a real lending or financial-advice system.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12. From the repository root:

```powershell
uv sync --locked --link-mode copy
uv run --link-mode copy loan-approval train
uv run --link-mode copy uvicorn loan_approval_prediction.api:app --reload
```

Get the training CSV from the Kaggle page and put it at `data/raw/train_u6lujuX_CVtuZ9i.csv` before training. Verify its SHA-256 against [data/README.md](data/README.md). The CSV is intentionally Git-ignored because the source's license is marked `Unknown`. Do not commit or redistribute it without permission.

The `--link-mode copy` option avoids the OneDrive hardlink error seen on this Windows workspace. On a normal filesystem, plain `uv sync --locked` and `uv run` also work. The notebook's dependencies live in the main dependency set, so `uv run --link-mode copy jupyter lab` opens `notebooks/Loan_default_prediction.ipynb` in the same environment.

## Workflow

The notebook is an independent exploration: it uses additional features and selects XGBoost. It is **not** the API's training or artifact-export path. The packaged pipeline below is authoritative for the served model.

1. `loan-approval train` validates the 614-row CSV, then makes a stratified 80/20 train/test split (seed 42).
2. Missing-value imputation, encoding, and scaling live inside each scikit-learn pipeline, so cross-validation fits them only on training folds. The candidates are logistic regression, random forest, and XGBoost. Five-fold balanced accuracy selects a candidate.
3. Sigmoid calibration and an approval threshold are fitted/selected using only the training portion. The held-out test set is evaluated once. The training command saves `models/loan_approval.joblib` and `models/loan_approval.report.json`, plus an MLflow run in the local ignored `mlflow.db`/`mlruns` storage.
4. The API loads the saved bundle at startup. `GET /health` checks readiness; `POST /predict` validates one application and returns the predicted historical label, estimated probability, threshold, and model name.

`Loan_ID`, `Gender`, and `Married` are not model inputs. The report retains subgroup error summaries for `Gender`, `Married`, and `Property_Area` to make limitations visible; this is **not** a fairness certification. Removing two attributes does not prevent proxy effects through other features.

## Results and limitations

The tracked [machine-readable evaluation report](reports/evaluation.json) records the packaged model's dataset checksum, split, selection scores, threshold, and test metrics. The full run report is regenerated locally by `loan-approval train`.

The reproducible local run on the documented CSV selected random forest. On 123 held-out rows it reached **0.770 balanced accuracy**, **0.854 accuracy**, and **0.780 ROC AUC**. Its confusion matrix was TN 21, FP 17, FN 1, TP 84, where `Y` is the positive class. Rejected-case recall was only **0.553** (21 of 38). See [MODEL_CARD.md](MODEL_CARD.md) for the full methodology and cautions.

These numbers are from one small split, with no external or temporal validation. The source labels may encode past policy or bias; no result here establishes creditworthiness, repayment risk, or suitability for actual applicants. Do not use the model to approve, reject, or rank real people.

## Use the API and demo

Start the API after training, then in another terminal run:

```powershell
uv run --link-mode copy streamlit run streamlit_app/app.py
```

Open the Streamlit URL shown in the terminal. API documentation is at `http://localhost:8000/docs`. Example request:

```powershell
$body = @{ ApplicantIncome = 5000; CoapplicantIncome = 1200; LoanAmount = 140; Loan_Amount_Term = 360; Dependents = '0'; Education = 'Graduate'; Self_Employed = 'No'; Credit_History = 1; Property_Area = 'Urban' } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType application/json -Body $body
```

`LoanAmount` is in the dataset's unconfirmed units. The model predicts the probability of a recorded `Y` label, not a real-world chance of receiving approval. The local CLI also accepts a JSON object with the same nine keys: `uv run --link-mode copy loan-approval predict --input application.json`.

To inspect training runs, execute `uv run --link-mode copy mlflow ui --backend-store-uri sqlite:///mlflow.db` from the project root. Generated models, data, logs, and tracking files are ignored by Git. Only load model bundles that you trained or otherwise trust: joblib files can execute code when deserialized.

## Container demo

Train locally first, then run `docker compose up --build`. Compose starts the API at `http://localhost:8000` and Streamlit at `http://localhost:8501`; the generated local `models/` directory is mounted read-only into the API container. Containers are for a local demo, not a production deployment. The raw dataset and training environment are not included in the image.

## Checks

```powershell
uv run --link-mode copy pytest
uv run --link-mode copy ruff check src tests streamlit_app
uv lock --check --offline
```

CI runs tests on synthetic data (no Kaggle file or trained artifact required) and lint on pushes and pull requests to `main`. The tests cover schema validation, training and saved-bundle inference, API health, prediction, and invalid requests.

## Project map

| Path | Purpose |
| --- | --- |
| `data/README.md` | Dataset source, checksum, schema, missingness |
| `notebooks/` | Original exploratory notebook |
| `reports/evaluation.json` | Tracked packaged-model evaluation |
| `src/loan_approval_prediction/` | Data validation, modeling, training, inference, CLI, API |
| `streamlit_app/` | Demo UI, calling the API |
| `tests/` | Synthetic-data unit and integration checks |
| `MODEL_CARD.md` | Evaluation and responsible-use notes |
| `Dockerfile`, `compose.yaml` | Local container demo |
| `.github/workflows/ci.yml` | Tests and lint in CI |

This project deliberately leaves out orchestration, cloud deployment, and monitoring infrastructure. The focus is a clear, reproducible learning workflow rather than a claim of production readiness.
