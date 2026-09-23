# Bank Loan Approval Predictor

[![CI](https://github.com/heyygauravai/bank-loan-approval-risk-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/heyygauravai/bank-loan-approval-risk-predictor/actions/workflows/ci.yml)

**[Try the live Streamlit app](https://bank-loan-approval-risk-predictor.streamlit.app/)** · [Model card](MODEL_CARD.md) · [Evaluation report](reports/evaluation.json) · [Dataset source](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset/data)

An end-to-end machine-learning project that predicts the **historical loan approval label** (`Y` or `N`) in the Kaggle Loan Prediction Problem dataset. It covers data checks, reproducible training, evaluation, experiment tracking, an API, a web demo, containers, and automated tests.

This is a portfolio learning project, **not a loan-default or repayment-risk model** and not a system for making real lending decisions.

## What I built

- A validated data-loading pipeline that checks the CSV schema, IDs, categories, numeric ranges, and unreadable files, while recording missingness before training.
- Leakage-safe preprocessing inside scikit-learn pipelines: median imputation and scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features.
- A reproducible comparison of logistic regression, random forest, and XGBoost using five-fold stratified cross-validation on the training split. The packaged workflow selected **random forest**.
- Sigmoid probability calibration and a decision threshold chosen from out-of-fold training predictions, followed by one evaluation on the held-out test split.
- A versioned model bundle, a machine-readable [evaluation report](reports/evaluation.json), and local MLflow experiment tracking.
- Three ways to make predictions: a command-line interface, a FastAPI service, and a Streamlit interface. The local Docker setup connects Streamlit to FastAPI.
- Logging and actionable error handling for bad datasets, invalid inputs, missing or corrupted model bundles, and prediction failures.
- Tests for data validation, training and inference, malformed bundles, API requests, and the standalone Streamlit prediction flow; GitHub Actions runs the tests, Ruff, and a Docker Compose configuration check.

## Results

The authoritative results come from the packaged workflow, not the exploratory notebook. The documented 614-row dataset was split into 491 training rows and 123 held-out test rows (stratified, seed 42). The selected random forest used a **0.42** approval threshold.

| Held-out metric | Result |
| --- | ---: |
| Balanced accuracy | 0.770 |
| Accuracy | 0.854 |
| ROC AUC | 0.780 |
| Approved (`Y`) recall | 0.988 |
| Rejected (`N`) recall | 0.553 |

The confusion matrix was TN 21, FP 17, FN 1, TP 84 (`Y` is the positive class). Rejected-case recall is a notable weakness despite the overall accuracy. See the [full report](reports/evaluation.json) and [model card](MODEL_CARD.md) for methodology, additional metrics, and limitations.

## How the pieces fit

The packaged path is: **Kaggle CSV → validation → preprocessing and model selection → calibration and threshold → held-out evaluation → model bundle and report → inference**.

The original [notebook](notebooks/Loan_default_prediction.ipynb) is preserved as a separately labeled exploration. It used extra features and selected XGBoost; it does **not** produce the model served by this project. The packaged training code and tracked evaluation report are the source of truth for the API and demo.

`Loan_ID`, `Gender`, and `Married` are excluded from model inputs. The training report includes subgroup error summaries for `Gender`, `Married`, and `Property_Area`, but these are not a fairness certification. Other features can still act as proxies.

### Local API and cloud demo

- **Locally and with Docker Compose:** Streamlit sends a request to FastAPI. FastAPI validates the input and calls the shared inference function with the trained model bundle.
- **On [Streamlit Community Cloud](https://bank-loan-approval-risk-predictor.streamlit.app/):** there is no separately hosted API. Streamlit calls that same inference function in-process. It downloads the pinned [v0.1.0 model release](https://github.com/heyygauravai/bank-loan-approval-risk-predictor/releases/tag/v0.1.0), verifies its SHA-256 before loading it, and caches it. This keeps the public demo usable without pretending that a local API is available on the internet.

The raw Kaggle CSV, generated local models, and MLflow files are Git-ignored. The model binary is published as a release asset rather than committed to Git; the evaluation report is committed.

## Run it locally

Requirements: **Python 3.12**, [uv](https://docs.astral.sh/uv/getting-started/installation/), and the training CSV from the [Kaggle dataset page](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset/data).

1. Clone the repository and place `train_u6lujuX_CVtuZ9i.csv` at `data/raw/train_u6lujuX_CVtuZ9i.csv`. Check its SHA-256 against [data/README.md](data/README.md); the raw file is not included in Git.
2. From the repository root, install the locked dependencies and train:

   ```powershell
   uv sync --locked
   uv run loan-approval train
   ```

3. Start FastAPI:

   ```powershell
   uv run uvicorn loan_approval_prediction.api:app --reload
   ```

4. In another PowerShell terminal, connect Streamlit to the local API:

   ```powershell
   $env:LOAN_API_URL = "http://localhost:8000/predict"
   uv run streamlit run streamlit_app/app.py
   ```

Open the Streamlit URL shown in the terminal. The local API has interactive documentation at [http://localhost:8000/docs](http://localhost:8000/docs) and a `GET /health` endpoint. For a direct API request:

```powershell
$body = @{ ApplicantIncome = 5000; CoapplicantIncome = 1200; LoanAmount = 140; Loan_Amount_Term = 360; Dependents = '0'; Education = 'Graduate'; Self_Employed = 'No'; Credit_History = 1; Property_Area = 'Urban' } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType application/json -Body $body
```

The CLI also supports `uv run loan-approval predict --input application.json` with those same nine feature keys. To inspect experiment runs, use `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`. To open the notebook in the same environment, use `uv run jupyter lab`.

If uv reports a hardlink error in a Windows OneDrive folder, set `$env:UV_LINK_MODE = "copy"` in each terminal before running the commands above.

### Docker Compose

After training locally, run `docker compose up --build`. The API is available at [http://localhost:8000](http://localhost:8000) and Streamlit at [http://localhost:8501](http://localhost:8501). Compose mounts the generated `models/` directory read-only into the API container and configures the UI to call that API. The raw dataset is not baked into the image.

## Check the project

```powershell
uv run pytest
uv run ruff check src tests streamlit_app
uv lock --check --offline
docker compose config --quiet
```

[GitHub Actions CI](https://github.com/heyygauravai/bank-loan-approval-risk-predictor/actions/workflows/ci.yml) runs the tests, lint, and Compose configuration check on pushes and pull requests to `main`. Its tests use synthetic data, so CI does not need the Kaggle CSV or the released model binary. Data and model-loading failures raise explicit errors, and the training/API paths log diagnostic information.

## Repository guide

| Path | Purpose |
| --- | --- |
| [data/README.md](data/README.md) | Dataset provenance, checksum, schema, and missingness |
| [notebooks/](notebooks/) | Original exploratory notebook |
| [src/loan_approval_prediction/](src/loan_approval_prediction/) | Validation, modeling, training, inference, CLI, and API |
| [streamlit_app/](streamlit_app/) | Web interface for local API or standalone cloud inference |
| [reports/evaluation.json](reports/evaluation.json) | Tracked packaged-model evaluation |
| [tests/](tests/) | Synthetic-data unit and integration tests |
| [MODEL_CARD.md](MODEL_CARD.md) | Detailed evaluation and responsible-use notes |
| [Dockerfile](Dockerfile) and [compose.yaml](compose.yaml) | Local container demo |

## Scope and limitations

This dataset is small and reflects historical approval decisions, which may contain bias. The test result comes from one split, with no external or temporal validation; its probabilities have not been independently validated for deployment. The source labels do not establish creditworthiness or repayment risk. `LoanAmount` is shown in the dataset's unconfirmed units. **Do not use this model to approve, reject, or rank real applicants.**

The Kaggle source marks its license as `Unknown`, so the raw CSV is documented but not redistributed. This project demonstrates an engineering workflow; it does not claim production readiness or include a separately deployed API, monitoring, or real-world lending controls.
