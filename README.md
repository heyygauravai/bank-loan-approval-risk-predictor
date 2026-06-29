# Bank loan default predictor
 
Predicts whether a loan applicant is likely to be **Approved or Rejected**, using classical ML. The models are deliberately optimized for **recall** (catching genuine defaulters) over raw accuracy, since a missed default costs a lender far more than a false alarm.
 
## Dataset
 
[Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) — 614 applications, ~70/30 class imbalance (Approved/Rejected). Credit History is the dominant predictor (80% approval with it, ~8% without).
 
## Approach
 
- **Preprocessing:** median/mode imputation for missing values, binary + one-hot encoding for categoricals, `StandardScaler` for feature scaling
- **Class imbalance — SMOTE** (Synthetic Minority Over-sampling): generates synthetic examples of the minority class (Rejected) so the model doesn't just learn to predict the majority class. Applied *inside* a pipeline (`imblearn.Pipeline`) so it's refit on each training fold only — preventing synthetic data from leaking into validation/test results.
- **Validation:** Stratified 5-fold cross-validation (CV) — the data is split 5 different ways and the model is tested on each held-out portion, giving a more reliable performance estimate than a single train/test split.
- **Tuning:** `GridSearchCV` scored explicitly on **recall** (% of actual defaulters correctly caught), not accuracy — since accuracy is misleading on imbalanced data.
- XGBoost was additionally calibrated for more reliable probabilities, with its decision threshold tuned via cross-validated predictions.
## Results
 
| Model | CV Recall | Test Recall | Test Precision | Test Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.565 | **0.68** | 0.70 | **0.81** |
| Random Forest | 0.507 | 0.66 | 0.71 | **0.81** |
| XGBoost (calibrated) | **0.591** | **0.68** | 0.68 | 0.80 |
 
**Key challenges per model:**
- **Logistic Regression** — sensitive to feature scale → fixed with `StandardScaler` in the pipeline
- **Random Forest** — lowest baseline recall, conservative on flagging Rejected → addressed with `class_weight='balanced'`, though tuning gains were marginal
- **XGBoost** — raw probabilities were unreliable for threshold decisions → fixed with `CalibratedClassifierCV` and a threshold tuned on out-of-fold CV predictions (not test data, to avoid leakage)
## Key Takeaways
 
- **No single model dominates** — all three converge to a similar recall ceiling (~0.50–0.59) under CV, suggesting the limit is the **dataset size**, not model choice.
- **Logistic Regression is the most business-friendly pick** — it matches XGBoost's test recall while being more stable across folds and fully interpretable (its coefficients directly show which factors drive a rejection, useful for explaining decisions to applicants).
## Tech Stack
 
`Python` · `Pandas` · `Scikit-learn` · `XGBoost` · `imbalanced-learn` · `Matplotlib` · `Seaborn`
 
## Run It
 
```bash
pip install -r requirements.txt
```
Open `Loan_default_prediction.ipynb` and run all cells (place the dataset CSV in the same directory, or `/content/` on Colab).
 
## Future Work
 
- Larger dataset to test if the recall ceiling is genuinely data-limited
- SHAP for instance-level explainability
- Streamlit app for interactive predictions
