# 🏦 Loan Approval Prediction System

An end-to-end Machine Learning project that predicts whether a loan application is likely to be **Approved** or **Rejected**. The project focuses on solving a real-world **imbalanced classification** problem where identifying risky applicants (Rejected) is more valuable than maximizing overall accuracy.

🚀 **Live Demo:** https://loan-default-prediction-by-gaurav.streamlit.app/

## Application Preview

### Home Page

[Home](screenshots/home.png)

### Prediction Result

[Prediction](screenshots/prediction.png)

## 📌 Project Overview

Financial institutions often face class imbalance when evaluating loan applications, where approved loans significantly outnumber rejected ones. Optimizing only for accuracy can produce misleading models that rarely identify risky applicants.

This project builds and compares multiple machine learning models while prioritizing **recall for the Rejected class**, ensuring that potentially risky loan applications are detected as effectively as possible.

---

## 📊 Dataset

**Dataset:** Loan Prediction Problem Dataset

https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

* **614 loan applications**
* Approximately **70:30** class imbalance (Approved vs Rejected)
* Combination of numerical and categorical features
* Missing values requiring preprocessing

---

## ⚙️ Machine Learning Pipeline

The complete workflow includes:

* Missing value imputation (Median & Mode)
* Binary encoding for binary categorical variables
* One-Hot Encoding (`drop_first=True`) for multi-class categorical variables
* Feature Scaling using `StandardScaler`
* Handling class imbalance using **SMOTE**
* `imblearn.Pipeline` to prevent data leakage
* Stratified 5-Fold Cross Validation
* Hyperparameter tuning using `GridSearchCV`
* Threshold Optimization
* Probability Calibration (XGBoost)
* Feature Contribution Visualization
* Interactive deployment using Streamlit

---

# 🚧 Challenges & How They Were Solved

### 1. Class Imbalance

**Problem**

The dataset contains considerably more approved loans than rejected ones, causing models to favor the majority class.

**Solution**

Implemented **SMOTE** inside an `imblearn.Pipeline` so synthetic samples were generated independently within every training fold, preventing information leakage into validation data.

---

### 2. Data Leakage

**Problem**

Applying preprocessing before cross-validation can leak information from validation folds into training.

**Solution**

Integrated preprocessing, SMOTE and the classifier into a single pipeline so every fold performs preprocessing independently.

---

### 3. Reliable Model Evaluation

**Problem**

A single train-test split can produce unstable performance estimates.

**Solution**

Used **Stratified 5-Fold Cross Validation**, ensuring every fold preserved the original class distribution.

---

### 4. Hyperparameter Optimization

**Problem**

Default parameters rarely provide optimal performance.

**Solution**

Applied **GridSearchCV** separately to Logistic Regression, Random Forest and XGBoost using recall as the optimization metric.

---

### 5. Decision Threshold Optimization

**Problem**

The default classification threshold (0.5) was not optimal for maximizing recall on rejected applicants.

**Solution**

Performed threshold tuning using **cross-validated predicted probabilities**, avoiding any use of the test set and preventing leakage.

---

### 6. Poor Probability Calibration (XGBoost)

**Problem**

Although XGBoost produced strong predictions, its probability estimates were not sufficiently reliable for threshold optimization.

**Solution**

Applied **CalibratedClassifierCV (Sigmoid/Platt Scaling)** before threshold tuning to improve probability calibration.

---

### 7. Model Explainability

**Problem**

Predictions alone provide little insight into why a loan was approved or rejected.

**Solution**

Built an interactive Streamlit dashboard that displays the contribution of every feature to the current prediction using Logistic Regression coefficients.

---

## 📈 Model Performance

| Model                | CV Recall | Test Recall | Test Precision | Test Accuracy |
| -------------------- | --------: | ----------: | -------------: | ------------: |
| Logistic Regression  |     0.565 |    **0.68** |           0.70 |      **0.81** |
| Random Forest        |     0.507 |        0.66 |           0.71 |      **0.81** |
| XGBoost (Calibrated) | **0.591** |    **0.68** |           0.68 |          0.80 |

---

## 💡 Key Findings

* Logistic Regression achieved the most balanced overall performance.
* SMOTE significantly improved minority class detection.
* Threshold tuning increased business usefulness by allowing the classifier to prioritize recall over default decision boundaries.
* Calibrating XGBoost probabilities improved threshold selection.
* Credit History remained the strongest predictor across all models.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit application where users can:

* Enter applicant information
* Predict loan approval in real time
* View approval probability
* View rejection probability
* See feature contribution analysis explaining the prediction

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* imbalanced-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

---

## 🚀 Running Locally

Install dependencies

```bash
pip install -r requirements.txt
```

Launch the application

```bash
streamlit run app.py
```

---

## 📂 Repository Structure

```text
├── app.py
├── Loan_Approval.ipynb
├── loan_approval_model.pkl
├── best_threshold.pkl
├── requirements.txt
├── README.md
```

---

## 🔮 Future Improvements

* SHAP-based model explanations
* Automated hyperparameter optimization with Optuna
* Additional ensemble models
* Larger and more balanced dataset
* Cloud deployment with continuous integration

---

## 👨‍💻 Author

**Gaurav Chaudhary**

AI / Machine Learning Enthusiast
