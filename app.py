from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Loan Approval Outcome Prediction",
    page_icon="🏦",
    layout="centered",
)

APPROVAL_LABEL = 1
RAW_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

PROJECT_ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "loan_approval_bundle.joblib"
MODEL_PATH = PROJECT_ROOT / "loan_approval_model.pkl"
THRESHOLD_PATH = PROJECT_ROOT / "best_threshold.pkl"


@st.cache_resource
def load_artifacts():
    if BUNDLE_PATH.exists():
        bundle = joblib.load(BUNDLE_PATH)
        required_keys = {
            "model",
            "approval_threshold",
            "selected_model",
            "feature_columns",
        }
        missing_keys = required_keys - set(bundle)
        if missing_keys:
            raise KeyError(f"Artifact bundle is missing: {sorted(missing_keys)}")
        return bundle, False

    if MODEL_PATH.exists() and THRESHOLD_PATH.exists():
        model = joblib.load(MODEL_PATH)
        threshold = float(joblib.load(THRESHOLD_PATH))
        feature_columns = list(getattr(model, "feature_names_in_", RAW_FEATURES))
        bundle = {
            "model": model,
            "approval_threshold": threshold,
            "selected_model": type(model).__name__,
            "feature_columns": feature_columns,
        }
        return bundle, set(feature_columns) != set(RAW_FEATURES)

    raise FileNotFoundError(
        "Model artifacts were not found. Run the notebook through the export section first."
    )


def legacy_encoded_frame(raw_frame, expected_columns):
    """Support artifacts created by the project's earlier manual encoding flow."""
    row = raw_frame.iloc[0]
    encoded = {
        "Gender": int(row["Gender"] == "Male"),
        "Married": int(row["Married"] == "Yes"),
        "Education": int(row["Education"] == "Graduate"),
        "Self_Employed": int(row["Self_Employed"] == "Yes"),
        "ApplicantIncome": row["ApplicantIncome"],
        "CoapplicantIncome": row["CoapplicantIncome"],
        "LoanAmount": row["LoanAmount"],
        "Loan_Amount_Term": row["Loan_Amount_Term"],
        "Credit_History": row["Credit_History"],
        "Dependents_1": int(row["Dependents"] == "1"),
        "Dependents_2": int(row["Dependents"] == "2"),
        "Dependents_3+": int(row["Dependents"] == "3+"),
        "Property_Area_Semiurban": int(row["Property_Area"] == "Semiurban"),
        "Property_Area_Urban": int(row["Property_Area"] == "Urban"),
    }
    return pd.DataFrame([encoded]).reindex(columns=expected_columns, fill_value=0)


try:
    artifacts, uses_legacy_schema = load_artifacts()
except (FileNotFoundError, KeyError, ValueError) as error:
    st.error(str(error))
    st.info("Open the notebook, choose **Runtime → Run all**, then copy the exported artifacts here.")
    st.stop()

model = artifacts["model"]
threshold = float(artifacts["approval_threshold"])
selected_model = artifacts["selected_model"]
expected_columns = artifacts["feature_columns"]

st.title("🏦 Loan Approval Outcome Prediction")
st.markdown(
    "Estimate whether an application resembles historically **approved** or "
    "**rejected** applications in the training data."
)
st.warning(
    "Educational prototype only. This predicts a historical approval outcome—not "
    "repayment, loss, or default risk—and must not be used for lending decisions."
)

with st.sidebar:
    st.header("Model information")
    st.write(f"**Selected model:** {selected_model}")
    st.write(f"**Approval threshold:** {threshold:.2f}")
    st.write("**Preprocessing:** imputation, categorical encoding, scaling, and SMOTENC")
    if uses_legacy_schema:
        st.caption("Using the earlier artifact schema. Re-run the notebook to export the new bundle.")

st.header("Applicant information")

left, right = st.columns(2)
with left:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self employed", ["No", "Yes"])
    property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])

with right:
    applicant_income = st.number_input("Applicant income", min_value=0.0, value=5000.0)
    coapplicant_income = st.number_input("Coapplicant income", min_value=0.0, value=0.0)
    loan_amount = st.number_input("Loan amount (thousands)", min_value=0.0, value=120.0)
    loan_term = st.selectbox(
        "Loan amount term",
        [12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0],
        index=8,
    )
    credit_history = st.selectbox("Credit history", [1.0, 0.0])

raw_input = pd.DataFrame([{
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
}])

if st.button("Predict historical outcome", type="primary", use_container_width=True):
    model_input = (
        legacy_encoded_frame(raw_input, expected_columns)
        if uses_legacy_schema
        else raw_input.reindex(columns=expected_columns)
    )

    probabilities = model.predict_proba(model_input)[0]
    classes = np.asarray(model.classes_)
    approval_matches = np.flatnonzero(classes == APPROVAL_LABEL)
    if approval_matches.size != 1:
        st.error(f"Expected class label 1, but the model exposes classes {classes.tolist()}.")
        st.stop()

    approval_probability = float(probabilities[approval_matches[0]])
    rejection_probability = 1.0 - approval_probability
    approved = approval_probability >= threshold

    st.subheader("Prediction result")
    if approved:
        st.success("Model outcome: historically approved")
    else:
        st.error("Model outcome: historically rejected")

    metric_left, metric_right = st.columns(2)
    metric_left.metric("Approval probability", f"{approval_probability:.1%}")
    metric_right.metric("Rejection probability", f"{rejection_probability:.1%}")

    st.caption(
        "These are model probabilities calibrated on the training data. They are not "
        "guarantees and should not be interpreted as default probabilities."
    )
