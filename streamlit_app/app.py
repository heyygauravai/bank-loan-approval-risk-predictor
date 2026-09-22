"""Streamlit interface for the loan approval prediction API."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

API_URL = os.getenv("LOAN_API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Loan approval predictor", page_icon="📋")
st.title("Loan approval predictor")
st.caption("Historical approval prediction—not repayment risk or a lending decision.")

with st.form("loan_features"):
    applicant_income = st.number_input("Applicant income", min_value=0.0, value=5000.0)
    coapplicant_income = st.number_input("Coapplicant income", min_value=0.0, value=0.0)
    loan_amount = st.number_input("Loan amount (dataset units)", min_value=1.0, value=150.0)
    loan_term = st.number_input("Loan term (dataset units)", min_value=1.0, value=360.0)
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self employed", ["No", "Yes"])
    credit_history = st.selectbox("Credit history in dataset", [1, 0])
    property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    payload = {
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            result = json.load(response)
    except HTTPError as exc:
        st.error(f"API rejected the input (HTTP {exc.code}). Check the API logs for details.")
    except (URLError, TimeoutError, ValueError) as exc:
        st.error(f"Could not get a prediction from the API: {exc}")
    else:
        label = "historically approved" if result["prediction"] == "Y" else "historically rejected"
        st.metric("Model prediction", label)
        st.write(f"Estimated approval-label probability: {result['approval_probability']:.1%}")
        st.caption(
            f"Decision threshold: {result['decision_threshold']:.2f} · "
            f"Selected model: {result['model']}"
        )
