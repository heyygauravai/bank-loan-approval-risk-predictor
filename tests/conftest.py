"""Synthetic data keeps CI independent of the private local CSV."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_data(tmp_path):
    rows = []
    for index in range(80):
        approved = index % 3 != 0
        rows.append(
            {
                "Loan_ID": f"SYN{index:04d}",
                "Gender": "Female" if index % 2 else "Male",
                "Married": "Yes" if index % 4 else "No",
                "Dependents": str(index % 3),
                "Education": "Graduate" if index % 5 else "Not Graduate",
                "Self_Employed": "No" if index % 6 else "Yes",
                "ApplicantIncome": 3000 + index * 41,
                "CoapplicantIncome": 0 if index % 2 else 1200,
                "LoanAmount": 100 + index % 45,
                "Loan_Amount_Term": 360 if index % 7 else 180,
                "Credit_History": 1 if approved else 0,
                "Property_Area": ["Urban", "Semiurban", "Rural"][index % 3],
                "Loan_Status": "Y" if approved else "N",
            }
        )
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
