from __future__ import annotations

import pandas as pd
import pytest

from loan_approval_prediction.data import (
    DataValidationError,
    load_dataset,
    validate_dataset,
)


def test_valid_synthetic_dataset(synthetic_data):
    data = load_dataset(synthetic_data)
    assert len(data) == 80
    assert set(data["Loan_Status"]) == {"Y", "N"}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda frame: frame.drop(columns="Loan_Status"), "Missing required columns"),
        (lambda frame: frame.assign(Loan_ID="duplicate"), "Loan_ID must"),
        (lambda frame: frame.assign(Loan_Status="Y"), "both classes"),
        (lambda frame: frame.assign(Credit_History=2), "Credit_History"),
        (lambda frame: frame.assign(ApplicantIncome=-1), "ApplicantIncome"),
    ],
)
def test_invalid_dataset_is_rejected(synthetic_data, change, message):
    frame = pd.read_csv(synthetic_data)
    with pytest.raises(DataValidationError, match=message):
        validate_dataset(change(frame))
