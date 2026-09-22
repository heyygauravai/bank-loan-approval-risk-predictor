from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from loan_approval_prediction.api import create_app
from loan_approval_prediction.inference import load_bundle, predict_one
from loan_approval_prediction.training import train


@pytest.fixture(scope="module")
def example_features():
    return {
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1200,
        "LoanAmount": 140,
        "Loan_Amount_Term": 360,
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "Credit_History": 1,
        "Property_Area": "Urban",
    }


def test_training_bundle_and_api(synthetic_data, tmp_path, example_features):
    model_path = tmp_path / "model.joblib"
    report = train(synthetic_data, model_path, cv_folds=2, track_mlflow=False)
    assert report["dataset_rows"] == 80
    assert report["train_rows"] + report["test_rows"] == 80
    assert 0 <= report["test_metrics"]["balanced_accuracy"] <= 1
    assert report["excluded_columns"] == ["Loan_ID", "Gender", "Married"]
    assert json.loads(model_path.with_suffix(".report.json").read_text())["selected_model"]

    bundle = load_bundle(model_path)
    prediction = predict_one(bundle, example_features)
    assert prediction["prediction"] in {"Y", "N"}
    assert 0 <= prediction["approval_probability"] <= 1

    with TestClient(create_app(model_path)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post("/predict", json=example_features)
        assert response.status_code == 200
        assert response.json() == prediction
        assert client.post("/predict", json={**example_features, "ApplicantIncome": -1}).status_code == 422
        assert client.post("/predict", json={**example_features, "Gender": "Male"}).status_code == 422


def test_missing_bundle_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run 'loan-approval train'"):
        load_bundle(tmp_path / "missing.joblib")
