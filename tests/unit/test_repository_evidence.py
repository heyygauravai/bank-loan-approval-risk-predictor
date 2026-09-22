"""Keep public project evidence internally consistent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loan_approval_prediction.data import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]


def test_tracked_evaluation_is_self_consistent():
    report = json.loads((ROOT / "reports" / "evaluation.json").read_text(encoding="utf-8"))
    matrix = report["test_metrics"]["confusion_matrix"]
    total = sum(matrix.values())
    assert total == report["test_rows"]
    assert report["dataset_rows"] == report["train_rows"] + report["test_rows"]
    assert report["test_metrics"]["accuracy"] == pytest.approx(
        (matrix["tn"] + matrix["tp"]) / total
    )
    assert report["test_metrics"]["balanced_accuracy"] == pytest.approx(
        (matrix["tn"] / (matrix["tn"] + matrix["fp"]) + matrix["tp"] / (matrix["tp"] + matrix["fn"])) / 2
    )
    assert report["selected_model"] == max(
        report["cv_results"], key=lambda name: report["cv_results"][name]["mean"]
    )
    assert report["feature_columns"] == list(FEATURE_COLUMNS)
    assert len(report["dataset_sha256"]) == 64


def test_notebook_is_labelled_as_exploration_not_artifact_source():
    notebook = json.loads(
        (ROOT / "notebooks" / "Loan_default_prediction.ipynb").read_text(encoding="utf-8")
    )
    introduction = "".join(notebook["cells"][0]["source"])
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "Exploratory notebook" in introduction
    assert "authoritative" in introduction
    assert "joblib.dump" not in code
    assert "loan_approval_model.pkl" not in code
    assert "best_threshold.pkl" not in code
