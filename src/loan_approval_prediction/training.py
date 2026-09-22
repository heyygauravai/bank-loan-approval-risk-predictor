"""Train, evaluate, and save one reproducible inference bundle."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

from loan_approval_prediction.data import (
    FEATURE_COLUMNS,
    SENSITIVE_COLUMNS,
    TARGET_COLUMN,
    file_sha256,
    load_dataset,
)
from loan_approval_prediction.modeling import candidate_models

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train_u6lujuX_CVtuZ9i.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "loan_approval.joblib"


def _choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    """Choose a threshold using only out-of-fold training predictions."""
    candidates = np.arange(0.20, 0.801, 0.02)
    scored = [
        (float(balanced_accuracy_score(y_true, probabilities >= value)), float(value))
        for value in candidates
    ]
    score, threshold = max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))
    return round(threshold, 2), score


def _test_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision_approved": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "recall_approved": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "f1_approved": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "precision_rejected": float(precision_score(y_true, predictions, pos_label=0, zero_division=0)),
        "recall_rejected": float(recall_score(y_true, predictions, pos_label=0, zero_division=0)),
        "f1_rejected": float(f1_score(y_true, predictions, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _subgroup_metrics(test: pd.DataFrame, predictions: np.ndarray) -> dict:
    """Describe errors by group without claiming a fairness certification."""
    result: dict[str, dict] = {}
    actual = (test[TARGET_COLUMN] == "Y").astype(int)
    for column in (*SENSITIVE_COLUMNS, "Property_Area"):
        groups = {}
        for group_value, rows in test.groupby(column, dropna=False).groups.items():
            label = "Missing" if pd.isna(group_value) else str(group_value)
            y_group = actual.loc[rows].to_numpy()
            pred_group = predictions[test.index.get_indexer(rows)]
            positives = int(y_group.sum())
            negatives = int(len(y_group) - positives)
            groups[label] = {
                "count": len(y_group),
                "predicted_approval_rate": float(pred_group.mean()),
                "recall_approved": (
                    float(pred_group[y_group == 1].mean()) if positives else None
                ),
                "false_approval_rate": (
                    float(pred_group[y_group == 0].mean()) if negatives else None
                ),
            }
        result[column] = groups
    return result


def _log_mlflow(report: dict, model_path: Path, report_path: Path, tracking_uri: str | None) -> str:
    import mlflow

    uri = (
        tracking_uri
        or os.getenv("MLFLOW_TRACKING_URI")
        or f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("loan-approval")
    with mlflow.start_run(run_name=report["selected_model"]) as run:
        mlflow.log_params(
            {
                "selected_model": report["selected_model"],
                "random_state": report["random_state"],
                "cv_folds": report["cv_folds"],
                "test_size": report["test_size"],
                "threshold": report["threshold"],
                "dataset_sha256": report["dataset_sha256"],
            }
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in report["test_metrics"].items() if isinstance(value, float)})
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(report_path), artifact_path="model")
        return run.info.run_id


def train(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    random_state: int = 42,
    test_size: float = 0.20,
    cv_folds: int = 5,
    track_mlflow: bool = True,
    tracking_uri: str | None = None,
) -> dict:
    """Select a model by CV, tune a threshold out of fold, and test once."""
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be between 0 and 0.5.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

    source = Path(data_path)
    output = Path(model_path)
    data = load_dataset(source)
    training, test = train_test_split(
        data,
        test_size=test_size,
        stratify=data[TARGET_COLUMN],
        random_state=random_state,
    )
    y_train = (training[TARGET_COLUMN] == "Y").astype(int)
    y_test = (test[TARGET_COLUMN] == "Y").astype(int)
    if y_train.value_counts().min() < cv_folds:
        raise ValueError("Each training class needs at least cv_folds rows.")

    x_train = training.loc[:, list(FEATURE_COLUMNS)]
    x_test = test.loc[:, list(FEATURE_COLUMNS)]
    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    candidates = candidate_models(random_state)
    cv_results = {}
    for name, pipeline in candidates.items():
        scores = cross_val_score(
            pipeline, x_train, y_train, scoring="balanced_accuracy", cv=folds, n_jobs=1,
        )
        cv_results[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
        LOGGER.info("%s CV balanced accuracy: %.3f ± %.3f", name, scores.mean(), scores.std())

    selected_name = max(candidates, key=lambda name: cv_results[name]["mean"])
    calibrated = CalibratedClassifierCV(
        estimator=clone(candidates[selected_name]), method="sigmoid", cv=3,
    )
    oof_probabilities = cross_val_predict(
        calibrated, x_train, y_train, cv=folds, method="predict_proba", n_jobs=1,
    )[:, 1]
    threshold, oof_score = _choose_threshold(y_train, oof_probabilities)
    calibrated.fit(x_train, y_train)

    test_probabilities = calibrated.predict_proba(x_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)
    metrics = _test_metrics(y_test, test_probabilities, threshold)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_sha256": file_sha256(source),
        "dataset_rows": len(data),
        "train_rows": len(training),
        "test_rows": len(test),
        "random_state": random_state,
        "test_size": test_size,
        "cv_folds": cv_folds,
        "selection_metric": "balanced_accuracy",
        "cv_results": cv_results,
        "selected_model": selected_name,
        "threshold": threshold,
        "threshold_oof_balanced_accuracy": oof_score,
        "test_metrics": metrics,
        "subgroup_metrics": _subgroup_metrics(test, test_predictions),
        "feature_columns": list(FEATURE_COLUMNS),
        "excluded_columns": ["Loan_ID", *SENSITIVE_COLUMNS],
    }
    bundle = {
        "model": calibrated,
        "threshold": threshold,
        "feature_columns": list(FEATURE_COLUMNS),
        "selected_model": selected_name,
        "dataset_sha256": report["dataset_sha256"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix(".report.json")
    joblib.dump(bundle, output)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    LOGGER.info("Saved evaluated model and report to %s", output.parent)

    if track_mlflow:
        report["mlflow_run_id"] = _log_mlflow(report, output, report_path, tracking_uri)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
