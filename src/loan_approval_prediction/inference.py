"""Load the evaluated bundle and make a single prediction."""

from __future__ import annotations

import logging
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from loan_approval_prediction.data import FEATURE_COLUMNS
from loan_approval_prediction.training import DEFAULT_MODEL_PATH

LOGGER = logging.getLogger(__name__)


def load_bundle(path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load a trusted local artifact; never load an untrusted joblib file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Model not found: {source}. Run 'loan-approval train' first.")
    try:
        bundle = joblib.load(source)
    except Exception as exc:
        LOGGER.exception("Could not load model bundle at %s", source)
        raise ValueError(f"Could not load model bundle: {source}") from exc
    if not isinstance(bundle, dict) or not {"model", "threshold", "feature_columns"} <= bundle.keys():
        raise ValueError(f"Invalid model bundle: {source}")
    if bundle["feature_columns"] != list(FEATURE_COLUMNS):
        raise ValueError("Model feature schema does not match this application version.")
    threshold = bundle["threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not isfinite(float(threshold))
        or not 0 < threshold < 1
    ):
        raise ValueError("Model bundle has an invalid decision threshold.")
    if not callable(getattr(bundle["model"], "predict_proba", None)):
        raise TypeError("Model bundle does not contain a probability classifier.")
    return bundle


def predict_one(bundle: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    """Predict an approval label and calibrated probability from validated features."""
    expected = set(FEATURE_COLUMNS)
    if set(features) != expected:
        raise ValueError(f"Expected exactly these features: {sorted(expected)}")
    frame = pd.DataFrame([{name: features[name] for name in FEATURE_COLUMNS}])
    probability = float(bundle["model"].predict_proba(frame)[0, 1])
    threshold = float(bundle["threshold"])
    return {
        "prediction": "Y" if probability >= threshold else "N",
        "approval_probability": probability,
        "decision_threshold": threshold,
        "model": bundle.get("selected_model", "unknown"),
    }
