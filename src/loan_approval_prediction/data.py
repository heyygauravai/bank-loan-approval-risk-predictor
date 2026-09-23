"""Load and validate the loan approval training dataset."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

LOGGER = logging.getLogger(__name__)

TARGET_COLUMN = "Loan_Status"
ID_COLUMN = "Loan_ID"
SENSITIVE_COLUMNS = ("Gender", "Married")
NUMERIC_FEATURES = (
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
)
CATEGORICAL_FEATURES = (
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area",
)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = (ID_COLUMN, *SENSITIVE_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN)

ALLOWED_CATEGORIES = {
    "Gender": {"Male", "Female"},
    "Married": {"Yes", "No"},
    "Dependents": {"0", "1", "2", "3+"},
    "Education": {"Graduate", "Not Graduate"},
    "Self_Employed": {"Yes", "No"},
    "Property_Area": {"Urban", "Semiurban", "Rural"},
}


class DataValidationError(ValueError):
    """The input dataset does not meet the project's documented schema."""


def file_sha256(path: Path) -> str:
    """Return a checksum without loading the full file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a training frame and return a typed copy of its numeric columns."""
    if frame.empty:
        raise DataValidationError("The dataset is empty.")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")

    data = frame.copy()
    if (
        data[ID_COLUMN].isna().any()
        or data[ID_COLUMN].astype(str).str.strip().eq("").any()
        or data[ID_COLUMN].duplicated().any()
    ):
        raise DataValidationError("Loan_ID must be present and unique for every row.")

    targets = set(data[TARGET_COLUMN].dropna().unique())
    if data[TARGET_COLUMN].isna().any() or targets != {"Y", "N"}:
        raise DataValidationError("Loan_Status must contain only Y and N, with both classes present.")

    for column, allowed in ALLOWED_CATEGORIES.items():
        unexpected = set(data[column].dropna().astype(str).unique()) - allowed
        if unexpected:
            raise DataValidationError(f"Unexpected {column} values: {sorted(unexpected)}")

    for column in (*NUMERIC_FEATURES, "Credit_History"):
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"{column} must be numeric.") from exc
        present = data[column].dropna()
        if not np.isfinite(present.to_numpy(dtype=float)).all():
            raise DataValidationError(f"{column} contains a non-finite value.")

    for column in ("ApplicantIncome", "CoapplicantIncome"):
        if (data[column].dropna() < 0).any():
            raise DataValidationError(f"{column} cannot be negative.")
    for column in ("LoanAmount", "Loan_Amount_Term"):
        if (data[column].dropna() <= 0).any():
            raise DataValidationError(f"{column} must be positive when present.")
    if not set(data["Credit_History"].dropna().unique()).issubset({0, 1}):
        raise DataValidationError("Credit_History must contain only 0, 1, or missing values.")

    missing_cells = int(data.isna().sum().sum())
    LOGGER.info("Validated %d rows; %d missing cells", len(data), missing_cells)
    return data


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read a CSV and check the expected schema without modifying the raw file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    try:
        frame = pd.read_csv(source)
    except (OSError, UnicodeError, EmptyDataError, ParserError) as exc:
        raise DataValidationError(f"Could not read dataset: {source}") from exc
    LOGGER.info("Loaded dataset %s (%d rows)", source.name, len(frame))
    return validate_dataset(frame)
