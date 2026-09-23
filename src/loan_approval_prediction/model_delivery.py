"""Fetch the pinned model artifact used by the standalone Streamlit demo."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

MODEL_URL = (
    "https://github.com/heyygauravai/bank-loan-approval-risk-predictor/"
    "releases/download/v0.1.0/loan_approval.joblib"
)
MODEL_SHA256 = "7d9754726ff26b503531443ddf65ab576e3783ddd33c32d8f4f8b8d447ca8ecd"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_pinned_model(cache_dir: str | Path | None = None) -> Path:
    """Cache a trusted release asset, verifying its hash before deserialization."""
    directory = Path(cache_dir) if cache_dir is not None else Path(tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"loan-approval-{MODEL_SHA256[:12]}.joblib"
    if destination.is_file() and _sha256(destination) == MODEL_SHA256:
        return destination

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix="loan-approval-", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            with urlopen(MODEL_URL, timeout=30) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    temporary.write(chunk)
        if _sha256(temporary_path) != MODEL_SHA256:
            raise ValueError("Downloaded model checksum does not match the pinned release.")
        os.replace(temporary_path, destination)
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
