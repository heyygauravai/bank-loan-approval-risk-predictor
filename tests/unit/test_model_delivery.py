"""The hosted demo only loads the expected release artifact."""

from __future__ import annotations

import hashlib
from io import BytesIO

import pytest

from loan_approval_prediction import model_delivery


def test_download_is_verified_and_cached(monkeypatch, tmp_path):
    content = b"trusted test artifact"
    monkeypatch.setattr(model_delivery, "MODEL_SHA256", hashlib.sha256(content).hexdigest())
    downloads = []

    def open_model(url, timeout):
        downloads.append((url, timeout))
        return BytesIO(content)

    monkeypatch.setattr(model_delivery, "urlopen", open_model)
    path = model_delivery.fetch_pinned_model(tmp_path)
    assert path.read_bytes() == content
    assert model_delivery.fetch_pinned_model(tmp_path) == path
    assert len(downloads) == 1


def test_wrong_download_checksum_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(model_delivery, "MODEL_SHA256", hashlib.sha256(b"expected").hexdigest())
    monkeypatch.setattr(model_delivery, "urlopen", lambda *_args, **_kwargs: BytesIO(b"other"))
    with pytest.raises(ValueError, match="checksum"):
        model_delivery.fetch_pinned_model(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_corrupted_cached_file_is_replaced(monkeypatch, tmp_path):
    content = b"trusted test artifact"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(model_delivery, "MODEL_SHA256", digest)
    path = tmp_path / f"loan-approval-{digest[:12]}.joblib"
    path.write_bytes(b"corrupted")
    monkeypatch.setattr(model_delivery, "urlopen", lambda *_args, **_kwargs: BytesIO(content))
    assert model_delivery.fetch_pinned_model(tmp_path) == path
    assert path.read_bytes() == content
