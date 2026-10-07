from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from heart_disease_mlops.api import app, load_model_bundle


@pytest.fixture
def client(tmp_path: Path, trained_model, monkeypatch: pytest.MonkeyPatch):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump(trained_model, model_path)
    metadata_path.write_text(
        json.dumps({"selected_model": "logistic_regression", "selected_run_id": "test-run"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_METADATA_PATH", str(metadata_path))
    load_model_bundle.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    load_model_bundle.cache_clear()


def test_health_and_prediction(client: TestClient, valid_payload: dict[str, int | float]) -> None:
    health = client.get("/health")
    response = client.post("/predict", json=valid_payload, headers={"X-Request-ID": "case-42"})

    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "case-42"
    assert response.json()["prediction"] in (0, 1)
    assert 0 <= response.json()["disease_probability"] <= 1


def test_invalid_payload_is_rejected(
    client: TestClient, valid_payload: dict[str, int | float]
) -> None:
    valid_payload["age"] = 999
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_prometheus_metrics(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "heart_api_requests_total" in response.text


def test_health_reports_missing_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    load_model_bundle.cache_clear()
    with TestClient(app) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 503
    load_model_bundle.cache_clear()
