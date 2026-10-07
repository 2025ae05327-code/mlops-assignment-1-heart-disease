"""FastAPI service for heart disease risk inference and operational metrics."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from heart_disease_mlops.config import (
    FEATURE_COLUMNS,
    MODEL_METADATA_PATH,
    configured_model_path,
    prediction_threshold,
)
from heart_disease_mlops.schemas import HealthResponse, PatientFeatures, PredictionResponse


class JsonFormatter(logging.Formatter):
    """Emit one parseable JSON object per request log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        for key in (
            "request_id",
            "method",
            "path",
            "status_code",
            "duration_ms",
            "prediction",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("heart_disease_api")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    return logger


LOGGER = configure_logging()
REQUEST_COUNT = Counter(
    "heart_api_requests_total",
    "Number of API requests",
    labelnames=("method", "path", "status"),
)
REQUEST_LATENCY = Histogram(
    "heart_api_request_duration_seconds",
    "API request duration in seconds",
    labelnames=("method", "path"),
)
PREDICTION_COUNT = Counter(
    "heart_api_predictions_total",
    "Number of model predictions",
    labelnames=("prediction",),
)


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    version: str


@lru_cache(maxsize=1)
def load_model_bundle() -> ModelBundle:
    """Load and validate the model once per process."""

    model_path = configured_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    model = joblib.load(model_path)
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("Packaged model must implement predict_proba")

    metadata_path = Path(os.getenv("MODEL_METADATA_PATH", str(MODEL_METADATA_PATH)))
    version = "unversioned"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_name = metadata.get("selected_model", "model")
        run_id = metadata.get("selected_run_id", "local")
        version = f"{model_name}:{run_id}"
    return ModelBundle(model=model, version=version)


app = FastAPI(
    title="Heart Disease Risk API",
    version="1.0.0",
    description="Cloud-ready inference for the UCI Cleveland heart disease classifier.",
)


@app.middleware("http")
async def observe_request(request: Request, call_next: Any) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        LOGGER.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
            },
        )
        raise
    finally:
        duration_seconds = time.perf_counter() - started
        REQUEST_COUNT.labels(request.method, request.url.path, str(status_code)).inc()
        REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration_seconds)
        LOGGER.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(duration_seconds * 1000, 3),
            },
        )


@app.get("/", tags=["service"])
def service_info() -> dict[str, Any]:
    return {
        "service": "heart-disease-risk-api",
        "version": app.version,
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    try:
        bundle = load_model_bundle()
    except (FileNotFoundError, TypeError, OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return HealthResponse(status="ok", model_loaded=True, model_version=bundle.version)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: PatientFeatures, request: Request) -> PredictionResponse:
    try:
        bundle = load_model_bundle()
    except (FileNotFoundError, TypeError, OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    frame = pd.DataFrame([payload.model_dump()], columns=FEATURE_COLUMNS)
    disease_probability = float(bundle.model.predict_proba(frame)[0, 1])
    threshold = prediction_threshold()
    prediction = int(disease_probability >= threshold)
    confidence = disease_probability if prediction == 1 else 1 - disease_probability
    PREDICTION_COUNT.labels(str(prediction)).inc()
    LOGGER.info(
        "prediction_completed",
        extra={"request_id": request.state.request_id, "prediction": prediction},
    )
    return PredictionResponse(
        prediction=prediction,
        risk_label="higher_risk" if prediction else "lower_risk",
        confidence=confidence,
        disease_probability=disease_probability,
        decision_threshold=threshold,
        model_version=bundle.version,
        request_id=request.state.request_id,
    )


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
