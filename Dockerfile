FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Heart Disease Risk API" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.description="FastAPI inference service for the UCI Cleveland model"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/heart_disease_pipeline.joblib \
    MODEL_METADATA_PATH=/app/models/model_metadata.json \
    PREDICTION_THRESHOLD=0.5

WORKDIR /app

COPY requirements-api.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app app

COPY --chown=app:app src ./src
COPY --chown=app:app models ./models

USER app
EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "heart_disease_mlops.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
