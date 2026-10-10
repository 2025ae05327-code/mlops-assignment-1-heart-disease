# Architecture and Design Decisions

## System Context

```mermaid
flowchart LR
    UCI[UCI Cleveland dataset] --> DQ[Acquisition and quality checks]
    DQ --> CSV[Clean CSV and SHA-256 lineage]
    CSV --> EDA[EDA artifacts]
    CSV --> CV[Stratified CV and grid search]
    CV --> LR[Logistic Regression]
    CV --> RF[Random Forest]
    LR --> MLF[MLflow tracking]
    RF --> MLF
    MLF --> SEL[Select by mean CV ROC-AUC]
    SEL --> PKG[Joblib preprocessing and model pipeline]
    PKG --> API[FastAPI inference]
    API --> IMG[Non-root Docker image]
    IMG --> K8S[Docker Compose or Kubernetes]
    K8S --> PROM[Prometheus]
    PROM --> GRAF[Grafana dashboard]
    API --> LOG[Structured request logs]
    GH[GitHub Actions] --> DQ
    GH --> CV
    GH --> IMG
```

## Training Flow

1. The downloader retries requests, writes atomically, and caches the UCI source.
2. Cleaning maps missing tokens, checks valid domains, removes duplicates, and binarizes the target.
3. A stratified split reserves 20% as an untouched test set.
4. Imputation, scaling, and encoding are fitted inside each cross-validation fold.
5. Logistic Regression and Random Forest hyperparameters are searched under the same folds.
6. Mean CV ROC-AUC selects the model; the test set estimates final generalization.
7. MLflow stores parameters, metrics, variation, plots, signatures, and candidate models.
8. The selected full pipeline and metadata are written atomically for serving.

## Serving Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Schema as Pydantic
    participant Model as Joblib Pipeline
    participant Metrics as Prometheus
    Client->>API: POST /predict plus JSON
    API->>Schema: Validate 13 fields and domains
    Schema-->>API: Ordered typed features
    API->>Model: predict_proba(DataFrame)
    Model-->>API: Disease probability
    API->>Metrics: Increment outcome and latency
    API-->>Client: Prediction, confidence, version, request ID
```

## Production Controls

| Concern | Control |
|---|---|
| Training/serving skew | Preprocessing and classifier serialized in one pipeline |
| Leakage | Imputation and encoding fitted within CV pipeline |
| Reproducibility | Pinned dependencies, fixed seed, source checksum, run metadata |
| Invalid input | Strict Pydantic schema, forbidden extra fields, bounded domains |
| Availability | Model-aware health endpoint and Kubernetes probes |
| Scalability | Stateless API, one worker per container, two Kubernetes replicas |
| Resource safety | Kubernetes CPU/memory requests and limits |
| Container safety | Non-root user, read-only root, no privilege escalation, dropped capabilities |
| Observability | JSON logs, request IDs, Prometheus metrics, Grafana dashboard |
| Privacy | Clinical feature values are not written to request logs |
| CI quality | Lint and tests block training and packaging on failure |

