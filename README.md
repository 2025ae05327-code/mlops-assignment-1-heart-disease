# Heart Disease MLOps Assignment

An end-to-end, reproducible MLOps solution for binary heart disease risk prediction using
the UCI Cleveland dataset. The project downloads and validates data, generates EDA,
compares tuned Logistic Regression and Random Forest pipelines, tracks experiments in
MLflow, serves the selected model through FastAPI, and deploys it with Docker or Minikube.

> This is an educational risk classifier, not a medical diagnosis system.

## Quick Start

Use Python 3.10, 3.11, or 3.12 from the project root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
ruff check src tests scripts
pytest -q
python -m heart_disease_mlops.data
python -m heart_disease_mlops.eda
python -m heart_disease_mlops.train
python scripts/build_report.py
```

## Requirement Traceability

| PDF requirement | Implementation and evidence |
|---|---|
| Data acquisition and EDA | `data.py`, SHA-256 lineage, cleaned CSV, four EDA views |
| Feature engineering and two models | One fitted preprocessing pipeline, Logistic Regression, Random Forest, tuning grids |
| Experiment tracking | Local MLflow runs with parameters, CV/test metrics, plots, signatures, and models |
| Packaging and reproducibility | Joblib pipeline, JSON metadata, pinned requirements, deterministic seeds |
| CI/CD and testing | Pytest suite, Ruff, GitHub Actions training/image workflow, uploaded evidence |
| Containerization | Non-root Dockerfile, model-aware health check, `/predict`, sample request |
| Production deployment | Kustomize manifests, two replicas, probes, limits, LoadBalancer service |
| Monitoring and logging | Structured JSON logs, Prometheus counters/histogram, provisioned Grafana dashboard |
| Documentation and reporting | Setup guide, architecture, report generator, screenshots, generated 10+ page DOCX |

## Project Layout

```text
assinment_1/
|-- .github/workflows/ci-cd.yml
|-- data/
|   |-- raw/                         # generated source data
|   |-- processed/                   # generated clean data and lineage
|   `-- sample_request.json
|-- artifacts/                       # generated EDA/training/test evidence
|-- docs/
|   |-- ARCHITECTURE.md
|   `-- Heart_Disease_MLOps_Report.docx
|-- k8s/                             # API, Prometheus, and Grafana manifests
|-- models/                          # generated pipeline and metadata
|-- monitoring/                      # local Prometheus/Grafana configuration
|-- screenshots/                     # add genuine execution evidence before submission
|-- scripts/                         # smoke test and DOCX report generator
|-- src/heart_disease_mlops/         # application package
|-- tests/                           # unit and integration tests
|-- Dockerfile
|-- docker-compose.yml
|-- Makefile
`-- pyproject.toml
```

## Pipeline Outputs

```bash
make pipeline
```

- Cleaned data: `data/processed/heart_disease_cleaned.csv`
- Data lineage: `data/processed/data_summary.json`
- EDA: `artifacts/eda/`
- Candidate diagnostics and comparison: `artifacts/training/`
- MLflow tracking store: `mlruns/`
- Selected pipeline: `models/heart_disease_pipeline.joblib`
- Model card metadata: `models/model_metadata.json`

Inspect experiments locally:

```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```





Before submission, commit the cleaned dataset, selected model, model metadata, generated
EDA/model plots, genuine CI/container/Kubernetes/monitoring screenshots, final DOCX or PDF,
repository URL, API access instructions, and short demonstration video link. Generated
metrics and screenshots must come from your own execution.
