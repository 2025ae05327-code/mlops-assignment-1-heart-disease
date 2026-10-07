# End-to-End MLOps for Heart Disease Risk Classification

**Student:** `<ENTER STUDENT NAME>`  
**Student ID:** `<ENTER STUDENT ID>`  
**Repository:** `<ENTER CODE REPOSITORY URL>`  
**Course:** AIMLCZG523 Machine Learning Operations, Assignment 01

> Generate the submitted DOCX with `python scripts/build_report.py` after completing the
> pipeline and adding genuine screenshots. This Markdown file is the editable narrative.

## 1. Executive Summary

This assignment implements the complete lifecycle of a binary heart disease classifier.
The workflow includes source-controlled data acquisition, explicit quality checks,
reproducible EDA, leakage-safe preprocessing, comparison and tuning of Logistic Regression
and Random Forest, MLflow experiment tracking, model packaging, API serving, automated
tests, CI/CD, Docker, Kubernetes, logging, Prometheus, and Grafana.

The objective is not simply a high metric. It is a traceable system in which the source
checksum, feature contract, folds, parameters, metrics, model artifact, image, deployment,
and runtime behavior can be inspected. This educational classifier is not a medical device
and must not be used for diagnosis.

## 2. Data Acquisition and Cleaning

The processed Cleveland file is downloaded directly from the UCI Machine Learning
Repository. The downloader uses retries and atomic replacement so interrupted transfers do
not leave a valid-looking partial file. A metadata artifact records the URL, UTC timestamp,
SHA-256 checksum, row count, duplicate count, missingness, and class balance.

The source has 13 predictors and a diagnosis field from zero to four. Zero is mapped to
absence and values one through four to presence. Question marks and nonnumeric tokens become
missing values. Out-of-domain codes become missing rather than silently entering the model.
Exact duplicate rows are removed. Missing predictor values remain in the clean CSV because
imputation must be learned from training folds rather than from all observations.

## 3. Exploratory Data Analysis

The EDA stage creates class balance, feature distributions separated by outcome, a Pearson
correlation matrix, and a missingness chart. A machine-readable summary accompanies the
plots. The class plot determines whether weighting is warranted. Distribution plots expose
range and skew differences. Correlation supports interpretation but is not used as an
automatic causal or feature-removal rule.

Insert observations from the generated plots here after running the pipeline. Reference
`artifacts/eda/class_balance.png`, `numeric_distributions.png`,
`correlation_heatmap.png`, and `missing_values.png`.

## 4. Feature Engineering and Validation

Continuous features (`age`, `trestbps`, `chol`, `thalach`, and `oldpeak`) receive median
imputation and standard scaling. Coded categorical features receive most-frequent
imputation and one-hot encoding. Unknown categories are ignored at transform time, while
the API still enforces the documented UCI domains.

The preprocessor and classifier form one scikit-learn Pipeline. Therefore each validation
fold fits its own imputer, scaler, and encoder. This avoids leakage and guarantees the same
transformations during inference. Data is split with stratification and random state 42;
the training split uses shuffled stratified cross-validation. Mean CV ROC-AUC is the
selection criterion. Accuracy, precision, recall, standard deviation, and final test metrics
provide additional context.

## 5. Model Development and Experiment Tracking

Logistic Regression is the interpretable linear baseline with balanced class weights.
Random Forest evaluates nonlinear thresholds and interactions with the same balanced-class
policy. The full run searches multiple regularization strengths and penalties for Logistic
Regression and tree counts, depths, and minimum leaf sizes for Random Forest.

Each candidate has an MLflow run containing best parameters, fold count, seed, CV metric
means and standard deviations, test metrics, fit duration, ROC curve, confusion matrix,
model signature, input example, and serialized candidate pipeline. Record the selected
model and measured results from `models/model_metadata.json` here. The test split must not
be used to change the selected model.

## 6. Packaging and API

The winning preprocessor-classifier pipeline is packaged with Joblib. JSON metadata stores
feature order, selection rule, candidate results, run ID, package version, and paths. FastAPI
loads the artifact lazily and reports an unhealthy status when it is absent or invalid.

The prediction request has all 13 features and forbids extras. Bounds and categorical
domains produce a clear HTTP 422 error before inference. A successful response contains the
binary output, risk label, disease probability, confidence in the predicted class, decision
threshold, model version/run ID, and request ID. `/metrics` exposes Prometheus text and
`/docs` exposes OpenAPI.

## 7. Testing and CI/CD

The deterministic test suite validates cleaning, domain handling, schema failures,
imputation, encoding, both classifiers, metric bounds, EDA outputs, MLflow packaging,
prediction success, invalid requests, model absence, and Prometheus output. Ruff enforces
errors, import order, bugbear checks, and Python modernization rules.

GitHub Actions runs lint and tests before downloading data, generating EDA, training both
models, creating the DOCX, and building the image. Failure stops dependent stages and leaves
clear logs. The evidence artifact includes JUnit, coverage, plots, clean data, models,
MLflow state, and report. Add a screenshot and URL for the successful submitted commit.

## 8. Containerization and Deployment

The serving image uses pinned dependencies, a slim Python base, one process per container,
a non-root account, and a model-aware health check. Docker Compose mounts the current model
read-only and starts API, Prometheus, and Grafana.

The Kustomize bundle deploys two API replicas with readiness and liveness checks, resource
requests/limits, no privilege escalation, dropped capabilities, a read-only root filesystem,
and a LoadBalancer Service. Minikube satisfies the local production-deployment option. A
cloud deployment only needs a registry image reference, pull policy update, and appropriate
Ingress/TLS and managed monitoring controls.

## 9. Monitoring and Logging

Structured JSON logs contain timestamp, level, request ID, method, route, status, latency,
and prediction class. They deliberately exclude raw patient attributes. Prometheus records
request totals by route and status, duration histograms, and predictions by output class.
Grafana is provisioned with request rate, p95 latency, status trend, and output distribution.

The output ratio is a triage signal, not conclusive drift detection. A real system should
capture governed input distributions, delayed outcomes, calibration, subgroup performance,
and alert thresholds while maintaining privacy controls.

## 10. Limitations, Reproducibility, and Conclusion

The Cleveland cohort is small, historical, and geographically limited. Predictive
association is not causal evidence. The reported probability may not be calibrated for a
new population. External validation, subgroup fairness analysis, threshold governance,
clinical review, authentication, encryption, durable storage, signed images, and secret
management are necessary before any real healthcare use.

Despite those limitations, the assignment demonstrates a coherent MLOps chain. A clean
environment can recreate the source lineage, plots, experiments, model, report, image,
Kubernetes workloads, logs, and dashboard from documented commands. The fitted preprocessing
pipeline eliminates training-serving skew, CI guards changes, and monitoring makes runtime
behavior visible.

## References

1. UCI Machine Learning Repository, Heart Disease dataset.
2. scikit-learn documentation for Pipeline, ColumnTransformer, GridSearchCV, and metrics.
3. MLflow tracking and model documentation.
4. FastAPI, Docker, Kubernetes, Prometheus, and Grafana official documentation.
