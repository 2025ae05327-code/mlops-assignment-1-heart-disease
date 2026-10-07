.RECIPEPREFIX := >
PYTHON ?= python3

.PHONY: install lint test download eda train pipeline api docker compose-up compose-down k8s report clean

install:
>$(PYTHON) -m pip install --upgrade pip
>$(PYTHON) -m pip install -r requirements-dev.txt
>$(PYTHON) -m pip install --no-deps -e .

lint:
>ruff check src tests scripts

test:
>pytest -q --cov=heart_disease_mlops --cov-report=term-missing

download:
>$(PYTHON) -m heart_disease_mlops.data

eda:
>$(PYTHON) -m heart_disease_mlops.eda

train:
>$(PYTHON) -m heart_disease_mlops.train

pipeline: download eda train

api:
>uvicorn heart_disease_mlops.api:app --host 0.0.0.0 --port 8000

docker:
>docker build -t assignment1-heart-api:1.0.0 .

compose-up:
>docker compose up --build -d

compose-down:
>docker compose down

k8s:
>kubectl apply -k k8s

report:
>$(PYTHON) scripts/build_report.py

clean:
>$(PYTHON) -c "import shutil; [shutil.rmtree(path, ignore_errors=True) for path in ('artifacts', 'mlruns', '.pytest_cache', '.ruff_cache')]"
