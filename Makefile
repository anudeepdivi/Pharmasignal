# PharmaSignal — common tasks. Requires uv (https://docs.astral.sh/uv/).
PY := .venv/bin/python
DATA := artifacts/data/seed42
MODEL := artifacts/models/m4
SIM := artifacts/data/simulated
MON := artifacts/monitoring

.SECONDARY:
.PHONY: help setup artifacts test lint dashboard api docker clean-artifacts

help:
	@echo "make setup      Install locked dependencies (incl. dashboard extras) into .venv"
	@echo "make artifacts  Generate synthetic data, train/track the model, build drift reports"
	@echo "make test       Run the full test suite (warnings are errors) with coverage"
	@echo "make lint       Ruff lint + format check"
	@echo "make dashboard  Launch the Streamlit dashboard on http://localhost:8501"
	@echo "make api        Launch the FastAPI service on http://127.0.0.1:8000 (docs at /docs)"
	@echo "make docker     Build and run the API container (needs a running Docker daemon)"

setup:
	uv sync --frozen --python 3.12 --extra dashboard

artifacts: $(MODEL)/metadata.json $(MON)/shifted/report.json $(MON)/unchanged/report.json

$(DATA)/manifest.json:
	$(PY) -m pharmasignal.data.build --config configs/data.json --output $(DATA)

$(MODEL)/metadata.json: $(DATA)/manifest.json
	$(PY) -m pharmasignal.models.tracking --data $(DATA) --output $(MODEL) --tracking-dir artifacts/mlflow

$(SIM)/%/manifest.json:
	$(PY) -m pharmasignal.monitoring.simulate --scenario $* --n 600 --output $(SIM)/$*

$(MON)/%/report.json: $(DATA)/manifest.json $(SIM)/%/manifest.json
	$(PY) -m pharmasignal.monitoring.report --reference $(DATA) --batch $(SIM)/$* \
		--config configs/monitoring.json --output $(MON)/$*

test:
	$(PY) -m pytest -W error --cov=pharmasignal --cov-report=term-missing

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

dashboard:
	.venv/bin/streamlit run src/pharmasignal/dashboard/app.py

api:
	$(PY) -m pharmasignal.serving --model-dir $(MODEL) --port 8000

docker: $(MODEL)/metadata.json
	docker build -t pharmasignal:latest .
	docker run --rm -p 127.0.0.1:8000:8000 pharmasignal:latest

clean-artifacts:
	rm -rf artifacts
