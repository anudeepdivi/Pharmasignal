# M5 — Local inference API

## Status and evidence

API implemented and verified locally. **165 tests passed** with warnings treated as errors; Ruff clean. Total coverage 91%; `serving/app.py` and `serving/requests.py` each 100%. CLI coverage shows 0% because its live subprocess is not instrumented; the real HTTP test executes it successfully.

**Update 2026-11-16 — Docker verified:** `docker build -t pharmasignal:latest .` succeeded (legacy builder, 13 steps, 1.56 GB unpacked / 356 MB content); the container ran as UID 10001, reported `healthy`, served `/health`, `/ready` and `/predict` (probability `0.7323832477775062` for the README example — identical to the local artifact), returned 422 without echoing input, and could not write to `/app/model`.

*Original M5 record:* **Docker build/run is blocked, not verified.** Docker client 29.7.2 was present, but the daemon socket `/var/run/docker.sock` was absent. An earlier build command was blocked by a tool safety-check timeout. The final direct `docker build -t pharmasignal:m5 .` executed and exited 1: `dial unix /var/run/docker.sock: connect: no such file or directory`. It also warned about the deprecated legacy builder. No successful image build, container startup, container HTTP result, or Podman availability is claimed. No daemon was installed or started. This is the explicit blocked-status option in the M5 acceptance criteria, not completed containerization.

## Methodology and implementation

- `serving/requests.py`: Pydantic model generated from `PREDICTORS` and `FIELDS_BY_NAME`, with strict types, finite numbers, ranges, categorical choices and nonblank/max-length narrative. Nullable fields accept null, but **all eight keys are required**. This corrects the initial M5 plan's omitted-as-null suggestion and preserves the documented M1 contract. Unknown keys, audit IDs and target labels are rejected.
- `serving/app.py`: app factory with lifespan loading of the trusted offline bundle via `load_artifact`. Failed loading aborts startup. No training, registry lookup, model download or MLflow connection at startup. Loaded model/metadata are cleared on shutdown.
- `/predict` uses `predict_records`, which validates/project predictors and applies the artifact's selected threshold, not sklearn's default 0.5. The response model validates the five returned fields.
- `serving/__main__.py`: one Uvicorn worker, no reload, default loopback binding; model selected by `--model-dir`, then `PHARMASIGNAL_MODEL_DIR`, then `artifacts/models/m4`.

The model remains synthetic report-triage classification, not population signal detection, causality or calibrated clinical risk. M5 performed only small-fixture training for tests, not another full-data training/test evaluation. Live HTTP equivalence was checked against a freshly trained small fixture; it is not evidence that the full-data model was container-tested.

## Endpoints

| Endpoint | Success | Failure |
|---|---|---|
| `GET /health` | 200, `{"status":"alive"}` | Process must be running/reachable |
| `GET /ready` | 200 with model/schema versions | 503 if unloaded |
| `POST /predict` | 200 with five versioned prediction fields | 422 validation; 503 unavailable; 500 inference failure |

With a bad startup artifact, normal Uvicorn startup fails entirely; it does not remain available just to answer health. Pre-load 503 behavior is tested separately in-process.

422 responses contain sanitized field locations, error types and a generic message, not Pydantic's raw input/context. Unexpected field names are replaced by `unknown`. Inference failures log only the exception class and return a generic 500. Uvicorn access logging is disabled to avoid logging query strings. These measures do not prevent an upstream proxy or custom server invocation from logging sensitive input.

## Run and request example

From project root:

```bash
.venv/bin/python -m pharmasignal.serving --model-dir artifacts/models/m4 --port 8000
```

Then, in another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
curl --fail http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data '{"age":64,"sex":"female","treatment":"treatment_a","dose_mg":50.0,"exposure_days":12,"baseline_egfr":62.0,"concomitant_med_count":3,"narrative":"Reports nausea after treatment started. Denies rash."}'
```

The same example payload and endpoints were exercised by HTTPX over a real loopback TCP connection to the CLI subprocess using a small fixture model. The literal curl commands above are provided for reproduction, not claimed as executed. Predictions return `predicted_signal`, `signal_probability`, `decision_threshold`, `model_version`, `schema_version`; numeric example outputs are intentionally not invented. `/docs` and `/openapi.json` expose FastAPI's generated API documentation.

## Container deliverables — unbuilt

`docker/requirements.txt` was generated successfully from the existing lockfile with:

```bash
uv export --frozen --offline --no-dev --no-emit-project --format requirements.txt --output-file docker/requirements.txt
```

The export preserves package pins and hashes; no dependency upgrade was performed. It includes the full runtime dependency set, including MLflow, favoring environment consistency over image size. It excludes the project itself; the image copies importable source and sets `PYTHONPATH=/app/src` rather than performing an unpinned build-backend install.

`Dockerfile` specifies Python **3.12.14** to match the trusted loader's exact Python-version check, hash-checked pip installation, UID/GID 10001, readiness healthcheck, source and only the three trusted M4 model files. `.dockerignore` allowlists those inputs, excluding datasets, tracking stores, caches, git and local environment files. No training or secrets are in the build steps. The base tag is not digest-pinned and its availability was not verified; dependency wheel availability on the container platform is likewise unverified.

When a working daemon is available, these are **unverified next commands**, not completed results:

```bash
docker build -t pharmasignal:m5 .
docker run --rm --name pharmasignal-m5 \
  -p 127.0.0.1:8000:8000 --cap-drop ALL \
  --security-opt no-new-privileges pharmasignal:m5
```

Binding 0.0.0.0 inside the container enables port forwarding; the host publish above is loopback-only. Reuse the curl commands after readiness. Stop with Ctrl+C. Preserve the model's compatible source/dependencies; recorded hashes do not authenticate artifacts or automatically enforce source compatibility.

## Tests and actual fixes

35 new serving cases plus the existing 130:
- Valid/null inputs and required presence of every predictor.
- 21 invalid-value/extra-field cases checked both through Pydantic and HTTP (including bools, numeric strings, nonfinite numbers and invalid narratives).
- Health/readiness before model loading.
- Lifespan loading/cleanup, exact API-versus-offline predictions, sanitized malformed JSON and inference errors.
- Missing/corrupt artifact startup failure.
- Real Uvicorn subprocess, bounded readiness polling, live GET/POST requests, exact prediction equivalence and process cleanup; no narrative in captured logs.

Commands actually passed:

```bash
.venv/bin/python -m pytest -W error --cov=pharmasignal --cov-report=term-missing
.venv/bin/ruff check src tests
```

Initial `TestClient` collection failed because installed Starlette deprecated its HTTPX fallback in favor of `httpx2`. An attempted dependency installation did not execute due to safety-check timeouts. Final tests use the existing HTTPX `ASGITransport` with lifespan explicitly entered, plus real TCP coverage; no warning suppression or dependency change was needed. Ruff line-length errors were corrected by formatting.

## Decisions, alternatives and limitations

- Schema-derived Pydantic fields avoid duplicating bounds; dynamic generation is less obvious to type checkers than an explicit static class. It does not make all future schema changes automatically correct: tests still matter.
- Single-report synchronous route uses FastAPI's worker threadpool; no batching, load test or throughput/latency claim. One process loads one model; additional workers would duplicate model memory.
- Fail-fast startup instead of serving degraded predictions; separate readiness and liveness concepts.
- Reuse complete offline pipeline instead of native MLflow serving to retain custom threshold and validation behavior.
- No authentication, TLS, rate limiting, global request-body limit, concurrency hardening or production monitoring. A 4,000-character narrative limit is not a total-body/transport limit. Local-only development use.
- Model loading remains trusted-only; checksums are not signatures. No arbitrary model upload endpoint exists.
- No Docker image or AWS deployment is verified. No clinical validation, drift or canary functionality.

M5 stops here with the Docker verification blocker explicitly recorded. M6 requires separate approval.
