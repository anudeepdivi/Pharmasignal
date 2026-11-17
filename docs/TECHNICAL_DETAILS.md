# PharmaSignal — technical details

> Full engineering write-up. The client-facing overview is the [project README](../README.md).

An educational adverse-event **report triage** pipeline combining structured fields and narrative features.

> **M7 production architecture designed (nothing provisioned); Docker image built and verified locally (2026-11-16).** **208 tests pass** with warnings treated as errors. Synthetic test ROC-AUC **0.7620**; selected-threshold precision **0.4214**, recall **0.7791**, F1 **0.5469**. Drift reports: unchanged batch **0 alerts** (`no_drift_detected`), shifted batch **6 alerts** on exactly the transformed fields (`drift_detected`). No clinical validation, canary or AWS deployment.

See [M6 drift methodology, observed results and limitations](M6_DRIFT.md).

See [M5 service, HTTP tests, container files and explicit build blocker](M5_SERVING.md).

See [M4 methodology, real run ID, tracked evidence and commands](M4_TRACKING.md).

See [M3 baseline, exact results, limitations and commands](M3_BASELINE.md).

See [M2 features, verified shapes and reproduction](M2_FEATURES.md).

See [M1 reproduction and implementation details](M1_DATA_PIPELINE.md).

## 1. Problem statement

Demonstrate how to prioritize fictional reports for review by combining tabular exposure information and symptom narratives. Although the project title says “signal detection,” a report-level classifier is not population-level pharmacovigilance signal detection and cannot establish drug-event causality.

## 2. Important disclaimer

Entirely synthetic, non-identifying data is planned. No clinical validation, patient-care use or real treatment safety conclusions. Scores will reflect an artificial data generator, not real adverse-event risk. This project is being built now; it must not be represented as previously deployed work.

## 3. Dataset

**Implemented (M1, verified):** 3,000 synthetic fictional reports from seed 42 via `python -m pharmasignal.data.build` — 1,800 train (1,293 negative / 507 positive), 600 validation (435/165), 600 test (428/172). Labels are stochastic synthetic review-priority labels from a documented generator formula, not expert adjudications. Train/validation/test partitioning is by narrative-template family (6/2/2 of 10 families); zero cross-split overlap in report IDs, families or normalized narratives (validated).

Families share symptom vocabulary and differ mainly in fixed wrapper phrasing, so the family holdout is a weak style-based check, not real-world text generalization. See [dataset strategy](DATASET_STRATEGY.md) and [M1 details](M1_DATA_PIPELINE.md). No external dataset or patient data is downloaded.

## 4. Architecture

```text
Synthetic records -> validation -> group-disjoint splits
 -> [structured imputation/encoding/scaling + spaCy rules + TF-IDF]
 -> sparse feature fusion -> logistic regression
 -> validation selection / frozen test evaluation -> MLflow + model artifact
 -> FastAPI -> local Docker service
 -> offline reference-versus-batch drift monitoring
```

[Full architecture and decisions](ARCHITECTURE.md).

## 5. Data schema

Predictors: `age`, `sex`, `treatment`, `dose_mg`, `exposure_days`, `baseline_egfr`, `concomitant_med_count`, `narrative`. Target: `signal` (0/1, synthetic review priority). `report_id` and `template_family` are audit-only, never predictors.

Implemented (M1): `src/pharmasignal/data/schema.py` defines types/bounds/nullability; `validation.py` enforces them strictly (complete keys, no coercion, booleans/nonfinite/out-of-range rejected, duplicate IDs rejected). See [schema table and example input](DATASET_STRATEGY.md#proposed-record-schema).

## 6. Feature engineering

Implemented numerical median imputation, missingness indicators and scaling; categorical missing-token handling and one-hot encoding. The unfitted sklearn pipeline is fit by the caller on training only. CSR fusion joins tabular, rule-derived and text features. The saved M1 training data produces 10 numeric, 7 categorical, 4 rule and 1,300 TF-IDF columns (1,321 total). No target/audit fields enter the feature matrix. All-missing numeric training columns retain a zero fallback; indicators exist only for columns missing during fit. See [M2 details](M2_FEATURES.md).

## 7. NLP methodology

Implemented word unigram/bigram TF-IDF (`min_df=2`, `max_features=3000`, sublinear TF) with `spacy.blank("en")` tokenization, retaining negation and temporal words. Four rule counts track unnegated symptoms, negated symptoms, history markers and token count. Rules preserve punctuation and use a four-token negation window. No trained clinical NER, language-model download, transformer, lemmatizer or parser. Pseudo-negation, long lists and historical attribution remain known limitations; unnegated does not mean new or treatment-caused.

## 8. Model choice

Logistic regression is a strong, understandable baseline for sparse text plus structured features. Compare structured-only, text-only and fused variants on validation. PyTorch adds no value to this baseline and is not included. Any resume technology list should reflect actual usage, not the other way around.

## 9. Evaluation

M3 compared three fixed logistic-regression pipelines using validation average precision: structured **0.4253**, text **0.4971**, fused **0.5240** (selected). All preprocessing and model fits use train only; no refit after validation selection.

A validation-only threshold of **0.13818775588643972** targets recall >=0.80. Test ROC-AUC **0.7620**, average precision **0.5613**. At the selected threshold: precision **0.4214**, recall **0.7791**, F1 **0.5469**, confusion matrix `[[244,184],[38,134]]` (true rows, predicted columns, order 0/1). At 0.5: `[[387,41],[94,78]]`. The lower threshold reduces false negatives but greatly increases false positives, and **test recall misses 0.80**. No test-based retuning.

False negatives can delay review; false positives consume reviewer time. These synthetic results demonstrate the artificial task, not clinical utility or calibrated risk. Three full-data runs were performed: the two disclosed M3 runs and one unchanged-settings M4 tracking run, with matching selection, metrics and error analysis. The one-test-pass guard applies per invocation, not across project history. Full error analysis and limitations are in [M3 notes](M3_BASELINE.md).

## 10. MLflow

Implemented M4 with MLflow 2.22.5: explicit local SQLite tracking plus filesystem artifacts, one run per baseline comparison, 22 parameters, 67 metrics, provenance and a retrievable complete model bundle. Verified FINISHED run: `39bcca5c03124296a602ec896d0418ee`. Every logged metric matches its saved report; downloaded files and predictions match the offline artifact. No autologging, native MLflow model flavor or registry promotion. Tracking failures propagate rather than silently succeeding. MLflow does not prevent leakage or establish clinical validity.

```bash
# New destination required; this is another full-data evaluation.
.venv/bin/python -m pharmasignal.models.tracking --data artifacts/data/seed42 --output artifacts/models/my_m4_run --tracking-dir artifacts/mlflow
```

Current tracked model: `artifacts/models/m4/`, version `baseline-a28dc00666102c16`. See [M4 details](M4_TRACKING.md) for readback without retraining and an optional localhost-only UI command.

## 11. API

Implemented FastAPI `POST /predict`, `GET /health` and `GET /ready`. All eight predictor keys are required, with explicit nulls where allowed. Strict validation rejects unknown fields, numeric coercion, nonfinite values and invalid narratives. Responses contain the predicted label, score, selected threshold and model/schema versions. Missing/incompatible artifacts abort startup; errors do not echo narratives. No training or MLflow server at startup.

```bash
.venv/bin/python -m pharmasignal.serving --model-dir artifacts/models/m4 --port 8000
```

Default bind is `127.0.0.1`; access logs are disabled. Real loopback HTTP requests and prediction equivalence passed using a small fixture model. [M5 documentation](M5_SERVING.md) includes request examples, curl reproduction commands and limitations.

## 12. Docker

Dockerfile, restrictive `.dockerignore` and hash-pinned `docker/requirements.txt` are present. The image specifies Python 3.12.14, non-root UID 10001 and the trusted M4 bundle. **Build/run verified locally (2026-11-16):** `make docker` builds the image and serves the API on `127.0.0.1:8000`; health check, predictions (identical to the local artifact), non-root user and read-only model files were confirmed. No cloud deployment is claimed. See [M5 serving notes](M5_SERVING.md).

## 13. Drift monitoring

Implemented M6 with SciPy/NumPy comparisons (no Evidently): reference is the 1,800-row training split; comparison batches are simulated demonstration draws (n=600, seeds 2025/2026) from the same synthetic generator. Numeric fields get the KS statistic plus an absolute standardized mean difference; categoricals and a fixed 9-alias symptom-vocabulary presence check get total-variation distance; six nullable fields get missingness-rate differences; text proxies cover narrative length, token count and negation-aware rule counts. Every row carries sample counts, the recorded (never gating) p-value where applicable, and a status from pre-registered thresholds in `configs/monitoring.json` (`min_samples=30`, `ks_alert=0.15`, `effect_alert=0.5`, `tv_alert=0.2`, `missingness_alert=0.15`).

**Observed:** the unchanged batch produced `no_drift_detected` (0 of 26 rows alerting; e.g. age KS 0.0449); the shifted batch produced `drift_detected` with 6 alerts on exactly the transformed dimensions (dose_mg KS 0.2827/effect 0.875; baseline_egfr KS 0.5223; exposure_days KS 0.3392; concomitant_med_count KS 0.3668; sex TV 0.3896; baseline_egfr missingness +0.310). In the unchanged batch, `token_count` shows p=0.0014 yet stays `ok` (KS 0.0894 < 0.15) — p-values are recorded but never gate status. No model scores are computed: no performance claim is possible from these reports.

- **Data drift:** inputs change. (Measured here.)
- **Concept drift:** the input-to-target relationship changes. (Not measurable without labels.)
- **Performance degradation:** predictive quality falls, requiring appropriate labels/evaluation to establish.

Data drift alone proves neither concept drift nor performance degradation. Evidently is an alternative, not an implemented dependency. No automatic retraining/promotion is planned or triggered. Batches are labeled simulated demonstrations, not observed production samples.

```bash
.venv/bin/python -m pharmasignal.monitoring.simulate --scenario shifted --n 600 --output artifacts/data/simulated/shifted
.venv/bin/python -m pharmasignal.monitoring.report --reference artifacts/data/seed42 --batch artifacts/data/simulated/shifted --config configs/monitoring.json --output artifacts/monitoring/shifted
```

## 14. Canary strategy

Designed in M7 ([full design](M7_PRODUCTION.md)): stable and candidate as separate target groups behind one ALB listener with weighted forwarding (95/5 start); promotion and rollback are reversible pointer/weight writes; operational alarms watched in minutes, labeled quality comparison only after delayed labels. Local routing is optional and not currently implemented. A design is not evidence of an actual safe rollout, and no rollback drill has been run.

## 15. AWS deployment architecture

Designed only in [M7 production design](M7_PRODUCTION.md): client -> TLS/ALB -> ECS Fargate in private subnets, ECR images, S3 versioned artifacts with promotion pointers, CloudWatch alarms -> SNS, least-privilege IAM; a single-EC2 alternative is compared with its cost trade-off. **M7 is a design milestone: no cloud resource, container build, router or deployment exists.** No credentials were inspected, no resources created, no costs incurred, and no deployment verified. Explicit approval is required before any billable resource creation.

## 16. Limitations

- Synthetic generation and labels do not represent real clinical evidence.
- Template-family holdout reduces but cannot eliminate generator shortcuts; families share symptom vocabulary, so it checks style robustness only.
- No population signal aggregation, causality analysis or expert adjudication.
- No calibrated clinical probabilities, approved threshold or external validation.
- Simple text rules can misread negation, temporality and symptom attribution.
- Drift monitoring is univariate and aggregate-level on simulated batches; no joint-distribution, semantic or per-segment analysis; alerts are informational only.
- The narrative embeds exposure duration, so a null structured `exposure_days` can be partly recovered from text; other missing fields have no such echo. This is a known synthetic-data artifact, documented rather than tuned away.
- Labels are arbitrary generator outputs; no real treatment is involved.
- No claimed use of PyTorch/Evidently; AWS and canary are design-only plans.

## 17. Future improvements

Only after the baseline is understood: appropriately licensed and adjudicated data, realistic patient/time/site holdouts, clinically reviewed terminology/negation handling, calibration, uncertainty and subgroup evaluation, and labeled production feedback. Do not add transformers or infrastructure without a demonstrated need.

## 18. Reproduction and current verification

### M0 files

```text
pharmasignal/
  pyproject.toml, uv.lock        dependency declaration and resolved lockfile
  README.md                      client-facing overview and quick start
  docs/                          architecture, dataset strategy, requirements, milestone notes, production design
  configs/data.json              local generator configuration (seed 42)
  src/pharmasignal/data/
    schema.py                    field types, bounds, nullability, PREDICTORS
    validation.py                strict record/dataset validation
    generator.py                 seeded synthetic reports and label functions
    splits.py                    family disjointness and narrative-overlap checks
    build.py                     CLI: writes JSONL + manifest, refuses overwrite
  src/pharmasignal/features/     text.py (spaCy/rules), fusion.py (sklearn pipeline)
  src/pharmasignal/models/       train.py, evaluation.py, artifact.py, tracking.py
  src/pharmasignal/provenance.py model provenance utilities
  src/pharmasignal/serving/      requests.py, app.py, __main__.py
  src/pharmasignal/monitoring/   config.py, metrics.py, simulate.py, report.py
  configs/monitoring.json        pre-registered drift thresholds
  Dockerfile, .dockerignore      container definition (build verified)
  docker/requirements.txt       hash-pinned runtime dependency export
  tests/test_monitoring.py       35 M6 drift cases
  tests/test_serving.py          35 M5 cases, including live HTTP
  tests/test_data.py             78 M1 cases
  tests/test_features.py         29 M2 cases
  tests/test_tracking.py         6 M4 MLflow persistence/retrieval cases
  artifacts/data/seed42/         generated dataset (git-ignored; rebuild locally)
  artifacts/data/simulated/      simulated unchanged/shifted batches (git-ignored)
  artifacts/monitoring/          recorded drift reports (git-ignored)
  artifacts/mlflow/              local SQLite tracking store and artifacts (git-ignored)
```

### Dependency-free M0 checks

From the project root, with Python 3.11+:

```bash
python3 -c 'import pathlib,tomllib; p=tomllib.loads(pathlib.Path("pyproject.toml").read_text()); print(p["project"]["name"], p["project"]["requires-python"])'
git status --short
git diff --check
```

These inspect configuration/worktree state; they do **not** run model tests. `git diff --check` does not inspect untracked file contents.

M0 checks passed for seven required deliverables, TOML structure, source syntax, whitespace and four local document link targets. A source-path import smoke test also passed:

```bash
PYTHONPATH=src python3 -c 'import pharmasignal, pharmasignal.data, pharmasignal.features, pharmasignal.models, pharmasignal.serving, pharmasignal.monitoring; print("scaffold imports OK")'
```

Plain `python3 -c 'import pharmasignal'` failed with `ModuleNotFoundError` because the project is not installed and uses a `src/` layout. The source-path smoke test imports empty packages only; it does not establish dependency compatibility or a runnable prediction system.

### Environment setup — verified in M1

```bash
uv sync --python 3.12   # creates .venv (Python 3.12.14) and installs the locked uv.lock
.venv/bin/python -c 'import numpy, pandas, sklearn, scipy, spacy; print("imports OK")'
```

`uv.lock` exists locally (no commit or push has been made); later environments use `uv sync --frozen --python 3.12`. Resolved versions at M1: numpy 2.5.3, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.18.1, spaCy 3.8.16. MLflow 2.22.5 became an implemented dependency at M4; FastAPI 0.141.1 and Uvicorn 0.53.0 are used at M5. `docker/requirements.txt` exports the locked runtime dependencies; container installation was verified on 2026-11-16.

### Build the dataset (verified in M1)

```bash
.venv/bin/python -m pharmasignal.data.build --config configs/data.json --output artifacts/data/seed42
```

Writes `train.jsonl`, `validation.jsonl`, `test.jsonl` and `manifest.json` (refuses to overwrite an existing directory). Output for the local config: dataset_id `31f04f04…37d3a9e`, with per-file SHA-256 hashes, config hash, source-file hashes and split checks in the manifest.

### Run the tests (verified in M6)

```bash
.venv/bin/python -m pytest -W error --cov=pharmasignal --cov-report=term-missing
.venv/bin/ruff check src tests
```

Current M6 verification: 200 passed, 93% total coverage, lint and format clean. Drift reports verified on simulated batches; local HTTP verified; Docker build/run verified separately on 2026-11-16. Historical M2/M3/M5 scoping is in [M2 reproduction](M2_FEATURES.md#exact-reproduction), [M3 reproduction](M3_BASELINE.md#reproduction) and [M5 documentation](M5_SERVING.md).

### Train and load the baseline (M3)

```bash
.venv/bin/python -m pharmasignal.models.train --data artifacts/data/seed42 --output artifacts/models/my_m3_run
```

Choose a new destination; existing directories are refused. Current verified artifact: `artifacts/models/m3_verified/`, version `baseline-61102426d59400c3`. Includes fitted `pipeline.joblib`, `report.json` and `metadata.json`. See [loading example and run disclosure](M3_BASELINE.md#reproduction). Load trusted artifacts only. No serving API yet.

## Milestone status

| Milestone | Status |
|---|---|
| M0 — design | Complete; reviewed |
| M1 — data | Complete; reviewed |
| M2 — features | Complete; reviewed |
| M3 — model | Complete; reviewed |
| M4 — MLflow | Complete; reviewed |
| M5 — API/Docker | Complete locally (165 tests at M5, live HTTP); Docker blocked at the time, verified later (2026-11-16); reviewed |
| M6 — drift | Complete locally (200 tests, both simulated reports recorded); no performance claim; human review required |
| M7 — production design | Design complete (`docs/M7_PRODUCTION.md`); **no resources, build still blocked**; reviewed |
| M8 — audit | Complete — all claims re-verified, 2 stale doc lines fixed, no functionality added |

Every milestone stops for review; approval of M0 does not authorize cloud provisioning.
