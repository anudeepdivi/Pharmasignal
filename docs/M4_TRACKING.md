# M4 — Local MLflow tracking

## Methodology and implementation

`src/pharmasignal/models/tracking.py` wraps the existing M3 `train_baselines` exactly once. One MLflow run represents the complete three-variant comparison. The training-only fits, validation AP selection, validation recall-constrained threshold and single test prediction pass per invocation are unchanged. Logging consumes the returned report; it does not predict again. Offline M3 training/loading remains available without a tracking database.

MLflow **2.22.5** uses an explicit `MlflowClient`, SQLite metadata at `artifacts/mlflow/mlflow.db` and local filesystem artifacts. Both tracking and registry URIs are explicitly local; ambient remote URI variables do not redirect the client. The experiment and run artifact locations are checked before uploads. There is no autologging, remote tracking server, model registry entry or deployment.

## Verified run

- Experiment: `pharmasignal-baselines`, ID `1`.
- Run: `39bcca5c03124296a602ec896d0418ee`, status **FINISHED**.
- Offline output: `artifacts/models/m4/`.
- Model version: `baseline-a28dc00666102c16`.
- Local tracking URI: `sqlite:///<project-root>/artifacts/mlflow/mlflow.db` (absolute path resolved at runtime).
- Artifact root: `artifacts/mlflow/artifacts/39bcca5c03124296a602ec896d0418ee/artifacts/`.
- Readback verified **22 parameters**, **67 metrics**, and matching dataset/model identities. Exact comparison of all 67 logged metrics against the matching report found **zero mismatches**.
- Downloaded bundle: `artifacts/verification/m4/model_bundle/`. All three files are byte-identical to the original; trusted loading and predictions on three training examples match. The bundle was also loaded in place from the artifact store root (`file:///.../39bcca5c03124296a602ec896d0418ee/artifacts/model_bundle`) without a download step, producing threshold-aware predictions with the expected model version.

One full-data M4 run was performed with unchanged settings. Together with the two M3 runs, this is **three full-data evaluations in project history**, plus small-fixture test runs. M4 is not a newly untouched holdout experiment. Its selection, validation/test metrics and errors match M3 exactly. No test-driven retuning occurred. The version differs because source/report provenance changed, not because predictions improved.

| Quantity | Observed result |
|---|---:|
| Selected variant | fused |
| Validation AP: structured / text / fused | 0.425342 / 0.497064 / 0.523960 |
| Selected threshold | 0.13818775588643972 |
| Test ROC-AUC / AP | 0.762008 / 0.561277 |
| Test precision / recall / F1 | 0.421384 / 0.779070 / 0.546939 |
| Test TN / FP / FN / TP | 244 / 184 / 38 / 134 |

Test recall still misses the illustrative 0.80 target. Tracking does not improve model quality or confer clinical validity.

## What is tracked

**Parameters (22):** seed, recall_target, C, solver, max_iter, class_weight, min_df, max_features, variants, penalty, tfidf_ngram_range, tfidf_sublinear_tf, numeric_imputation, categorical_encoding, dataset_config_sha256, selection_criterion, training_scope, dataset_id, dataset_manifest_sha256, configuration_sha256, lockfile_sha256, schema_version. Non-string values use JSON strings (`null`, `true`, lists), not ambiguous Python representations.

**Metrics (67):** six groups, each containing rows, threshold, precision, recall, F1, ROC-AUC, AP and TN/FP/FN/TP, plus `test_passes`:
- `validation.structured.at_0_5.*`
- `validation.text.at_0_5.*`
- `validation.fused.at_0_5.*`
- `validation.selected.*`
- `test.at_0_5.*`
- `test.selected.*`

Undefined AUC/AP values are omitted from numeric logging; explicit status remains in the JSON report. Subgroup metrics and coefficient extremes are in the report artifact, not separate UI metrics. No training metrics are fabricated: M3 does not produce them. Only the selected model receives test evaluation.

**Tags:** milestone, synthetic status, MLflow version, run name, selected variant, model version, git revision/dirty state and clinical-validation status. There is no git commit: revision is `unknown`, dirty is `True`.

**Six artifact files:**

```text
model_bundle/
  pipeline.joblib          fitted preprocessing + classifier
  report.json              selection, metrics, errors, coefficients, provenance
  metadata.json            threshold, versions, checksums, predictor schema
evidence/
  dataset_manifest.json    generator config, split hashes/counts/families
  configuration.json       M3 training configuration
  provenance.json          dependencies, source hashes, git/lock identity, MLflow version
```

`tracking.json` is a separate local receipt in the offline output; it is not part of the checksummed bundle and not uploaded. Query MLflow for run status; a receipt alone does not prove success. Raw narrative rows and dataset JSONL files are not uploaded. The fitted vocabulary necessarily contains learned text tokens; this is synthetic data, not a privacy-preserving model technique. The parameter/hash manifest records identity but does not archive the entire source tree or lockfile; preserve those separately for reconstruction.

## Reproduction and inspection

From project root, use the existing locked environment:

```bash
uv sync --frozen --python 3.12
.venv/bin/python -m pytest -W error --cov=pharmasignal --cov-report=term-missing
.venv/bin/ruff check src tests
# New output required. This performs another disclosed test evaluation.
.venv/bin/python -m pharmasignal.models.tracking \
  --data artifacts/data/seed42 --output artifacts/models/my_m4_run \
  --tracking-dir artifacts/mlflow
```

Inspect the existing run without retraining:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from mlflow.tracking import MlflowClient
from pharmasignal.models.tracking import report_metrics
root = Path('artifacts/models/m4')
receipt = json.loads((root / 'tracking.json').read_text())
client = MlflowClient(tracking_uri=receipt['tracking_uri'], registry_uri=receipt['tracking_uri'])
run = client.get_run(receipt['run_id'])
report = json.loads((root / 'report.json').read_text())
assert run.info.status == 'FINISHED'
assert run.data.metrics == report_metrics(report)
print(run.info.run_id, run.info.status, run.data.params, run.data.metrics)
print(client.list_artifacts(run.info.run_id, 'model_bundle'))
PY
```

`client.download_artifacts(run_id, 'model_bundle', destination)` retrieves the complete bundle; create the destination first. Load using `load_artifact(Path(downloaded_path))` and apply `predict_records`, not bare estimator `predict`, which would ignore the custom threshold. Only load trusted outputs. Fresh-process retrieval/inference is covered in tests.

Optional local UI command (flags verified with `--help`; UI itself was **not started or browser-tested**):

```bash
.venv/bin/mlflow ui --backend-store-uri "sqlite:///$PWD/artifacts/mlflow/mlflow.db" \
  --default-artifact-root "file://$PWD/artifacts/mlflow/artifacts" \
  --no-serve-artifacts --host 127.0.0.1 --port 5000 --workers 1
```

Tracking works without a running UI/server. Do not expose this development UI publicly.

## Tests and observed results

**130 passed**, warnings treated as errors; **92% total coverage**, **86% tracking-module coverage**; Ruff clean. Six new cases cover:
1. Actual temporary SQLite run, parameter/metric readback, manifest retrieval, byte-identical bundle, fresh-process predictions, exactly one training call, repeated runs in one experiment, overwrite refusal, and ambient remote URI isolation.
2. Training failure marks the real run FAILED.
3. Artifact upload failure marks FAILED while preserving the offline model.
4. Failure to mark FAILED preserves the original error and explicitly notes termination failure; the test observes RUNNING rather than claiming success.
5. Unexpected remote experiment artifact location is rejected before a run/upload.
6. Undefined AUC/AP are not logged as numeric values.

Existing 124 tests still pass, including the M3 ordering guard. Two initial Ruff line-length findings were fixed by formatting. A test invocation without `-W error` displayed upstream MLflow/Pydantic deprecation warnings; the specified full-suite `-W error` invocation passes in this environment, with no project warning suppression added. Dependency compatibility is not guaranteed beyond the locked environment.

## Decisions, alternatives and limitations

- **Manual client vs autologging:** explicit split names and threshold semantics are easier to audit; more mapping code must be maintained.
- **SQLite vs file store or remote server:** structured persistent queries with no daemon; not a tested concurrent multi-user service. Concurrent experiment creation can fail; no retry/locking layer is added.
- **One comparison run vs nested runs:** keeps selection and the one selected-model test result together. Baselines are namespaced metrics, not separate model rows in the UI.
- **Custom bundle vs native MLflow sklearn/pyfunc flavor:** preserves the tested threshold-aware loader. There is no `MLmodel` signature/flavor, `mlflow.pyfunc.load_model` support, registration or promotion. Those are alternatives, not completed work.
- **Failure visibility vs best-effort logging:** partial tracking is an error. Training/output failure may leave incomplete files; upload failure may leave a valid offline model and partial MLflow artifacts. SQLite, filesystem uploads and receipt writes are not atomic. Inspect the run and files; do not automatically rerun training just to repair tracking. There is no resume/import command.
- A crash or unavailable store can leave RUNNING; exception cleanup is best effort. Even a receipt can exist before final status is persisted. No secrets/raw exception payloads are uploaded as failure tags.
- Absolute local artifact URIs are not portable by moving the SQLite file alone. Back up the database and artifact tree consistently; preserve source and lockfile. No authentication, remote storage, signed artifact provenance, retention or production reliability claims.
- MLflow is experiment tracking, not deployment, leakage prevention, clinical validation or guaranteed reproducibility. Hashes are not authentication; joblib loading executes trusted code. Existing M3 synthetic-data and evaluation limitations remain.

Later milestones (API, drift monitoring, dashboard) build on this tracked artifact; see [technical details](TECHNICAL_DETAILS.md).
