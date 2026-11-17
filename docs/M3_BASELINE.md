# M3 — baseline classification

## Implemented

Three fixed logistic-regression baselines: structured-only, text (rules + TF-IDF), fused. L2 regularization, C=1, liblinear solver, max_iter=2000, seed=42, no class weighting. Convergence warnings fail training. No hyperparameter search, transformer, PyTorch or cloud deployment.

Each candidate fits the entire feature/classifier pipeline on train only. Validation average precision selects the variant; exact ties prefer structured, text, then fused. The winner's threshold maximizes validation precision subject to recall >=0.80; ties prefer recall, then the larger threshold. Labels use score >= threshold. The 0.80 target is illustrative, not clinically approved. No refit on train+validation: refitting would change scores and invalidate the selected threshold.

After freezing selection, a per-invocation guard permits one test read and one prediction pass. Metrics at 0.5 and the selected threshold, errors and subgroups reuse the same scores. This is a workflow guard, not global prevention of repeated experiments. M1/M2 already inspected test data for schema and feature checks. Do not claim it was never accessed.

## Actual synthetic results

Dataset: `31f04f04aa894358f111bb50e4bcdd393eb56be562f77929a6bf8a15937d3a9e`.

| Validation variant | Average precision | ROC-AUC | F1 at 0.5 |
|---|---:|---:|---:|
| Structured | 0.425342 | 0.668687 | 0.192893 |
| Text | 0.497064 | 0.726534 | 0.454545 |
| Fused (selected) | 0.523960 | 0.741595 | 0.464945 |

Selected threshold: **0.13818775588643972**. Validation precision 0.363144, recall 0.812121, F1 0.501873.

| Test operating point | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Threshold 0.5 | 0.655462 | 0.453488 | 0.536082 | 387 | 41 | 94 | 78 |
| Selected threshold | 0.421384 | 0.779070 | 0.546939 | 244 | 184 | 38 | 134 |

Test ROC-AUC **0.7620082590741144**, average precision **0.5612770087196366**, identical at both thresholds because they use continuous scores. Confusion matrices have true rows and predicted columns, ordered 0,1. Undefined precision/recall/F1 divisions return zero. Single-class subgroup ROC-AUC and average precision are null with an explicit status.

The selected threshold catches 56 additional positives but adds 143 false positives. Test recall **misses 0.80**. No adjustment was made using test outcomes. False negatives could delay review, while false positives consume review capacity. Neither actual clinical costs nor calibrated patient risk can be inferred here.

## Error analysis

`report.json` contains all 222 errors at the selected threshold (38 FN, 184 FP), with report IDs/scores rather than duplicated narratives. It also contains sex, treatment and template-family metrics with row counts. Original synthetic narratives can be looked up locally by report ID.

Family 05 (275 rows): FP56/FN19, precision0.5214, recall0.7625. Family 10 (325): FP128/FN19, precision0.3632, recall0.7935. The second family produces substantially more false positives; this is descriptive and cannot isolate style from case composition.

Coefficient extremes expose shortcuts and correlation: token count has coefficient about +1.731; history-marker count -1.049; unnegated-symptom count -0.726 despite positive text terms such as “reports.” Sparse numeric tokens such as “216” also have large weights. Correlated text/rule features and arbitrary templates mean coefficient signs are conditional, not medical explanations. Recorded sex has no generator risk multiplier yet acquires coefficients from finite-sample associations. No fairness claim follows.

## Artifact and run history

Local first run: `artifacts/models/m3/`. A second disclosed run used unchanged model settings after adding report git provenance/formatting: `artifacts/models/m3_verified/`. Both produced exactly matching selection, metrics and error analysis. The first artifact is preserved. Tests additionally use small synthetic fixtures, including repeatability tests; “one pass” is per invocation, not across the whole project lifetime.

Use `m3_verified` as the current artifact. Verified model version: `baseline-61102426d59400c3`. It contains:
- `pipeline.joblib`: full fitted preprocessing + classifier.
- `report.json`: configuration/hash, dataset identity, source hashes, lockfile hash, dependency versions, validation comparisons, frozen selection, test metrics and error analysis.
- `metadata.json`: artifact/schema versions, predictor order, model/report hashes, threshold, variant, dependency versions and content-derived model version. Written last; an incomplete directory without metadata must not be used.

No directory is overwritten. No git revision exists; current report records revision null and dirty true. Version IDs may differ after source/report changes even with identical predictions. A fitted sklearn model requires compatible code as well as dependencies: source hashes are recorded but automatic source-code matching/migration is not implemented.

Loader checks schema, predictors, exact recorded dependency versions, threshold and checksums before unpickling. These checks detect accidental corruption; they do not authenticate an artifact. Joblib can execute code: load only trusted local outputs. Files must not be writable by an untrusted concurrent actor. This is not a secure model registry. M1's original provenance implementation was left unchanged to preserve dataset evidence; the new provenance utility currently serves models only.

## Reproduction

From project root:

```bash
uv sync --frozen --python 3.12
.venv/bin/python -m pytest -W error --cov=pharmasignal.models --cov-report=term-missing
.venv/bin/ruff check src tests
# Use a NEW destination; reruns are additional disclosed test evaluations.
.venv/bin/python -m pharmasignal.models.train --data artifacts/data/seed42 --output artifacts/models/my_m3_run
```

This command reproduces the fixed policy, not a search for better test results. To load without retraining:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from pharmasignal.models.artifact import load_artifact, predict_records
model, metadata = load_artifact(Path('artifacts/models/m3_verified'))
# Training examples for a loading smoke test, not another test evaluation.
rows = [json.loads(line) for line in Path('artifacts/data/seed42/train.jsonl').read_text().splitlines()][:3]
print(predict_records(model, metadata, rows))
PY
```

Verified: **124 tests pass**, warnings treated as errors; **88% model-module coverage**, lint clean. Tests cover workflow ordering, hand-computed metrics, threshold policy, real model training, all ablation branches, deterministic repeat runs, fresh-process saved predictions, checksum/dependency failures and overwrite refusal. No model performance claim is encoded as an expected favorable test score.

M3 stops here. MLflow (M4), API/Docker (M5), drift (M6) and deployment work remain unimplemented. No commits/pushes or AWS resources.
