# PharmaSignal architecture — M0 design

Status: proposed architecture, not a running system. See [technical details](TECHNICAL_DETAILS.md#milestone-status) for milestone status.

## Scope and prediction unit

One input is one fictional treatment-exposure report at review time. The binary output is a synthetic report-review priority label. It is not drug-event causality, an incidence estimate, or a population-level safety signal. The historical project title uses “signal”; API documentation must explain this narrower meaning wherever `predicted_signal` is exposed.

No patient care, real patient data, external NLP service, or cloud resources are involved. A true pharmacovigilance system would require expert adjudication, case deduplication, terminology mapping, bias controls, and aggregate evidence beyond this classifier.

## Planned flow

```text
Seeded synthetic generator -> validated records + provenance manifest
  -> deterministic group-disjoint train / validation / test split
       |
       +-> training-only numerical imputation + scaling
       +-> training-only categorical imputation + one-hot encoding
       +-> spaCy tokenization + small rule-based symptom/negation features
       +-> training-only TF-IDF vocabulary and inverse document frequencies
                 |
          sparse feature concatenation (ColumnTransformer / Pipeline)
                 |
          logistic regression
                 |
       validation: threshold choice and limited model comparison
       test: frozen-model final evaluation and error analysis
                 |
       MLflow run + trusted pipeline artifact + version metadata
                 |
       FastAPI -> Docker (local)
                 |
       batch drift comparison against training reference
```

## Component boundaries

- `data/`: generator, schema validation, split manifest; never model fitting.
- `features/`: deterministic spaCy transformer and sklearn-compatible feature assembly. Use `spacy.blank("en")` tokenization and explicit rules; do not claim a trained clinical NER model or linguistic parsing.
- `models/`: train, evaluate, serialize entire preprocessing + estimator pipeline.
- `serving/`: typed requests, shared validation, load a trusted artifact at startup, versioned prediction response. No training at startup.
- `monitoring/`: offline reference-versus-batch reports. No automatic retraining or promotion.
- `configs/`: checked-in configuration; record seed, settings and configuration checksum.
- `scripts/`: thin entry points only where useful; no parallel implementation of core logic.
- `tests/`: unit tests and a small deterministic integration dataset.

Empty package directories in M0 reserve these boundaries; they do not implement them.

## Features and estimator

Numerical fields: age, dose, exposure duration, baseline renal-function marker, concomitant medication count. Median imputation (with missingness indicators) then standard scaling, fit on training only. Explicitly handle an all-missing training column. Missingness at deployment is monitored separately from values.

Categorical fields: sex and fictional treatment. Use a missing token and one-hot encoding with unknown-category handling. Schema validation rejects unsupported treatment codes; the encoder guard is not permission to silently accept arbitrary inputs.

Text: lowercase tokens and whitespace normalization, retain negation and temporal words. Word unigrams/bigrams in TF-IDF; limit vocabulary through configuration. Missing/blank narrative is rejected. Do not remove “no”, “not”, or “denies”. Add small symptom-mention and scoped-negation counts using explicit rules. These are imperfect linguistic proxies, not validated clinical entities.

Concatenate sparse text/structured/rule features with sklearn. Logistic regression is a practical high-dimensional sparse baseline. Select a sparse-compatible solver and fixed seed at M3. No PyTorch, transformer, or GPU is needed. Coefficients and ablations help explain behavior but do not demonstrate causality.

## Evaluation and leakage controls

- Design target: 3,000 independent fictional reports; 60/20/20 train/validation/test by report-template family, not simple row splitting.
- Keep report IDs, labels, generator latent variables and template-family identifiers out of predictors.
- Reserve disjoint template families for each partition; hash record and normalized-narrative content to detect accidental overlap. Record realized sizes and class balance. Fail visibly on one-class partitions; never silently regenerate favorable test data.
- Fit every learned preprocessing operation on training only. No vocabulary fitting on validation/test.
- Compare structured-only, text-only and fused baselines on validation. Keep comparisons small and record them. Do not inspect test scores to select a model.
- Report threshold 0.5 as a baseline. Choose any operational threshold using validation only. A demonstration recall objective (for example 0.80) is arbitrary, not clinically approved; record achieved precision and whether that objective is feasible. Freeze threshold before final test evaluation.
- Precision, recall, F1, confusion matrix (true rows/predicted columns, order 0 then 1), ROC-AUC from scores when both classes exist, and PR-AUC/average precision for imbalance. Define zero-division handling explicitly and mark undefined AUC as unavailable, not zero.
- False negatives may delay review of concerning reports; false positives increase reviewer workload. Actual costs cannot be quantified from synthetic data. Probability is model output for this generator, not clinical risk; calibration is not assumed.
- Inspect false positives/negatives and performance by template family and selected structured subgroups, reporting sample counts and avoiding fairness/clinical claims from small synthetic strata.

## Artifacts and reproducibility

Save the entire fitted sklearn pipeline with joblib, plus JSON metadata: model version, training run ID, data hash, configuration hash, feature/schema version, seed, selected threshold, dependency versions and source revision when available. Do not invent a git commit for an uncommitted tree; record that state. Custom transformers must live in importable package modules.

MLflow M4 will record parameters, split/configuration manifests, metrics with train/validation/test prefixes and model artifacts. Local tracking is sufficient; no registry or remote server required. The initial model artifact in M3 remains useful without MLflow.

Joblib/pickle loading can execute code: load only locally produced, trusted artifacts, never arbitrary uploaded model files. Checksums detect changes but do not prove source authenticity. Keep narratives out of default service logs. No credentials in code/images.

Use Python 3.12 as the target runtime, a checked-in `uv.lock` after a successful resolve and a frozen install for reproducibility. M0 has only dependency constraints, not a lockfile. Docker M5 should use the same Python/dependency set, run non-root and include a trusted model explicitly. Do not claim a build without running it.

## Drift: implemented in M6

Implemented in [docs/M6_DRIFT.md](M6_DRIFT.md): SciPy/NumPy distribution comparisons rather than adding Evidently solely for a resume keyword. Compare training-reference versus supplied/simulated batch distributions for numeric values, categorical proportions, missingness, text length and fixed-vocabulary coverage. Numeric KS statistics plus an effect-size measure and categorical total-variation distance are understandable starting points. Define minimum sample sizes and alert thresholds in configuration, show sample counts, handle constant/all-missing columns and unseen categories, and expose insufficient-data status. Multiple tests and large samples can generate nuisance alerts; p-values alone are not promotion criteria.

Data drift is a change in inputs. Concept drift is a change in the relationship between input and target. Performance degradation requires appropriate outcomes/labels and evaluation. Feature drift does not prove either of the latter. Simulate an explicitly labeled unchanged batch and a shifted batch; shifts are demonstrations, not observed incidents. Drift does not automatically trigger rollback.

## Versions, canary and AWS: designed in M7

Designed in [docs/M7_PRODUCTION.md](M7_PRODUCTION.md) (M7 is a design milestone; nothing provisioned). Version artifact and response metadata from M3/M5 onward. The design keeps a stable and candidate container behind an ALB, with ECS Fargate as the compute service; ECR for images, S3 for versioned artifacts, CloudWatch for operational metrics, scoped IAM roles and managed secrets if needed. TLS, access control, retention, trusted artifact supply and private service networking are requirements for real deployment, not implemented features. A single VM is simpler but requires more host management; Fargate reduces that work at recurring cost. No infrastructure is provisioned.

A canary design sends a small, explicitly configured traffic fraction to a candidate version, logs version-tagged operational outcomes without raw narratives, and retains the stable target for rollback. Gates (schema/artifact compatibility, offline quality, image scan) precede promotion; feature drift is informational, never a gate. Operational errors and latency are monitored immediately; delayed adjudicated labels are required to compare predictive quality. Small canaries have weak statistical power. No automated “safe promotion” claim is made from feature drift or unlabeled traffic. A local router remains optional, unimplemented, and cannot stand in for an AWS rollout; no rollback drill has been run.
