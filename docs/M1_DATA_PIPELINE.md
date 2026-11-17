# M1 — implemented data pipeline

## Reproduction

From the project root with uv installed:

```bash
uv sync --frozen --python 3.12
.venv/bin/python -m pytest --cov=pharmasignal.data --cov-report=term-missing
.venv/bin/ruff check src tests
.venv/bin/python -m pharmasignal.data.build --config configs/data.json --output artifacts/data/seed42
```

The initial `uv sync --python 3.12` succeeded and produced `.venv` and `uv.lock`. Python 3.12.14; core imports verified. The lockfile and source are local files, not committed or pushed. No dependency vulnerability audit has been performed.

The build command was executed successfully. If its destination already exists, choose a new path (for example `artifacts/data/repeat42`); never overwrite existing evidence. The manifest is written last. An interrupted write can leave an incomplete directory without a manifest; do not consume it. This is not a transactional object store.

## Actual dataset, not model results

| Split | Rows | Negative | Positive | Positive fraction |
|---|---:|---:|---:|---:|
| Train | 1800 | 1293 | 507 | 28.1667% |
| Validation | 600 | 435 | 165 | 27.5% |
| Test | 600 | 428 | 172 | 28.6667% |

Dataset ID: `31f04f04aa894358f111bb50e4bcdd393eb56be562f77929a6bf8a15937d3a9e`.

Default family assignments: train 02/03/06/07/08/09, validation 01/04, test 05/10. IDs, family IDs and normalized exact narratives have zero cross-split overlap. There are zero within-split normalized exact narrative duplicates in this default dataset; within-split duplicates are counted, not rejected by policy. Exact checks do not detect all near duplicates.

Counts and manifest hashes were checked against persisted JSONL files, not just in-memory generation. `git_revision` is null and `git_dirty` true because no commit exists. Manifest records synthetic/schema/generator flags, effective configuration, configuration hash, per-file/record/narrative hashes, source-module hashes, lockfile hash, Python/NumPy versions and split checks. Dataset identity is the SHA-256 of canonical JSON containing the split file hashes; it is content identity, not a clinical dataset identifier or an authenticity signature.

## Generation mechanics

`configs/data.json` stores the effective defaults. For each partition, separate seeded RNG streams handle structured values, latent state/labels, narrative rendering, missingness and family assignment. A seventh fixed-key stream allocates ten authored families 6/2/2 before sampling. Changing train size does not change validation/test; changing missingness or underreporting does not change labels.

Structured generation: age rounded normal(54,16), clipped 18–90; sex sampled with probabilities 0.49/0.49/0.02; fictional treatments uniformly sampled; dose uniform 10–400 rounded to one decimal; exposure uniform integer 1–365; eGFR normal(75,20) clipped 15–140 and rounded; medication count Poisson(4), capped at 15. Generator clipping constructs bounded synthetic values; validation of supplied records never clips them.

Define `d=(dose-205)/195`, `e=log(1+days)/log(366)`, `a=(age-54)/36`, `r=(75-egfr)/65`, `m=med_count/15`, and sigmoid(z)=1/(1+exp(-z)).

```text
p_event = sigmoid(-1.6 + 0.9*d + 0.7*e + 0.5*a + 0.8*r + 0.5*m
                  + 0.25*[treatment == treatment_b])
event ~ Bernoulli(p_event)
p_signal = sigmoid(-2.6 + 2.6*event + 0.6*d + 0.4*e + 0.5*r + 0.3*m)
signal ~ Bernoulli(p_signal)
```

All coefficients are arbitrary demonstration choices, not fitted or clinical estimates. Sex is absent from these functions; chance associations may still occur in finite samples. No continuous severity score is implemented. Latent `event`, probabilities and intermediate normalized values are not exported as predictors.

For event reports, 15% omit asserted symptoms; otherwise mention 1–3. Every report denies 1–3 other symptoms. For no-event reports, 25% mention an unchanged historical symptom. An unrelated unchanged joint-stiffness mention occurs with probability 0.4. Labels remain stochastic, so asserted symptoms are not equivalent to positive labels.

After labels and narratives exist, each nullable structured field is independently set to null at its configured probability: age .05, sex .04, dose .06, days .05, eGFR .12, medication count .08. This is MCAR by construction, not a claim that realized missingness is exactly independent in a finite sample. Narrative always includes duration and treatment: duration is therefore recoverable when its structured value is missing. This limits missing-data realism.

## Validation and preprocessing boundary

All record keys must be present; explicit null is permitted only for nullable fields. The schema's `required` attribute describes non-nullability, not whether the key may be omitted. Unknown keys, unsupported categories, numeric strings/booleans, fractional integers, nonfinite values, invalid ranges, blank/oversized text and invalid targets fail validation. IDs must be unique within and across splits. Reject one-class partitions without searching for a better seed.

M1 preprocessing is only deterministic audit normalization: Unicode NFKC, case folding and collapsed whitespace for duplicate hashes. Original narratives remain in the JSONL. No imputation, scaling, TF-IDF, spaCy feature extraction or model fitting occurs until M2/M3.

## Tests and limitations

78 tests passed; data-module statement coverage 93% (325 statements, 22 uncovered). Tests cover boundaries, required/null keys, malformed values, duplicate IDs, seed equality/change, partition resize isolation, text/missingness versus label independence, group/content overlap, degenerate labels, configuration failures, JSONL round trips, hashes and overwrite refusal. Coverage is not proof of correctness; this is not model evaluation.

Families differ mainly by wrapper and sentence order while sharing internal symptom syntax. The holdout is deliberately modest, not independent clinical language or an unseen hospital. All rows come from the same generator, with no repeated patients, temporal/site effects, reporting biases or expert labels. No model, MLflow run, API, Docker image, drift report, canary or AWS resource exists. M1 ends here for human review.
