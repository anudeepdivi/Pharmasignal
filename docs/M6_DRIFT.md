# M6 — Offline drift monitoring (simulated batches)

Status: implemented and verified locally on synthetic data. Both demonstration reports were
generated with the exact commands below. No model scores are computed anywhere in this
milestone, so no performance, calibration or clinical claim is possible from these reports.
This document records observed statistics only.

## 1. What was built

A transparent SciPy/NumPy reference-versus-batch comparison (no Evidently, no new runtime
dependency) with four parts:

| Module | Role |
|---|---|
| `src/pharmasignal/monitoring/config.py` | Frozen `DriftConfig`: `min_samples=30`, `ks_alert=0.15`, `effect_alert=0.5`, `tv_alert=0.2`, `missingness_alert=0.15`; JSON loader rejects unknown keys and invalid values |
| `src/pharmasignal/monitoring/metrics.py` | Pure comparisons: numeric KS + standardized mean difference, categorical total-variation distance, per-field missingness rates |
| `src/pharmasignal/monitoring/simulate.py` | Labeled unchanged/shifted batches (fixed seeds 2025/2026), JSONL + manifest writer, CLI |
| `src/pharmasignal/monitoring/report.py` | Report assembly and CLI; batch loading with checksum verification |

Reference = the training split (`train.jsonl`, 1,800 rows) of dataset
`31f04f04…37d3a9e`, loaded through `read_split` (the same loader training used). Batches are
simulated demonstration draws from the same M1 generator (different seeds: 2025 unchanged,
2026 shifted); the shifted scenario applies documented transforms. No model artifact, no
scores, no performance statement exists in this milestone.

## 2. Methodology

### Numeric features (`numeric`, `text_proxies` sections)

For each numeric field (and each text proxy), non-null values are compared with:

- **Two-sample Kolmogorov–Smirnov statistic** (`scipy.stats.ks_2samp`, `method="asymp"`):
  the maximum absolute vertical distance between the two empirical CDFs, in `[0, 1]`.
  `method="asymp"` is used deliberately: scipy's exact-attempt fallback emits a
  `RuntimeWarning` at moderate n, and the suite runs with warnings as errors. The statistic
  is identical either way; only small-sample p-value accuracy differs, and the p-value
  never gates status.
- **Standardized mean difference (effect size)**: `|mean(reference) − mean(batch)| /
  sqrt((var(reference) + var(batch)) / 2)` — the root mean of the two sample variances,
  which equals the classic pooled standard deviation at equal sample sizes. The absolute
  value is deliberate: drift in either direction is drift; a signed statistic would
  silently suppress upward shifts (this was a real bug found during development and fixed).
  When both variances are zero the effect size is `null` (status can still be `alert` via
  KS) rather than implying zero effect.
- The KS p-value is recorded for transparency only. It is never a gate: at n=1,800 vs 600
  even trivial differences become "significant". The recorded unchanged batch demonstrates
  this: `token_count` has p = 0.0014 but KS statistic 0.0894 (< 0.15) and stays `ok`.

### Categorical features (`categorical`, `vocabulary_presence`)

- **Total-variation distance** over the union of observed categories:
  `TV = ½ Σ_c |p_ref(c) − p_bat(c)|`, in `[0, 1]`.
- **Unseen categories** (present in batch, absent from reference) are flagged in
  `unseen_categories` and included in the distance — the monitor is deliberately permissive
  about new categories where the M5 API schema would reject them.
- Nulls are excluded per field before counting (sample counts show the reduction).

### Fixed-vocabulary coverage (`vocabulary_presence`)

For each of the 9 symptom aliases in `SYMPTOM_LEXICON` (longest match first), each record
contributes one `present`/`absent` observation; each alias is compared as a categorical
TV-distance row. This is a coverage proxy over a fixed vocabulary, not learned or clinical
NER. The alias `rash` is always beaten by the longer `skin rash` match, so it is constant
(`absent` on both sides, TV 0.0) — a degenerate column handled without crashing.

### Missingness (`missingness`)

For each of the 6 nullable fields: null rate on each side and the absolute difference.

### Statuses

Every row carries one status decided by pre-registered thresholds:

- `ok` — no threshold crossed.
- `alert` — `ks_statistic ≥ 0.15` **or** `effect_size ≥ 0.5` (numeric); `tv_distance ≥ 0.2`
  (categorical); `absolute_difference ≥ 0.15` (missingness).
- `insufficient_data` — either side has fewer than `min_samples` (30) non-null observations;
  no alert is claimed either way.
- `undefined_constant` — effect size undefined (both sides constant); KS can still alert.

Overall status precedence: `insufficient_data` > `drift_detected` > `no_drift_detected`.

### Threshold rationale (pre-registered)

`ks_alert=0.15`: a visible ECDF overlap gap, far above sampling noise at these sample sizes.
`effect_alert=0.5`: half a pooled standard deviation — a practically visible shift, not
noise. `tv_alert=0.2`: a fifth of probability mass moved between categories. 
`missingness_alert=0.15`: an absolute 15-point null-rate change. `min_samples=30`: minimum
size at which these fixed thresholds are meaningful. All five live in
`configs/monitoring.json` and were fixed before the recorded runs below; nothing was
retuned afterwards. Known and accepted: fixed thresholds produce nuisance alerts at small
samples — a development fixture of 80 rows vs 120 did alert on `age` (KS 0.2248). Tests use
larger draws where the deterministic outcome matches design intent; the real 600-row
unchanged batch is clean at these thresholds (observed, not engineered).

## 3. Simulated scenarios (demonstrations, not incidents)

Both batches: n=600, generated by the untouched M1 generator with scenario seeds, then
(schema-bounded) post-transforms for `shifted`. Provenance in every batch manifest and every
report:

| Transform | Detail |
|---|---|
| `dose_mg` | +90 (additive) |
| `baseline_egfr` | ×0.7 multiplicative (generator floor 15 → ≥10.5, above schema minimum 10) |
| `concomitant_med_count` | +2 |
| `exposure_days` | ×1.5, capped at schema max 365 |
| `baseline_egfr` missingness | extra 0.35 null probability |
| `sex` remix | (0.8, 0.1, 0.1) toward `female` |

The unchanged scenario applies no transforms: same generator, same distributions, new seed
(2025 ≠ 42). Every batch carries a manifest (`synthetic: true`, scenario, seed, n,
transforms, `batch_sha256` over the exact JSONL bytes, generator config) written after the
data file. Reports embed the batch manifest summary plus `batch_manifest_sha256`,
`batch_sha256`, `configuration_sha256`, reference `dataset_id` and manifest hash, and
dependency versions. Batches are labeled demonstrations: the note "Simulated demonstration
batch, not an observed production sample" appears in the manifest and every report.

## 4. Recorded runs (observed results)

```bash
.venv/bin/python -m pharmasignal.monitoring.simulate --scenario unchanged --n 600 \
  --output artifacts/data/simulated/unchanged
# Wrote 600 simulated rows; sha256=cfdf33111560ad7235d142a215014255bb728b1885fc77ffe3e1f6594e817578
.venv/bin/python -m pharmasignal.monitoring.simulate --scenario shifted --n 600 \
  --output artifacts/data/simulated/shifted
# Wrote 600 simulated rows; sha256=2fa2c5d034485069b6ffe17ef230b6805f2f253d581b302141a77b18b8399f20
.venv/bin/python -m pharmasignal.monitoring.report --reference artifacts/data/seed42 \
  --batch artifacts/data/simulated/unchanged --config configs/monitoring.json \
  --output artifacts/monitoring/unchanged
.venv/bin/python -m pharmasignal.monitoring.report --reference artifacts/data/seed42 \
  --batch artifacts/data/simulated/shifted --config configs/monitoring.json \
  --output artifacts/monitoring/shifted
```

**Unchanged batch → `no_drift_detected`** (0 alerts, 0 insufficient, 0 undefined).
Selected observed rows (reference n=1,800; batch n=600):

| Feature | Statistic | Effect / TV | p-value (not a gate) | Status |
|---|---|---|---|---|
| age | KS 0.0449 | 0.018 | 0.347 | ok |
| dose_mg | KS 0.0601 | 0.096 | 0.090 | ok |
| baseline_egfr | KS 0.0476 | 0.081 | 0.323 | ok |
| sex (TV) | 0.0208 | — | — | ok |
| token_count | KS 0.0894 | 0.152 | 0.0014 | ok — significant p-value, below statistic threshold |
| narrative_length | KS 0.0711 | 0.157 | 0.020 | ok |
| vocabulary (max) | dizziness TV 0.0428 | — | — | ok |

All 30 numeric/proxy rows, 2 categorical rows, 9 vocabulary rows and 6 missingness rows are
`ok`. Largest missingness difference: `baseline_egfr` 0.0217 (10.7% → 12.8%).

**Shifted batch → `drift_detected`** (6 alerts, 0 insufficient, 0 undefined). Exactly the
six transformed dimensions alert; nothing else does:

| Feature | Statistic | Effect / TV | Alert via |
|---|---|---|---|
| dose_mg | KS 0.2827 | 0.875 | KS + effect |
| exposure_days | KS 0.3392 | 0.511 | KS + effect |
| baseline_egfr | KS 0.5223 | 1.335 (n_batch 350 after extra missingness) | KS + effect |
| concomitant_med_count | KS 0.3668 | 0.948 | KS + effect |
| sex | TV 0.3896 (female 0.474→0.785, male 0.505→0.115) | — | TV |
| baseline_egfr missingness | diff 0.310 (10.7% → 41.7%) | — | missingness |

Untouched dimensions correctly stay `ok`: `treatment` TV 0.0306, `age` KS 0.0263, all four
text proxies (narratives were not transformed; largest KS 0.0583), all 9 vocabulary rows.
Recorded p-values on alerted fields are tiny (e.g. dose_mg 3.4e-30) — recorded, never used.

## 5. Interpretation: three kinds of drift

- **Data drift** (what this report measures): inputs change. Detected above.
- **Concept drift**: the input→target relationship changes. Not measurable here — no labels
  are consumed, and the synthetic generator's label formula is outside the report.
- **Performance degradation**: predictive quality falls. Requires labeled evaluation. This
  report computes no model scores, so **no performance or calibration claim is possible
  from it** — the absence of model artifacts is explicit in the report kind and caveats.

A drift alert is an investigation trigger, not proof the model is failing. Conversely, no
alert does not prove the model is performing well. Multiple comparisons across ~30 monitored
rows mean some false alerts are expected over repeated batches; thresholds trade that against
sensitivity. There is no automatic retraining, rollback or promotion wired to this report.

## 6. Robustness handling (required acceptance cases)

- **Constant columns**: effect size `null` + `undefined_constant` when both sides are
  constant; KS still gates (a shifted constant column alerts via KS — tested).
- **All-missing/empty inputs**: `insufficient_data`; missingness rows use `null` rates and
  `null` difference rather than crashing (tested).
- **Unseen categories**: flagged and included in TV distance (tested: TV 0.4 with unseen `c`).
- **Sub-minimum samples**: `insufficient_data` at `min_samples=30` (600-row batches are
  comfortably above it; sample counts are printed on every row).
- **Batch integrity**: `_load_batch` verifies `batch_sha256` over the exact bytes, requires
  `synthetic: true` and a matching row count, and reuses the strict structural validator —
  with unseen string categories temporarily mapped to a valid choice so the monitor stays
  permissive where the API is strict. Tampered bytes are rejected (tested).

## 7. Report structure

`report.json` sections: `numeric` (5), `categorical` (2), `text_proxies` (4),
`vocabulary_presence` (9), `missingness` (6) — 26 monitored rows. `summary.json` is written
first, `report.json` last as the completion marker; the CLI refuses to overwrite. Identity
block carries both manifests' hashes, the configuration hash and dependency versions. No
narrative text, report ID or label value appears anywhere in a report (asserted by tests).
A fixed caveats block states: feature drift proves neither concept drift nor performance
degradation; batches are simulated demonstrations; alerts are informational; no scores → no
performance claim; no automatic retraining/rollback/promotion.

## 8. Tests and verification

35 tests in `tests/test_monitoring.py` (suite total **200 passed**, warnings as errors, 93%
coverage; `metrics.py` and `config.py` 100%, `report.py`/`simulate.py` 99% — only the
`__main__` guards uncovered):

- Identical distributions → no alert; shifted → alert (both directions; effect size is a
  magnitude); nulls excluded from counts; constant → `undefined_constant`; small samples →
  `insufficient_data` with no claimed alert; p-value recorded but never gating (large-n
  tiny shift stays `ok` with p=0.0014-level behavior; +200/−200 shifts alert identically).
- Categorical TV 0.4 with unseen category flagged; identical categories `ok`.
- Missingness rates/difference, configurable threshold (alert at default 0.15, `ok` at 0.4,
  `insufficient_data` below `min_samples`), all-missing and empty inputs.
- Configuration validation: non-positive/non-finite/out-of-range thresholds, bool for int,
  non-numeric types all rejected; JSON round-trip; unknown keys rejected.
- Vocabulary presence: longest-match wins, absence counted, one observation per record.
- Unchanged scenario (600 vs 300-row fixture) → zero alerts; shifted → alerts exactly on
  `dose_mg`, `baseline_egfr`, `concomitant_med_count`, `exposure_days`, `sex` (6 alert rows).
- Report determinism: two independent builds produce identical canonical bytes; no
  reference narrative or report ID appears anywhere in the report JSON.
- Label-flipping a batch leaves sections unchanged (labels are not inputs to the monitor).
- Batch loader: checksum mismatch, empty batch, row-count mismatch, non-synthetic manifest,
  non-numeric values, non-object rows all rejected; `write_batch` refuses overwrite.
- Both CLIs: success output, failure exit codes, overwrite refusal preserving prior bytes.

## 9. Limitations

- Simulated batches only. No observed production traffic exists in this project; a real
  deployment must define batch composition, arrival windows and reference refresh.
- Fixed thresholds, no seasonality/segmentation, no per-family breakdown; the monitor is
  aggregate-level. Multiple-comparison nuisance alerts are expected at small samples
  (demonstrated at 80-vs-120 during development).
- KS/SMD/TV are univariate; correlations between fields are not compared (a joint
  distribution can shift while every marginal looks fine).
- Text proxies are length/token/rule-count and a fixed 9-alias vocabulary — not semantic
  drift detection. New phrasings with familiar symptoms are invisible.
- The monitor reads batch records as JSONL with structural validation, but it does not and
  cannot verify that a supplied batch was drawn from any particular population.
- No storage of historical reports, no trend/escalation logic, no notification surface.
- No performance monitoring at all: that requires labels and is out of M6 scope.
- Evidently remains an unimplemented alternative; no claim of its usage is made.

## 10. Commands

```bash
# Simulate a labeled demonstration batch (no-overwrite directory; manifest last)
.venv/bin/python -m pharmasignal.monitoring.simulate --scenario unchanged --n 600 \
  --output artifacts/data/simulated/unchanged
# Build a drift report against the training reference
.venv/bin/python -m pharmasignal.monitoring.report --reference artifacts/data/seed42 \
  --batch artifacts/data/simulated/shifted --config configs/monitoring.json \
  --output artifacts/monitoring/shifted
```

Existing outputs are refused rather than overwritten; choose new directories. The recorded
reports live in `artifacts/monitoring/unchanged/` and `artifacts/monitoring/shifted/`.
