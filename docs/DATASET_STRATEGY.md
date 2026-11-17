# Dataset strategy and schema — proposed for M1

## Provenance and purpose

Use entirely synthetic data, generated locally from authored templates and a fixed NumPy random seed (`42`). No public dataset is downloaded, no restricted license is relied on, and no patient record or real narrative is copied. Fictional treatments `treatment_a`, `treatment_b`, and `treatment_c` deliberately avoid suggesting real drug safety relationships. No names, dates of birth, addresses, identifiers from real systems, or other PHI.

The generator will demonstrate joint structured/narrative information and uncertainty. It cannot validate clinical relationships. Even plausible units and ranges below are software constraints for the demonstration, not medical advice or real population distributions.

## Proposed generation process

1. Predefine a sufficient set of narrative template families and a seeded 60/20/20 family allocation. Aim for 3,000 reports with independent exposures and one row per fictional person. Use independent seeded generation streams so changing one partition does not unintentionally change the others.
2. Sample structured attributes, a latent symptom/severity state, and reporting circumstances. Render narratives with varied symptom combinations, explicit absence/negation and simple temporal context. Template wording/style is sampled independently of labels. Do not put target words such as “signal positive” or disposition such as “requires escalation” in the narrative.
3. Generate `signal` from a documented stochastic function of structured exposure factors and latent event state. Both positive and negative labels must be possible for similar observed inputs. The generator function is illustrative, not derived clinical knowledge. Sex must not be a hard-coded risk multiplier.
4. Inject small, documented missingness in selected structured inputs after label generation; this avoids intentionally encoding the label through missingness. Never omit target or narrative. Add ambiguity/irrelevant symptom mentions so a single token is not designed as a perfect label.
5. Store `report_id` and `template_family` for traceability and splitting only. Do not expose latent generation variables in model inputs. Record exact generator configuration and revision/version in a manifest.
6. Validate schema/ranges/uniqueness, group disjointness, exact narrative overlap, row counts and class counts. A fixed seed does not guarantee valid class balance. Fail clearly if an evaluation partition has one class or required checks fail; document any generator changes rather than hiding unsuccessful trials.

Proposed imbalance is approximately 20–35% positive, not a promised outcome or an estimate of adverse-event prevalence. Record actual prevalence when data exists. Held-out wording families make testing less trivial, but the same underlying generator still limits generalization. Shared legitimate symptom words across partitions are not leakage by themselves.

## Proposed record schema

| Field | Type / unit | Allowed values and missingness | Role |
|---|---|---|---|
| `report_id` | string | Unique generated identifier, e.g. `syn_000001`; required | Audit only |
| `template_family` | string | Authored family ID; required | Split/audit only |
| `age` | integer, years | 18–90 or null | Numeric |
| `sex` | categorical string | `female`, `male`, `other_or_unknown`, or null | Categorical; synthetic recorded field, not inferred identity |
| `treatment` | categorical string | `treatment_a`, `treatment_b`, `treatment_c`; required | Categorical |
| `dose_mg` | float, mg/day | >0 and <=500, or null | Numeric; fictional dosing only |
| `exposure_days` | integer, days | 1–365 or null | Numeric |
| `baseline_egfr` | float, mL/min/1.73 m² | 10–150 or null | Baseline numeric marker |
| `concomitant_med_count` | integer | 0–15 or null | Numeric |
| `narrative` | string | Nonblank after trimming, max 4,000 characters; required | Text |
| `signal` | integer | Exactly 0 or 1; required for labeled data | Synthetic review-priority target |

Reject infinity/NaN in interchange JSON, booleans in numeric fields, invalid categories and out-of-range values. Represent supported missing values as JSON null. Missing fields and explicitly null fields need consistent handling in dataset and API validators. No silent clipping or string-to-number coercion. Training files include all declared columns, with nullable cells where allowed. M5 inference accepts only predictor fields (required keys, null where allowed), not targets, template families or audit IDs; reject extra fields. These are planned validation rules; no validator exists in M0.

A fictional example of predictors (not a prediction or a generated dataset record):

```json
{
  "age": 64,
  "sex": "female",
  "treatment": "treatment_a",
  "dose_mg": 50.0,
  "exposure_days": 12,
  "baseline_egfr": 62.0,
  "concomitant_med_count": 3,
  "narrative": "Reports nausea after treatment started. Denies rash. Symptoms have persisted for two days."
}
```

All predictors are assumed available at the time of review. Baseline eGFR is pre-exposure; narrative describes information already reported at review time. Exclude future outcomes, adjudication decisions and post-review interventions. This is report triage, not prospective prediction before symptoms occur.

## Planned persisted outputs

- `artifacts/data/`: train, validation and test records plus a manifest.
- Manifest: synthetic flag, generator/schema version, seed and settings, template assignments, split report IDs, row/class counts, file SHA-256 hashes and normalized-content overlap checks.
- Configuration and generator source tracked in git; generated records ignored by git and regenerated locally. No fabricated “dataset version” before actual content exists.
- Labels remain in offline datasets only. Store unlabeled reference predictors separately for drift monitoring; do not accidentally include the target in drift features.

## M1 acceptance tests

Same configuration/seed produces identical records and split membership; different seeds change the sample. All generated data validates. Reject duplicate IDs, blank/oversized narrative, unsupported treatment, out-of-range numbers, booleans, nonfinite values and invalid labels. Valid nullable inputs pass. Assert disjoint IDs, template families and normalized exact narrative content; both labels present in every partition; no audit/target fields in declared predictor list. Tests must verify the generator, not merely hard-code expected favorable metrics.
