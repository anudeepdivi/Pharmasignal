# M2 — NLP, structured features and fusion

Implemented and tested, with no classifier fitted. Source: `features/text.py`, `features/fusion.py`; tests: `tests/test_features.py`. Existing M1 data and generator were not changed.

## Pipeline

`assemble_features(records)` selects the eight schema predictors in canonical order, validates values before pandas coercion and returns a DataFrame. Labels/audit/extra fields are excluded; this helper intentionally does not validate them. Validate full labeled datasets with M1 validators first. The future API must reject extra fields separately.

`build_feature_pipeline()` returns an unfitted sklearn Pipeline:

1. **Numeric:** numeric type guard -> median imputation with missing indicators -> standard scaling. Five values, and an indicator only for each column with missing values during fit. An all-missing training column is retained with a zero fallback (not a clinical estimate). A newly missing column at inference is imputed without adding a new indicator. Scalers learn training means/variances only.
2. **Categorical:** replace null with `__missing__` without coercing other types -> sparse one-hot encoding. Categories are learned from training. A valid but unseen category yields zeros in that field's block. Missing also yields zeros if the sentinel was never seen at fit. Upstream schema validation still rejects unsupported category codes.
3. **Rules:** four deterministic counts -> standard scaling. Counts are `unnegated_symptom_count`, `negated_symptom_count`, `history_marker_count`, `token_count`. The scaler is learned even though extraction is stateless.
4. **TF-IDF:** spaCy tokenizer, lowercase, no stopword removal, word unigrams/bigrams, `min_df=2`, at most 3,000 features, sublinear term frequency, sklearn default smoothed IDF and L2 normalization. `min_df` and `max_features` are explicit constructor options, not tuned results.
5. **Fusion:** ColumnTransformer concatenates blocks; explicit final CSR conversion guarantees sparse output even for tiny/dense examples. Do not center the fused sparse matrix. Dense scaling happens only on the small numeric/rule blocks before concatenation.

Every learned operation fits training only. Tests inject a held-out-only token and compare the full fitted-state hash before/after transform, as well as training median statistics. A caller can still misuse `.fit` on held-out data; sklearn cannot enforce the experiment protocol for them.

## What spaCy actually does

`spacy.blank("en")` supplies tokenization, without a downloaded model, NER, parser or lemmatizer. A lazy process-local tokenizer is recreated after deserialization, not stored in the artifact. Custom functions/classes live in importable modules for serialization.

Two views of tokens serve distinct purposes:
- TF-IDF drops punctuation but preserves negation, numbers and temporal words. Bigrams can cross a sentence boundary after punctuation removal; this is a baseline limitation.
- Rules retain punctuation. Negation cues (`no`, `not`, `denies`, `without`, `never`) cover up to four following tokens, stopping at `. ! ? ; : ,` or `but/however/yet`. A new cue resets the window. Longest symptom match wins, so `skin rash` is not counted twice through the `rash` alias. Multiword matches cannot cross punctuation; a match is classified by its first token's scope.

History markers count `unchanged`, `pre-existing`, `long-standing`, recognizing spaCy's hyphen tokens. They do not alter the symptom scope or establish onset/causality. “Unnegated” was chosen instead of “adverse” or “new” to avoid a clinical interpretation.

Examples (unnegated, negated, history):
- `No nausea. Vomiting.` -> (1, 1, 0).
- `Denies nausea and skin rash. Reports vomiting.` -> (1, 2, 0).
- `Long-standing nausea, unchanged. Pre-existing fatigue.` -> (2, 0, 3).
- Known failure: `Not only nausea but vomiting` -> (1, 1, 0), incorrectly negating nausea.
- Fixed-window failure: `No one two three four fever` -> (1, 0, 0).
- Stopping at a comma avoids some over-scoping but under-negates lists such as `No nausea, fever`.

The lexicon is authored for this generator, not independently clinically curated. Rule tests document exact behavior, including known failures, not clinical accuracy.

## Verified results

Full suite: **107 tests passed** with `-W error` (78 M1 + 29 M2 parameterized cases). Feature-module statement coverage **98%**: 128 statements, two uncovered. Text module 100%, fusion 97%. Ruff passed. Fresh-process joblib round trip reproduces transforms and feature names. This is a feature artifact compatibility test, not model loading or an inference service.

On the unchanged saved seed-42 dataset, fit on train only:

| Split | Shape | Nonzeros | Density |
|---|---|---:|---:|
| Train | 1800 × 1321 | 93744 | 0.039425 |
| Validation | 600 × 1321 | 25982 | 0.032781 |
| Test | 600 × 1321 | 25486 | 0.032155 |

Final refit check: independent fits on identical training data produced identical transformed matrices for all three partitions (maximum difference 0). An initially over-strict bitwise comparison between `fit_transform(train)` and subsequent `transform(train)` failed: maximum difference was 2.220446049250313e-16, while fitted state was unchanged. They agree with explicit absolute/relative tolerance 1e-12. This is floating-point numerical agreement, not a claim of universal bitwise reproducibility.

Composition: **10 numeric + 7 categorical + 4 rules + 1300 TF-IDF = 1321** columns. Finite CSR outputs and unchanged fitted-state hash verified. Dataset ID: `31f04f04aa894358f111bb50e4bcdd393eb56be562f77929a6bf8a15937d3a9e`. Shapes are structural checks, not predictive performance. No feature setting was selected using test performance; no classifier exists.

## Exact reproduction

From the project root, after M1 dataset creation:

```bash
uv sync --frozen --python 3.12
.venv/bin/python -m pytest -W error --cov=pharmasignal.features --cov-report=term-missing
.venv/bin/ruff check src tests
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import joblib
import numpy as np
from scipy import sparse
from pharmasignal.data.build import sha256
from pharmasignal.data.splits import validate_splits
from pharmasignal.features.fusion import (
    assemble_features, build_feature_pipeline, describe_fitted_features,
)
root = Path('artifacts/data/seed42')
manifest = json.loads((root / 'manifest.json').read_text())
data = {}
for split in ('train', 'validation', 'test'):
    payload = (root / f'{split}.jsonl').read_bytes()
    assert sha256(payload) == manifest['files_sha256'][f'{split}.jsonl']
    data[split] = [json.loads(line) for line in payload.splitlines()]
validate_splits(data, manifest['family_assignments'])
features = build_feature_pipeline().fit(assemble_features(data['train']))
state = joblib.hash(features)
print(describe_fitted_features(features))
for split, rows in data.items():
    matrix = features.transform(assemble_features(rows))
    assert sparse.isspmatrix_csr(matrix) and np.isfinite(matrix.data).all()
    assert joblib.hash(features) == state
    print(split, matrix.shape, matrix.nnz)
PY
```

Blank narrative is rejected by assembly; low-level rule extraction accepts empty strings as zero counts for unit testing. Punctuation-only training text or too-small corpora can yield an empty vocabulary: fail explicitly rather than fitting on held-out text. Low-level DataFrame transforms are not a substitute for schema validation. Empty prediction batches are not supported by the public assembly helper.

## Milestone boundary

No trained classifier, evaluation metrics, MLflow run, API, Docker image, drift monitoring, AWS resources or canary deployment. Feature fits in this milestone were smoke tests; a versioned complete model artifact belongs to M3. No commits or pushes. Stop for human review.
