"""Dataset schema: single source of truth for record structure.

The synthetic generator produces records matching this schema, validation
rejects anything else, and later milestones derive the predictor list for
feature handling and the API request model from :data:`PREDICTORS`.

`report_id` and `template_family` are audit/split fields only: they are never
model inputs and never logged as predictors.
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "1.0.0"
GENERATOR_VERSION = "1.0.0"

TARGET = "signal"
AUDIT_FIELDS: tuple[str, ...] = ("report_id", "template_family")

#: Fields the model is allowed to see at inference time (M2+). Excludes the
#: target and audit-only fields by construction.
PREDICTORS: tuple[str, ...] = (
    "age",
    "sex",
    "treatment",
    "dose_mg",
    "exposure_days",
    "baseline_egfr",
    "concomitant_med_count",
    "narrative",
)

NARRATIVE_MAX_CHARS = 4000


@dataclass(frozen=True)
class FieldSpec:
    """One column of the dataset schema.

    `kind` drives validation:
    integer/float -> numeric range checks (booleans rejected),
    categorical   -> exact membership in `choices`,
    text          -> nonblank after trimming, bounded length,
    identifier    -> nonempty bounded string (audit only),
    target        -> exactly 0 or 1 (synthetic review-priority label).
    """

    name: str
    kind: str
    required: bool
    min_value: float | None = None
    max_value: float | None = None
    exclusive_min: bool = False
    choices: tuple[str, ...] | None = None
    max_length: int | None = None


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("report_id", "identifier", required=True, max_length=64),
    FieldSpec("template_family", "identifier", required=True, max_length=64),
    FieldSpec("age", "integer", required=False, min_value=18, max_value=90),
    FieldSpec(
        "sex",
        "categorical",
        required=False,
        choices=("female", "male", "other_or_unknown"),
    ),
    FieldSpec(
        "treatment",
        "categorical",
        required=True,
        choices=("treatment_a", "treatment_b", "treatment_c"),
    ),
    # Dose is strictly positive: a report describes an exposure that happened.
    FieldSpec(
        "dose_mg",
        "float",
        required=False,
        min_value=0.0,
        max_value=500.0,
        exclusive_min=True,
    ),
    FieldSpec("exposure_days", "integer", required=False, min_value=1, max_value=365),
    FieldSpec(
        "baseline_egfr",
        "float",
        required=False,
        min_value=10.0,
        max_value=150.0,
    ),
    FieldSpec("concomitant_med_count", "integer", required=False, min_value=0, max_value=15),
    FieldSpec("narrative", "text", required=True, max_length=NARRATIVE_MAX_CHARS),
    FieldSpec("signal", "target", required=True),
)

FIELDS_BY_NAME: dict[str, FieldSpec] = {f.name: f for f in FIELDS}
FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in FIELDS)

# Fields that may be null in the data (missingness is injected after label
# generation so that a null never encodes the label).
NULLABLE_FIELDS: frozenset[str] = frozenset(f.name for f in FIELDS if not f.required)

INTEGER_FIELDS: frozenset[str] = frozenset(f.name for f in FIELDS if f.kind == "integer")
FLOAT_FIELDS: frozenset[str] = frozenset(f.name for f in FIELDS if f.kind == "float")
