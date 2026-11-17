"""Strict record and dataset validation; no coercion or silent repair."""

from __future__ import annotations

import math
from numbers import Integral, Real

from pharmasignal.data.schema import FIELDS, FIELDS_BY_NAME, FieldSpec


class ValidationError(ValueError):
    """Schema violations, with a short summary and complete errors attribute."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        shown = errors[:5]
        more = len(errors) - len(shown)
        suffix = f" (+{more} more)" if more > 0 else ""
        super().__init__("; ".join(shown) + suffix)


def _check_range(errors: list[str], spec: FieldSpec, value: Real) -> None:
    if spec.min_value is not None:
        below = value <= spec.min_value if spec.exclusive_min else value < spec.min_value
        if below:
            operator = ">" if spec.exclusive_min else ">="
            errors.append(f"{spec.name} must be {operator} {spec.min_value}")
    if spec.max_value is not None and value > spec.max_value:
        errors.append(f"{spec.name} must be <= {spec.max_value}")


def validate_record(record: dict) -> None:
    """Require every declared key, allowing None only on nullable fields.

    Dataset records have a fixed schema. A missing key is distinct from an
    explicitly null value. Numeric strings, bools, nonfinite numbers, unknown
    fields/categories and floating-point target labels are rejected. Text is
    inspected after trimming but is not modified here.
    """
    if not isinstance(record, dict):
        raise ValidationError(["record must be a dict"])
    errors: list[str] = []
    for name in record:
        if name not in FIELDS_BY_NAME:
            errors.append(f"unknown field: {name!r}")

    for spec in FIELDS:
        if spec.name not in record:
            errors.append(f"missing field: {spec.name}")
            continue
        value = record[spec.name]
        if value is None:
            if spec.required:
                errors.append(f"required field is null: {spec.name}")
            continue

        if spec.kind in ("integer", "float", "target"):
            expected = Integral if spec.kind in ("integer", "target") else Real
            if isinstance(value, bool) or not isinstance(value, expected):
                errors.append(
                    f"{spec.name} must be {'an integer' if expected is Integral else 'a number'}"
                )
                continue
            # Arbitrarily large Python integers are finite, but can overflow
            # float conversion. Check their bounds directly instead.
            if not isinstance(value, Integral) and not math.isfinite(value):
                errors.append(f"{spec.name} must be finite")
            elif spec.kind == "target" and value not in (0, 1):
                errors.append(f"{spec.name} must be exactly 0 or 1")
            else:
                _check_range(errors, spec, value)
        elif spec.kind == "categorical":
            if not isinstance(value, str) or value not in (spec.choices or ()):
                errors.append(f"{spec.name} must be one of {list(spec.choices or ())}")
        elif spec.kind in ("text", "identifier"):
            if not isinstance(value, str):
                errors.append(f"{spec.name} must be a string")
            else:
                if not value.strip():
                    errors.append(f"{spec.name} must be nonblank")
                if spec.max_length is not None and len(value) > spec.max_length:
                    errors.append(f"{spec.name} exceeds {spec.max_length} characters")
    if errors:
        raise ValidationError(errors)


def validate_records(records: list[dict]) -> None:
    """Validate a nonempty dataset and reject duplicate report identifiers."""
    if not isinstance(records, list) or not records:
        raise ValidationError(["records must be a nonempty list"])
    seen: set[str] = set()
    for index, record in enumerate(records):
        try:
            validate_record(record)
        except ValidationError as exc:
            raise ValidationError([f"row {index}: {error}" for error in exc.errors]) from exc
        report_id = record["report_id"]
        if report_id in seen:
            raise ValidationError([f"duplicate report_id: {report_id}"])
        seen.add(report_id)
