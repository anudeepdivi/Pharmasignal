"""Strict predictor-only HTTP schema derived from the dataset field definitions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from pharmasignal.data.schema import FIELDS_BY_NAME, PREDICTORS


class _RequestBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    @field_validator("narrative", check_fields=False)
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("narrative must be nonblank")
        return value


def _request_fields() -> dict:
    fields = {}
    for name in PREDICTORS:
        spec = FIELDS_BY_NAME[name]
        constraints = {}
        if spec.kind in ("integer", "float"):
            annotation = int if spec.kind == "integer" else float
            constraints["gt" if spec.exclusive_min else "ge"] = spec.min_value
            constraints["le"] = spec.max_value
        elif spec.kind == "categorical":
            annotation = Literal[spec.choices]
        else:
            annotation = str
            constraints["max_length"] = spec.max_length
        if not spec.required:
            annotation = annotation | None
        # Nullable is not optional: all eight keys must be supplied, as in M1.
        fields[name] = (annotation, Field(..., **constraints))
    return fields


PredictionRequest = create_model("PredictionRequest", __base__=_RequestBase, **_request_fields())


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    predicted_signal: Literal[0, 1]
    signal_probability: float = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    model_version: str
    schema_version: str
