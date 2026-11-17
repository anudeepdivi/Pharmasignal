"""Unfitted sklearn feature pipeline; caller must fit on training records only."""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from pharmasignal.data.schema import PREDICTORS
from pharmasignal.data.validation import validate_record
from pharmasignal.features.text import NarrativeRuleFeatures, tokenize

NUMERIC_FIELDS = ("age", "dose_mg", "exposure_days", "baseline_egfr", "concomitant_med_count")
CATEGORICAL_FIELDS = ("sex", "treatment")
TEXT_FIELD = "narrative"
MISSING_CATEGORY = "__missing__"


def _to_numeric(values: pd.DataFrame) -> pd.DataFrame:
    """Handle pandas null representation, rejecting numeric strings and bools.

    Schema/range validation belongs at the records boundary; this guard also
    prevents silent casting of malformed direct DataFrame inputs.
    """
    for value in values.to_numpy(dtype=object).flat:
        if value is None or value is pd.NA:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("numeric block requires numbers or null, not strings/bools")
        if np.isinf(value):
            raise ValueError("numeric block cannot contain infinity")
    return values.astype("float64")


def _fill_categories(values: pd.DataFrame) -> pd.DataFrame:
    """Fixed null sentinel, not a learned category or string coercion."""
    for value in values.to_numpy(dtype=object).flat:
        if not pd.isna(value) and not isinstance(value, str):
            raise TypeError("categorical block requires strings or null")
    return values.astype(object).where(values.notna(), MISSING_CATEGORY)


def build_feature_pipeline(
    *,
    variant: str = "fused",
    min_df: int = 2,
    max_features: int = 3000,
) -> Pipeline:
    """Return a fresh, unfitted pipeline with explicit sparse CSR output.

    ``variant`` selects feature blocks: "structured" (numeric+categorical),
    "text" (rules+tfidf) or "fused" (all). "fused" is the primary model; the
    other two support M3 ablation comparisons on identical folds.

    Median fallback for all-missing numeric training columns is zero. Indicators
    are added only for columns missing during fit (sklearn's documented default).
    A new missing value in another column is imputed but gets no new indicator.
    An unseen category maps to zeros. Record validation still rejects codes
    outside the schema; the encoder supports valid categories absent in train.
    """
    if type(min_df) is not int or min_df < 1:
        raise ValueError("min_df must be a positive integer")
    if type(max_features) is not int or max_features < 1:
        raise ValueError("max_features must be a positive integer")
    if variant not in ("structured", "text", "fused"):
        raise ValueError("variant must be structured, text or fused")
    numeric = Pipeline(
        [
            ("cast", FunctionTransformer(_to_numeric, feature_names_out="one-to-one")),
            (
                "impute",
                SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True),
            ),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("fill", FunctionTransformer(_fill_categories, feature_names_out="one-to-one")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    rules = Pipeline([("extract", NarrativeRuleFeatures()), ("scale", StandardScaler())])
    tfidf = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    branches = []
    if variant in ("structured", "fused"):
        branches.extend(
            [
                ("numeric", numeric, list(NUMERIC_FIELDS)),
                ("categorical", categorical, list(CATEGORICAL_FIELDS)),
            ]
        )
    if variant in ("text", "fused"):
        branches.extend([("text_rules", rules, TEXT_FIELD), ("tfidf", tfidf, TEXT_FIELD)])
    fusion = ColumnTransformer(
        branches,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
    return Pipeline(
        [
            ("fusion", fusion),
            ("csr", FunctionTransformer(sparse.csr_matrix, feature_names_out="one-to-one")),
        ]
    )


def assemble_features(records: list[dict]) -> pd.DataFrame:
    """Project predictors and validate BEFORE pandas can coerce their types.

    Accept labeled dataset records or predictor-only records. Audit/target and
    other extra fields are deliberately excluded, not checked; use M1 dataset
    validation for labeled inputs. This is not the future API request validator.
    """
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty list")
    projected = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("each record must be a dict")
        values = {name: record[name] for name in PREDICTORS}  # Missing keys fail.
        validate_record(
            {"report_id": "feature_input", "template_family": "unused", "signal": 0, **values}
        )
        projected.append(values)
    return pd.DataFrame(projected, columns=list(PREDICTORS))


def feature_names(pipeline: Pipeline) -> list[str]:
    check_is_fitted(pipeline.named_steps["fusion"], "transformers_")
    return list(pipeline.get_feature_names_out())


def describe_fitted_features(pipeline: Pipeline) -> dict:
    names = feature_names(pipeline)
    return {
        "total": len(names),
        "blocks": {
            block: sum(name.startswith(f"{block}__") for name in names)
            for block in ("numeric", "categorical", "text_rules", "tfidf")
        },
    }
