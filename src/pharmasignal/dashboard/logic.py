"""Pure helpers behind the Streamlit dashboard (kept free of Streamlit for testing)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError
from sklearn.metrics import precision_recall_curve, roc_curve

from pharmasignal.data.schema import FIELDS_BY_NAME, PREDICTORS
from pharmasignal.features.fusion import assemble_features, feature_names
from pharmasignal.serving.requests import PredictionRequest

RULE_LABELS = {
    "unnegated_symptom_count": "Narrative: symptom mentions (not negated)",
    "negated_symptom_count": "Narrative: negated symptom mentions",
    "history_marker_count": "Narrative: pre-existing / unchanged markers",
    "token_count": "Narrative: length (tokens)",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pretty_feature(name: str) -> str:
    """Turn a fitted pipeline column name into a client-readable label."""
    block, _, rest = name.partition("__")
    if block == "numeric":
        if rest.startswith("missingindicator_"):
            field = rest.removeprefix("missingindicator_")
            return f"{field.replace('_', ' ').capitalize()} is missing"
        return rest.replace("_", " ").capitalize()
    if block == "categorical":
        for field in ("sex", "treatment"):
            if rest.startswith(field + "_"):
                value = rest.removeprefix(field + "_")
                value = "missing" if value == "__missing__" else value
                return f"{field.capitalize()} = {value}"
        return rest
    if block == "text_rules":
        return RULE_LABELS.get(rest, rest)
    if block == "tfidf":
        return f'Narrative phrase "{rest}"'
    return name


def explain_record(model, record: dict, top: int = 8) -> pd.DataFrame:
    """Per-feature contribution (scaled value x coefficient) to the log-odds.

    Exact for a linear model; it explains the score, not real-world causality.
    """
    features = model.named_steps["features"]
    row = features.transform(assemble_features([record])).tocsr()
    coef = model.named_steps["classifier"].coef_[0]
    names = feature_names(features)
    contributions = [
        (names[col], float(value * coef[col]))
        for col, value in zip(row.indices, row.data, strict=True)
        if value * coef[col] != 0
    ]
    contributions.sort(key=lambda item: abs(item[1]), reverse=True)
    frame = pd.DataFrame(contributions[:top], columns=["feature", "contribution"])
    frame["label"] = frame["feature"].map(pretty_feature)
    frame["direction"] = np.where(frame["contribution"] > 0, "raises priority", "lowers priority")
    return frame


def _coerce(name: str, value):
    """CSV cell -> JSON-like value matching the strict request schema."""
    spec = FIELDS_BY_NAME[name]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str) and not value.strip() and spec.kind != "text":
        return None
    if spec.kind == "integer":
        number = float(value)
        if not number.is_integer():
            raise ValueError(f"{name} must be a whole number")
        return int(number)
    if spec.kind == "float":
        return float(value)
    return str(value).strip() if spec.kind == "categorical" else str(value)


def parse_upload(frame: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """Validate uploaded rows; return (valid records, per-row error table).

    Every predictor column is required; nullable fields may have empty cells.
    Extra columns (e.g. an ID) are ignored for scoring but kept by the caller.
    """
    missing = [name for name in PREDICTORS if name not in frame.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))
    valid, errors = [], []
    for position, (_, row) in enumerate(frame.iterrows()):
        try:
            record = {name: _coerce(name, row[name]) for name in PREDICTORS}
            PredictionRequest(**record)
        except (ValueError, TypeError, ValidationError) as exc:
            if isinstance(exc, ValidationError):
                fields = sorted({str(err["loc"][0]) for err in exc.errors() if err["loc"]})
                message = "Invalid value(s): " + ", ".join(fields)
            else:
                message = str(exc)
            errors.append({"row": position + 1, "error": message})
            continue
        valid.append({"row": position + 1, **record})
    return valid, pd.DataFrame(errors, columns=["row", "error"])


def score_rows(model, threshold: float, rows: list[dict]) -> pd.DataFrame:
    records = [{name: row[name] for name in PREDICTORS} for row in rows]
    scores = model.predict_proba(assemble_features(records))[:, 1]
    return pd.DataFrame(
        {
            "row": [row["row"] for row in rows],
            "priority_score": np.round(scores, 4),
            "flag_for_review": np.where(scores >= threshold, "YES", "no"),
        }
    )


def threshold_metrics(labels, scores, threshold: float) -> dict:
    y = np.asarray(labels, dtype=int)
    predicted = np.asarray(scores) >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flagged_share": float(np.mean(predicted)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def curves(labels, scores) -> dict:
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}


DRIFT_SECTIONS = {
    "numeric": ("Numeric field", "ks_statistic", "KS statistic"),
    "categorical": ("Category mix", "tv_distance", "TV distance"),
    "missingness": ("Missing-value rate", "absolute_difference", "Rate change"),
    "text_proxies": ("Narrative text", "ks_statistic", "KS statistic"),
    "vocabulary_presence": ("Symptom vocabulary", "tv_distance", "TV distance"),
}


def drift_table(report: dict) -> pd.DataFrame:
    rows = []
    for section, (group, key, metric) in DRIFT_SECTIONS.items():
        for item in report["sections"].get(section, []):
            value = item.get(key)
            rows.append(
                {
                    "check": group,
                    "feature": item["feature"],
                    "metric": metric,
                    "value": None if value is None else round(float(value), 4),
                    "status": item["status"],
                }
            )
    return pd.DataFrame(rows, columns=["check", "feature", "metric", "value", "status"])
