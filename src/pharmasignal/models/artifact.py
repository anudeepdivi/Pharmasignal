"""Trusted local model artifacts. Never load uploaded/untrusted joblib files."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from pharmasignal.data.schema import PREDICTORS, SCHEMA_VERSION
from pharmasignal.features.fusion import assemble_features, feature_names
from pharmasignal.provenance import canonical_json, dependency_versions, sha256

ARTIFACT_VERSION = "1.0"


def coefficient_summary(model, limit: int = 15) -> list[dict]:
    names = feature_names(model.named_steps["features"])
    weights = model.named_steps["classifier"].coef_[0]
    indices = sorted(set(np.argsort(weights)[:limit]) | set(np.argsort(weights)[-limit:]))
    return [{"feature": names[i], "coefficient": float(weights[i])} for i in indices]


def save_artifact(output: Path, model, report: dict) -> dict:
    """New destination only; write metadata last as completion marker."""
    output.mkdir(parents=True, exist_ok=False)
    model_path = output / "pipeline.joblib"
    joblib.dump(model, model_path)
    report_bytes = canonical_json(report)
    (output / "report.json").write_bytes(report_bytes)
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "predictors": list(PREDICTORS),
        "dependencies": dependency_versions(),
        "model_sha256": sha256(model_path.read_bytes()),
        "report_sha256": sha256(report_bytes),
        "dataset_id": report["dataset_id"],
        "variant": report["selection"]["variant"],
        "threshold": report["selection"]["threshold"],
    }
    metadata["model_version"] = "baseline-" + sha256(canonical_json(metadata))[:16]
    (output / "metadata.json").write_bytes(canonical_json(metadata))
    return metadata


def load_artifact(output: Path):
    """Integrity/compatibility checks precede unpickling; not authentication.

    The caller must trust the directory's source. An attacker can replace
    both a pickle and its checksum. Exact recorded dependency versions are
    required; migration needs a deliberate revalidation rather than a warning.
    """
    metadata = json.loads((output / "metadata.json").read_text())
    if (
        metadata["artifact_version"] != ARTIFACT_VERSION
        or metadata["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("incompatible artifact/schema version")
    if metadata["predictors"] != list(PREDICTORS):
        raise ValueError("incompatible predictor schema")
    if metadata["dependencies"] != dependency_versions():
        raise ValueError("incompatible dependency versions")
    threshold = metadata["threshold"]
    if type(threshold) not in (int, float) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("invalid artifact threshold")
    model_bytes = (output / "pipeline.joblib").read_bytes()
    if sha256(model_bytes) != metadata["model_sha256"]:
        raise ValueError("model checksum mismatch")
    if sha256((output / "report.json").read_bytes()) != metadata["report_sha256"]:
        raise ValueError("report checksum mismatch")
    identity = {k: v for k, v in metadata.items() if k != "model_version"}
    if metadata["model_version"] != "baseline-" + sha256(canonical_json(identity))[:16]:
        raise ValueError("metadata identity mismatch")
    model = joblib.load(output / "pipeline.joblib")
    if list(model.classes_) != [0, 1]:
        raise ValueError("incompatible model classes")
    return model, metadata


def predict_records(model, metadata: dict, records: list[dict]) -> list[dict]:
    scores = model.predict_proba(assemble_features(records))[:, 1]
    return [
        {
            "predicted_signal": int(score >= metadata["threshold"]),
            "signal_probability": float(score),
            "decision_threshold": metadata["threshold"],
            "model_version": metadata["model_version"],
            "schema_version": metadata["schema_version"],
        }
        for score in scores
    ]
