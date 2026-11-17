"""Reference-versus-batch drift report. No model scores: no performance claim.

Usage: python -m pharmasignal.monitoring.report --reference artifacts/data/seed42 \
  --batch artifacts/data/simulated/shifted --config configs/monitoring.json \
  --output artifacts/monitoring/shifted
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pharmasignal.data.schema import FIELDS_BY_NAME, NULLABLE_FIELDS
from pharmasignal.data.validation import validate_records
from pharmasignal.features.fusion import CATEGORICAL_FIELDS, NUMERIC_FIELDS
from pharmasignal.features.text import SYMPTOM_LEXICON, raw_tokens, rule_features
from pharmasignal.models.train import read_split
from pharmasignal.monitoring.config import DriftConfig
from pharmasignal.monitoring.metrics import (
    ALERT,
    INSUFFICIENT,
    UNDEFINED,
    categorical_comparison,
    missingness_comparison,
    numeric_comparison,
)
from pharmasignal.monitoring.simulate import manifest_summary
from pharmasignal.provenance import canonical_json, dependency_versions, sha256

CAVEATS = [
    "Feature drift is a change in inputs; it proves neither concept drift nor "
    "performance degradation.",
    "Comparison batches are simulated demonstrations with authored shifts, not "
    "observed production data.",
    "Alerts are informational; p-values are recorded but never gate status.",
    "No model scores are computed here, so no performance or calibration claim "
    "is possible from this report.",
    "No automatic retraining, rollback, or promotion is triggered by this report.",
]


def _numeric_values(records: list[dict], field: str) -> list[float]:
    return [float(r[field]) if r[field] is not None else float("nan") for r in records]


def _symptom_presence(records: list[dict]) -> dict[str, list[str]]:
    """One present/absent observation per record per fixed lexicon alias."""
    aliases = sorted(SYMPTOM_LEXICON, key=len, reverse=True)
    columns = {" ".join(alias): [] for alias in aliases}
    for record in records:
        tokens = raw_tokens(record["narrative"])
        present = set()
        position = 0
        while position < len(tokens):
            for alias in aliases:
                if tuple(tokens[position : position + len(alias)]) == alias:
                    present.add(" ".join(alias))
                    position += len(alias)
                    break
            else:
                position += 1
        for name in columns:
            columns[name].append("present" if name in present else "absent")
    return columns


def build_report(
    reference_records: list[dict],
    reference_manifest: dict,
    batch_records: list[dict],
    batch_manifest: dict,
    config: DriftConfig,
) -> dict:
    """Assemble all comparison sections; text/IDs never enter the output."""
    numeric = [
        numeric_comparison(
            field,
            _numeric_values(reference_records, field),
            _numeric_values(batch_records, field),
            config,
        )
        for field in NUMERIC_FIELDS
    ]
    categorical = [
        categorical_comparison(
            field,
            [r[field] for r in reference_records if r[field] is not None],
            [r[field] for r in batch_records if r[field] is not None],
            config,
        )
        for field in CATEGORICAL_FIELDS
    ]
    reference_length = [len(r["narrative"]) for r in reference_records]
    batch_length = [len(r["narrative"]) for r in batch_records]
    reference_rules = [rule_features(r["narrative"]) for r in reference_records]
    batch_rules = [rule_features(r["narrative"]) for r in batch_records]
    text = [
        numeric_comparison("narrative_length", reference_length, batch_length, config),
        numeric_comparison(
            "token_count", [r[3] for r in reference_rules], [r[3] for r in batch_rules], config
        ),
        numeric_comparison(
            "unnegated_symptom_count",
            [r[0] for r in reference_rules],
            [r[0] for r in batch_rules],
            config,
        ),
        numeric_comparison(
            "negated_symptom_count",
            [r[1] for r in reference_rules],
            [r[1] for r in batch_rules],
            config,
        ),
    ]
    reference_presence = _symptom_presence(reference_records)
    batch_presence = _symptom_presence(batch_records)
    vocabulary = [
        categorical_comparison(name, reference_presence[name], batch_presence[name], config)
        for name in reference_presence
    ]
    missingness = missingness_comparison(
        reference_records, batch_records, tuple(sorted(NULLABLE_FIELDS)), config
    )
    sections = {
        "numeric": numeric,
        "categorical": categorical,
        "text_proxies": text,
        "vocabulary_presence": vocabulary,
        "missingness": missingness,
    }
    counts = {status: 0 for status in (ALERT, INSUFFICIENT, UNDEFINED)}
    for rows in sections.values():
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
    alerts = counts.get(ALERT, 0)
    if counts.get(INSUFFICIENT, 0):
        overall = "insufficient_data"
    elif alerts:
        overall = "drift_detected"
    else:
        overall = "no_drift_detected"
    return {
        "kind": "reference_vs_batch_drift_report",
        "synthetic": True,
        "configuration": config.to_dict(),
        "identity": {
            "reference_dataset_id": reference_manifest["dataset_id"],
            "reference_manifest_sha256": sha256(canonical_json(reference_manifest)),
            "batch": manifest_summary(batch_manifest),
            "batch_manifest_sha256": sha256(canonical_json(batch_manifest)),
            "batch_sha256": batch_manifest["batch_sha256"],
            "configuration_sha256": sha256(canonical_json(config.to_dict())),
            "dependencies": dependency_versions(),
        },
        "sections": sections,
        "summary": {
            "alerts": alerts,
            "insufficient": counts.get(INSUFFICIENT, 0),
            "undefined": counts.get(UNDEFINED, 0),
            "overall_status": overall,
        },
        "caveats": CAVEATS,
        "note": "Offline demonstration on synthetic data; not a production monitoring service.",
    }


def _load_batch(directory: Path) -> tuple[list[dict], dict]:
    payload = (directory / "batch.jsonl").read_bytes()
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha256(payload) != manifest["batch_sha256"]:
        raise ValueError("batch checksum mismatch")
    records = [json.loads(line) for line in payload.splitlines()]
    if not records:
        raise ValueError("batch is empty")
    if manifest.get("synthetic") is not True or manifest.get("n") != len(records):
        raise ValueError("expected a synthetic batch with a matching row count")
    # Reuse structural validation, but permit new string-valued categories in
    # the monitor. Validate copies; original categories enter the comparisons.
    checked = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("batch records must be objects")
        row = dict(record)
        for field in CATEGORICAL_FIELDS:
            if isinstance(row.get(field), str) and row[field].strip():
                row[field] = FIELDS_BY_NAME[field].choices[0]
        checked.append(row)
    validate_records(checked)
    return records, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/monitoring.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(f"destination already exists: {args.output}")
        manifest = json.loads((args.reference / "manifest.json").read_text())
        reference = read_split(args.reference, manifest, "train")
        batch_records, batch_manifest = _load_batch(args.batch)
        report = build_report(
            reference, manifest, batch_records, batch_manifest, DriftConfig.from_json(args.config)
        )
        args.output.mkdir(parents=True, exist_ok=False)
        # Manifest-style completion marker: report.json written last.
        (args.output / "summary.json").write_bytes(canonical_json(report["summary"]))
        (args.output / "report.json").write_bytes(canonical_json(report))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Drift report failed: {exc}\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
