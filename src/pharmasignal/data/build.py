"""Build a new, immutable local dataset directory and provenance manifest.

Usage: python -m pharmasignal.data.build --output artifacts/data/seed42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from pharmasignal.data.generator import (
    GeneratorConfig,
    SignalCoefficients,
    assign_families,
    generate_dataset,
)
from pharmasignal.data.schema import GENERATOR_VERSION, PREDICTORS, SCHEMA_VERSION
from pharmasignal.data.splits import narrative_hash, validate_splits


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_config(path: Path) -> GeneratorConfig:
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("configuration must be a JSON object")
    if "coefficients" in values:
        values["coefficients"] = SignalCoefficients(**values["coefficients"])
    return GeneratorConfig(**values)


def _source_identity() -> dict:
    source = Path(__file__).parent
    hashes = {p.name: sha256(p.read_bytes()) for p in sorted(source.glob("*.py"))}
    root = source.parents[2]
    lock = root / "uv.lock"
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    return {
        "source_sha256": hashes,
        "git_revision": revision,
        "git_dirty": dirty,
        "lockfile_sha256": sha256(lock.read_bytes()) if lock.exists() else None,
    }


def build_dataset(output: Path, config: GeneratorConfig) -> dict:
    """Never overwrite an existing destination, including an empty directory.

    Write the manifest last as a completion marker. A failed write can leave
    an incomplete directory without a manifest; it must not be consumed.
    """
    if output.exists():
        raise FileExistsError(f"destination already exists: {output}; choose a new path")
    partitions = generate_dataset(config)
    assignments = assign_families(config)
    checks = validate_splits(partitions, assignments)
    payloads = {
        name: b"".join(canonical_json(row) for row in rows) for name, rows in partitions.items()
    }
    file_hashes = {f"{name}.jsonl": sha256(payload) for name, payload in payloads.items()}
    configuration = config.to_manifest_dict()
    manifest = {
        "synthetic": True,
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "dataset_id": sha256(canonical_json(file_hashes)),
        "config": configuration,
        "config_sha256": sha256(canonical_json(configuration)),
        "files_sha256": file_hashes,
        "family_assignments": assignments,
        "predictors": PREDICTORS,
        "checks": checks,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        **_source_identity(),
        "records": {
            name: [
                {
                    "report_id": row["report_id"],
                    "record_sha256": sha256(canonical_json(row)),
                    "narrative_sha256": narrative_hash(row["narrative"]),
                }
                for row in rows
            ]
            for name, rows in partitions.items()
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        (output / f"{name}.jsonl").write_bytes(payload)
    (output / "manifest.json").write_bytes(canonical_json(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    try:
        config = load_config(args.config) if args.config else GeneratorConfig()
        manifest = build_dataset(args.output, config)
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(1, f"Dataset build failed: {exc}\n")
    print(
        json.dumps({"dataset_id": manifest["dataset_id"], "checks": manifest["checks"]}, indent=2)
    )


if __name__ == "__main__":
    main()
