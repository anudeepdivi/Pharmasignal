"""Labeled simulated batches: demonstrations of distribution change, not incidents."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pharmasignal.data.generator import GeneratorConfig, assign_families, generate_partition
from pharmasignal.data.validation import validate_records

#: Fixed scenario seeds so both simulations are reproducible and differ from
#: the reference seed (42): fresh draws, same generator distributions.
SCENARIO_SEED = {"unchanged": 2025, "shifted": 2026}

#: Documented post-generation transforms, applied only within schema bounds.
SHIFT_TRANSFORMS = (
    ("dose_mg", "add", 90.0),
    # Generator floor 15 becomes 10.5, still above the schema minimum 10.
    ("baseline_egfr", "multiply", 0.7),
    ("concomitant_med_count", "add", 2),
    ("exposure_days", "multiply", 1.5),
)

EXTRA_MISSINGNESS = {"baseline_egfr": 0.35}
SEX_REMIX = (0.8, 0.1, 0.1)


def _apply_shifts(records: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng([seed, 9])
    shifted = []
    for record in records:
        row = dict(record)
        for field, operation, parameter in SHIFT_TRANSFORMS:
            if row[field] is None:
                continue
            if operation == "add":
                row[field] = row[field] + parameter
            elif field == "baseline_egfr":
                row[field] = round(row[field] * parameter, 1)
            else:
                row[field] = min(365, int(round(row[field] * parameter)))
        if row["baseline_egfr"] is None or rng.random() < EXTRA_MISSINGNESS["baseline_egfr"]:
            row["baseline_egfr"] = None
        row["sex"] = str(rng.choice(("female", "male", "other_or_unknown"), p=SEX_REMIX))
        shifted.append(row)
    return shifted


def _transform_log() -> list[dict]:
    return [
        {"field": field, "operation": operation, "parameter": parameter}
        for field, operation, parameter in SHIFT_TRANSFORMS
    ] + [
        {
            "field": "baseline_egfr",
            "operation": "extra_missingness",
            "parameter": EXTRA_MISSINGNESS,
        },
        {"field": "sex", "operation": "remix", "parameter": list(SEX_REMIX)},
    ]


def simulate_batch(
    scenario: str, *, n: int = 600, seed: int | None = None
) -> tuple[list[dict], dict]:
    """Return (records, provenance) for a labeled demonstration scenario."""
    if scenario not in SCENARIO_SEED:
        raise ValueError(f"scenario must be one of {sorted(SCENARIO_SEED)}")
    if type(n) is not int or n < 2:
        raise ValueError("n must be an integer >= 2")
    seed = SCENARIO_SEED[scenario] if seed is None else seed
    config = GeneratorConfig(seed=seed, n_train=n, n_validation=2, n_test=2)
    records = generate_partition("train", assign_families(config)["train"], n, config)
    transforms = _transform_log() if scenario == "shifted" else []
    if transforms:
        records = _apply_shifts(records, seed)
    provenance = {
        "scenario": scenario,
        "seed": seed,
        "n": len(records),
        "transforms": transforms,
        "generator_config": config.to_manifest_dict(),
        "note": (
            "Simulated demonstration batch, not an observed production sample. "
            "Same synthetic generator as the reference; shifts are authored."
        ),
    }
    validate_records(records)
    return records, provenance


def write_batch(directory, records: list[dict], provenance: dict) -> dict:
    """JSONL + manifest (hash last) into a no-overwrite directory."""
    from pharmasignal.provenance import canonical_json, sha256

    payload = b"".join(canonical_json(row) for row in records)
    manifest = {
        "synthetic": True,
        "kind": "simulated_monitoring_batch",
        "scenario": provenance["scenario"],
        "seed": provenance["seed"],
        "n": provenance["n"],
        "transforms": provenance["transforms"],
        "note": provenance["note"],
        "batch_sha256": sha256(payload),
        "generator_config": provenance["generator_config"],
    }
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "batch.jsonl").write_bytes(payload)
    (directory / "manifest.json").write_bytes(canonical_json(manifest))
    return manifest


def manifest_summary(manifest: dict) -> dict:
    """Identity subset of a batch manifest safe for embedding in reports."""
    return {key: manifest[key] for key in ("scenario", "seed", "n", "transforms", "note")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIO_SEED, required=True)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        records, provenance = simulate_batch(args.scenario, n=args.n, seed=args.seed)
        manifest = write_batch(args.output, records, provenance)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Simulation failed: {exc}\n")
    print(f"Wrote {manifest['n']} simulated rows; sha256={manifest['batch_sha256']}")


if __name__ == "__main__":
    main()
