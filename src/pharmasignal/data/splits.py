"""Group-isolation and exact normalized-content overlap checks."""

from __future__ import annotations

import hashlib
import unicodedata
from itertools import combinations

from pharmasignal.data.validation import ValidationError, validate_records


def normalize_narrative(text: str) -> str:
    """Conservative audit normalization, not learned NLP preprocessing."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def narrative_hash(text: str) -> str:
    return hashlib.sha256(normalize_narrative(text).encode("utf-8")).hexdigest()


def validate_splits(
    partitions: dict[str, list[dict]], assignments: dict[str, tuple[str, ...]] | None = None
) -> dict:
    expected = {"train", "validation", "test"}
    if set(partitions) != expected:
        raise ValidationError(["expected exactly train, validation and test partitions"])
    if assignments is not None and set(assignments) != expected:
        raise ValidationError(["family assignments must name all three partitions"])
    identities = {}
    summary = {}
    for name, records in partitions.items():
        validate_records(records)
        labels = {record["signal"] for record in records}
        if labels != {0, 1}:
            raise ValidationError([f"{name}: both target classes are required; got {labels}"])
        families = {record["template_family"] for record in records}
        if assignments is not None and not families <= set(assignments[name]):
            raise ValidationError([f"{name}: record uses an unassigned template family"])
        hashes = {narrative_hash(record["narrative"]) for record in records}
        identities[name] = {
            "report_ids": {record["report_id"] for record in records},
            "template_families": families,
            "normalized_narratives": hashes,
        }
        positive = sum(record["signal"] for record in records)
        summary[name] = {
            "rows": len(records),
            "class_counts": {"0": len(records) - positive, "1": positive},
            "positive_fraction": positive / len(records),
            "observed_families": sorted(families),
            "within_split_duplicate_narratives": len(records) - len(hashes),
        }
    overlaps = {}
    for left, right in combinations(sorted(partitions), 2):
        if assignments is not None and set(assignments[left]) & set(assignments[right]):
            raise ValidationError([f"{left}/{right}: assigned template families overlap"])
        counts = {
            key: len(identities[left][key] & identities[right][key]) for key in identities[left]
        }
        overlaps[f"{left}/{right}"] = counts
        if any(counts.values()):
            raise ValidationError([f"{left}/{right}: cross-split overlap: {counts}"])
    return {"splits": summary, "cross_split_overlap_counts": overlaps}
