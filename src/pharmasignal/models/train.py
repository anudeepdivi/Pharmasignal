"""Train three fixed baselines, freeze validation selection, evaluate test once.

Usage: python -m pharmasignal.models.train --data artifacts/data/seed42 --output artifacts/models/m3
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pharmasignal.data.schema import GENERATOR_VERSION, SCHEMA_VERSION
from pharmasignal.data.splits import narrative_hash
from pharmasignal.data.validation import validate_records
from pharmasignal.features.fusion import assemble_features, build_feature_pipeline
from pharmasignal.models.artifact import coefficient_summary, save_artifact
from pharmasignal.models.evaluation import error_analysis, evaluate, select_threshold
from pharmasignal.provenance import canonical_json, dependency_versions, sha256, source_identity

VARIANTS = ("structured", "text", "fused")


class TestAccessGuard:
    """One test access per training invocation after validation selection.

    This is workflow control, not a security boundary or a global holdout
    registry. A user can rerun training; such reruns must be disclosed.
    """

    __test__ = False

    def __init__(self):
        self.frozen = False
        self.consumed = False

    def freeze_selection(self):
        if self.frozen:
            raise RuntimeError("selection already frozen")
        self.frozen = True

    def consume(self):
        if not self.frozen:
            raise RuntimeError("selection must be frozen before test access")
        if self.consumed:
            raise RuntimeError("test can be consumed only once")
        self.consumed = True


def read_split(root: Path, manifest: dict, split: str) -> list[dict]:
    payload = (root / f"{split}.jsonl").read_bytes()
    if sha256(payload) != manifest["files_sha256"][f"{split}.jsonl"]:
        raise ValueError(f"{split} checksum mismatch")
    rows = [json.loads(line) for line in payload.splitlines()]
    validate_records(rows)
    if {r["signal"] for r in rows} != {0, 1}:
        raise ValueError(f"{split} requires both classes")
    if not {r["template_family"] for r in rows} <= set(manifest["family_assignments"][split]):
        raise ValueError(f"{split} family assignment mismatch")
    return rows


def _check_disjoint(left: list[dict], right: list[dict]) -> None:
    for field in ("report_id", "template_family"):
        if {r[field] for r in left} & {r[field] for r in right}:
            raise ValueError(f"cross-split {field} overlap")
    if {narrative_hash(r["narrative"]) for r in left} & {
        narrative_hash(r["narrative"]) for r in right
    }:
        raise ValueError("cross-split narrative overlap")


def make_model(variant: str, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("features", build_feature_pipeline(variant=variant)),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    penalty="l2",
                    max_iter=2000,
                    random_state=seed,
                    class_weight=None,
                ),
            ),
        ]
    )


def train_baselines(
    data_path: Path, output: Path, *, seed: int = 42, recall_target: float = 0.8
) -> dict:
    if output.exists():
        raise FileExistsError(f"destination already exists: {output}")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32-1]")
    if not np.isfinite(recall_target) or not 0 < recall_target <= 1:
        raise ValueError("recall_target must be in (0, 1]")
    manifest = json.loads((data_path / "manifest.json").read_text())
    if (
        manifest["synthetic"] is not True
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["generator_version"] != GENERATOR_VERSION
    ):
        raise ValueError("unsupported dataset provenance/schema")
    if sha256(canonical_json(manifest["files_sha256"])) != manifest["dataset_id"]:
        raise ValueError("dataset identity mismatch")
    training = read_split(data_path, manifest, "train")
    validation = read_split(data_path, manifest, "validation")
    _check_disjoint(training, validation)
    X_train, X_validation = assemble_features(training), assemble_features(validation)
    y_train = [r["signal"] for r in training]
    y_validation = [r["signal"] for r in validation]
    models, validation_scores, comparisons = {}, {}, {}
    for variant in VARIANTS:
        model = make_model(variant, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X_train, y_train)
        scores = model.predict_proba(X_validation)[:, 1]
        models[variant], validation_scores[variant] = model, scores
        comparisons[variant] = evaluate(y_validation, scores)

    # Primary selection metric is validation average precision, threshold-free
    # and appropriate for imbalance. Fixed order breaks exact ties.
    winner = max(VARIANTS, key=lambda v: comparisons[v]["average_precision"])
    threshold_info = select_threshold(y_validation, validation_scores[winner], recall_target)
    selection = {
        "variant": winner,
        **threshold_info,
        "criterion": "validation_average_precision; ties: structured,text,fused",
        "training_scope": "train_only_no_refit",
    }
    guard = TestAccessGuard()
    guard.freeze_selection()
    guard.consume()
    test = read_split(data_path, manifest, "test")
    _check_disjoint(training, test)
    _check_disjoint(validation, test)
    test_scores = models[winner].predict_proba(assemble_features(test))[:, 1]
    # Both operating points and error analysis reuse the SAME test score vector.
    y_test = [r["signal"] for r in test]
    source_root = Path(__file__).parents[1]
    source_hashes = {
        str(p.relative_to(source_root)): sha256(p.read_bytes())
        for p in sorted(source_root.rglob("*.py"))
    }
    project_root = source_root.parents[1]
    lock = project_root / "uv.lock"
    report = {
        "synthetic": True,
        "dataset_id": manifest["dataset_id"],
        "source_identity": source_identity(Path(__file__).parent, project_root),
        "dataset_manifest_sha256": sha256((data_path / "manifest.json").read_bytes()),
        "source_sha256": source_hashes,
        "dependencies": dependency_versions(),
        "lockfile_sha256": sha256(lock.read_bytes()) if lock.exists() else None,
        "configuration": {
            "seed": seed,
            "recall_target": recall_target,
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 2000,
            "class_weight": None,
            "min_df": 2,
            "max_features": 3000,
            "variants": list(VARIANTS),
        },
        "validation_comparison_at_0_5": comparisons,
        "selection": selection,
        "test_at_0_5": evaluate(y_test, test_scores),
        "test_at_selected_threshold": evaluate(y_test, test_scores, selection["threshold"]),
        "test_error_analysis": error_analysis(test, test_scores, selection["threshold"]),
        "coefficient_extremes": coefficient_summary(models[winner]),
        "test_passes": 1,
        "note": "Synthetic demonstration, no clinical validation. No refit after selection.",
    }
    report["configuration_sha256"] = sha256(canonical_json(report["configuration"]))
    save_artifact(output, models[winner], report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall-target", type=float, default=0.8)
    args = parser.parse_args()
    try:
        report = train_baselines(
            args.data, args.output, seed=args.seed, recall_target=args.recall_target
        )
    except (ValueError, OSError, RuntimeError, ConvergenceWarning) as exc:
        parser.exit(1, f"Training failed: {exc}\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "dataset_id",
                    "validation_comparison_at_0_5",
                    "selection",
                    "test_at_0_5",
                    "test_at_selected_threshold",
                    "test_passes",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
