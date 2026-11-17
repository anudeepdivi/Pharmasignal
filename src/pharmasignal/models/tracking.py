"""Explicit local MLflow tracking around the unchanged M3 training workflow.

Usage: python -m pharmasignal.models.tracking --data artifacts/data/seed42 \
--output artifacts/models/m4 --tracking-dir artifacts/mlflow
"""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

from pharmasignal.models.train import train_baselines
from pharmasignal.provenance import canonical_json, sha256

EXPERIMENT_NAME = "pharmasignal-baselines"
BUNDLE_FILES = ("pipeline.joblib", "report.json", "metadata.json")


def local_client(tracking_dir: Path) -> tuple[MlflowClient, str, str]:
    """Use local SQLite and files, independent of ambient tracking/registry URIs."""
    root = tracking_dir.resolve()
    # SQLAlchemy parses '?' as a query separator in URL strings.
    if "?" in str(root) or "#" in str(root):
        raise ValueError("tracking directory cannot contain '?' or '#'")
    root.mkdir(parents=True, exist_ok=True)
    uri = "sqlite:///" + str(root / "mlflow.db")
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    artifact_uri = (root / "artifacts").as_uri()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_uri)
    else:
        if experiment.lifecycle_stage != "active" or experiment.artifact_location != artifact_uri:
            raise ValueError("experiment must be active with the expected local artifact location")
        experiment_id = experiment.experiment_id
    return client, uri, experiment_id


def report_metrics(report: dict) -> dict[str, float]:
    """Flatten already-computed metrics; never read data or predict again."""
    groups = {
        **{
            f"validation.{variant}.at_0_5": values
            for variant, values in report["validation_comparison_at_0_5"].items()
        },
        "validation.selected": report["selection"]["validation_metrics"],
        "test.at_0_5": report["test_at_0_5"],
        "test.selected": report["test_at_selected_threshold"],
    }
    metrics = {"test_passes": float(report["test_passes"])}
    for prefix, values in groups.items():
        for key in (
            "rows",
            "threshold",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        ):
            if values[key] is not None:
                metrics[f"{prefix}.{key}"] = float(values[key])
        tn, fp = values["confusion_matrix"][0]
        fn, tp = values["confusion_matrix"][1]
        for key, value in zip(("tn", "fp", "fn", "tp"), (tn, fp, fn, tp), strict=True):
            metrics[f"{prefix}.{key}"] = float(value)
    return metrics


def train_tracked(
    data_path: Path,
    output: Path,
    tracking_dir: Path,
    *,
    seed: int = 42,
    recall_target: float = 0.8,
) -> dict:
    """One run per comparison; tracking failures are errors, never hidden success.

    SQLite, uploaded files and the offline output are not one transaction.
    Completed offline files survive tracking failure. No automatic retry/refit.
    """
    if output.exists():
        raise FileExistsError(f"destination already exists: {output}")
    client, uri, experiment_id = local_client(tracking_dir)
    run = client.create_run(
        experiment_id,
        run_name="fixed-baseline-comparison",
        tags={"milestone": "M4", "synthetic": "true", "mlflow_version": version("mlflow")},
    )
    run_id = run.info.run_id
    try:
        # Validate the run location too, before any artifact upload.
        expected = (tracking_dir.resolve() / "artifacts" / run_id / "artifacts").as_uri()
        if run.info.artifact_uri != expected:
            raise ValueError("run must use the expected local artifact location")
        client.log_batch(
            run_id,
            params=[Param("seed", str(seed)), Param("recall_target", str(recall_target))],
        )
        manifest_bytes = (data_path / "manifest.json").read_bytes()
        report = train_baselines(data_path, output, seed=seed, recall_target=recall_target)
        if sha256(manifest_bytes) != report["dataset_manifest_sha256"]:
            raise ValueError("dataset manifest changed during tracking")
        metadata = json.loads((output / "metadata.json").read_text())
        params = {
            **report["configuration"],
            "penalty": "l2",
            "tfidf_ngram_range": [1, 2],
            "tfidf_sublinear_tf": True,
            "numeric_imputation": "median_keep_empty_add_indicator",
            "categorical_encoding": "one_hot_ignore_unknown",
            "dataset_config_sha256": json.loads(manifest_bytes)["config_sha256"],
            "selection_criterion": report["selection"]["criterion"],
            "training_scope": report["selection"]["training_scope"],
            "dataset_id": report["dataset_id"],
            "dataset_manifest_sha256": report["dataset_manifest_sha256"],
            "configuration_sha256": report["configuration_sha256"],
            "lockfile_sha256": report["lockfile_sha256"],
            "schema_version": metadata["schema_version"],
        }
        tags = {
            "selected_variant": report["selection"]["variant"],
            "model_version": metadata["model_version"],
            "git_revision": report["source_identity"]["git_revision"],
            "git_dirty": report["source_identity"]["git_dirty"],
            "clinical_validation": "none",
        }
        timestamp = int(time.time() * 1000)
        client.log_batch(
            run_id,
            params=[
                Param(k, json.dumps(v) if not isinstance(v, str) else v) for k, v in params.items()
            ],
            metrics=[Metric(k, v, timestamp, 0) for k, v in report_metrics(report).items()],
            tags=[RunTag(k, "unknown" if v is None else str(v)) for k, v in tags.items()],
        )
        for filename in BUNDLE_FILES:
            client.log_artifact(run_id, str(output / filename), artifact_path="model_bundle")
        client.log_dict(run_id, json.loads(manifest_bytes), "evidence/dataset_manifest.json")
        client.log_dict(run_id, report["configuration"], "evidence/configuration.json")
        client.log_dict(
            run_id,
            {
                "dependencies": report["dependencies"],
                "source_sha256": report["source_sha256"],
                "source_identity": report["source_identity"],
                "mlflow_version": version("mlflow"),
            },
            "evidence/provenance.json",
        )
        receipt = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "tracking_uri": uri,
            "artifact_uri": run.info.artifact_uri,
            "model_version": metadata["model_version"],
        }
        # Receipt identifies the run; query MLflow for its authoritative status.
        with (output / "tracking.json").open("xb") as stream:
            stream.write(canonical_json(receipt))
        client.set_terminated(run_id, status="FINISHED")
        return receipt
    except BaseException as exc:
        exc.add_note(f"MLflow run {run_id}; tracking URI {uri}; no automatic retry")
        try:
            client.set_terminated(run_id, status="FAILED")
        except Exception as termination_error:
            exc.add_note(f"Could not mark run FAILED: {type(termination_error).__name__}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracking-dir", type=Path, default=Path("artifacts/mlflow"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall-target", type=float, default=0.8)
    args = parser.parse_args()
    print(
        json.dumps(
            train_tracked(
                args.data,
                args.output,
                args.tracking_dir,
                seed=args.seed,
                recall_target=args.recall_target,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
