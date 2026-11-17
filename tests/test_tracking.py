"""Real local MLflow persistence and downloaded offline inference."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from mlflow.tracking import MlflowClient

from pharmasignal.data.build import build_dataset
from pharmasignal.data.generator import GeneratorConfig
from pharmasignal.models.artifact import load_artifact, predict_records
from pharmasignal.models.tracking import report_metrics, train_tracked


def test_local_tracking_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://unused.invalid")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "https://unused.invalid")
    data = tmp_path / "data"
    output = tmp_path / "model"
    build_dataset(data, GeneratorConfig(n_train=120, n_validation=60, n_test=60))
    from pharmasignal.models import tracking

    with patch.object(tracking, "train_baselines", wraps=tracking.train_baselines) as training:
        receipt = train_tracked(data, output, tmp_path / "tracking")
    training.assert_called_once_with(data, output, seed=42, recall_target=0.8)
    client = MlflowClient(tracking_uri=receipt["tracking_uri"])
    run = client.get_run(receipt["run_id"])
    assert run.info.status == "FINISHED"
    assert client.get_experiment(run.info.experiment_id).name == "pharmasignal-baselines"
    report = json.loads((output / "report.json").read_text())
    assert run.data.metrics == report_metrics(report)
    assert run.data.params["dataset_id"] == report["dataset_id"]
    assert run.data.params["seed"] == "42"
    assert run.data.tags["selected_variant"] == report["selection"]["variant"]
    assert {item.path for item in client.list_artifacts(run.info.run_id, "model_bundle")} == {
        "model_bundle/pipeline.joblib",
        "model_bundle/report.json",
        "model_bundle/metadata.json",
    }
    downloaded = tmp_path / "download"
    downloaded.mkdir()
    bundle = Path(client.download_artifacts(run.info.run_id, "model_bundle", str(downloaded)))
    model, metadata = load_artifact(bundle)
    original, original_metadata = load_artifact(output)
    rows = [json.loads(line) for line in (data / "train.jsonl").read_text().splitlines()[:3]]
    predictions = predict_records(original, original_metadata, rows)
    assert predict_records(model, metadata, rows) == predictions
    script = (
        "import json,sys; from pathlib import Path; "
        "from pharmasignal.models.artifact import load_artifact,predict_records; "
        "m,meta=load_artifact(Path(sys.argv[1])); "
        "print(json.dumps(predict_records(m,meta,json.loads(sys.argv[2]))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(bundle), json.dumps(rows)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == predictions
    manifest = Path(client.download_artifacts(run.info.run_id, "evidence/dataset_manifest.json"))
    assert json.loads(manifest.read_text()) == json.loads((data / "manifest.json").read_text())
    for filename in ("pipeline.joblib", "metadata.json", "report.json"):
        assert (bundle / filename).read_bytes() == (output / filename).read_bytes()
    with pytest.raises(FileExistsError):
        train_tracked(data, output, tmp_path / "tracking")
    second = train_tracked(data, tmp_path / "second", tmp_path / "tracking")
    assert second["experiment_id"] == receipt["experiment_id"]
    assert second["run_id"] != receipt["run_id"]
    assert len(client.search_runs([receipt["experiment_id"]])) == 2


@pytest.mark.parametrize("phase", ["training", "upload", "termination"])
def test_failed_runs_preserve_original_error(tmp_path, phase):
    from pharmasignal.models import tracking

    data = tmp_path / "data"
    build_dataset(data, GeneratorConfig(n_train=120, n_validation=60, n_test=60))
    client, _, experiment_id = tracking.local_client(tmp_path / "tracking")
    output = tmp_path / "model"
    if phase == "training":
        target, name = tracking, "train_baselines"
    else:
        target, name = MlflowClient, "log_artifact"
    error = RuntimeError("intentional failure")
    with patch.object(target, name, side_effect=error):
        if phase == "termination":
            with patch.object(MlflowClient, "set_terminated", side_effect=OSError("store offline")):
                with pytest.raises(RuntimeError, match="intentional failure") as raised:
                    train_tracked(data, output, tmp_path / "tracking")
            assert any("Could not mark" in note for note in raised.value.__notes__)
        else:
            with pytest.raises(RuntimeError, match="intentional failure"):
                train_tracked(data, output, tmp_path / "tracking")
    (run,) = client.search_runs([experiment_id])
    assert run.info.status == ("RUNNING" if phase == "termination" else "FAILED")
    if phase == "training":
        assert not output.exists()
    else:
        load_artifact(output)
    assert not (output / "tracking.json").exists()


def test_reject_unexpected_experiment_location(tmp_path):
    from pharmasignal.models.tracking import EXPERIMENT_NAME, local_client

    root = tmp_path / "tracking"
    root.mkdir()
    uri = "sqlite:///" + str(root / "mlflow.db")
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment_id = client.create_experiment(
        EXPERIMENT_NAME, artifact_location="https://unused.invalid"
    )
    with pytest.raises(ValueError, match="expected local"):
        local_client(root)
    assert client.search_runs([experiment_id]) == []


def test_undefined_metrics_are_not_logged():
    from pharmasignal.models.evaluation import evaluate

    values = evaluate([0, 0], [0.1, 0.2])
    report = {
        "validation_comparison_at_0_5": {"structured": values},
        "selection": {"validation_metrics": values},
        "test_at_0_5": values,
        "test_at_selected_threshold": values,
        "test_passes": 1,
    }
    metrics = report_metrics(report)
    assert not any(key.endswith(("roc_auc", "average_precision")) for key in metrics)
    assert metrics["test.selected.tn"] == 2
