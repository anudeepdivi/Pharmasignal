"""Metric contracts and real-model artifact integrity/round-trip tests."""

import json
import subprocess
import sys

import numpy as np
import pytest
from sklearn.metrics import precision_score, recall_score

from pharmasignal.data.build import build_dataset
from pharmasignal.data.generator import GeneratorConfig
from pharmasignal.features.fusion import build_feature_pipeline, describe_fitted_features
from pharmasignal.models.artifact import load_artifact, predict_records
from pharmasignal.models.evaluation import error_analysis, evaluate, select_threshold
from pharmasignal.models.train import train_baselines


def test_metrics_exact_and_undefined():
    result = evaluate([0, 0, 1, 1], [0.1, 0.7, 0.3, 0.9])
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["precision"] == result["recall"] == result["f1"] == 0.5
    assert result["roc_auc"] == 0.75
    assert result["average_precision"] == pytest.approx(5 / 6)
    for labels in ([0, 0], [1, 1]):
        one_class = evaluate(labels, [0.1, 0.2])
        assert one_class["roc_auc"] is None and one_class["average_precision"] is None
    assert evaluate([0, 1], [0.1, 0.2])["precision"] == 0
    assert evaluate([0, 1], [0.5, 0.5])["confusion_matrix"] == [[0, 1], [0, 1]]


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        ([], []),
        ([0], [0.1, 0.2]),
        ([2], [0.5]),
        ([0], [float("nan")]),
        ([1], [1.1]),
        ([[0]], [[0.1]]),
    ],
)
def test_bad_metric_inputs(labels, scores):
    with pytest.raises(ValueError):
        evaluate(labels, scores)


def test_threshold_policy():
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.8])
    selected = select_threshold(y, p, 0.8)
    options = []
    for t in np.unique(np.r_[0, p]):
        predicted = p >= t
        recall = recall_score(y, predicted)
        if recall >= 0.8:
            options.append((precision_score(y, predicted), recall, t))
    assert selected["threshold"] == max(options)[2]
    assert selected["validation_metrics"]["recall"] >= 0.8
    with pytest.raises(ValueError):
        select_threshold([0, 0], [0.1, 0.2])
    for t in (0, 2, float("nan")):
        with pytest.raises(ValueError):
            select_threshold(y, p, t)
    with pytest.raises(ValueError):
        evaluate(y, p, -1)


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("real_model")
    build_dataset(root / "data", GeneratorConfig(n_train=120, n_validation=60, n_test=60))
    report = train_baselines(root / "data", root / "model")
    return root, report


def test_real_roundtrip_and_fresh_process(real_run):
    root, report = real_run
    model, metadata = load_artifact(root / "model")
    # Use train examples to verify saved inference, not another test evaluation.
    rows = [json.loads(line) for line in (root / "data/train.jsonl").read_text().splitlines()][:3]
    predictions = predict_records(model, metadata, rows)
    assert all(0 <= row["signal_probability"] <= 1 for row in predictions)
    assert metadata["threshold"] == report["selection"]["threshold"]
    script = (
        "import json,sys; from pathlib import Path; "
        "from pharmasignal.models.artifact import load_artifact,predict_records; "
        'root=Path(sys.argv[1]); m,meta=load_artifact(root/"model"); '
        'lines=(root/"data/train.jsonl").read_text().splitlines(); '
        "rows=[json.loads(line) for line in lines[:3]]; "
        "print(json.dumps(predict_records(m,meta,rows)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(root)], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == predictions
    repeat = train_baselines(root / "data", root / "repeat")
    assert report["selection"] == repeat["selection"]
    assert report["test_at_selected_threshold"] == repeat["test_at_selected_threshold"]
    reloaded, meta = load_artifact(root / "repeat")
    assert [r["signal_probability"] for r in predict_records(reloaded, meta, rows)] == [
        r["signal_probability"] for r in predictions
    ]
    with pytest.raises(FileExistsError):
        train_baselines(root / "data", root / "model")


@pytest.mark.parametrize("kind", ["model", "report", "metadata", "dependency"])
def test_tampered_artifact_rejected(real_run, tmp_path, kind):
    import shutil

    root, _ = real_run
    destination = tmp_path / "copy"
    shutil.copytree(root / "model", destination)
    if kind in ("model", "report"):
        filename = "pipeline.joblib" if kind == "model" else "report.json"
        with (destination / filename).open("ab") as f:
            f.write(b"corruption")
    else:
        path = destination / "metadata.json"
        metadata = json.loads(path.read_text())
        if kind == "dependency":
            metadata["dependencies"]["sklearn"] = "incompatible"
        else:
            metadata["threshold"] = 0.99
        path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_artifact(destination)


def test_error_analysis_and_variants(real_run):
    root, report = real_run
    analysis = report["test_error_analysis"]
    cm = report["test_at_selected_threshold"]["confusion_matrix"]
    assert len(analysis["errors"]) == cm[0][1] + cm[1][0]
    assert all("narrative" not in item for item in analysis["errors"])
    rows = [
        {
            "report_id": "a",
            "template_family": "f",
            "sex": None,
            "treatment": "treatment_a",
            "signal": 1,
        }
    ]
    assert error_analysis(rows, [0.1], 0.5)["errors"][0]["error"] == "false_negative"
    from pharmasignal.features.fusion import assemble_features

    train = [json.loads(line) for line in (root / "data/train.jsonl").read_text().splitlines()]
    X = assemble_features(train)
    for variant in ("structured", "text", "fused"):
        pipeline = build_feature_pipeline(variant=variant).fit(X)
        blocks = describe_fitted_features(pipeline)["blocks"]
        assert bool(blocks["numeric"]) == (variant != "text")
        assert bool(blocks["tfidf"]) == (variant != "structured")
    with pytest.raises(ValueError):
        build_feature_pipeline(variant="bogus")


def test_dataset_tampering_rejected(real_run, tmp_path):
    import shutil

    root, _ = real_run
    data = tmp_path / "data"
    shutil.copytree(root / "data", data)
    with (data / "train.jsonl").open("ab") as f:
        f.write(b"\n")
    with pytest.raises(ValueError, match="checksum"):
        train_baselines(data, tmp_path / "model")
    assert not (tmp_path / "model").exists()
