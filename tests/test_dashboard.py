"""Dashboard helper logic plus a headless end-to-end run of the Streamlit app."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmasignal.dashboard.logic import (
    drift_table,
    explain_record,
    parse_upload,
    pretty_feature,
    read_jsonl,
    score_rows,
    threshold_metrics,
)
from pharmasignal.data.build import build_dataset
from pharmasignal.data.generator import GeneratorConfig
from pharmasignal.data.schema import PREDICTORS
from pharmasignal.models.artifact import load_artifact
from pharmasignal.models.train import train_baselines
from pharmasignal.monitoring.config import DriftConfig
from pharmasignal.monitoring.report import build_report
from pharmasignal.monitoring.simulate import simulate_batch, write_batch

APP = Path(__file__).resolve().parents[1] / "src" / "pharmasignal" / "dashboard" / "app.py"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("dashboard")
    build_dataset(root / "data", GeneratorConfig(n_train=120, n_validation=60, n_test=60))
    train_baselines(root / "data", root / "model")
    return root


@pytest.fixture(scope="module")
def loaded(workspace):
    return load_artifact(workspace / "model")


def test_pretty_feature_labels():
    assert pretty_feature("numeric__dose_mg") == "Dose mg"
    assert pretty_feature("numeric__missingindicator_age") == "Age is missing"
    assert pretty_feature("categorical__sex___missing__") == "Sex = missing"
    assert pretty_feature("categorical__treatment_treatment_b") == "Treatment = treatment_b"
    assert pretty_feature("text_rules__token_count") == "Narrative: length (tokens)"
    assert pretty_feature("tfidf__nausea") == 'Narrative phrase "nausea"'


def test_explanation_sums_to_log_odds(loaded, workspace):
    model, _ = loaded
    record = {k: read_jsonl(workspace / "data" / "test.jsonl")[0][k] for k in PREDICTORS}
    full = explain_record(model, record, top=10_000)
    from pharmasignal.features.fusion import assemble_features

    margin = model.decision_function(assemble_features([record]))[0]
    intercept = model.named_steps["classifier"].intercept_[0]
    assert full["contribution"].sum() + intercept == pytest.approx(margin)
    top = explain_record(model, record, top=3)
    assert len(top) <= 3
    assert list(top["contribution"].abs()) == sorted(top["contribution"].abs(), reverse=True)


def test_parse_upload_coerces_and_reports_errors():
    good = {
        "age": "64",
        "sex": "",
        "treatment": "treatment_a",
        "dose_mg": "50.5",
        "exposure_days": "12.0",
        "baseline_egfr": None,
        "concomitant_med_count": "3",
        "narrative": "Reports nausea.",
        "my_id": "A1",
    }
    bad_range = {**good, "age": "200"}
    bad_type = {**good, "exposure_days": "1.5"}
    bad_cat = {**good, "treatment": "aspirin"}
    frame = pd.DataFrame([good, bad_range, bad_type, bad_cat])
    valid, errors = parse_upload(frame)
    assert len(valid) == 1
    assert valid[0]["row"] == 1
    assert valid[0]["age"] == 64 and valid[0]["exposure_days"] == 12
    assert valid[0]["sex"] is None and valid[0]["baseline_egfr"] is None
    assert list(errors["row"]) == [2, 3, 4]
    assert "age" in errors.loc[0, "error"]
    assert "whole number" in errors.loc[1, "error"]


def test_parse_upload_requires_columns():
    with pytest.raises(ValueError, match="narrative"):
        parse_upload(pd.DataFrame([{"age": 1}]))


def test_score_rows_matches_threshold(loaded, workspace):
    model, metadata = loaded
    rows = [
        {"row": i + 1, **{k: r[k] for k in PREDICTORS}}
        for i, r in enumerate(read_jsonl(workspace / "data" / "test.jsonl")[:10])
    ]
    scored = score_rows(model, metadata["threshold"], rows)
    assert list(scored["row"]) == list(range(1, 11))
    flagged = scored["priority_score"] >= round(metadata["threshold"], 4)
    assert ((scored["flag_for_review"] == "YES") == flagged).all()


def test_threshold_metrics():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.2, 0.8, 0.1])
    m = threshold_metrics(labels, scores, 0.5)
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["precision"] == m["recall"] == m["f1"] == 0.5
    assert threshold_metrics(labels, scores, 0.95)["precision"] == 0.0


@pytest.fixture(scope="module")
def drift_reports(workspace):
    """Simulated batches + drift reports laid out like the real artifacts tree."""
    reference = read_jsonl(workspace / "data" / "train.jsonl")
    manifest = json.loads((workspace / "data" / "manifest.json").read_text())
    reports = {}
    for scenario in ("shifted", "unchanged"):
        batch, provenance = simulate_batch(scenario, n=120)
        batch_manifest = write_batch(workspace / "sim" / scenario, batch, provenance)
        report = build_report(reference, manifest, batch, batch_manifest, DriftConfig())
        (workspace / "mon" / scenario).mkdir(parents=True)
        (workspace / "mon" / scenario / "report.json").write_text(json.dumps(report))
        reports[scenario] = report
    return reports


def test_drift_table_flattens_every_section(drift_reports):
    report = drift_reports["shifted"]
    table = drift_table(report)
    total = sum(len(rows) for rows in report["sections"].values())
    assert len(table) == total
    assert set(table["status"]) <= {"ok", "alert", "insufficient", "undefined"}
    assert (table["status"] == "alert").any()


def test_streamlit_app_runs_end_to_end(workspace, drift_reports, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("PHARMASIGNAL_MODEL_DIR", str(workspace / "model"))
    monkeypatch.setenv("PHARMASIGNAL_DATA_DIR", str(workspace / "data"))
    monkeypatch.setenv("PHARMASIGNAL_MONITORING_DIR", str(workspace / "mon"))
    monkeypatch.setenv("PHARMASIGNAL_SIMULATED_DIR", str(workspace / "sim"))
    app = testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert app.metric[0].label == "ROC-AUC (held-out test)"
    for index in range(len(app.selectbox[0].options)):
        app.selectbox[0].set_value(index).run()
        assert not app.exception
    app.slider[0].set_value(0.5).run()
    app.radio[0].set_value("unchanged").run()
    assert not app.exception
