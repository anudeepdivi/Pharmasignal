"""Strict HTTP inputs, trusted startup and threshold-aware inference."""

import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from pharmasignal.data.build import build_dataset
from pharmasignal.data.generator import GeneratorConfig
from pharmasignal.data.schema import PREDICTORS
from pharmasignal.models.artifact import load_artifact, predict_records
from pharmasignal.models.train import train_baselines
from pharmasignal.serving.app import create_app
from pharmasignal.serving.requests import PredictionRequest


def request(app, method, path, **kwargs):
    async def send():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


@pytest.fixture
def record():
    return {
        "age": 64,
        "sex": "female",
        "treatment": "treatment_a",
        "dose_mg": 50.0,
        "exposure_days": 12,
        "baseline_egfr": 62.0,
        "concomitant_med_count": 3,
        "narrative": "Reports nausea after treatment started. Denies rash.",
    }


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("serving")
    build_dataset(root / "data", GeneratorConfig(n_train=120, n_validation=60, n_test=60))
    train_baselines(root / "data", root / "model")
    return root / "model"


def test_request_valid_and_nullable(record):
    assert PredictionRequest(**record).model_dump() == record
    for name in set(PREDICTORS) - {"treatment", "narrative"}:
        record[name] = None
    assert PredictionRequest(**record).model_dump() == record


@pytest.mark.parametrize("name", PREDICTORS)
def test_request_requires_every_key(record, name):
    del record[name]
    with pytest.raises(ValidationError):
        PredictionRequest(**record)
    response = request(create_app(Path("unused")), "POST", "/predict", json=record)
    assert response.status_code == 422
    assert "input" not in response.text


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("age", "64"),
        ("age", 64.0),
        ("age", True),
        ("age", 91),
        ("dose_mg", True),
        ("dose_mg", "50"),
        ("dose_mg", 0),
        ("dose_mg", float("nan")),
        ("baseline_egfr", float("inf")),
        ("baseline_egfr", False),
        ("exposure_days", 366),
        ("concomitant_med_count", -1),
        ("sex", "unsupported"),
        ("treatment", "real_drug"),
        ("treatment", None),
        ("narrative", None),
        ("narrative", " \t"),
        ("narrative", "x" * 4001),
        ("report_id", "audit"),
        ("signal", 1),
        ("template_family", "family_01"),
    ],
)
def test_request_rejects_bad_values(record, name, value):
    record[name] = value
    with pytest.raises(ValidationError):
        PredictionRequest(**record)
    response = request(
        create_app(Path("unused")),
        "POST",
        "/predict",
        content=json.dumps(record),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "input" not in response.text


def test_health_and_unready(record):
    app = create_app(Path("unused"))
    assert request(app, "GET", "/health").json() == {"status": "alive"}
    assert request(app, "GET", "/ready").status_code == 503
    assert request(app, "POST", "/predict", json=record).status_code == 503


def test_prediction_and_lifecycle(artifact, record, caplog):
    app = create_app(artifact)
    original, metadata = load_artifact(artifact)

    async def exercise():
        # ASGITransport does not run startup/shutdown; exercise lifespan explicitly.
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                assert (await client.get("/ready")).json()["model_version"] == metadata[
                    "model_version"
                ]
                response = await client.post("/predict", json=record)
                assert response.status_code == 200
                assert response.json() == predict_records(original, metadata, [record])[0]
                with patch(
                    "pharmasignal.serving.app.predict_records",
                    side_effect=ValueError(record["narrative"]),
                ):
                    failed = await client.post("/predict", json=record)
                assert failed.status_code == 500
                assert record["narrative"] not in failed.text
                bad = await client.post("/predict", content='{"narrative": SECRET')
                assert bad.status_code == 422 and "SECRET" not in bad.text
        assert app.state.model is None

    asyncio.run(exercise())
    assert record["narrative"] not in caplog.text


@pytest.mark.parametrize("kind", ["missing", "corrupt"])
def test_startup_failure(tmp_path, artifact, kind):
    import shutil

    target = tmp_path / "model"
    if kind == "corrupt":
        shutil.copytree(artifact, target)
        (target / "pipeline.joblib").write_bytes(b"invalid")
    app = create_app(target)

    async def start():
        async with app.router.lifespan_context(app):
            pytest.fail("startup should not succeed")

    with pytest.raises(RuntimeError, match="unavailable or incompatible"):
        asyncio.run(start())
    assert app.state.model is None


def test_live_http(artifact, record, tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = tmp_path / "server.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pharmasignal.serving",
                "--model-dir",
                str(artifact),
                "--port",
                str(port),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2) as client:
                deadline = time.monotonic() + 30
                while True:
                    if process.poll() is not None:
                        pytest.fail(f"Server exited: {log.read_text()}")
                    try:
                        ready = client.get("/ready")
                        if ready.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() > deadline:
                        pytest.fail("Server readiness timed out")
                    time.sleep(0.1)
                assert client.get("/health").status_code == 200
                response = client.post("/predict", json=record)
                model, metadata = load_artifact(artifact)
                assert response.status_code == 200
                assert response.json() == predict_records(model, metadata, [record])[0]
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    assert record["narrative"] not in log.read_text()
