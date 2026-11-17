"""Pure metric contracts, scenario behavior and report assembly."""

import json
import math

import pytest

from pharmasignal.data.build import build_dataset
from pharmasignal.data.generator import GeneratorConfig
from pharmasignal.monitoring.config import DriftConfig
from pharmasignal.monitoring.metrics import (
    ALERT,
    INSUFFICIENT,
    OK,
    UNDEFINED,
    categorical_comparison,
    missingness_comparison,
    numeric_comparison,
)
from pharmasignal.monitoring.report import build_report
from pharmasignal.monitoring.simulate import simulate_batch, write_batch
from pharmasignal.provenance import canonical_json, sha256

CFG = DriftConfig(min_samples=5)


def test_identical_numeric_distributions_do_not_alert():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    row = numeric_comparison("x", values, values, CFG)
    assert row["status"] == OK and row["ks_statistic"] == 0 and row["effect_size"] == 0


def test_shifted_numeric_field_alerts_only_on_shift():
    reference = [float(value) for value in range(10, 30)]
    batch = [value + 20 for value in reference]
    assert numeric_comparison("x", reference, batch, CFG)["status"] == ALERT
    assert numeric_comparison("x", reference, reference, CFG)["status"] == OK


def test_nulls_are_excluded_from_counts():
    reference = [1.0, math.nan, 2.0, 3.0, 4.0, 5.0, 6.0]
    row = numeric_comparison("x", reference, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0], CFG)
    assert row["n_reference"] == 6 and row["n_batch"] == 6


def test_constant_column_is_undefined_not_zero_effect():
    row = numeric_comparison("x", [5.0] * 10, [5.0] * 10, CFG)
    assert row["effect_size"] is None and row["status"] == UNDEFINED


def test_small_samples_are_insufficient_and_claim_no_alert():
    row = numeric_comparison("x", [1.0, 2.0], [3.0, 4.0], CFG)
    assert row["status"] == INSUFFICIENT and row["ks_statistic"] is None


def test_pvalue_is_recorded_but_never_gates_status():
    reference = [float(value) for value in range(1000)]
    row = numeric_comparison("x", reference, [value + 0.05 for value in reference], CFG)
    assert 0 <= row["ks_pvalue"] <= 1
    assert row["effect_size"] < 0.5 and row["status"] == OK
    # A +200 shift alerts via the statistic threshold (KS 0.2 >= 0.15) even
    # though a large n makes the p-value tiny; the p-value is never the gate.
    up = numeric_comparison("x", reference, [value + 200 for value in reference], CFG)
    assert up["ks_statistic"] >= 0.15 and up["status"] == ALERT
    # The effect size is a magnitude: an equal downward shift alerts identically.
    down = numeric_comparison("x", reference, [value - 200 for value in reference], CFG)
    assert down["status"] == ALERT and down["effect_size"] == up["effect_size"]


def test_categorical_tv_distance_and_unseen():
    reference = ["a"] * 6 + ["b"] * 4
    batch = ["a"] * 3 + ["b"] * 3 + ["c"] * 4
    row = categorical_comparison("sex", reference, batch, CFG)
    assert row["unseen_categories"] == ["c"]
    # |0.6-0.3| + |0.4-0.3| + |0-0.4| = 0.8; TV = 0.8 / 2 = 0.4.
    assert row["tv_distance"] == pytest.approx(0.4)
    assert row["status"] == ALERT
    assert row["proportions"]["c"]["reference"] == 0


def test_identical_categories_are_ok():
    reference = batch = ["a", "b", "a", "b", "a", "b"]
    row = categorical_comparison("treatment", reference, batch, CFG)
    assert row["status"] == OK and row["tv_distance"] == 0


def test_missingness_rates_and_difference():
    reference = [{"f": None}, {"f": 1}, {"f": 2}, {"f": 3}]
    batch = [{"f": None}, {"f": None}, {"f": 2}, {"f": 3}]
    row = missingness_comparison(reference, batch, ("f",), CFG)[0]
    assert row["reference_rate"] == 0.25 and row["batch_rate"] == 0.5
    assert row["absolute_difference"] == pytest.approx(0.25)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    root = tmp_path_factory.mktemp("m6")
    build_dataset(root / "data", GeneratorConfig(n_train=600, n_validation=60, n_test=60))
    records = [json.loads(line) for line in (root / "data/train.jsonl").read_text().splitlines()]
    manifest = json.loads((root / "data/manifest.json").read_text())
    return records, manifest, root / "data"


def _report(reference, scenario, tmp_path):
    records, manifest, _ = reference
    batch, provenance = simulate_batch(scenario, n=300)
    batch_manifest = write_batch(tmp_path / scenario, batch, provenance)
    return build_report(records, manifest, batch, batch_manifest, DriftConfig())


def test_unchanged_scenario_no_alerts(reference, tmp_path):
    report = _report(reference, "unchanged", tmp_path)
    assert report["summary"]["overall_status"] == "no_drift_detected"
    assert report["summary"]["alerts"] == 0


def test_shifted_scenario_alerts_transformed_fields(reference, tmp_path):
    report = _report(reference, "shifted", tmp_path)
    assert report["summary"]["overall_status"] == "drift_detected"
    alerted = {
        row["feature"]
        for section in report["sections"].values()
        for row in section
        if row["status"] == ALERT
    }
    assert alerted == {"dose_mg", "baseline_egfr", "concomitant_med_count", "exposure_days", "sex"}
    assert report["summary"]["alerts"] == 6


def test_report_determinism_and_no_text_leak(reference, tmp_path):
    first = _report(reference, "shifted", tmp_path / "first")
    second = _report(reference, "shifted", tmp_path / "second")
    assert canonical_json(first) == canonical_json(second)
    blob = json.dumps(first)
    records, _, _ = reference
    assert records[0]["narrative"] not in blob
    assert records[0]["report_id"] not in blob
    assert "narrative" not in first["identity"]["batch"]


def test_report_records_pvalue_transparently(reference, tmp_path):
    report = _report(reference, "unchanged", tmp_path)
    numeric = report["sections"]["numeric"]
    assert all("ks_pvalue" in row for row in numeric)


def test_constant_shift_still_alerts_via_ks():
    row = numeric_comparison("x", [1.0] * 10, [2.0] * 10, CFG)
    assert row["effect_size"] is None
    assert row["ks_statistic"] == 1 and row["status"] == ALERT
    # One constant side still has a defined pooled variance.
    row = numeric_comparison("x", list(range(10)), [20.0] * 10, CFG)
    assert row["effect_size"] is not None and row["status"] == ALERT


def test_all_missing_and_empty_inputs():
    row = numeric_comparison("x", [math.nan] * 10, [math.nan] * 10, CFG)
    assert row["status"] == INSUFFICIENT and row["n_reference"] == 0
    assert categorical_comparison("x", [], ["new"], CFG)["status"] == INSUFFICIENT
    row = missingness_comparison([], [], ("f",), CFG)[0]
    assert row["status"] == INSUFFICIENT
    assert row["reference_rate"] is None and row["absolute_difference"] is None
    full = [{"f": None}] * 10
    row = missingness_comparison(full, full, ("f",), CFG)[0]
    assert row["status"] == OK and row["reference_rate"] == 1


def test_missingness_threshold_is_configurable():
    reference = [{"f": 1}] * 10
    batch = [{"f": None}] * 3 + [{"f": 1}] * 7
    assert missingness_comparison(reference, batch, ("f",), CFG)[0]["status"] == ALERT
    relaxed = DriftConfig(min_samples=5, missingness_alert=0.4)
    assert missingness_comparison(reference, batch, ("f",), relaxed)[0]["status"] == OK
    assert missingness_comparison(reference, batch[:2], ("f",), CFG)[0]["status"] == INSUFFICIENT


@pytest.mark.parametrize(
    "values",
    [
        {"min_samples": 1},
        {"min_samples": True},
        {"ks_alert": 0},
        {"tv_alert": 1.1},
        {"effect_alert": math.inf},
        {"missingness_alert": math.nan},
        {"effect_alert": "0.5"},
    ],
)
def test_invalid_config(values):
    with pytest.raises(ValueError):
        DriftConfig(**values)


def test_config_json(tmp_path):
    path = tmp_path / "config.json"
    config = DriftConfig(effect_alert=2.0)
    path.write_bytes(canonical_json(config.to_dict()))
    assert DriftConfig.from_json(path) == config
    for invalid in ({"unknown": 1}, []):
        path.write_bytes(canonical_json(invalid))
        with pytest.raises(ValueError):
            DriftConfig.from_json(path)


def test_vocabulary_counts_absence_and_longest_match():
    from pharmasignal.monitoring.report import _symptom_presence

    columns = _symptom_presence(
        [
            {"narrative": "No skin rash. Nausea and nausea."},
            {"narrative": "Nothing reported."},
        ]
    )
    assert columns["skin rash"] == ["present", "absent"]
    assert columns["rash"] == ["absent", "absent"]
    assert columns["nausea"] == ["present", "absent"]
    assert all(len(values) == 2 for values in columns.values())


def test_report_insufficient_and_ignores_labels(reference, tmp_path):
    records, manifest, _ = reference
    batch, provenance = simulate_batch("unchanged", n=2)
    batch_manifest = write_batch(tmp_path / "batch", batch, provenance)
    report = build_report(records, manifest, batch, batch_manifest, DriftConfig())
    assert report["summary"]["overall_status"] == "insufficient_data"
    assert report["summary"]["alerts"] == 0
    for row in batch:
        row["signal"] = 1 - row["signal"]
    other = build_report(records, manifest, batch, batch_manifest, DriftConfig())
    assert report["sections"] == other["sections"]


def test_batch_integrity_and_unseen_categories(tmp_path):
    from pharmasignal.monitoring.report import _load_batch

    batch, provenance = simulate_batch("unchanged", n=10)
    batch[0]["treatment"] = "new_treatment"
    directory = tmp_path / "batch"
    manifest = write_batch(directory, batch, provenance)
    loaded, actual = _load_batch(directory)
    assert loaded == batch and actual == manifest
    assert actual["generator_config"] == provenance["generator_config"]
    assert actual["batch_sha256"] == sha256((directory / "batch.jsonl").read_bytes())
    with pytest.raises(FileExistsError):
        write_batch(directory, batch, provenance)
    with (directory / "batch.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="checksum"):
        _load_batch(directory)


@pytest.mark.parametrize("mutation", ["empty", "count", "nonsynthetic", "numeric", "object"])
def test_batch_loader_rejects_invalid_input(tmp_path, mutation):
    from pharmasignal.monitoring.report import _load_batch

    records, provenance = simulate_batch("unchanged", n=10)
    directory = tmp_path / "batch"
    manifest = write_batch(directory, records, provenance)
    if mutation == "empty":
        records = []
    elif mutation == "count":
        manifest["n"] = 9
    elif mutation == "nonsynthetic":
        manifest["synthetic"] = False
    elif mutation == "numeric":
        records[0]["dose_mg"] = "invalid"
    else:
        records[0] = 1
    payload = b"".join(canonical_json(row) for row in records)
    manifest["batch_sha256"] = sha256(payload)
    (directory / "batch.jsonl").write_bytes(payload)
    (directory / "manifest.json").write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError):
        _load_batch(directory)


@pytest.mark.parametrize("kwargs", [{"scenario": "invalid"}, {"scenario": "unchanged", "n": 1}])
def test_simulation_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        simulate_batch(**kwargs)


def test_monitoring_clis(reference, tmp_path, monkeypatch, capsys):
    from pharmasignal.monitoring.report import main as report_main
    from pharmasignal.monitoring.simulate import main as simulate_main

    directory = tmp_path / "batch"
    output = tmp_path / "report"
    monkeypatch.setattr(
        "sys.argv", ["simulate", "--scenario", "unchanged", "--n", "40", "--output", str(directory)]
    )
    simulate_main()
    assert "40 simulated rows" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        simulate_main()
    assert error.value.code == 1
    assert "Simulation failed" in capsys.readouterr().err
    monkeypatch.setattr(
        "sys.argv",
        [
            "report",
            "--reference",
            str(reference[2]),
            "--batch",
            str(directory),
            "--output",
            str(output),
        ],
    )
    report_main()
    report = json.loads((output / "report.json").read_text())
    assert json.loads(capsys.readouterr().out) == report["summary"]
    assert json.loads((output / "summary.json").read_text()) == report["summary"]
    assert report["identity"]["configuration_sha256"] == sha256(
        canonical_json(DriftConfig().to_dict())
    )
    before = (output / "report.json").read_bytes()
    with pytest.raises(SystemExit) as error:
        report_main()
    assert error.value.code == 1
    assert "destination already exists" in capsys.readouterr().err
    assert (output / "report.json").read_bytes() == before
