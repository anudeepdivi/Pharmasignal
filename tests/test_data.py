"""M1 contracts: strict inputs, deterministic generation and split isolation."""

import copy
import json
from dataclasses import replace

import pytest

from pharmasignal.data.build import build_dataset, canonical_json, load_config, sha256
from pharmasignal.data.generator import (
    GeneratorConfig,
    SignalCoefficients,
    _probabilities,
    assign_families,
    generate_dataset,
)
from pharmasignal.data.schema import AUDIT_FIELDS, FIELD_NAMES, NULLABLE_FIELDS, PREDICTORS
from pharmasignal.data.splits import normalize_narrative, validate_splits
from pharmasignal.data.validation import ValidationError, validate_record, validate_records


@pytest.fixture
def record():
    return {
        "report_id": "syn_example",
        "template_family": "family_01",
        "age": 40,
        "sex": "female",
        "treatment": "treatment_a",
        "dose_mg": 50.0,
        "exposure_days": 12,
        "baseline_egfr": 62.0,
        "concomitant_med_count": 3,
        "narrative": "Denies nausea.",
        "signal": 0,
    }


@pytest.fixture(scope="module")
def config():
    return GeneratorConfig(n_train=120, n_validation=60, n_test=60)


@pytest.fixture(scope="module")
def dataset(config):
    return generate_dataset(config)


def test_schema_and_nulls(record):
    validate_record(record)
    for name in NULLABLE_FIELDS:
        record[name] = None
    validate_record(record)
    assert set(record) == set(FIELD_NAMES)
    assert set(PREDICTORS).isdisjoint((*AUDIT_FIELDS, "signal"))
    assert len(PREDICTORS) == 8


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("age", 17),
        ("age", 91),
        ("age", True),
        ("age", 40.0),
        ("age", "40"),
        ("age", float("nan")),
        ("dose_mg", 0),
        ("dose_mg", 501),
        ("dose_mg", True),
        ("dose_mg", float("inf")),
        ("dose_mg", float("-inf")),
        ("dose_mg", float("nan")),
        ("dose_mg", 10**400),
        ("exposure_days", 0),
        ("exposure_days", 366),
        ("baseline_egfr", 9),
        ("baseline_egfr", 151),
        ("concomitant_med_count", -1),
        ("concomitant_med_count", 16),
        ("sex", "invalid"),
        ("sex", []),
        ("treatment", "real_drug"),
        ("treatment", None),
        ("narrative", " \n\t"),
        ("narrative", "x" * 4001),
        ("narrative", 123),
        ("report_id", " "),
        ("template_family", "x" * 65),
        ("signal", 2),
        ("signal", 1.0),
        ("signal", True),
        ("signal", "1"),
        ("signal", None),
    ],
)
def test_invalid_values(record, name, value):
    record[name] = value
    with pytest.raises(ValidationError, match=name):
        validate_record(record)


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_all_keys_required(record, name):
    del record[name]
    with pytest.raises(ValidationError, match="missing field"):
        validate_record(record)


def test_extra_fields_and_record_type(record):
    with pytest.raises(ValidationError, match="unknown field"):
        validate_record({**record, "future_outcome": 1})
    with pytest.raises(ValidationError, match="dict"):
        validate_record([])
    with pytest.raises(ValidationError, match="duplicate"):
        validate_records([record, record.copy()])
    with pytest.raises(ValidationError, match="nonempty"):
        validate_records([])


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("age", 18),
        ("age", 90),
        ("dose_mg", 0.1),
        ("dose_mg", 500),
        ("exposure_days", 1),
        ("exposure_days", 365),
        ("baseline_egfr", 10),
        ("baseline_egfr", 150),
        ("concomitant_med_count", 0),
        ("concomitant_med_count", 15),
        ("narrative", "x" * 4000),
        ("signal", 1),
    ],
)
def test_valid_boundaries(record, name, value):
    validate_record({**record, name: value})


def test_generation_reproducibility_and_isolation(config, dataset):
    assert dataset == generate_dataset(config)
    other = generate_dataset(replace(config, seed=43))
    assert dataset != other
    assert assign_families(config) != assign_families(replace(config, seed=43))
    resized = generate_dataset(replace(config, n_train=121))
    assert resized["validation"] == dataset["validation"]
    assert resized["test"] == dataset["test"]
    assert resized["train"][:120] == dataset["train"]
    checks = validate_splits(dataset, assign_families(config))
    assert all(not any(counts.values()) for counts in checks["cross_split_overlap_counts"].values())
    for name, rows in dataset.items():
        assert len(rows) == getattr(config, f"n_{name}")
        validate_records(rows)
        assert {r["signal"] for r in rows} == {0, 1}
        assert all(set(r) == set(FIELD_NAMES) for r in rows)


def test_text_and_missingness_do_not_change_labels(config, dataset):
    changed = generate_dataset(
        replace(
            config,
            missingness={key: 1 for key in NULLABLE_FIELDS},
            p_underreport=1,
        )
    )
    for name in dataset:
        assert [r["signal"] for r in changed[name]] == [r["signal"] for r in dataset[name]]
        assert all(r[key] is None for r in changed[name] for key in NULLABLE_FIELDS)
        assert all("Reports " not in r["narrative"] for r in changed[name])


def test_sex_not_in_label_function(record):
    coef = SignalCoefficients()
    for event in (0, 1):
        assert _probabilities(record, coef, event) == _probabilities(
            {**record, "sex": "male"}, coef, event
        )
        assert all(0 < value < 1 for value in _probabilities(record, coef, event))


@pytest.mark.parametrize("field", ["report_id", "template_family", "narrative"])
def test_overlap_rejected(dataset, field):
    data = copy.deepcopy(dataset)
    data["test"][0][field] = data["train"][0][field]
    if field == "narrative":
        data["test"][0][field] = "  " + data["test"][0][field].upper() + "\n"
    with pytest.raises(ValidationError, match="overlap"):
        validate_splits(data)


def test_single_class_and_bad_assignments(config, dataset):
    data = copy.deepcopy(dataset)
    for row in data["test"]:
        row["signal"] = 0
    with pytest.raises(ValidationError, match="both target classes"):
        validate_splits(data)
    with pytest.raises(ValidationError, match="both target classes"):
        generate_dataset(replace(config, coefficients=SignalCoefficients(b0=-1000)))
    assignments = assign_families(config)
    assignments["test"] = assignments["train"]
    with pytest.raises(ValidationError, match="unassigned"):
        validate_splits(dataset, assignments)
    with pytest.raises(ValidationError, match="exactly"):
        validate_splits({"train": dataset["train"]})


def test_normalization():
    assert normalize_narrative("  ＮＯ\tNausea.\n") == "no nausea."


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"seed": -1},
        {"n_train": 0},
        {"n_test": 2.5},
        {"p_underreport": float("nan")},
        {"p_false_alarm": 2},
        {"p_benign_mention": True},
        {"missingness": {"signal": 0.1}},
        {"missingness": {"age": -0.1}},
        {"coefficients": SignalCoefficients(b0=float("inf"))},
    ],
)
def test_config_rejected(kwargs):
    with pytest.raises(ValueError):
        GeneratorConfig(**kwargs)


def test_persistence_manifest_and_no_overwrite(tmp_path, config, dataset):
    output = tmp_path / "data"
    manifest = build_dataset(output, config)
    saved = json.loads((output / "manifest.json").read_text())
    assert saved["synthetic"] is True
    assert saved["config_sha256"] == sha256(canonical_json(config.to_manifest_dict()))
    assert saved["dataset_id"] == sha256(canonical_json(saved["files_sha256"]))
    for name, rows in dataset.items():
        payload = (output / f"{name}.jsonl").read_bytes()
        assert sha256(payload) == saved["files_sha256"][f"{name}.jsonl"]
        assert [json.loads(line) for line in payload.splitlines()] == rows
        assert all(
            item["record_sha256"] == sha256(canonical_json(row))
            for item, row in zip(saved["records"][name], rows, strict=True)
        )
    assert build_dataset(tmp_path / "repeat", config)["dataset_id"] == manifest["dataset_id"]
    with pytest.raises(FileExistsError):
        build_dataset(output, config)
    assert json.loads((output / "manifest.json").read_text()) == saved
    path = tmp_path / "config.json"
    path.write_bytes(canonical_json(config.to_manifest_dict()))
    assert load_config(path) == config
    path.write_text('{"unknown": 1}')
    with pytest.raises(TypeError):
        load_config(path)


def test_invalid_config_does_not_write(tmp_path, config):
    invalid = replace(config, coefficients=SignalCoefficients(b0=-1000))
    output = tmp_path / "bad"
    with pytest.raises(ValidationError):
        build_dataset(output, invalid)
    assert not output.exists()
