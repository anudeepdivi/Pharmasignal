"""M3 ordering contract: train fit -> validation selection -> one test pass."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from pharmasignal.data.build import build_dataset
from pharmasignal.data.generator import GeneratorConfig
from pharmasignal.models.train import TestAccessGuard, train_baselines


def test_test_access_guard():
    guard = TestAccessGuard()
    with pytest.raises(RuntimeError, match="frozen"):
        guard.consume()
    guard.freeze_selection()
    guard.consume()
    with pytest.raises(RuntimeError, match="once"):
        guard.consume()
    with pytest.raises(RuntimeError):
        guard.freeze_selection()


def test_train_validation_then_exactly_one_test_pass(tmp_path):
    data_path = tmp_path / "data"
    build_dataset(data_path, GeneratorConfig(n_train=120, n_validation=60, n_test=60))
    output = tmp_path / "model"
    events = []
    train_ids = set()
    validation_ids = set()
    test_ids = set()
    for name, destination in (
        ("train", train_ids),
        ("validation", validation_ids),
        ("test", test_ids),
    ):
        destination.update(
            r["narrative"]
            for r in map(json.loads, (data_path / f"{name}.jsonl").read_text().splitlines())
        )

    class SpyPipeline:
        def __init__(self, variant):
            self.variant = variant

        def fit(self, X, y):
            assert set(X["narrative"]) <= train_ids
            events.append(("fit", self.variant))
            return self

        def predict_proba(self, X):
            narratives = set(X["narrative"])
            split = "validation" if narratives <= validation_ids else "test"
            if split == "test":
                assert narratives <= test_ids
                assert events[-1][0] == "test_read"
            events.append(("predict", split))
            probabilities = np.linspace(0.1, 0.9, len(X))
            return np.column_stack((1 - probabilities, probabilities))

    from pharmasignal.models import train

    original_read = train.read_split
    original_freeze = TestAccessGuard.freeze_selection

    def traced_read(root, manifest, split):
        if split == "test":
            assert events[-1][0] == "frozen"
            events.append(("test_read", split))
        return original_read(root, manifest, split)

    def traced_freeze(self):
        assert events[-1] == ("predict", "validation")
        original_freeze(self)
        events.append(("frozen", None))

    # Persistence is exercised separately with a real sklearn model. The spy
    # verifies control flow independent of what an estimator learns.
    with (
        patch.object(train, "make_model", side_effect=lambda variant, seed: SpyPipeline(variant)),
        patch.object(train, "read_split", side_effect=traced_read),
        patch.object(TestAccessGuard, "freeze_selection", traced_freeze),
        patch.object(train, "save_artifact"),
        patch.object(train, "coefficient_summary", return_value=[]),
    ):
        result = train_baselines(data_path, output)
    assert [event for event in events if event[0] == "fit"] == [
        ("fit", "structured"),
        ("fit", "text"),
        ("fit", "fused"),
    ]
    assert events.count(("predict", "validation")) == 3
    assert events.count(("test_read", "test")) == 1
    assert events.count(("predict", "test")) == 1
    assert events[-1] == ("predict", "test")
    assert result["selection"]["training_scope"] == "train_only_no_refit"
