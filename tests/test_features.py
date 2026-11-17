"""M2 contracts: exact linguistic cases, train-only fitting and stable fusion."""

import copy
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from pharmasignal.data.generator import GeneratorConfig, generate_dataset
from pharmasignal.data.schema import PREDICTORS
from pharmasignal.data.validation import ValidationError
from pharmasignal.features.fusion import (
    MISSING_CATEGORY,
    NUMERIC_FIELDS,
    assemble_features,
    build_feature_pipeline,
    describe_fitted_features,
    feature_names,
)
from pharmasignal.features.text import (
    RULE_FEATURE_NAMES,
    NarrativeRuleFeatures,
    count_symptom_mentions,
    negation_mask,
    raw_tokens,
    rule_features,
    tokenize,
)


@pytest.fixture(scope="module")
def records():
    return generate_dataset(GeneratorConfig(n_train=300, n_validation=60, n_test=60))["train"]


@pytest.fixture(scope="module")
def frame(records):
    return assemble_features(records)


@pytest.fixture(scope="module")
def fitted(frame):
    return build_feature_pipeline().fit(frame)


def test_tokenize():
    assert tokenize(" Denies nausea. Reports vomiting. ") == [
        "denies",
        "nausea",
        "reports",
        "vomiting",
    ]
    assert tokenize("No headache, without fever on day 12.") == [
        "no",
        "headache",
        "without",
        "fever",
        "on",
        "day",
        "12",
    ]
    assert tokenize("") == []
    for function in (tokenize, raw_tokens, rule_features):
        with pytest.raises(TypeError):
            function(None)


def test_raw_negation_contract():
    tokens = raw_tokens("No nausea and dizziness. Reports fatigue.")
    assert tokens == ["no", "nausea", "and", "dizziness", ".", "reports", "fatigue", "."]
    assert negation_mask(tokens) == [False, True, True, True, False, False, False, False]
    with pytest.raises(ValueError, match="equal lengths"):
        count_symptom_mentions(tokens, [])


@pytest.mark.parametrize(
    ("text", "unnegated", "negated", "history"),
    [
        ("No nausea. Vomiting.", 1, 1, 0),
        ("Denies nausea and skin rash. Reports vomiting.", 1, 2, 0),
        ("No headache, reports dizziness.", 1, 1, 0),
        ("No fever; nausea.", 1, 1, 0),
        ("No nausea but vomiting.", 1, 1, 0),
        ("No nausea without fever and dizziness.", 0, 3, 0),
        ("skin rash and rash", 2, 0, 0),
        ("abdominal. pain", 0, 0, 0),
        ("No one two three four fever", 1, 0, 0),  # Fixed window limitation.
        ("Denies nausea and abdominal pain", 0, 2, 0),
        ("Long-standing nausea, unchanged. Pre-existing fatigue.", 2, 0, 3),
        ("Not only nausea but vomiting", 1, 1, 0),  # Known pseudo-negation failure.
        ("No history of fever", 0, 1, 0),  # Does not distinguish current/history absence.
        ("", 0, 0, 0),
    ],
)
def test_rule_counts(text, unnegated, negated, history):
    assert rule_features(text) == [unnegated, negated, history, len(tokenize(text))]


def test_rule_transformer():
    transformer = NarrativeRuleFeatures()
    with pytest.raises(NotFittedError):
        transformer.transform(["x"])
    output = transformer.fit_transform(pd.Series(["No nausea.", "Fever."]))
    assert output.shape == (2, 4)
    np.testing.assert_array_equal(output[:, :2], [[0, 1], [1, 0]])
    assert list(transformer.get_feature_names_out()) == list(RULE_FEATURE_NAMES)
    assert transformer.transform([]).shape == (0, 4)
    with pytest.raises(ValueError, match="one-dimensional"):
        transformer.transform(pd.DataFrame({"narrative": ["No nausea."]}))


def test_assembly_projects_and_validates(records):
    before = copy.deepcopy(records[:2])
    result = assemble_features(before)
    assert list(result.columns) == list(PREDICTORS)
    assert before == records[:2]
    changed = [
        {**r, "signal": 999, "template_family": "ignored", "future": "ignored"} for r in before
    ]
    pd.testing.assert_frame_equal(result, assemble_features(changed))
    predictor_only = [{name: r[name] for name in PREDICTORS} for r in before]
    pd.testing.assert_frame_equal(result, assemble_features(predictor_only))
    for invalid in ([], None):
        with pytest.raises(ValueError):
            assemble_features(invalid)
    with pytest.raises(TypeError):
        assemble_features([None])
    with pytest.raises(KeyError):
        assemble_features([{}])
    for name, value in (
        ("age", "40"),
        ("age", True),
        ("dose_mg", float("inf")),
        ("narrative", " "),
        ("sex", "unknown-code"),
        ("treatment", None),
    ):
        with pytest.raises(ValidationError):
            assemble_features([{**before[0], name: value}])


def test_fit_shapes_names_and_determinism(frame, fitted):
    matrix = fitted.transform(frame)
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape[0] == len(frame)
    assert np.isfinite(matrix.data).all()
    names = feature_names(fitted)
    summary = describe_fitted_features(fitted)
    assert len(names) == len(set(names)) == matrix.shape[1] == summary["total"]
    assert summary["blocks"]["numeric"] == 10
    assert summary["blocks"]["text_rules"] == 4
    assert all(value > 0 for value in summary["blocks"].values())
    assert sum(summary["blocks"].values()) == summary["total"]
    np.testing.assert_array_equal(matrix.toarray(), fitted.transform(frame).toarray())
    np.testing.assert_allclose(matrix.toarray(), clone(fitted).fit_transform(frame).toarray())
    # ColumnTransformer uses names, not positional order.
    np.testing.assert_array_equal(
        matrix.toarray(), fitted.transform(frame[frame.columns[::-1]]).toarray()
    )
    for field in NUMERIC_FIELDS:
        assert f"numeric__missingindicator_{field}" in names
    with pytest.raises(NotFittedError):
        feature_names(build_feature_pipeline())


def test_actual_holdout_does_not_change_fitted_state(frame, fitted):
    before = joblib.hash(fitted)
    fusion = fitted.named_steps["fusion"]
    numeric = fusion.named_transformers_["numeric"]
    np.testing.assert_allclose(
        numeric.named_steps["impute"].statistics_, frame[list(NUMERIC_FIELDS)].median()
    )
    tfidf = fusion.named_transformers_["tfidf"]
    assert "no" in tfidf.vocabulary_ and "denies" in tfidf.vocabulary_
    holdout = frame.iloc[:2].copy()
    holdout["narrative"] = "uniquetestwordxyz fever"
    holdout["age"] = 90
    holdout["dose_mg"] = 500
    transformed = fitted.transform(holdout)
    assert transformed.shape[1] == fitted.transform(frame).shape[1]
    assert "uniquetestwordxyz" not in tfidf.vocabulary_
    assert joblib.hash(fitted) == before  # Imputer, categories, scalers, IDF all unchanged.


def test_unseen_valid_category_and_nulls(frame):
    train = frame.copy()
    train["sex"] = "female"
    fitted = build_feature_pipeline().fit(train)
    categorical = fitted.named_steps["fusion"].named_transformers_["categorical"]
    names = list(categorical.get_feature_names_out())
    sex_columns = [i for i, name in enumerate(names) if name.startswith("sex_")]
    heldout = train.iloc[:2].copy()
    heldout["sex"] = ["male", None]  # Neither seen during fit, but valid schema values.
    encoded = categorical.transform(heldout[["sex", "treatment"]]).toarray()
    assert np.all(encoded[:, sex_columns] == 0)
    assert "sex_male" not in names
    train.loc[train.index[0], "sex"] = None
    categorical = (
        build_feature_pipeline().fit(train).named_steps["fusion"].named_transformers_["categorical"]
    )
    assert f"sex_{MISSING_CATEGORY}" in categorical.get_feature_names_out()
    encoded = categorical.transform(heldout[["sex", "treatment"]]).toarray()
    missing_index = list(categorical.get_feature_names_out()).index(f"sex_{MISSING_CATEGORY}")
    assert encoded[1, missing_index] == 1


def test_all_missing_numeric_and_categorical(frame):
    train = frame.copy()
    train.loc[:, list(NUMERIC_FIELDS)] = np.nan
    train["sex"] = None
    fitted = build_feature_pipeline().fit(train)
    matrix = fitted.transform(train)
    assert sparse.isspmatrix_csr(matrix) and np.isfinite(matrix.data).all()
    numeric = fitted.named_steps["fusion"].named_transformers_["numeric"]
    np.testing.assert_array_equal(numeric.named_steps["impute"].statistics_, np.zeros(5))
    assert numeric.transform(train[list(NUMERIC_FIELDS)]).shape[1] == 10
    assert np.isfinite(fitted.transform(frame).data).all()


def test_new_missingness_keeps_shape(frame):
    train = frame.copy()
    train.loc[:, list(NUMERIC_FIELDS)] = train[list(NUMERIC_FIELDS)].fillna(30)
    fitted = build_feature_pipeline().fit(train)
    holdout = train.iloc[:1].copy()
    holdout["age"] = np.nan
    assert fitted.transform(holdout).shape[1] == fitted.transform(train).shape[1]
    assert "numeric__missingindicator_age" not in feature_names(fitted)


@pytest.mark.parametrize("value", ["fifty", "50", True, float("inf")])
def test_direct_numeric_block_rejects_malformed(frame, fitted, value):
    bad = frame.iloc[:1].copy().astype(object)
    bad.loc[bad.index[0], "dose_mg"] = value
    with pytest.raises((TypeError, ValueError)):
        fitted.transform(bad)


def test_tiny_corpus_and_empty_vocabulary(frame):
    tiny = frame.iloc[:1].copy()
    tiny["narrative"] = "fever"
    with pytest.raises(ValueError):
        build_feature_pipeline().fit(tiny)  # Default min_df=2 needs sufficient support.
    output = build_feature_pipeline(min_df=1).fit_transform(tiny)
    assert sparse.isspmatrix_csr(output)
    tiny["narrative"] = "!!!"
    with pytest.raises(ValueError, match="empty vocabulary"):
        build_feature_pipeline(min_df=1).fit(tiny)
    for kwargs in ({"min_df": 0}, {"min_df": True}, {"max_features": -1}):
        with pytest.raises(ValueError):
            build_feature_pipeline(**kwargs)


def test_serialization_in_fresh_process(frame, fitted, tmp_path):
    path = tmp_path / "features.joblib"
    inputs = tmp_path / "inputs.json"
    output = tmp_path / "output.npz"
    joblib.dump(fitted, path)
    frame.iloc[:3].to_json(inputs, orient="records")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import joblib,pandas as pd; from scipy.sparse import save_npz; "
            "import sys; p=joblib.load(sys.argv[1]); "
            "save_npz(sys.argv[3],p.transform(pd.read_json(sys.argv[2])))",
            str(path),
            str(inputs),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    np.testing.assert_allclose(
        sparse.load_npz(output).toarray(), fitted.transform(frame.iloc[:3]).toarray()
    )
    assert feature_names(joblib.load(path)) == feature_names(fitted)
