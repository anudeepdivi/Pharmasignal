"""Transparent tokenization and mention counts, not clinical NER.

Blank spaCy English supplies tokenization only. Negation covers at most four
following tokens, stops at punctuation/contrast, and resets on another cue.
History markers are separate counts: an unnegated mention is not necessarily
new, caused by treatment, or clinically concerning.
"""

from __future__ import annotations

import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

_NLP: spacy.Language | None = None
NEGATION_CUES = frozenset({"no", "not", "denies", "without", "never"})
SCOPE_ENDS = frozenset({".", "!", "?", ";", ":", ",", "but", "however", "yet"})
NEGATION_WINDOW = 4
# Authored vocabulary covers the generator's symptoms plus the rash alias.
# This is demonstration-specific, not independently clinically curated.
SYMPTOM_LEXICON = (
    ("skin", "rash"),
    ("abdominal", "pain"),
    ("nausea",),
    ("vomiting",),
    ("rash",),
    ("headache",),
    ("dizziness",),
    ("fatigue",),
    ("fever",),
)
BENIGN_MARKERS = (("unchanged",), ("pre", "-", "existing"), ("long", "-", "standing"))
RULE_FEATURE_NAMES = (
    "unnegated_symptom_count",
    "negated_symptom_count",
    "history_marker_count",
    "token_count",
)


def get_tokenizer() -> spacy.Language:
    """Lazy, process-local tokenizer; never serialized inside a transformer."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.blank("en")
    return _NLP


def raw_tokens(text: str) -> list[str]:
    """Lowercase tokens including punctuation, excluding whitespace tokens."""
    if not isinstance(text, str):
        raise TypeError("narrative must be a string")
    return [token.lower_ for token in get_tokenizer().make_doc(text) if not token.is_space]


def tokenize(text: str) -> list[str]:
    """TF-IDF tokens: retain numbers, negation and temporality, omit punctuation.

    This can form bigrams across sentence boundaries; the rule features use
    the separate punctuation-preserving view instead.
    """
    if not isinstance(text, str):
        raise TypeError("narrative must be a string")
    return [
        token.lower_
        for token in get_tokenizer().make_doc(text)
        if not (token.is_punct or token.is_space)
    ]


def negation_mask(tokens: list[str]) -> list[bool]:
    """Scope mask for raw_tokens output; stripped punctuation cannot be recovered."""
    remaining = 0
    mask = []
    for token in tokens:
        if token in SCOPE_ENDS:
            remaining = 0
            mask.append(False)
        elif token in NEGATION_CUES:
            remaining = NEGATION_WINDOW
            mask.append(False)
        else:
            mask.append(remaining > 0)
            remaining = max(0, remaining - 1)
    return mask


def count_symptom_mentions(tokens: list[str], mask: list[bool]) -> tuple[int, int]:
    """Longest match wins; classify a mention using its first token's scope."""
    if len(tokens) != len(mask):
        raise ValueError("tokens and mask must have equal lengths")
    unnegated = negated = position = 0
    while position < len(tokens):
        for entry in sorted(SYMPTOM_LEXICON, key=len, reverse=True):
            if tuple(tokens[position : position + len(entry)]) == entry:
                negated += int(mask[position])
                unnegated += int(not mask[position])
                position += len(entry)
                break
        else:
            position += 1
    return unnegated, negated


def rule_features(text: str) -> list[float]:
    tokens = raw_tokens(text)
    unnegated, negated = count_symptom_mentions(tokens, negation_mask(tokens))
    history = sum(
        tuple(tokens[index : index + len(entry)]) == entry
        for index in range(len(tokens))
        for entry in BENIGN_MARKERS
    )
    return [float(unnegated), float(negated), float(history), float(len(tokenize(text)))]


class NarrativeRuleFeatures(TransformerMixin, BaseEstimator):
    """Four deterministic counts per string; input must be one-dimensional."""

    def fit(self, X, y=None):
        self.n_features_out_ = len(RULE_FEATURE_NAMES)
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, "n_features_out_")
        texts = np.asarray(X, dtype=object)
        if texts.ndim != 1:
            raise ValueError("expected a one-dimensional sequence of narratives")
        return np.asarray([rule_features(text) for text in texts], dtype=float).reshape(
            len(texts), len(RULE_FEATURE_NAMES)
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_out_")
        return np.asarray(RULE_FEATURE_NAMES, dtype=object)
