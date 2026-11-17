"""Seeded fictional reports, not medical knowledge or patient observations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from pharmasignal.data.schema import NULLABLE_FIELDS
from pharmasignal.data.validation import validate_records

PARTITIONS = ("train", "validation", "test")
SYMPTOMS = (
    "nausea",
    "vomiting",
    "skin rash",
    "headache",
    "dizziness",
    "fatigue",
    "fever",
    "abdominal pain",
)
# Families differ in documentation style, not label policy. They intentionally
# share symptom vocabulary. Fixed wrappers also make this an artificial task.
TEMPLATES = {
    "family_01": "Clinic note. {context} {observations}",
    "family_02": "Patient message: {observations} Current exposure: {context}",
    "family_03": "Intake summary | {context} | {observations}",
    "family_04": "During review, the following was documented. {observations} {context}",
    "family_05": "Telephone documentation: {context} Findings: {observations}",
    "family_06": "Caregiver report. {observations} Treatment details: {context}",
    "family_07": "Pharmacy follow-up. {context} Reported information: {observations}",
    "family_08": "Study visit entry. {observations} Exposure summary: {context}",
    "family_09": "Nursing contact record. {context} Review notes: {observations}",
    "family_10": "Portal update. {observations} Regimen information: {context}",
}


@dataclass(frozen=True)
class SignalCoefficients:
    """Arbitrary coefficients for this software demonstration only."""

    g0: float = -1.6
    g1_dose: float = 0.9
    g2_exposure: float = 0.7
    g3_age: float = 0.5
    g4_egfr: float = 0.8
    g5_meds: float = 0.5
    g6_treatment_b: float = 0.25
    b0: float = -2.6
    b1_event: float = 2.6
    b2_dose: float = 0.6
    b3_exposure: float = 0.4
    b4_egfr: float = 0.5
    b5_meds: float = 0.3


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 42
    n_train: int = 1800
    n_validation: int = 600
    n_test: int = 600
    coefficients: SignalCoefficients = field(default_factory=SignalCoefficients)
    p_underreport: float = 0.15
    p_false_alarm: float = 0.25
    p_benign_mention: float = 0.4
    missingness: dict[str, float] = field(
        default_factory=lambda: {
            "age": 0.05,
            "sex": 0.04,
            "dose_mg": 0.06,
            "exposure_days": 0.05,
            "baseline_egfr": 0.12,
            "concomitant_med_count": 0.08,
        }
    )

    def __post_init__(self) -> None:
        for name in ("seed", "n_train", "n_validation", "n_test"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 2
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.coefficients, SignalCoefficients):
            raise ValueError("coefficients must be SignalCoefficients")
        for name, value in asdict(self.coefficients).items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"coefficient {name} must be finite")
        if not isinstance(self.missingness, dict):
            raise ValueError("missingness must be a dict")
        if set(self.missingness) - NULLABLE_FIELDS:
            raise ValueError("missingness may only name nullable fields")
        rates = {
            name: getattr(self, name)
            for name in (
                "p_underreport",
                "p_false_alarm",
                "p_benign_mention",
            )
        }
        rates.update(self.missingness)
        for name, value in rates.items():
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite probability in [0, 1]")

    def to_manifest_dict(self) -> dict:
        return asdict(self)


def assign_families(config: GeneratorConfig) -> dict[str, tuple[str, ...]]:
    rng = np.random.default_rng([config.seed, 777])
    shuffled = [str(name) for name in rng.permutation(list(TEMPLATES))]
    return dict(
        zip(
            PARTITIONS,
            (tuple(shuffled[:6]), tuple(shuffled[6:8]), tuple(shuffled[8:])),
            strict=True,
        )
    )


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _sample_structured(rng: np.random.Generator) -> dict:
    return {
        "age": int(np.clip(round(float(rng.normal(54, 16))), 18, 90)),
        "sex": str(rng.choice(("female", "male", "other_or_unknown"), p=(0.49, 0.49, 0.02))),
        "treatment": str(rng.choice(("treatment_a", "treatment_b", "treatment_c"))),
        "dose_mg": round(float(rng.uniform(10, 400)), 1),
        "exposure_days": int(rng.integers(1, 366)),
        "baseline_egfr": round(float(np.clip(rng.normal(75, 20), 15, 140)), 1),
        "concomitant_med_count": int(min(rng.poisson(4), 15)),
    }


def _probabilities(values: dict, coef: SignalCoefficients, event: int = 0) -> tuple[float, float]:
    dose = (values["dose_mg"] - 205) / 195
    exposure = math.log1p(values["exposure_days"]) / math.log(366)
    age = (values["age"] - 54) / 36
    egfr = (75 - values["baseline_egfr"]) / 65
    meds = values["concomitant_med_count"] / 15
    p_event = _sigmoid(
        coef.g0
        + coef.g1_dose * dose
        + coef.g2_exposure * exposure
        + coef.g3_age * age
        + coef.g4_egfr * egfr
        + coef.g5_meds * meds
        + coef.g6_treatment_b * (values["treatment"] == "treatment_b")
    )
    p_signal = _sigmoid(
        coef.b0
        + coef.b1_event * event
        + coef.b2_dose * dose
        + coef.b3_exposure * exposure
        + coef.b4_egfr * egfr
        + coef.b5_meds * meds
    )
    return p_event, p_signal


def _render(
    rng: np.random.Generator, values: dict, event: int, family: str, config: GeneratorConfig
) -> str:
    pool = [str(s) for s in rng.permutation(SYMPTOMS)]
    asserted = []
    if event and rng.random() >= config.p_underreport:
        asserted = [pool.pop() for _ in range(int(rng.integers(1, 4)))]
    denied = [pool.pop() for _ in range(int(rng.integers(1, 4)))]
    observations = []
    if asserted:
        observations.append("Reports " + " and ".join(asserted) + " after treatment began.")
    observations.append(
        str(rng.choice(("Denies ", "No ", "Without "))) + " and ".join(denied) + "."
    )
    if not event and rng.random() < config.p_false_alarm:
        observations.append(f"Long-standing {pool.pop()}, unchanged from baseline.")
    if rng.random() < config.p_benign_mention:
        observations.append("Pre-existing joint stiffness, unchanged.")
    context = f"{values['treatment']}, day {values['exposure_days']} of therapy."
    return TEMPLATES[family].format(context=context, observations=" ".join(observations))


def generate_partition(
    partition: str, families: tuple[str, ...], n: int, config: GeneratorConfig
) -> list[dict]:
    if partition not in PARTITIONS or not families or set(families) - TEMPLATES.keys():
        raise ValueError("invalid partition or template family assignment")
    if type(n) is not int or n < 2:
        raise ValueError("partition size must be an integer >= 2")
    code = PARTITIONS.index(partition) + 1
    # Independent streams also keep labels unchanged when missingness or text
    # rendering settings change. Use fixed integer stream keys, not hash().
    structured_rng = np.random.default_rng([config.seed, code, 1])
    label_rng = np.random.default_rng([config.seed, code, 2])
    text_rng = np.random.default_rng([config.seed, code, 3])
    missing_rng = np.random.default_rng([config.seed, code, 4])
    family_rng = np.random.default_rng([config.seed, code, 5])
    records = []
    for index in range(n):
        values = _sample_structured(structured_rng)
        family = str(family_rng.choice(families))
        p_event, _ = _probabilities(values, config.coefficients)
        event = int(label_rng.random() < p_event)
        _, p_signal = _probabilities(values, config.coefficients, event)
        signal = int(label_rng.random() < p_signal)
        narrative = _render(text_rng, values, event, family, config)
        record = {
            "report_id": f"syn_{partition}_{index + 1:06d}",
            "template_family": family,
            **values,
            "narrative": narrative,
            "signal": signal,
        }
        for name in sorted(NULLABLE_FIELDS):
            if missing_rng.random() < config.missingness.get(name, 0):
                record[name] = None
        records.append(record)
    validate_records(records)
    return records


def generate_dataset(config: GeneratorConfig | None = None) -> dict[str, list[dict]]:
    from pharmasignal.data.splits import validate_splits

    config = config or GeneratorConfig()
    config.__post_init__()  # Recheck mutable nested configuration before use.
    assignments = assign_families(config)
    partitions = {
        name: generate_partition(name, assignments[name], getattr(config, f"n_{name}"), config)
        for name in PARTITIONS
    }
    validate_splits(partitions, assignments)
    return partitions
