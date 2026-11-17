"""Drift thresholds decided before seeing any batch; p-values are never gates."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

FIELDS = (
    "min_samples",
    "ks_alert",
    "effect_alert",
    "tv_alert",
    "missingness_alert",
)


@dataclass(frozen=True)
class DriftConfig:
    """Pre-registered comparison thresholds.

    Alerts fire on the KS statistic, standardized mean difference or total-
    variation distance crossing these constants — never on a p-value, which
    grows significant with sample size regardless of practical importance.
    """

    min_samples: int = 30
    ks_alert: float = 0.15
    effect_alert: float = 0.5
    tv_alert: float = 0.2
    missingness_alert: float = 0.15

    def __post_init__(self) -> None:
        if type(self.min_samples) is not int or self.min_samples < 2:
            raise ValueError("min_samples must be an integer >= 2")
        for name in FIELDS[1:]:
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            if name != "effect_alert" and value > 1:
                raise ValueError(f"{name} must be in (0, 1]")

    @classmethod
    def from_json(cls, path: Path) -> DriftConfig:
        values = json.loads(Path(path).read_text())
        if not isinstance(values, dict) or set(values) - set(FIELDS):
            raise ValueError(f"configuration must contain only {list(FIELDS)}")
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)
