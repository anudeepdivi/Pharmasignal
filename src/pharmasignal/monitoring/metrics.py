"""Pure distribution comparisons: KS, standardized mean difference, TV distance.

Every row carries sample counts and a status decided by pre-registered
thresholds. Narratives and identifiers never appear in returned rows.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import stats

from pharmasignal.monitoring.config import DriftConfig

OK, ALERT, INSUFFICIENT, UNDEFINED = "ok", "alert", "insufficient_data", "undefined_constant"


def numeric_comparison(name: str, reference, batch, config: DriftConfig) -> dict:
    """Two-sample Kolmogorov-Smirnov statistic and standardized mean difference.

    Effect size is the absolute mean difference divided by the root mean
    of the two sample variances. It is undefined when both variances are zero;
    KS can still alert in that case. The p-value never determines status.
    """
    a = np.asarray(reference, dtype=float)
    b = np.asarray(batch, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    row = {"feature": name, "n_reference": int(a.size), "n_batch": int(b.size)}
    if min(a.size, b.size) < config.min_samples:
        row.update(status=INSUFFICIENT, ks_statistic=None, ks_pvalue=None, effect_size=None)
        return row
    # method="asymp" avoids scipy's exact-attempt fallback, which emits a
    # RuntimeWarning on moderate n (the suite runs with warnings as errors).
    # The KS statistic is identical either way; only the recorded p-value's
    # small-sample accuracy differs, and the p-value never gates status.
    ks = stats.ks_2samp(a, b, method="asymp")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    # Absolute value: drift in either direction is drift; a one-sided sign
    # would silently suppress alerts on upward shifts.
    effect = abs(float(a.mean() - b.mean())) / float(pooled) if pooled > 0 else None
    row.update(
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        effect_size=effect,
        status=(
            ALERT
            if ks.statistic >= config.ks_alert
            or (effect is not None and effect >= config.effect_alert)
            else UNDEFINED
            if effect is None
            else OK
        ),
    )
    return row


def categorical_comparison(name: str, reference, batch, config: DriftConfig) -> dict:
    """Total-variation distance over the union of observed categories.

    Unseen batch categories are flagged and included in the distance; they are
    expected artifacts of simulated batches, not input errors to reject here.
    """
    a, b = Counter(reference), Counter(batch)
    categories = sorted(set(a) | set(b))
    n_a, n_b = sum(a.values()), sum(b.values())
    row = {"feature": name, "n_reference": n_a, "n_batch": n_b, "unseen_categories": []}
    if min(n_a, n_b) < config.min_samples:
        row.update(status=INSUFFICIENT, tv_distance=None, proportions=None)
        return row
    proportions = {}
    distance = 0.0
    for category in categories:
        p_a, p_b = a[category] / n_a, b[category] / n_b
        proportions[category] = {"reference": p_a, "batch": p_b}
        if category not in a:
            row["unseen_categories"].append(category)
        distance += abs(p_a - p_b)
    row.update(
        tv_distance=distance / 2,
        proportions=proportions,
        status=ALERT if distance / 2 >= config.tv_alert else OK,
    )
    return row


def missingness_comparison(
    reference_records: list[dict], batch_records: list[dict], fields, config: DriftConfig
) -> list[dict]:
    """Per-field null rates; a missingness shift is data drift, not a label."""
    rows = []
    for field in fields:
        a = [r[field] is None for r in reference_records]
        b = [r[field] is None for r in batch_records]
        row = {
            "feature": field,
            "n_reference": len(a),
            "n_batch": len(b),
            "reference_rate": sum(a) / len(a) if a else None,
            "batch_rate": sum(b) / len(b) if b else None,
        }
        row["absolute_difference"] = (
            abs(row["reference_rate"] - row["batch_rate"]) if a and b else None
        )
        row["status"] = (
            INSUFFICIENT
            if min(len(a), len(b)) < config.min_samples
            else (ALERT if row["absolute_difference"] >= config.missingness_alert else OK)
        )
        rows.append(row)
    return rows
