"""Binary score evaluation and validation-only operating-threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def checked_arrays(labels, scores):
    y = np.asarray(labels)
    p = np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("labels and scores must be nonempty equal-length vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("scores must be finite probabilities")
    return y.astype(int), p


def evaluate(labels, scores, threshold: float = 0.5) -> dict:
    y, p = checked_arrays(labels, scores)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    predicted = (p >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        predicted,
        average="binary",
        zero_division=0,
    )
    both = len(np.unique(y)) == 2
    return {
        "rows": len(y),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "auc_status": "defined" if both else "undefined_single_class",
    }


def select_threshold(labels, scores, recall_target: float = 0.8) -> dict:
    """Maximize precision subject to recall >= target on validation only.

    Ties prefer recall, then the larger threshold. Score >= threshold is
    positive. Threshold zero ensures a feasible all-positive baseline when
    positives exist. No claim that this arbitrary target is clinically safe.
    """
    y, p = checked_arrays(labels, scores)
    if len(np.unique(y)) != 2:
        raise ValueError("threshold selection requires both classes")
    if not np.isfinite(recall_target) or not 0 < recall_target <= 1:
        raise ValueError("recall_target must be in (0, 1]")
    candidates = [evaluate(y, p, float(t)) for t in np.unique(np.r_[0.0, p])]
    feasible = [row for row in candidates if row["recall"] >= recall_target]
    best = max(feasible, key=lambda row: (row["precision"], row["recall"], row["threshold"]))
    return {
        "threshold": best["threshold"],
        "recall_target": recall_target,
        "target_met": True,
        "validation_metrics": best,
    }


def error_analysis(records: list[dict], scores, threshold: float) -> dict:
    """Audit errors without raw narratives; subgroup counts are descriptive only."""
    y, p = checked_arrays([r["signal"] for r in records], scores)
    predicted = (p >= threshold).astype(int)
    errors = [
        {
            "report_id": row["report_id"],
            "template_family": row["template_family"],
            "label": int(label),
            "prediction": int(prediction),
            "score": float(score),
            "error": "false_negative" if label else "false_positive",
        }
        for row, label, prediction, score in zip(records, y, predicted, p, strict=True)
        if label != prediction
    ]
    subgroups = {}
    for field in ("template_family", "sex", "treatment"):
        subgroups[field] = {}
        values = ["__missing__" if r[field] is None else r[field] for r in records]
        for value in sorted(set(values)):
            mask = np.asarray([v == value for v in values])
            subgroups[field][value] = evaluate(y[mask], p[mask], threshold)
    return {
        "errors": errors,
        "subgroups": subgroups,
        "note": "Synthetic descriptive analysis, not clinical or fairness validation.",
    }
