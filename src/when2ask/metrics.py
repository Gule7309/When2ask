from __future__ import annotations

from collections.abc import Iterable

from .compatibility import candidate_set_contains_gt
from .models import DecisionRecord


def candidate_recall(records: Iterable[DecisionRecord]) -> float:
    records = list(records)
    if not records:
        raise ValueError("candidate_recall requires at least one record")
    hits = sum(
        candidate_set_contains_gt(record.candidates, record.ground_truth)
        for record in records
    )
    return hits / len(records)


def max_candidate_confidence(record: DecisionRecord) -> float | None:
    confidences = [c.confidence for c in record.candidates if c.confidence is not None]
    return max(confidences) if confidences else None


def high_confidence_wrong_execution_rate(
    records: Iterable[DecisionRecord], threshold: float
) -> float:
    """Fraction of all records that execute wrongly at confidence >= threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    records = list(records)
    if not records:
        raise ValueError("metric requires at least one record")

    failures = 0
    for record in records:
        if record.executed_candidate is None or record.executed_correctly is not False:
            continue
        confidence = record.executed_candidate.confidence
        if confidence is not None and confidence >= threshold:
            failures += 1
    return failures / len(records)


def conditional_execution_accuracy(
    records: Iterable[DecisionRecord], *, gt_present: bool
) -> float | None:
    selected = [
        record
        for record in records
        if candidate_set_contains_gt(record.candidates, record.ground_truth) == gt_present
        and record.executed_correctly is not None
    ]
    if not selected:
        return None
    return sum(record.executed_correctly is True for record in selected) / len(selected)
