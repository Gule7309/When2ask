from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def load_probe_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: record must be an object")
            records.append(item)
    if not records:
        raise ValueError("probe file contains no records")
    return records


def best_is_gt_compatible(record: dict[str, Any]) -> bool:
    decision = record["pre_question_decision"]
    best_tool = decision["best_tool"]
    best_args = decision["best_arguments"]

    for candidate in record.get("candidates", []):
        if (
            candidate.get("tool_name") == best_tool
            and candidate.get("arguments", {}) == best_args
        ):
            return bool(candidate.get("gt_compatible", False))
    raise ValueError(
        f"{record.get('sample_id')}: best candidate not found in candidate list"
    )


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_initial_probe(
    records: list[dict[str, Any]],
    thresholds: list[float],
) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one probe record is required")

    n = len(records)
    gt_present = [bool(r.get("gt_present")) for r in records]
    best_correct = [best_is_gt_compatible(r) for r in records]
    viability = [
        float(r["pre_question_decision"]["best_viability"]) for r in records
    ]
    execute = [
        r["pre_question_decision"]["action"] == "execute" for r in records
    ]

    absent_indices = [i for i, present in enumerate(gt_present) if not present]
    present_indices = [i for i, present in enumerate(gt_present) if present]

    def rate(indices: list[int], flags: list[bool]) -> float | None:
        if not indices:
            return None
        return sum(flags[i] for i in indices) / len(indices)

    high_wrong: dict[str, float] = {}
    for threshold in thresholds:
        count = sum(
            (not best_correct[i]) and viability[i] >= threshold
            for i in range(n)
        )
        high_wrong[str(threshold)] = count / n

    absent_high_wrong: dict[str, float | None] = {}
    for threshold in thresholds:
        if not absent_indices:
            absent_high_wrong[str(threshold)] = None
            continue
        count = sum(
            (not best_correct[i]) and viability[i] >= threshold
            for i in absent_indices
        )
        absent_high_wrong[str(threshold)] = count / len(absent_indices)

    return {
        "n_records": n,
        "candidate_recall": sum(gt_present) / n,
        "best_candidate_gt_compatible_rate": sum(best_correct) / n,
        "pre_question_execute_rate": sum(execute) / n,
        "n_gt_present": len(present_indices),
        "n_gt_absent": len(absent_indices),
        "execute_rate_gt_present": rate(present_indices, execute),
        "execute_rate_gt_absent": rate(absent_indices, execute),
        "best_gt_compatible_rate_given_gt_present": rate(
            present_indices, best_correct
        ),
        "mean_best_viability_gt_present": _mean_or_none(
            [viability[i] for i in present_indices]
        ),
        "mean_best_viability_gt_absent": _mean_or_none(
            [viability[i] for i in absent_indices]
        ),
        "high_viability_wrong_best_rate_all": high_wrong,
        "high_viability_wrong_best_rate_given_gt_absent": absent_high_wrong,
    }
