from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Candidate, DecisionRecord, GroundTruthCall


def _candidate_from_dict(data: dict[str, Any]) -> Candidate:
    return Candidate(
        tool_name=data["tool_name"],
        arguments=data.get("arguments", data.get("parameters", {})),
        confidence=data.get("confidence", data.get("viability")),
    )


def record_from_dict(data: dict[str, Any]) -> DecisionRecord:
    gt_data = data["ground_truth"]
    executed_data = data.get("executed_candidate")
    return DecisionRecord(
        sample_id=str(data["sample_id"]),
        candidates=tuple(_candidate_from_dict(c) for c in data.get("candidates", [])),
        ground_truth=GroundTruthCall(
            tool_name=gt_data["tool_name"],
            parameters=gt_data.get("parameters", {}),
        ),
        executed_candidate=(
            _candidate_from_dict(executed_data) if executed_data is not None else None
        ),
        executed_correctly=data.get("executed_correctly"),
    )


def load_jsonl(path: str | Path) -> list[DecisionRecord]:
    path = Path(path)
    records: list[DecisionRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(record_from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid trace record") from exc
    return records
