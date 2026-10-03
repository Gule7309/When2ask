from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import GroundTruthCall


@dataclass(frozen=True)
class ClarifyBenchSample:
    sample_id: str
    user_query: str
    ground_truth_tool_calls: tuple[GroundTruthCall, ...]
    primary_api: str | None
    raw: dict[str, Any]


def load_sample(path: str | Path) -> ClarifyBenchSample:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    query = data.get("user_query", data.get("initial_query"))
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{path}: missing user_query/initial_query")

    calls = []
    for item in data.get("ground_truth_tool_calls", []):
        calls.append(
            GroundTruthCall(
                tool_name=item["tool_name"],
                parameters=item.get("parameters", {}),
            )
        )

    return ClarifyBenchSample(
        sample_id=path.stem,
        user_query=query,
        ground_truth_tool_calls=tuple(calls),
        primary_api=data.get("primary_api"),
        raw=data,
    )
