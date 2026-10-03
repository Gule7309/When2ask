from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

UNK = "<UNK>"


@dataclass(frozen=True)
class Candidate:
    """A candidate tool invocation considered by the agent at one decision step."""

    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    confidence: float | None = None

    def with_confidence(self, confidence: float | None) -> "Candidate":
        return Candidate(self.tool_name, dict(self.arguments), confidence)


@dataclass(frozen=True)
class GroundTruthCall:
    """Ground-truth tool invocation from a ClarifyBench sample."""

    tool_name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionRecord:
    """Minimal per-step record needed for candidate-set diagnostics."""

    sample_id: str
    candidates: tuple[Candidate, ...]
    ground_truth: GroundTruthCall
    executed_candidate: Candidate | None = None
    executed_correctly: bool | None = None
