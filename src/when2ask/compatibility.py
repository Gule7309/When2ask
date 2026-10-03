from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .models import Candidate, GroundTruthCall, UNK


def _equal_value(left: Any, right: Any) -> bool:
    """Conservative structural equality for tool arguments."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(_equal_value(left[k], right[k]) for k in left)
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
        and not isinstance(right, (str, bytes))
    ):
        return len(left) == len(right) and all(
            _equal_value(a, b) for a, b in zip(left, right)
        )
    return left == right


def is_gt_compatible(candidate: Candidate, ground_truth: GroundTruthCall) -> bool:
    """Return whether a partial candidate can still represent the ground truth.

    Compatibility is weaker than exact-match correctness. A candidate is compatible
    when it selects the correct tool and every specified argument agrees with the
    ground-truth value. Missing arguments and <UNK> remain unresolved and therefore
    do not count against compatibility.
    """
    if candidate.tool_name != ground_truth.tool_name:
        return False

    for name, value in candidate.arguments.items():
        if value == UNK:
            continue
        if name not in ground_truth.parameters:
            return False
        if not _equal_value(value, ground_truth.parameters[name]):
            return False
    return True


def candidate_set_contains_gt(
    candidates: Sequence[Candidate], ground_truth: GroundTruthCall
) -> bool:
    return any(is_gt_compatible(candidate, ground_truth) for candidate in candidates)


def make_oracle_partial_candidate(
    ground_truth: GroundTruthCall,
    observed_arguments: Mapping[str, Any] | None = None,
) -> Candidate:
    """Construct a GT-compatible partial candidate without revealing hidden values."""
    observed_arguments = observed_arguments or {}
    arguments: dict[str, Any] = {}
    for name in ground_truth.parameters:
        arguments[name] = observed_arguments.get(name, UNK)
    return Candidate(tool_name=ground_truth.tool_name, arguments=arguments)
