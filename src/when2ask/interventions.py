from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .compatibility import candidate_set_contains_gt, is_gt_compatible
from .models import Candidate, GroundTruthCall


@dataclass(frozen=True)
class InterventionResult:
    candidates: tuple[Candidate, ...]
    changed: bool
    reason: str


def oracle_complete(
    candidates: Sequence[Candidate],
    ground_truth: GroundTruthCall,
    oracle_candidate: Candidate,
) -> InterventionResult:
    """Ensure at least one GT-compatible candidate is present."""
    current = tuple(candidates)
    if candidate_set_contains_gt(current, ground_truth):
        return InterventionResult(current, False, "ground truth already represented")
    if not is_gt_compatible(oracle_candidate, ground_truth):
        raise ValueError("oracle_candidate must be ground-truth compatible")
    return InterventionResult(
        current + (oracle_candidate,), True, "added ground-truth-compatible candidate"
    )


def forced_miss(
    candidates: Sequence[Candidate],
    ground_truth: GroundTruthCall,
    replacement: Candidate | None = None,
) -> InterventionResult:
    """Remove all GT-compatible candidates, optionally preserving set size."""
    current = tuple(candidates)
    kept = tuple(c for c in current if not is_gt_compatible(c, ground_truth))
    removed = len(current) - len(kept)
    if removed == 0:
        return InterventionResult(current, False, "ground truth not represented")

    if replacement is not None:
        if is_gt_compatible(replacement, ground_truth):
            raise ValueError("replacement must not be ground-truth compatible")
        kept = kept + (replacement,)

    return InterventionResult(kept, True, f"removed {removed} compatible candidate(s)")
