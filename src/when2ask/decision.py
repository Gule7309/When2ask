from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .belief import CandidateBelief, best_belief


class PreQuestionAction(str, Enum):
    EXECUTE = "execute"
    NEEDS_QUESTION_SCORING = "needs_question_scoring"


@dataclass(frozen=True)
class PreQuestionDecision:
    action: PreQuestionAction
    best: CandidateBelief
    tau_exec: float


def decide_before_question_generation(
    beliefs: Sequence[CandidateBelief],
    *,
    tau_exec: float,
) -> PreQuestionDecision:
    """Implement SAGE Step 1's execution-threshold check.

    This is intentionally not labelled a final ASK decision. If the threshold is
    not met, the paper proceeds to question generation and EVPI scoring, which is
    reconstructed in a later phase.
    """
    if not 0.0 <= tau_exec <= 1.0:
        raise ValueError("tau_exec must be between 0 and 1")
    best = best_belief(beliefs)
    action = (
        PreQuestionAction.EXECUTE
        if best.viability >= tau_exec
        else PreQuestionAction.NEEDS_QUESTION_SCORING
    )
    return PreQuestionDecision(action=action, best=best, tau_exec=tau_exec)
