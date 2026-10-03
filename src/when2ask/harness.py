from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .belief import CandidateBelief, compute_beliefs
from .compatibility import candidate_set_contains_gt, is_gt_compatible
from .decision import PreQuestionDecision, decide_before_question_generation
from .generation import JsonGenerator, generate_candidates
from .models import Candidate, GroundTruthCall
from .schema import ToolSpec


@dataclass(frozen=True)
class SageProbeResult:
    sample_id: str
    candidates: tuple[Candidate, ...]
    beliefs: tuple[CandidateBelief, ...]
    gt_present: bool | None
    gt_compatibility: tuple[bool, ...] | None
    pre_question_decision: PreQuestionDecision

    def to_dict(self) -> dict:
        rows = []
        for index, belief in enumerate(self.beliefs):
            row = {
                "index": index,
                "tool_name": belief.candidate.tool_name,
                "arguments": dict(belief.candidate.arguments),
                "viability": belief.viability,
                "normalized_share": belief.normalized_share,
            }
            if self.gt_compatibility is not None:
                row["gt_compatible"] = self.gt_compatibility[index]
            rows.append(row)

        best = self.pre_question_decision.best
        return {
            "sample_id": self.sample_id,
            "gt_present": self.gt_present,
            "candidates": rows,
            "pre_question_decision": {
                "action": self.pre_question_decision.action.value,
                "tau_exec": self.pre_question_decision.tau_exec,
                "best_tool": best.candidate.tool_name,
                "best_arguments": dict(best.candidate.arguments),
                "best_viability": best.viability,
                "best_normalized_share": best.normalized_share,
            },
        }


def run_probe(
    generator: JsonGenerator,
    *,
    sample_id: str,
    user_query: str,
    observations: Sequence[str],
    tools: Mapping[str, ToolSpec],
    n_candidates: int,
    tau_exec: float,
    epsilon: float = 1e-4,
    temperature: float = 0.5,
    max_tokens: int = 2000,
    ground_truth: GroundTruthCall | None = None,
    include_optional: bool = True,
) -> SageProbeResult:
    candidates = generate_candidates(
        generator,
        user_query=user_query,
        observations=observations,
        tools=tools,
        n_candidates=n_candidates,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    beliefs = compute_beliefs(
        candidates,
        tools,
        epsilon=epsilon,
        include_optional=include_optional,
    )
    decision = decide_before_question_generation(beliefs, tau_exec=tau_exec)

    if ground_truth is None:
        gt_present = None
        compatibility = None
    else:
        gt_present = candidate_set_contains_gt(candidates, ground_truth)
        compatibility = tuple(
            is_gt_compatible(candidate, ground_truth) for candidate in candidates
        )

    return SageProbeResult(
        sample_id=sample_id,
        candidates=candidates,
        beliefs=beliefs,
        gt_present=gt_present,
        gt_compatibility=compatibility,
        pre_question_decision=decision,
    )
