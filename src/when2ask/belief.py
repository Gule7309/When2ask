from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, prod
from typing import Mapping, Sequence

from .models import Candidate, UNK
from .schema import ToolSpec


@dataclass(frozen=True)
class CandidateBelief:
    """Structured score for one candidate.

    viability follows the product used by the paper's appendix:
    specified arguments contribute 1; unresolved finite arguments contribute 1/|D|;
    unresolved continuous/unbounded arguments contribute epsilon.

    normalized_share is logged separately because the paper calls pi a belief
    distribution while its appendix also uses the unnormalized product as a score
    that reaches 1 when all parameters are specified.
    """

    candidate: Candidate
    viability: float
    normalized_share: float


def argument_certainty(
    value: object,
    *,
    domain_size: float,
    epsilon: float,
) -> float:
    if value != UNK:
        return 1.0
    if isfinite(domain_size):
        if domain_size <= 0:
            raise ValueError("finite domain size must be positive")
        return 1.0 / domain_size
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    return epsilon


def candidate_viability(
    candidate: Candidate,
    tool: ToolSpec,
    *,
    epsilon: float = 1e-4,
    include_optional: bool = True,
) -> float:
    """Compute the paper-reconstructed product score for one candidate.

    Missing schema arguments are treated as unresolved (<UNK>). Extra candidate
    arguments are rejected because their domains are undefined.

    The main paper writes the product over the tool parameter set Theta, so the
    default includes optional parameters as well. include_optional=False is kept
    only for a sensitivity analysis.
    """
    schema_args = tool.argument_map
    unknown_names = set(candidate.arguments) - set(schema_args)
    if unknown_names:
        raise ValueError(
            f"candidate for {tool.name!r} contains unknown arguments: "
            f"{sorted(unknown_names)}"
        )

    terms: list[float] = []
    for argument in tool.arguments:
        if not include_optional and not argument.required:
            continue
        value = candidate.arguments.get(argument.name, UNK)
        terms.append(
            argument_certainty(
                value,
                domain_size=argument.domain.size,
                epsilon=epsilon,
            )
        )

    return float(prod(terms)) if terms else 1.0


def compute_beliefs(
    candidates: Sequence[Candidate],
    tools: Mapping[str, ToolSpec],
    *,
    epsilon: float = 1e-4,
    include_optional: bool = True,
) -> tuple[CandidateBelief, ...]:
    if not candidates:
        raise ValueError("at least one candidate is required")

    scored: list[tuple[Candidate, float]] = []
    for candidate in candidates:
        if candidate.tool_name not in tools:
            raise ValueError(f"unknown tool: {candidate.tool_name}")
        score = candidate_viability(
            candidate,
            tools[candidate.tool_name],
            epsilon=epsilon,
            include_optional=include_optional,
        )
        scored.append((candidate, score))

    total = sum(score for _, score in scored)
    if total <= 0:
        raise ValueError("candidate scores must have positive total mass")

    return tuple(
        CandidateBelief(
            candidate=candidate,
            viability=score,
            normalized_share=score / total,
        )
        for candidate, score in scored
    )


def best_belief(beliefs: Sequence[CandidateBelief]) -> CandidateBelief:
    if not beliefs:
        raise ValueError("beliefs cannot be empty")
    return max(beliefs, key=lambda item: item.viability)
