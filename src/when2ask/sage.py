"""Paper-reconstructed decision steps; no import of upstream ClarifyBench agents."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from .decision import PreQuestionAction
from .generation import JsonGenerator, _tool_payload
from .harness import run_probe
from .models import GroundTruthCall, UNK
from .questioning import (Aspect, QUESTION_PROMPT_VERSION, QUESTION_SCORE_METHOD,
                          generate_questions, score_questions)
from .schema import ToolSpec

HARNESS_VERSION = "paper_reconstructed_sage_step_v1"


@dataclass(frozen=True)
class SageSettings:
    tau_exec: float  # unresolved in paper: caller must choose explicitly
    max_steps: int   # maximum number of questions; steps start at zero
    n_candidates: int = 5
    n_questions: int = 3
    lambda_cost: float = 0.5
    alpha: float = 0.1
    epsilon: float = 1e-4
    temperature: float = 0.5
    max_tokens: int = 2000
    include_optional: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.tau_exec <= 1:
            raise ValueError("tau_exec must be in [0, 1]")
        if not 0 < self.epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")
        for name in ("lambda_cost", "alpha", "temperature"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("n_candidates", "n_questions", "max_tokens", "max_steps"):
            value = getattr(self, name)
            minimum = 0 if name == "max_steps" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


class _AuditedGenerator:
    def __init__(self, delegate: JsonGenerator):
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, prompt: str, *, temperature: float, max_tokens: int) -> Mapping[str, Any]:
        payload = self.delegate.generate_json(prompt, temperature=temperature, max_tokens=max_tokens)
        self.calls.append({"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                           "response": json.loads(json.dumps(payload, allow_nan=False))})
        return payload


def run_step(
    generator: JsonGenerator, *, sample_id: str, step: int, user_query: str,
    observations: Sequence[str], tools: Mapping[str, ToolSpec], settings: SageSettings,
    aspect_counts: Mapping[Aspect, int] | None = None,
    ground_truth: GroundTruthCall | None = None,
) -> dict[str, Any]:
    """Return a serializable ask/execute plan, not a tool execution result.

    A caller supplies observations and current schemas on each step. Ground truth
    is used only by compatibility instrumentation after candidate generation.
    """
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    counts = dict(aspect_counts or {})
    if any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("aspect counts must be nonnegative integers")
    audited = _AuditedGenerator(generator)
    probe = run_probe(
        audited, sample_id=sample_id, user_query=user_query, observations=observations,
        tools=tools, n_candidates=settings.n_candidates, tau_exec=settings.tau_exec,
        epsilon=settings.epsilon, temperature=settings.temperature,
        max_tokens=settings.max_tokens, ground_truth=ground_truth,
        include_optional=settings.include_optional,
    )
    best = probe.pre_question_decision.best
    question_scores = ()
    selected_question = None
    if probe.pre_question_decision.action == PreQuestionAction.EXECUTE:
        action, reason = "execute", "execution_threshold"
    elif step >= settings.max_steps:
        action, reason = "execute", "question_budget_exhausted"
    else:
        questions = generate_questions(
            audited, user_query=user_query, observations=observations,
            candidates=probe.candidates, tools=tools, n_questions=settings.n_questions,
            temperature=settings.temperature, max_tokens=settings.max_tokens,
        )
        question_scores = score_questions(
            questions, probe.beliefs, tools, aspect_counts=counts,
            lambda_cost=settings.lambda_cost, epsilon=settings.epsilon,
            include_optional=settings.include_optional,
        )
        if not question_scores:
            action, reason = "execute", "no_questions"
        else:
            best_question = max(question_scores, key=lambda q: q.score)
            if best_question.score < settings.alpha * best.viability:
                action, reason = "execute", "question_gain_below_threshold"
            else:
                action, reason = "ask", "question_selected"
                selected_question = best_question.question

    record = probe.to_dict()
    record.update({
        "schema_version": 1, "implementation": HARNESS_VERSION, "step": step,
        "ground_truth": None if ground_truth is None else {
            "tool_name": ground_truth.tool_name, "parameters": dict(ground_truth.parameters)},
        "observations": list(observations), "tool_schemas": _tool_payload(tools),
        "questions": [q.to_dict() for q in question_scores],
        "aspect_counts": [{**a.to_dict(), "count": n} for a, n in sorted(counts.items())],
        "generation_calls": audited.calls,
        "settings": {**vars(settings), "candidate_prompt_version": "reconstructed_v1",
                     "question_prompt_version": QUESTION_PROMPT_VERSION,
                     "question_score_method": QUESTION_SCORE_METHOD,
                     "pi_c_quantity": "raw_viability"},
        "decision": {"action": action, "reason": reason,
                     "selected_candidate_index": next(i for i, b in enumerate(probe.beliefs) if b is best),
                     "question": None if selected_question is None else selected_question.to_dict(),
                     "stopping_threshold": settings.alpha * best.viability},
        # Only an environment adapter can populate actual execution outcomes.
        "executed_candidate": None, "executed_correctly": None,
    })
    for row in record["candidates"]:
        row["pi_c"] = row["viability"]
        row["unresolved_arguments"] = [arg.name for arg in tools[row["tool_name"]].arguments
                                       if row["arguments"].get(arg.name, UNK) == UNK]
    record["decision"]["selected_candidate_fully_specified"] = not any(
        arg.required and best.candidate.arguments.get(arg.name, UNK) == UNK
        for arg in tools[best.candidate.tool_name].arguments
    )
    return record
