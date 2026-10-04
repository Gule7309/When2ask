"""Independent, explicitly heuristic reconstruction of SAGE question selection."""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from .belief import CandidateBelief, argument_certainty, best_belief
from .generation import JsonGenerator, _tool_payload
from .models import Candidate, UNK
from .schema import ToolSpec

QUESTION_PROMPT_VERSION = "reconstructed_questions_v1"
QUESTION_SCORE_METHOD = "optimistic_perfect_resolution_v1"


@dataclass(frozen=True, order=True)
class Aspect:
    tool_name: str
    argument: str

    def to_dict(self) -> dict:
        return {"tool_name": self.tool_name, "argument": self.argument}


@dataclass(frozen=True)
class Question:
    text: str
    candidate_index: int
    aspects: tuple[Aspect, ...]

    def to_dict(self) -> dict:
        return {"text": self.text, "candidate_index": self.candidate_index,
                "aspects": [a.to_dict() for a in self.aspects]}


@dataclass(frozen=True)
class QuestionScore:
    question: Question
    resolution_gain: float
    redundancy_cost: float

    @property
    def score(self) -> float:
        return self.resolution_gain - self.redundancy_cost

    def to_dict(self) -> dict:
        return {**self.question.to_dict(), "resolution_gain": self.resolution_gain,
                "redundancy_cost": self.redundancy_cost, "score": self.score}


def generate_questions(
    generator: JsonGenerator, *, user_query: str, observations: Sequence[str],
    candidates: Sequence[Candidate], tools: Mapping[str, ToolSpec],
    n_questions: int, temperature: float, max_tokens: int,
) -> tuple[Question, ...]:
    context = {
        "user_query": user_query, "observations": list(observations),
        "tools": _tool_payload(tools),
        "candidates": [{"index": i, "tool_name": c.tool_name,
                        "arguments": dict(c.arguments)} for i, c in enumerate(candidates)],
    }
    prompt = f"""Generate up to {n_questions} concise clarification questions for the
current candidate tool calls. Target schema arguments that are unresolved or
ambiguous. Do not assume missing user preferences. Each question must identify
a zero-based candidate_index and a nonempty list of distinct aspects, each with
tool_name and argument. The target candidate's tool must occur in the aspects.
Use only tools and arguments in the provided schema. Return JSON only, with
this shape: {{"questions": [{{"text": "Which booking?", "candidate_index": 0,
"aspects": [{{"tool_name": "cancel_booking", "argument": "booking_id"}}]}}]}}.
Context:
{json.dumps(context, ensure_ascii=False, indent=2)}
"""
    payload = generator.generate_json(prompt, temperature=temperature, max_tokens=max_tokens)
    return parse_questions(payload, candidates=candidates, tools=tools, n_questions=n_questions)


def parse_questions(
    payload: Mapping[str, Any], *, candidates: Sequence[Candidate],
    tools: Mapping[str, ToolSpec], n_questions: int,
) -> tuple[Question, ...]:
    if n_questions < 1:
        raise ValueError("n_questions must be >= 1")
    rows = payload.get("questions")
    if not isinstance(rows, list):
        raise ValueError("question response must contain a questions list")
    questions = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each question must be an object")
        text, index, raw_aspects = row.get("text"), row.get("candidate_index"), row.get("aspects")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("question text must be nonempty")
        if type(index) is not int or not 0 <= index < len(candidates):
            raise ValueError("invalid question candidate_index")
        if not isinstance(raw_aspects, list) or not raw_aspects:
            raise ValueError("question aspects must be a nonempty list")
        aspects = []
        for item in raw_aspects:
            if not isinstance(item, Mapping):
                raise ValueError("each aspect must be an object")
            tool, argument = item.get("tool_name"), item.get("argument")
            if not isinstance(tool, str) or tool not in tools:
                raise ValueError("aspect references unknown tool")
            if not isinstance(argument, str) or argument not in tools[tool].argument_map:
                raise ValueError("aspect references unknown argument")
            aspect = Aspect(tool, argument)
            if aspect in aspects:
                raise ValueError("duplicate question aspect")
            aspects.append(aspect)
        if not any(a.tool_name == candidates[index].tool_name for a in aspects):
            raise ValueError("question aspects do not target its candidate's tool")
        questions.append(Question(text.strip(), index, tuple(aspects)))
    return tuple(questions[:n_questions])


def score_questions(
    questions: Sequence[Question], beliefs: Sequence[CandidateBelief],
    tools: Mapping[str, ToolSpec], *, aspect_counts: Mapping[Aspect, int],
    lambda_cost: float = 0.5, epsilon: float = 1e-4, include_optional: bool = True,
) -> tuple[QuestionScore, ...]:
    """Optimistic gain in raw viability under simultaneous perfect resolution.

    This is NOT the response-weighted EVPI in Definition 4. No response model
    is specified by the minimal harness. The assumption is emitted in traces.
    Recompute the product to avoid infinite multipliers for continuous domains.
    """
    if not isfinite(lambda_cost) or lambda_cost < 0:
        raise ValueError("lambda_cost must be finite and nonnegative")
    if any(type(n) is not int or n < 0 for n in aspect_counts.values()):
        raise ValueError("aspect counts must be nonnegative integers")
    current = best_belief(beliefs).viability
    scores = []
    for question in questions:
        resolved_scores = []
        for belief in beliefs:
            candidate = belief.candidate
            value = 1.0
            for arg in tools[candidate.tool_name].arguments:
                if not include_optional and not arg.required:
                    continue
                if Aspect(candidate.tool_name, arg.name) in question.aspects:
                    continue  # perfect resolution contributes unit certainty
                value *= argument_certainty(candidate.arguments.get(arg.name, UNK),
                                            domain_size=arg.domain.size, epsilon=epsilon)
            resolved_scores.append(value)
        gain = max(resolved_scores) - current
        cost = lambda_cost * sum(aspect_counts.get(a, 0) for a in question.aspects)
        scores.append(QuestionScore(question, gain, cost))
    return tuple(scores)
