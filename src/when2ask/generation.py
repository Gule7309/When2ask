from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .models import Candidate
from .schema import ToolSpec


class JsonGenerator(Protocol):
    def generate_json(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> Mapping[str, Any]:
        ...


def _tool_payload(tools: Mapping[str, ToolSpec]) -> list[dict[str, Any]]:
    payload = []
    for tool in tools.values():
        payload.append(
            {
                "name": tool.name,
                "description": tool.description,
                "arguments": [
                    {
                        "name": arg.name,
                        "description": arg.description,
                        "required": arg.required,
                        "domain": {
                            "type": arg.domain.kind,
                            "values": arg.domain.values,
                        },
                    }
                    for arg in tool.arguments
                ],
            }
        )
    return payload


def build_candidate_prompt(
    *,
    user_query: str,
    observations: Sequence[str],
    tools: Mapping[str, ToolSpec],
    n_candidates: int,
) -> str:
    """Build the explicit reconstruction prompt for SAGE Step 1.

    The paper specifies the inputs (u, O_t, T), N candidates, and <UNK> semantics,
    but does not provide a separate candidate-generation prompt. This prompt is
    therefore a documented reconstruction, not a verbatim paper prompt.
    """
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")

    observations_text = "\n".join(f"- {x}" for x in observations) or "None"
    tools_json = json.dumps(_tool_payload(tools), ensure_ascii=False, indent=2)

    return f"""You are selecting possible tool calls for a user's current request.

User request:
{user_query}

Previous observations:
{observations_text}

Available tool schemas:
{tools_json}

Generate {n_candidates} alternative candidate tool calls that could represent the
user's intended next action. Candidates are alternatives, not a sequence of actions.

For every candidate:
- choose exactly one available tool;
- include the tool's arguments;
- copy a concrete value only when it is supported by the request or observations;
- use "<UNK>" when an argument is not sufficiently specified;
- do not invent a missing user preference just to make the call executable.

Return JSON only:
{{
  "candidates": [
    {{
      "tool_name": "tool_name",
      "arguments": {{"arg_name": "<UNK>"}}
    }}
  ]
}}
"""


def parse_candidates(
    payload: Mapping[str, Any],
    *,
    tools: Mapping[str, ToolSpec],
    n_candidates: int | None = None,
) -> tuple[Candidate, ...]:
    raw = payload.get("candidates", [])
    if not isinstance(raw, list):
        raise ValueError("candidate response must contain a candidates list")

    candidates: list[Candidate] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("each candidate must be an object")
        tool_name = item.get("tool_name")
        arguments = item.get("arguments", {})
        if tool_name not in tools:
            raise ValueError(f"candidate references unknown tool: {tool_name!r}")
        if not isinstance(arguments, Mapping):
            raise ValueError("candidate arguments must be an object")
        candidates.append(Candidate(str(tool_name), dict(arguments)))

    if n_candidates is not None:
        candidates = candidates[:n_candidates]
    if not candidates:
        raise ValueError("model returned no valid candidates")
    return tuple(candidates)


def generate_candidates(
    generator: JsonGenerator,
    *,
    user_query: str,
    observations: Sequence[str],
    tools: Mapping[str, ToolSpec],
    n_candidates: int,
    temperature: float = 0.5,
    max_tokens: int = 2000,
) -> tuple[Candidate, ...]:
    prompt = build_candidate_prompt(
        user_query=user_query,
        observations=observations,
        tools=tools,
        n_candidates=n_candidates,
    )
    payload = generator.generate_json(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return parse_candidates(
        payload,
        tools=tools,
        n_candidates=n_candidates,
    )
