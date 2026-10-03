from when2ask.generation import (
    build_candidate_prompt,
    generate_candidates,
    parse_candidates,
)
from when2ask.schema import ArgumentSpec, DomainSpec, ToolSpec


TOOLS = {
    "touch": ToolSpec(
        "touch",
        (ArgumentSpec("file_name", DomainSpec("string")),),
        "Create a file",
    )
}


class FakeGenerator:
    def __init__(self):
        self.prompt = None

    def generate_json(self, prompt, *, temperature, max_tokens):
        self.prompt = prompt
        return {
            "candidates": [
                {"tool_name": "touch", "arguments": {"file_name": "<UNK>"}}
            ]
        }


def test_prompt_marks_candidates_as_alternatives_and_supports_unknown():
    prompt = build_candidate_prompt(
        user_query="Create it.",
        observations=[],
        tools=TOOLS,
        n_candidates=3,
    )
    assert "alternatives, not a sequence" in prompt
    assert "<UNK>" in prompt
    assert "3 alternative candidate" in prompt


def test_generate_candidates_uses_json_generator_contract():
    generator = FakeGenerator()
    candidates = generate_candidates(
        generator,
        user_query="Create it.",
        observations=[],
        tools=TOOLS,
        n_candidates=1,
        temperature=0.5,
    )
    assert candidates[0].tool_name == "touch"
    assert generator.prompt is not None


def test_parser_rejects_unknown_tools():
    try:
        parse_candidates(
            {"candidates": [{"tool_name": "delete", "arguments": {}}]},
            tools=TOOLS,
        )
    except ValueError as exc:
        assert "unknown tool" in str(exc)
    else:
        raise AssertionError("expected ValueError")
