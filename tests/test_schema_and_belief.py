import math

from when2ask.belief import candidate_viability, compute_beliefs
from when2ask.models import Candidate, UNK
from when2ask.schema import ArgumentSpec, DomainSpec, ToolSpec


def _book_tool():
    return ToolSpec(
        name="book",
        arguments=(
            ArgumentSpec("seat", DomainSpec("finite", ["economy", "business"])),
            ArgumentSpec("date", DomainSpec("string")),
            ArgumentSpec("flexible", DomainSpec("boolean"), required=False),
        ),
    )


def test_unknown_finite_and_unbounded_arguments_follow_paper_rule():
    tool = _book_tool()
    candidate = Candidate(
        "book",
        {"seat": UNK, "date": UNK, "flexible": True},
    )
    score = candidate_viability(candidate, tool, epsilon=1e-4)
    assert math.isclose(score, 0.5 * 1e-4)


def test_specified_arguments_have_unit_certainty():
    tool = _book_tool()
    candidate = Candidate(
        "book",
        {"seat": "economy", "date": "2026-10-04", "flexible": False},
    )
    assert candidate_viability(candidate, tool) == 1.0


def test_missing_schema_argument_is_treated_as_unknown():
    tool = _book_tool()
    candidate = Candidate("book", {"seat": "economy", "date": "2026-10-04"})
    assert candidate_viability(candidate, tool) == 0.5


def test_required_only_is_explicit_sensitivity_mode():
    tool = _book_tool()
    candidate = Candidate("book", {"seat": "economy", "date": "2026-10-04"})
    assert candidate_viability(candidate, tool, include_optional=False) == 1.0


def test_compute_beliefs_logs_raw_and_normalized_scores():
    tool = _book_tool()
    tools = {"book": tool}
    candidates = (
        Candidate("book", {"seat": "economy", "date": "2026-10-04", "flexible": False}),
        Candidate("book", {"seat": UNK, "date": "2026-10-04", "flexible": False}),
    )
    beliefs = compute_beliefs(candidates, tools)
    assert beliefs[0].viability == 1.0
    assert beliefs[1].viability == 0.5
    assert math.isclose(sum(x.normalized_share for x in beliefs), 1.0)
