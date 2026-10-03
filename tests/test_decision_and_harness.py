from when2ask.decision import (
    PreQuestionAction,
    decide_before_question_generation,
)
from when2ask.harness import run_probe
from when2ask.models import GroundTruthCall
from when2ask.schema import ArgumentSpec, DomainSpec, ToolSpec


TOOLS = {
    "cancel_booking": ToolSpec(
        "cancel_booking",
        (ArgumentSpec("booking_id", DomainSpec("finite", ["B1", "B2"])),),
    ),
    "retrieve_invoice": ToolSpec(
        "retrieve_invoice",
        (ArgumentSpec("booking_id", DomainSpec("finite", ["B1", "B2"])),),
    ),
}


class FakeGenerator:
    def generate_json(self, prompt, *, temperature, max_tokens):
        return {
            "candidates": [
                {
                    "tool_name": "retrieve_invoice",
                    "arguments": {"booking_id": "B1"},
                },
                {
                    "tool_name": "cancel_booking",
                    "arguments": {"booking_id": "<UNK>"},
                },
            ]
        }


def test_threshold_decision_uses_raw_viability_not_normalized_share():
    result = run_probe(
        FakeGenerator(),
        sample_id="s",
        user_query="Cancel it.",
        observations=(),
        tools=TOOLS,
        n_candidates=2,
        tau_exec=0.9,
        ground_truth=GroundTruthCall("cancel_booking", {"booking_id": "B1"}),
    )
    assert result.pre_question_decision.action == PreQuestionAction.EXECUTE
    assert result.pre_question_decision.best.candidate.tool_name == "retrieve_invoice"
    assert result.pre_question_decision.best.viability == 1.0


def test_probe_records_gt_presence_even_when_best_candidate_is_wrong():
    result = run_probe(
        FakeGenerator(),
        sample_id="s",
        user_query="Cancel it.",
        observations=(),
        tools=TOOLS,
        n_candidates=2,
        tau_exec=0.9,
        ground_truth=GroundTruthCall("cancel_booking", {"booking_id": "B1"}),
    )
    assert result.gt_present is True
    payload = result.to_dict()
    assert payload["candidates"][1]["gt_compatible"] is True
