import json
import subprocess
import sys
from pathlib import Path

import pytest

from when2ask.models import Candidate, GroundTruthCall, UNK
from when2ask.belief import compute_beliefs
from when2ask.questioning import Aspect, Question, parse_questions, score_questions
from when2ask.sage import SageSettings, run_step
from when2ask.schema import ArgumentSpec, DomainSpec, ToolSpec
from when2ask.trace_io import record_from_dict


TOOLS = {
    "cancel": ToolSpec("cancel", (ArgumentSpec("id", DomainSpec("finite", ["A", "B"])),)),
    "lookup": ToolSpec("lookup", (ArgumentSpec("id", DomainSpec("finite", ["A", "B"])),)),
}
SETTINGS = SageSettings(tau_exec=0.8, max_steps=3, n_candidates=1)
QUESTION = {"text": "Which booking?", "candidate_index": 0,
            "aspects": [{"tool_name": "cancel", "argument": "id"}]}


class Replay:
    def __init__(self, candidates=None, questions=None):
        self.prompts = []
        self.responses = iter([
            {"candidates": candidates or [{"tool_name": "cancel", "arguments": {"id": UNK}}]},
            {"questions": [QUESTION] if questions is None else questions},
        ])

    def generate_json(self, prompt, *, temperature, max_tokens):
        self.prompts.append(prompt)
        return next(self.responses)


def step(generator=None, **kwargs):
    defaults = dict(sample_id="s", step=0, user_query="Cancel it", observations=[],
                    tools=TOOLS, settings=SETTINGS)
    defaults.update(kwargs)
    return run_step(generator or Replay(), **defaults)


def test_unknown_candidate_asks_and_emits_auditable_scores():
    record = step(ground_truth=GroundTruthCall("cancel", {"id": "A"}))
    assert record["decision"]["action"] == "ask"
    assert record["gt_present"] is True
    assert record["candidates"][0]["pi_c"] == 0.5
    assert record["candidates"][0]["normalized_share"] == 1
    assert record["questions"][0]["resolution_gain"] == 0.5
    assert len(record["generation_calls"]) == 2
    assert record["settings"]["question_score_method"] == "optimistic_perfect_resolution_v1"
    assert len(record["generation_calls"][0]["prompt_sha256"]) == 64
    json.dumps(record, allow_nan=False)


def test_threshold_execute_skips_question_generation_and_does_not_claim_execution():
    generator = Replay([{"tool_name": "lookup", "arguments": {"id": "A"}}])
    record = step(generator, ground_truth=GroundTruthCall("cancel", {"id": "A"}))
    assert record["decision"]["reason"] == "execution_threshold"
    assert record["gt_present"] is False
    assert len(generator.prompts) == 1
    assert record["executed_candidate"] is None
    assert record["executed_correctly"] is None
    assert record_from_dict(record).candidates[0].confidence == 1


def test_redundancy_cost_and_strict_stopping_boundary():
    record = step(aspect_counts={Aspect("cancel", "id"): 1})
    assert record["questions"][0]["redundancy_cost"] == 0.5
    assert record["decision"]["reason"] == "question_gain_below_threshold"
    assert not record["decision"]["selected_candidate_fully_specified"]
    # Score == alpha * max(pi) must ask; paper uses a strict < stop rule.
    record = step(settings=SageSettings(tau_exec=0.8, max_steps=3, alpha=1))
    assert record["decision"]["action"] == "ask"


def test_budget_execute_skips_questions_and_empty_questions_are_explicit():
    generator = Replay()
    assert step(generator, step=3)["decision"]["reason"] == "question_budget_exhausted"
    assert len(generator.prompts) == 1
    assert step(Replay(questions=[]))["decision"]["reason"] == "no_questions"


def test_ground_truth_is_diagnostic_only():
    left, right = Replay(), Replay()
    present = step(left, ground_truth=GroundTruthCall("cancel", {"id": "SECRET"}))
    absent = step(right, ground_truth=GroundTruthCall("lookup", {"id": "OTHER_SECRET"}))
    assert left.prompts == right.prompts
    assert "SECRET" not in "\n".join(left.prompts)
    assert present["decision"] == absent["decision"]
    assert present["gt_present"] and not absent["gt_present"]


def test_response_context_can_drive_the_next_step_without_upstream_imports():
    first = step()
    second = step(Replay([{"tool_name": "cancel", "arguments": {"id": "A"}}]),
                  step=1, observations=[first["decision"]["question"]["text"], "Booking A"],
                  aspect_counts={Aspect("cancel", "id"): 1})
    assert second["step"] == 1
    assert second["observations"][-1] == "Booking A"
    assert second["decision"]["action"] == "execute"
    assert second["candidates"][0]["pi_c"] == 1


def test_continuous_resolution_is_finite_and_optional_mode_is_preserved():
    tools = {"cancel": ToolSpec("cancel", (ArgumentSpec("id", DomainSpec("string")),))}
    record = step(tools=tools)
    assert record["questions"][0]["resolution_gain"] == pytest.approx(1 - 1e-4)
    tools = {"cancel": ToolSpec("cancel", (ArgumentSpec("id", DomainSpec("string"), required=False),))}
    record = step(tools=tools, settings=SageSettings(tau_exec=0.8, max_steps=3, include_optional=False))
    assert record["decision"]["action"] == "execute"


def test_aspects_are_tool_qualified_and_candidate_ties_use_generation_order():
    # Resolving cancel.id must not resolve lookup.id even though the names match.
    tools = dict(TOOLS)
    tools["cancel"] = ToolSpec("cancel", (
        ArgumentSpec("id", DomainSpec("finite", ["A", "B"])),
        ArgumentSpec("date", DomainSpec("finite", [1, 2, 3, 4])),
    ))
    beliefs = compute_beliefs([Candidate("cancel", {"id": UNK, "date": UNK}),
                              Candidate("lookup", {"id": UNK})], tools)
    scores = score_questions([Question("Which?", 0, (Aspect("cancel", "id"),))],
                             beliefs, tools, aspect_counts={})
    assert scores[0].resolution_gain == 0  # lookup stays best at 0.5
    record = step(Replay([{"tool_name": "lookup", "arguments": {"id": "B"}},
                          {"tool_name": "cancel", "arguments": {"id": "A"}}]),
                  settings=SageSettings(tau_exec=0.8, max_steps=3, n_candidates=2))
    assert record["decision"]["selected_candidate_index"] == 0


def test_unlabeled_steps_emit_null_gt_and_question_ties_use_generation_order():
    other = {**QUESTION, "text": "Please identify the booking."}
    record = step(Replay(questions=[QUESTION, other]))
    assert record["gt_present"] is None
    assert record["ground_truth"] is None
    assert "gt_compatible" not in record["candidates"][0]
    assert record["decision"]["question"]["text"] == QUESTION["text"]


def test_invalid_history_is_rejected_before_model_calls():
    generator = Replay()
    with pytest.raises(ValueError):
        step(generator, aspect_counts={Aspect("cancel", "id"): -1})
    assert generator.prompts == []


@pytest.mark.parametrize("change", [
    {"candidate_index": -1}, {"candidate_index": True}, {"text": ""},
    {"aspects": []}, {"aspects": [{"tool_name": "cancel", "argument": "missing"}]},
    {"aspects": [{"tool_name": "lookup", "argument": "id"}]},
    {"aspects": QUESTION["aspects"] * 2},
])
def test_invalid_questions_are_not_silently_used(change):
    with pytest.raises(ValueError):
        parse_questions({"questions": [{**QUESTION, **change}]},
                        candidates=[Candidate("cancel")], tools=TOOLS, n_questions=3)


@pytest.mark.parametrize("change", [
    {"epsilon": 0}, {"epsilon": float("nan")}, {"tau_exec": float("nan")},
    {"alpha": -1}, {"lambda_cost": float("inf")}, {"n_candidates": 0}, {"max_steps": -1},
])
def test_invalid_settings_fail_before_model_calls(change):
    with pytest.raises(ValueError):
        SageSettings(**{**vars(SETTINGS), **change})


def test_replay_cli_emits_jsonl_consumable_by_existing_parser(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "trace.jsonl"
    result = subprocess.run([
        sys.executable, str(root / "scripts/run_sage_step.py"),
        "--input", str(root / "examples/sage_step.json"),
        "--replay", str(root / "examples/sage_responses.json"), "--output", str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["decision"]["action"] == "ask"
    assert record["run"]["empirical"] is False
    assert record_from_dict(record).ground_truth.tool_name == "cancel_booking"
