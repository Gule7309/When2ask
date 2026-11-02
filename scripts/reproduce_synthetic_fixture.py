"""Reproduce one synthetic SAGE mechanism check; NEVER report benchmark accuracy.

This uses fixed JSON responses and no model server, tool side effects or private
dataset. It checks candidate-set intervention invariants and writes a
deterministic provenance record suitable for CI artifact inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from when2ask.compatibility import candidate_set_contains_gt, make_oracle_partial_candidate
from when2ask.interventions import forced_miss, oracle_complete
from when2ask.models import Candidate, GroundTruthCall
from when2ask.sage import SageSettings, run_step
from when2ask.schema import load_tool_specs


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples" / "sage_step.json"
REPLAY = ROOT / "examples" / "sage_responses.json"


class FixedReplay:
    """Consumes pre-recorded synthetic responses; no network/model invocation."""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def generate_json(self, prompt: str, *, temperature: float, max_tokens: int):
        self.calls += 1
        try:
            return next(self.responses)
        except StopIteration as exc:
            raise RuntimeError("Synthetic fixture exhausted") from exc


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_set(label: str, candidates, ground_truth: GroundTruthCall) -> dict:
    return {
        "condition": label,
        "candidate_count": len(candidates),
        "ground_truth_compatible_candidate_present": candidate_set_contains_gt(
            candidates, ground_truth
        ),
    }


def reproduce(input_file: Path = INPUT, replay_file: Path = REPLAY) -> dict:
    data = json.loads(input_file.read_text(encoding="utf-8"))
    responses = json.loads(replay_file.read_text(encoding="utf-8"))
    generator = FixedReplay(responses)
    settings = SageSettings(**data["settings"])
    gt = GroundTruthCall(**data["ground_truth"])
    record = run_step(
        generator,
        sample_id=data["sample_id"],
        step=data.get("step", 0),
        user_query=data["user_query"],
        observations=data.get("observations", []),
        tools=load_tool_specs(data["tools"]),
        settings=settings,
        ground_truth=gt,
    )

    natural = tuple(
        Candidate(c["tool_name"], c["arguments"], c.get("viability"))
        for c in record["candidates"]
    )
    # This explicit, intentionally imperfect decoy differs in the booking ID.
    # It matches the tool/argument count, NOT the unresolved-argument pattern.
    decoy = Candidate("cancel_booking", {"booking_id": "B2"})
    partial = make_oracle_partial_candidate(gt)
    oracle = oracle_complete(natural, gt, partial)
    miss = forced_miss(natural, gt, replacement=decoy)
    recovery = oracle_complete(miss.candidates, gt, partial)

    conditions = [
        summarize_set("natural", natural, gt),
        summarize_set("oracle_complete", oracle.candidates, gt),
        summarize_set("forced_miss", miss.candidates, gt),
        summarize_set("oracle_recovery_from_forced_miss", recovery.candidates, gt),
    ]
    assert len(natural) == 1, "Expected a one-candidate fixture"
    assert [r["ground_truth_compatible_candidate_present"] for r in conditions] == [
        True, True, False, True
    ], "The synthetic intervention structural invariants failed"
    assert [r["candidate_count"] for r in conditions] == [1, 1, 1, 2]
    assert oracle.changed is False and miss.changed and recovery.changed
    assert generator.calls == 2, "Expected candidate and question JSON response calls"
    assert record["decision"]["action"] == "ask"
    assert record["decision"]["reason"] == "question_selected"
    assert record["candidates"][0]["pi_c"] == 0.5
    assert record["candidates"][0]["normalized_share"] == 1.0
    assert record["gt_present"] is True
    assert record["executed_candidate"] is None
    assert record["executed_correctly"] is None

    return {
        "schema": "synthetic_mechanism_audit_v1",
        "scope": "single_synthetic_decision_point_plumbing_only",
        "empirical": False,
        "benchmark_dataset": None,
        "benchmark_items_evaluated": 0,
        "model_calls": 0,
        "real_tool_executions": 0,
        "source": {
            "repository": "Gule7309/When2ask",
            "commit_sha": os.environ.get("GITHUB_SHA", "unversioned-local"),
            "input_path": str(input_file.relative_to(ROOT)),
            "input_sha256": sha256(input_file),
            "replay_path": str(replay_file.relative_to(ROOT)),
            "replay_sha256": sha256(replay_file),
        },
        "parameters": {
            "tau_exec": settings.tau_exec,
            "max_steps": settings.max_steps,
            "n_candidates": settings.n_candidates,
            "temperature": settings.temperature,
            "lambda_cost": settings.lambda_cost,
            "alpha": settings.alpha,
            "epsilon": settings.epsilon,
            "question_score_method": record["settings"]["question_score_method"],
            "generation": "fixed_synthetic_replay",
        },
        "single_decision": {
            "sample_id": record["sample_id"],
            "action": record["decision"]["action"],
            "reason": record["decision"]["reason"],
            "raw_viability": record["candidates"][0]["pi_c"],
            "normalized_share": record["candidates"][0]["normalized_share"],
            "questions_generated": len(record["questions"]),
            "synthetic_response_count": generator.calls,
            "ground_truth_candidate_present": record["gt_present"],
            "executed_correctly": None,
        },
        "candidate_set_checks": conditions,
        "intervention_control_notes": {
            "forced_miss_decoy": "Same tool and argument count; known B2 differs from synthetic ground truth B1.",
            "decoy_unknown_pattern_matched": False,
            "oracle_recovery_candidate_count_matched": False,
            "ground_truth_used_in_model_prompt": False,
        },
        "test_outcome": "PASS",
        "limitations": [
            "No LLM call, benchmark sample, user simulator, or tool execution was performed.",
            "Candidate-set compatibility is not equivalent to task success.",
            "Interventions were not followed by a new model inference or full SAGE turn.",
            "The one-example setup cannot estimate accuracy, calibration, HCWE, or causal effect.",
            "Full response-weighted EVPI is not implemented in this reconstruction.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reproduce()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(json.dumps({
        "status": result["test_outcome"],
        "scope": result["scope"],
        "input": result["source"]["input_path"],
        "output": str(args.output),
        "conditions": result["candidate_set_checks"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
