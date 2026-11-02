"""Executed synthetic smoke must stay deterministic and never claim empirical outcomes."""
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/reproduce_synthetic_fixture.py"


def run(output: Path, *, env: dict | None = None):
    return subprocess.run(
        [sys.executable, str(RUNNER), "--output", str(output)],
        cwd=ROOT, env={**os.environ, **(env or {})},
        capture_output=True, text=True,
    )


def test_synthetic_audit_is_deterministic_with_fixed_git_commit(tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    env = {"GITHUB_SHA": "a" * 40}
    result1, result2 = run(first, env=env), run(second, env=env)
    assert result1.returncode == 0, result1.stderr
    assert result2.returncode == 0, result2.stderr
    assert first.read_bytes() == second.read_bytes()
    data = json.loads(first.read_text(encoding="utf-8"))
    assert data["schema"] == "synthetic_mechanism_audit_v1"
    assert data["empirical"] is False
    assert data["benchmark_items_evaluated"] == 0
    assert data["model_calls"] == 0
    assert data["real_tool_executions"] == 0
    assert data["single_decision"]["action"] == "ask"
    assert data["single_decision"]["raw_viability"] == 0.5
    assert data["single_decision"]["normalized_share"] == 1.0
    assert data["single_decision"]["executed_correctly"] is None
    assert [x["ground_truth_compatible_candidate_present"] for x in
            data["candidate_set_checks"]] == [True, True, False, True]
    assert [x["candidate_count"] for x in data["candidate_set_checks"]] == [1, 1, 1, 2]
    assert len(data["source"]["input_sha256"]) == 64
    assert len(data["source"]["replay_sha256"]) == 64


def test_audit_refuses_overwriting_existing_report(tmp_path):
    output = tmp_path / "report.json"
    assert run(output).returncode == 0
    prior = output.read_bytes()
    assert run(output).returncode != 0
    assert output.read_bytes() == prior
