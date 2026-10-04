# Candidate trace schema

The intervention study consumes one JSON object per SAGE decision point. JSONL is used so that long runs can be streamed and audited without loading an entire experiment into memory.

Minimal record:

```json
{
  "sample_id": "example-001:turn-1",
  "ground_truth": {
    "tool_name": "cancel_booking",
    "parameters": {"booking_id": "B1"}
  },
  "candidates": [
    {
      "tool_name": "cancel_booking",
      "arguments": {"booking_id": "<UNK>"},
      "confidence": 0.72
    }
  ],
  "executed_candidate": null,
  "executed_correctly": null
}
```

If execution occurs, `executed_candidate` should contain the candidate as it was evaluated at the execution decision, including its structured confidence. `executed_correctly` is a boolean derived from the benchmark's execution/evaluation logic, not from the candidate compatibility rule.

The initial parser accepts this minimal schema and ignores extra instrumentation. Records consumed by it require a non-null `ground_truth`; unlabeled live decision steps can be serialized but cannot be used for GT-dependent metrics.

The compatibility rule is used only to ask whether the intended tool call is represented in the candidate set. It must not be substituted for the benchmark's final correctness metrics.

## Instrumented reconstruction schema, version 1

`sage.run_step` returns a JSON-compatible record for each explicit decision point. `scripts/run_sage_step.py` writes it as one JSONL line. The initial ClarifyBench probe also emits this schema in `--decision-mode sage-step` mode.

| Field | Meaning |
| --- | --- |
| `schema_version`, `implementation` | Version of this independent reconstruction |
| `sample_id`, `step` | Request identifier and zero-based count of prior questions |
| `candidates` | Ordered `C_t`; index, tool, arguments, raw `pi_c`/`viability`, `normalized_share`, unresolved argument names, and optional GT compatibility |
| `gt_present` | GT-compatible candidate exists in `C_t`; null without a GT label |
| `ground_truth` | Evaluation-only label, never part of a model prompt |
| `pre_question_decision` | Original threshold-stage result, preserved for Phase 1 analysis |
| `decision` | Final `ask`/`execute` plan, reason, best candidate index, selected question, dynamic stopping threshold, and required-argument completeness flag |
| `questions` | Generated questions, target indices/aspects, optimistic resolution gains, redundancy costs, and final scores |
| `aspect_counts` | Tool-qualified counts supplied by caller, before this decision |
| `observations`, `tool_schemas` | Current history and argument domains used for generation/scoring |
| `settings` | Candidate/question limits, budget, temperature, epsilon, lambda, alpha, tau, optional-argument mode, prompt versions, scoring assumption and `pi_c` convention |
| `generation_calls` | Parsed model responses and SHA-256 hashes of actual UTF-8 prompts, in call order |
| `run` | CLI-supplied source metadata; replay/live status or model/backend, scope and condition |
| `executed_candidate`, `executed_correctly` | Null until actual environment execution and evaluation |

`pi_c` is an explicit alias for raw viability. `normalized_share` sums to 1 over the returned list and is not used by this reconstruction's thresholds. The `question_score_method` is always `optimistic_perfect_resolution_v1`; the `resolution_gain` field must not be interpreted as response-weighted EVPI.

The threshold and budget branches produce an empty `questions` list and one generation call. Other branches generate questions in a second call. Candidate and question ties use list order. Actual candidate count is `len(candidates)` and can be smaller than the requested upper bound.

The standalone CLI's `run.empirical=false` marks synthetic replay records. Live records still describe proposed decisions, not successful tool executions or full benchmark scores. The existing summary script continues to summarize **pre-question** threshold decisions even for enriched records; its execution-rate fields are not final action rates.

Parsed responses are logged rather than raw HTTP envelopes or model token probabilities. Provider/model revision, sampling reproducibility, upstream SHA and this repository SHA must still be recorded in an empirical run manifest. Prompt hashes identify the prompt used but do not guarantee deterministic model outputs. The current CLI does not supply a simulator or posterior domain updater.
