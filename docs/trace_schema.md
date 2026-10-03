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

The trace should eventually also log experiment metadata (condition, candidate count, model, prompt version, SAGE hyperparameters, simulator type, and run seed). Those fields are deliberately not required by the initial parser because the upstream SAGE integration has not yet been reconstructed.

The compatibility rule is used only to ask whether the intended tool call is represented in the candidate set. It must not be substituted for the benchmark's final correctness metrics.
