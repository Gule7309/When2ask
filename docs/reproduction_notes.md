# Reproduction notes

This file records mismatches and unresolved choices rather than silently normalizing them. The goal is to keep later experimental conclusions separate from reconstruction assumptions.

## Upstream snapshot

Public repository: `MananSuri27/ClarifyBench`

Pinned commit:

```text
a85d4f9df1fc87d05c713fd408f8a57d72bcc348
```

Commit message: `metrics updated`.

The experiment should record this SHA in every run manifest. We do not vendor the upstream benchmark into this repository.

## Paper / public-code differences observed so far

### Model temperature

The paper reports temperature `0.5` for the main GPT-4o and Qwen2.5-14B comparison. The current public `config.py` uses `0.7`.

For paper-oriented reproduction runs, use `0.5` and record the override. For public-code replication runs, preserve `0.7`. Do not mix the two under one result label.

### Execution threshold

The SAGE description uses an execution threshold `tau_exec`, but the numerical value was not specified in the experiment description we inspected. We therefore treat it as unresolved and use a predeclared sensitivity sweep instead of selecting one value post hoc.

### Maximum SAGE steps

The paper describes a maximum number of SAGE steps but the numerical setting was not identified in the inspected method/evaluation text. The public harness currently exposes `max_turns = 10` and `max_clarifications_per_request = 5`; these should not automatically be assumed to be the paper's SAGE maximum-step setting.

### Public agent coverage

The inspected public experiment configuration registers `core.baseline_agent`. A complete SAGE implementation is not currently registered there. This project should therefore distinguish code copied/reused from upstream from SAGE components reconstructed from the paper.

### Dataset snapshot

The public repository snapshot should be treated as its own versioned artifact. Dataset counts must be measured from the pinned checkout and reported directly rather than assumed to match the paper table.

## Current ClarifyBench JSON schema observed

The public loader accepts either `user_query` or `initial_query` and expects `ground_truth_tool_calls` as a list of:

```json
{
  "tool_name": "...",
  "parameters": {"...": "..."}
}
```

Current sample files also contain fields such as `potential_follow_ups`, `user_intention`, `initial_config`, and `primary_api`.

## Reproduction discipline

Every empirical run should save at least:

- this repository commit SHA;
- upstream ClarifyBench SHA;
- model identifier;
- model temperature;
- prompt/version identifier;
- candidate count `N`;
- `lambda`, `alpha`, `epsilon`, and `tau_exec`;
- simulator type;
- dataset directory and number of evaluated decision points;
- random seed where applicable.

If a setting is unknown, label it as an assumption or sensitivity parameter. Do not back-fill it from whichever value gives the cleanest result.
