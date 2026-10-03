# When2ask

**Candidate-set incompleteness in structured-uncertainty tool agents**

This repository is a research reproduction and stress-test project built around *Structured Uncertainty guided Clarification for LLM Agents* (Suri et al., Findings of ACL 2026). The immediate goal is not to propose a new agent. It is to test a narrower assumption in SAGE-Agent: whether confidence over a model-generated candidate set remains meaningful when the intended tool call is not represented in that set.

The working hypothesis is that SAGE's structured belief can be well-defined *conditional on the current candidate set* while still being overconfident about the task as a whole. If the correct interpretation is absent from the candidate set, downstream uncertainty scoring may have no explicit way to represent that omission.

> **Status:** experiment scaffold in progress. No empirical result is claimed in this repository yet.

## Research question

**RQ1.** Does SAGE-Agent's structured uncertainty remain reliable when the correct tool-call interpretation is absent from its generated candidate set?

We separate this into four testable hypotheses:

- **H1 — candidate completeness:** performance should be substantially higher when a ground-truth-compatible candidate is present.
- **H2 — overconfidence:** candidate omission can coexist with high internal confidence, producing high-confidence wrong execution.
- **H3 — clarification limitation:** clarification over an incomplete hypothesis set may fail to recover the intended action unless candidate generation itself is revised.
- **H4 — open-world fix (later):** explicitly modeling candidate-set incompleteness should reduce high-confidence wrong executions without a large increase in clarification burden.

H4 is intentionally postponed until H1–H3 are tested. The first phase is diagnostic, not method-building.

## Experimental design

The main study uses matched interventions on the candidate set while holding the task, tool schema, model, prompts, and downstream SAGE logic fixed.

| Condition | Intervention | Purpose |
| --- | --- | --- |
| Natural | use the model-generated candidate set | observational baseline |
| Oracle-complete | add a ground-truth-compatible **partial** candidate when it is missing | isolate candidate omission as a bottleneck |
| Forced-miss | remove ground-truth-compatible candidates and, where possible, replace them with a structurally matched decoy | causal stress test |

A ground-truth-compatible candidate is not required to contain hidden argument values. It must select the correct tool and must not contradict any ground-truth value on arguments it does specify. Unresolved arguments remain `<UNK>`. This prevents the oracle condition from leaking the final answer.

Primary evaluation is planned on ClarifyBench-Ambiguous. We retain the paper's Coverage, Tool Match Rate (TMR), Parameter Match Rate (PMR), and average number of clarification questions, and add diagnostics specific to the attack:

- candidate recall;
- execution accuracy conditioned on ground-truth candidate presence;
- high-confidence wrong execution rate (HCWE);
- confidence/correctness calibration split by candidate presence;
- oracle recovery gap.

See [`docs/experiment_plan.md`](docs/experiment_plan.md) for the preregistered analysis plan.

## Why candidate-set incompleteness?

SAGE first generates a finite candidate set and then computes structured uncertainty over those candidates. This is a reasonable and tractable design, but it raises an open-world question: internal confidence may measure which candidate is best *among those considered*, rather than whether the intended candidate was considered at all.

The experiment is designed to distinguish three possibilities:

1. candidate omission is rare and practically unimportant;
2. omission hurts task accuracy but is already reflected by lower SAGE confidence;
3. omission can produce high-confidence errors, indicating a missing source of uncertainty.

Only the third outcome would strongly support the proposed attack.

## Repository layout

```text
.
├── configs/
│   └── experiment.toml
├── docs/
│   ├── experiment_plan.md
│   ├── reproduction_notes.md
│   └── trace_schema.md
├── scripts/
│   ├── bootstrap_clarifybench.py
│   └── inspect_candidate_recall.py
├── src/when2ask/
│   ├── clarifybench.py
│   ├── compatibility.py
│   ├── interventions.py
│   ├── metrics.py
│   ├── models.py
│   └── trace_io.py
└── tests/
```

The code currently focuses on the parts that should be model-independent and easy to audit: ground-truth compatibility, candidate-set interventions, data loading, and diagnostic metrics. Agent integration will be added only after these components are stable. The current JSONL trace schema and diagnostic CLI are documented in [`docs/trace_schema.md`](docs/trace_schema.md).

## Reproducibility baseline

The public ClarifyBench repository is pinned to:

```text
a85d4f9df1fc87d05c713fd408f8a57d72bcc348
```

Bootstrap it with:

```bash
python scripts/bootstrap_clarifybench.py
```

The upstream code and the paper are not perfectly aligned. In particular, the public configuration currently uses temperature `0.7`, while the paper reports temperature `0.5` for the main comparison. The paper also introduces an execution threshold but does not provide a numerical value in the experiment description we inspected. These are treated as reproduction uncertainties rather than silently resolved; details are recorded in [`docs/reproduction_notes.md`](docs/reproduction_notes.md).

## Development

The core package has no runtime dependency beyond Python 3.10+. Tests use `pytest`.

```bash
python -m pip install -e '.[dev]'
pytest
```

At the initial scaffold commit, the model-independent test suite contains 13 tests covering compatibility, interventions, metrics, trace parsing, and the current ClarifyBench JSON schema.

## Upstream

- Paper: *Structured Uncertainty guided Clarification for LLM Agents*, Findings of ACL 2026.
- Public benchmark/code: <https://github.com/MananSuri27/ClarifyBench>

This repository is an independent research project and is not an official implementation maintained by the paper authors.
