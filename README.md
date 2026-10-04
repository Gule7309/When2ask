# When2ask

**Candidate-set incompleteness in structured-uncertainty tool agents**

This repository is a research reconstruction and controlled stress-test project built around *Structured Uncertainty guided Clarification for LLM Agents* (Suri et al., Findings of ACL 2026). The immediate goal is not to propose a new agent. It is to test a narrower assumption in SAGE-Agent: whether confidence over a model-generated candidate set remains meaningful when the intended tool call is not represented in that set.

The working hypothesis is that SAGE's structured belief can be well-defined *conditional on the current candidate set* while still being overconfident about the task as a whole. If the correct interpretation is absent from the candidate set, downstream uncertainty scoring may have no explicit way to represent that omission.

> **Status:** Minimal instrumented SAGE decision steps implemented; no empirical result is claimed yet. Candidate generation and structured viability are followed by aspect-targeted question generation and an explicitly labelled optimistic perfect-resolution score. This is a mechanism reconstruction, not a complete reproduction of the paper's response-weighted EVPI or multi-turn benchmark.

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

See [`docs/experiment_plan.md`](docs/experiment_plan.md) for the preregistered analysis plan and [`docs/sage_reconstruction.md`](docs/sage_reconstruction.md) for the assumptions required to reconstruct SAGE from the paper.

## Model policy

The original paper evaluates GPT-4o and Qwen2.5-14B-Instruct. This repository distinguishes three kinds of runs:

1. **paper reproduction** — matching the paper's base model and serving condition;
2. **mechanism reconstruction** — keeping the reconstructed SAGE mechanism fixed while using another explicitly reported model;
3. **cross-model robustness** — repeating the same candidate-set intervention across multiple base models.

A run with another model must not be reported as a reproduction of the paper's Table 3.

For local pipeline validation, the current development condition is `qwen2.5:7b` served through Ollama's OpenAI-compatible endpoint. The model choice and local quantization are part of the experimental condition, not an implementation detail. See [`docs/local_model_protocol.md`](docs/local_model_protocol.md).

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
│   ├── local_model_protocol.md
│   ├── reproduction_notes.md
│   ├── sage_reconstruction.md
│   └── trace_schema.md
├── scripts/
│   ├── bootstrap_clarifybench.py
│   ├── inspect_candidate_recall.py
│   ├── run_initial_candidate_probe.py
│   ├── run_sage_step.py
│   ├── run_qwen25_7b_smoke.ps1
│   └── summarize_initial_probe.py
├── src/when2ask/
│   ├── belief.py
│   ├── clarifybench.py
│   ├── compatibility.py
│   ├── decision.py
│   ├── generation.py
│   ├── harness.py
│   ├── interventions.py
│   ├── metrics.py
│   ├── models.py
│   ├── probe_analysis.py
│   ├── provider.py
│   ├── questioning.py
│   ├── sage.py
│   ├── schema.py
│   ├── trace_io.py
│   └── upstream.py
└── tests/
```

The independent reconstruction lives in `src/when2ask/`; `upstream.py` only adapts upstream tool schemas and context. The decision core imports no ClarifyBench agent. Ground-truth compatibility, interventions, and diagnostic metrics remain separate analysis components.

## Instrumented decision steps

`when2ask.sage.run_step` accepts a request, observation history, current tool schemas, a zero-based step index, and aspect-query counts. Each result records `C_t` as `candidates`, raw `pi_c`, `normalized_share`, `gt_present`, and a final `decision.action` (`ask` or `execute`). It also records the question scores, stopping reason, settings, schema snapshot, and parsed model responses with prompt hashes. Ground truth is used only for diagnostic labels and is never passed to either generation prompt.

Run the synthetic example without a model or credentials:

```bash
python -m pip install -e '.[dev]'
python scripts/run_sage_step.py --input examples/sage_step.json \
  --replay examples/sage_responses.json --output results/synthetic-step.jsonl
```

This emits an `ask` decision with raw viability `0.5` and normalized share `1.0`. The fixture is a plumbing check, not an experimental result. Use a fresh output path for each invocation; the CLI refuses to overwrite a trace. For live generation, replace `--replay ...` with `--model MODEL` and optionally `--base-url URL`, after installing `.[llm]`.

The initial ClarifyBench probe can also emit final decision plans:

```bash
python scripts/run_initial_candidate_probe.py --model MODEL \
  --decision-mode sage-step --tau-exec 0.8 --max-steps 3 \
  --output results/initial-sage-steps.jsonl
```

Both `0.8` and `3` are illustrative sensitivity settings, not recovered paper values. The default `threshold` mode preserves the original first-stage probe. Neither CLI invokes domain tools or simulates user answers. `execute` denotes a selected plan; actual execution outcomes remain null. A caller can repeat `run_step` after supplying an answer in observations, updated schemas, and incremented aspect counts. Automatic constraint extraction and full multi-turn request alignment remain future work.

The question score assumes simultaneous perfect resolution of targeted arguments and measures the increase in maximum raw viability, minus redundancy cost. It is deliberately named `optimistic_perfect_resolution_v1`, not full EVPI. The paper does not fully specify the response distribution required to implement that expectation. See the reconstruction notes for this limitation, tie handling, and stopping assumptions.

## Phase 1: initial candidate-set probe

The first empirical step is a 20-example smoke test, followed by manual audit. On Windows with Ollama and Qwen2.5-7B:

```powershell
python -m pip install -e ".[dev,llm]"
powershell -ExecutionPolicy Bypass -File scripts/run_qwen25_7b_smoke.ps1
```

The helper script pulls `qwen2.5:7b`, bootstraps the pinned ClarifyBench checkout if needed, runs the initial probe with temperature `0.5`, and prints a Phase 1 summary.

The default `tau_exec=0.8` in the smoke test is only a diagnostic sensitivity point. It is **not** claimed to be the paper's execution threshold.

The current probe aligns the initial user query to the first ground-truth tool call only. Its purpose is to measure initial candidate recall and inspect reconstructed confidence behavior before implementing the full multi-turn intervention experiment. It must not be reported as full ClarifyBench performance.

The Phase 1 summary reports candidate recall separately from the more mechanism-specific quantity:

```text
high-viability wrong-best rate given GT candidate absent
```

That distinction is important because base-model quality directly affects candidate recall, whereas the conditional behavior after omission is the central target of the attack.

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

The upstream harness also updates data-dependent tool domains from each sample context before agent execution. The local adapter mirrors this behavior because SAGE's structured viability depends directly on domain cardinality.

## Development

The core package has no runtime dependency beyond Python 3.10+. Tests use `pytest`; live candidate generation uses the optional `openai` client.

```bash
python -m pip install -e '.[dev,llm]'
pytest
```

The test suite covers compatibility, interventions, structured viability, candidate generation, first-stage and final decision plans, question validation, redundancy costs, stopping boundaries, GT isolation, replay traces, local JSON parsing, Phase 1 analysis, and the current ClarifyBench JSON schema. GitHub Actions runs the suite on pushes and pull requests.

## Upstream

- Paper: *Structured Uncertainty guided Clarification for LLM Agents*, Findings of ACL 2026.
- Public benchmark/code: <https://github.com/MananSuri27/ClarifyBench>
- Local development model: <https://ollama.com/library/qwen2.5:7b>

This repository is an independent research project and is not an official implementation maintained by the paper authors.
