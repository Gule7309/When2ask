# SAGE reconstruction notes

This document separates statements that are explicit in the paper from choices required to make the current diagnostic harness executable.

## Scope of the current reconstruction

The original `run_probe` implements the first part of SAGE's decision loop:

1. generate a finite candidate set from the current user request, observations, and tool schemas;
2. assign concrete argument values or <UNK>;
3. compute the structured argument-uncertainty score;
4. apply the execution-threshold check.

The first-stage probe still records a low score as `needs_question_scoring`. The new `sage.run_step` continues with aspect-targeted question generation and an optimistic perfect-resolution score to produce a final ask/execute plan. Response-weighted EVPI, automatic domain update after a user response, tool execution, and error recovery remain outside this minimal reconstruction.

This distinction matters for the candidate-set attack: the first measurement we need is whether the intended interpretation is represented in the initial hypothesis set and what internal score SAGE assigns before downstream clarification logic can act.

## Source version and implementation boundary

The sources inspected for this phase are the [published Findings of ACL 2026 paper](https://aclanthology.org/2026.findings-acl.2028.pdf), Sections 4 and 5, and the [arXiv v1 method](https://arxiv.org/html/2511.08798v1), Sections 3 and 5. They do not give identical question-scoring descriptions: arXiv v1 specifies a candidate-partition expression, while the published Step 3 describes perfect-resolution multipliers followed by an expected maximum. We do not silently combine these into a purported exact algorithm.

All candidate-generation, belief, question, and decision code is authored independently in `src/when2ask`. The upstream adapter supplies schema/context only. No upstream agent is registered or modified by this phase.

## Candidate generation

### Explicit in the paper

SAGE Step 1 states that an LLM receives the current user request, observation history, and toolkit and produces N candidate tool calls. Arguments may be concrete values or <UNK>.

### Reconstruction choice

The appendix provides a ReAct reasoning prompt and a clarification-question prompt, but the inspected paper text does not provide a separate verbatim prompt that asks for N alternative candidate invocations.

`generation.build_candidate_prompt` is therefore a reconstruction. It makes one additional semantic constraint explicit: candidates are alternative hypotheses for the intended next action, not a sequence of tool calls.

This prompt must be versioned in later runs. Any material prompt change creates a new experimental condition.

## What is pi?

The paper creates a genuine reconstruction ambiguity.

The main text calls the object a structured belief distribution and writes the candidate score as proportional to the product of parameter certainties. However, the appendix viability proof treats the product itself as pi and relies on the property that the score reaches 1 when every parameter is specified.

These interpretations differ if multiple candidates are present.

The harness therefore logs both quantities:

- `viability`: the raw product of argument certainties;
- `normalized_share`: viability divided by the total viability over the generated candidate set.

The first-stage execution threshold currently uses raw viability. This choice follows the appendix completeness property and the perfect-resolution multiplication described in the method. It is not presented as the only possible interpretation of the paper.

A later sensitivity analysis should repeat the candidate-set experiment using normalized share if the conclusion depends on this choice.

## Argument certainty

The implemented rule is:

- specified argument: 1;
- unresolved finite argument: 1 / |D|;
- unresolved unbounded / continuous argument: epsilon.

The paper reports epsilon = 1e-4 in its main experimental setup.

Missing candidate arguments are interpreted as <UNK>. Candidate arguments not present in the selected tool schema are rejected because no paper-defined domain exists for them.

## Optional parameters

The main equation multiplies over the tool's parameter set, while the tool definition separately identifies a required-parameter subset. The inspected method text does not say to restrict the uncertainty product to required parameters.

The default reconstruction therefore includes optional parameters. `include_optional=False` exists only as an explicit sensitivity condition.

This setting must be logged because optional parameters can substantially change the raw viability score.

## Uniform tool prior

The paper assumes a uniform tool prior. Because the same constant prior factor applies to every candidate, it is omitted from the raw product used for ranking and threshold reconstruction.

This is not equivalent to claiming that the uniform-prior assumption is harmless in general; it only means the common multiplicative constant is not represented twice in this implementation.

## Execution threshold

The paper defines an execution threshold but the numerical value was not identified in the inspected experimental specification. The harness therefore refuses to infer a canonical value. Experiment scripts require an explicit value, and the preregistered study uses a sensitivity sweep.

## Initial diagnostic scope

`scripts/run_initial_candidate_probe.py` currently aligns the initial user query with the first ground-truth tool call only.

This is a deliberately narrow diagnostic. It is useful for checking candidate generation, tool-domain loading, GT compatibility, and score behavior, but it is not a replacement for the full multi-turn ClarifyBench protocol. The intervention experiment will require explicit request-to-tool-call alignment before full benchmark claims are made.

## Question generation and perfect-resolution approximation

Published Step 2 specifies the inputs and the `(question, target candidate, aspects)` output. Our `reconstructed_questions_v1` prompt supplies those inputs and requests explicit schema-qualified aspects. Candidate indices are zero-based. Malformed questions, unknown tools/arguments, repeated aspects, and invalid target indices raise errors instead of being silently repaired. An empty question list is a separate recorded condition.

Let `v_c` denote raw viability. For each question, we recompute `v_c^q` with the certainty of every targeted argument set to 1, leaving all other factors unchanged. Tool-qualified aspects affect only the matching tool. The score is:

```text
resolution_gain(q) = max_c v_c^q - max_c v_c
score(q) = resolution_gain(q) - lambda * sum_a n_a
```

This mirrors the published finite-domain perfect-resolution multiplier, but makes an additional **optimistic simultaneous-resolution assumption**. We have not supplied `P(r | c, q)`, sampled possible user answers, eliminated contradictory candidates, or computed the expectation in Definition 4. Consequently this quantity is a heuristic proxy, not exact EVPI. In particular it does not resolve tool-choice uncertainty between fully specified alternatives, and may overestimate useful gain when candidate interpretations conflict. It must not be used to claim reproduction of the full SAGE Table 3 results.

For an unbounded domain, resolving an unknown changes its certainty from epsilon to 1. We recompute the product directly rather than multiplying by infinity. Optional-parameter sensitivity applies to question scoring as well as viability.

## Final decision and per-step contract

- Execute if maximum raw viability meets `tau_exec`; skip the question model call.
- Otherwise execute if the zero-based step index reaches `max_steps`; skip question generation.
- Otherwise generate and score questions. Ask the first highest-scoring question unless its score is strictly below `alpha * max_viability`. Equality therefore asks, as in the published strict inequality.
- If no questions are returned, record `no_questions` and select the best candidate. This fallback is an explicit reconstruction assumption.

Ties in both candidate and question ranking use generation order. Duplicates in the candidate list retain their multiplicity, as in the existing probe; normalized share is conditional on that list. `n_candidates` is an upper bound enforced by truncation, and an undersized response is retained and visible in the trace. It is not supplemented using GT.

`max_steps` is interpreted as a maximum number of prior questions for the current request. It and `tau_exec` must be supplied explicitly. A caller must advance `step`, increment the counts of aspects actually asked, add the answer to observations, and supply narrowed domains where appropriate. `run_step` has no persistent state and performs no hidden response extraction. Candidate generation is repeated from current observations at each invocation; no Bayesian posterior is carried between changing candidate lists.

GT is excluded from both prompts, question scoring, and selection. Compatibility labels are computed after generation. `execute` is an action plan: it may still contain unknown required arguments when a budget or gain stopping rule fires. The trace flags this with `selected_candidate_fully_specified`; no hidden GT values or defaults are inserted. Actual `executed_candidate` and `executed_correctly` remain null until an environment adapter executes and evaluates the plan.
