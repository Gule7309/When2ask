# SAGE reconstruction notes

This document separates statements that are explicit in the paper from choices required to make the current diagnostic harness executable.

## Scope of the current reconstruction

The current code implements only the first part of SAGE's decision loop:

1. generate a finite candidate set from the current user request, observations, and tool schemas;
2. assign concrete argument values or <UNK>;
3. compute the structured argument-uncertainty score;
4. apply the execution-threshold check.

Question generation, EVPI scoring, domain update after a user response, and error recovery are intentionally not implemented yet. A low first-stage score is therefore recorded as `needs_question_scoring`, not as a final decision to ask the user.

This distinction matters for the candidate-set attack: the first measurement we need is whether the intended interpretation is represented in the initial hypothesis set and what internal score SAGE assigns before downstream clarification logic can act.

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
