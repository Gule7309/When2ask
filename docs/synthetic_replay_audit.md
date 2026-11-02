# Executable synthetic mechanism audit (Phase 0)

**Scope:** one *synthetic booking example* with fixed JSON responses. This is an **offline plumbing and intervention-invariant check**, not a ClarifyBench experiment, original-paper reproduction, or evidence for H1–H4.

## Run without API credentials or model downloads

From the repository root:

```bash
python -m pip install -e '.[dev]'
pytest
python scripts/reproduce_synthetic_fixture.py --output results/synthetic-audit.json
```

The final command refuses to overwrite an existing file. The run reads the checked-in `examples/sage_step.json` and `examples/sage_responses.json`. It performs zero LLM calls, zero actual tool executions and evaluates **zero benchmark samples**.

The `synthetic_replay` job in [GitHub Actions](../.github/workflows/tests.yml) runs the same command and uploads `when2ask-synthetic-mechanism-audit` containing `synthetic-mechanism-audit.json`. The record includes a **source commit SHA**, input/replay **SHA-256 hashes**, reconstruction thresholds, the decision trace, structural intervention checks and known limitations. CI checks failures via assertions; JSON schemas do not masquerade as empirical significance estimates.

## Expected checks (fixture-defined, not observed model behavior)

| Controlled state | Number of candidates | GT-compatible present | How obtained |
| --- | ---: | --- | --- |
| Natural synthetic replay | 1 | Yes | One partial `cancel_booking(<UNK>)` candidate |
| Oracle-complete | 1 | Yes | No-op when a compatible candidate already exists |
| Forced-miss | 1 | No | Replace the compatible candidate with `cancel_booking(B2)`; GT is B1 |
| Oracle recovery | 2 | Yes | Add one partial compatible candidate to Forced-miss |

The forced-miss decoy matches the **tool and argument count**, **not** the unresolved-argument pattern. Oracle recovery changes candidate-set size. The subsequent SAGE questions/viability are **not** re-evaluated for the intervened sets, so these records do not estimate the effect on accuracy, calibration, or high-confidence wrong execution.

The fixed replay's *one natural decision point* is expected to select **ASK** with raw viability `0.5` and normalized candidate share `1.0`. It is only a demonstration of the choice between a threshold and question generation, not the paper's EVPI or any final tool execution. The output must keep `executed_correctly = null`.

## What must happen before research results are shown

1. Audit actual ClarifyBench ambiguous decision-point alignment and record the upstream dataset SHA.
2. Run a fixed identifiable model (version/quantization), predeclare thresholds and input seed, and save actual model-generated candidate sets.
3. Run matched interventions through the **same downstream** decision logic and simulator, ensuring matched decoy properties, then compare paired outcomes.
4. Report sample count, confidence intervals, GT compatibility and adjudication rules. Do not infer HCWE/accuracy from a synthetic fixture.

Only then should the personal portfolio claim empirical H1–H3 evidence. A visible synthetic audit can demonstrate *mechanism readiness*, but must always carry the above distinction.
