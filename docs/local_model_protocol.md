# Local-model protocol

The original SAGE evaluation uses GPT-4o and Qwen2.5-14B-Instruct. This project does not treat access to those exact serving conditions as a prerequisite for the candidate-set stress test.

The experimental distinction is:

- **paper reproduction condition** — same base model and serving setup as the paper, if available;
- **mechanism reconstruction condition** — same reconstructed SAGE mechanism with a different, explicitly reported base model;
- **cross-model robustness condition** — repeat the same intervention across multiple base models.

Results from the second or third category must not be described as reproducing the paper's Table 3.

## Development model

The first local smoke-test condition is:

```text
backend: Ollama (OpenAI-compatible endpoint)
model tag: qwen2.5:7b
candidate temperature: 0.5
candidate count: 5
```

Ollama currently publishes `qwen2.5:7b` as a 4.7 GB Q4_K_M model. The upstream Qwen2.5-7B-Instruct model is an instruction-tuned 7B-class model released under Apache-2.0. The local quantized Ollama build is therefore **not numerically identical** to the full-precision Hugging Face checkpoint and must be reported as its own model condition.

Official references:

- https://ollama.com/library/qwen2.5:7b
- https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- https://ollama.com/blog/openai-compatibility

## Why start with 7B?

The 7B condition is a pipeline-validation model, not the final empirical claim. It is used to verify:

1. candidate generation and JSON parsing;
2. `<UNK>` behavior;
3. tool-schema loading and data-dependent domains;
4. GT-compatible candidate labeling;
5. viability-score logging;
6. the first-stage execute / continue decision.

Once the 20-example audit passes, the same protocol can be repeated with a stronger local model such as `qwen2.5:14b` if resources permit. The model change must be treated as a new experimental condition.

## What changes when the base model changes?

The candidate generator directly changes the probability that the intended hypothesis is represented:

```text
P(GT candidate in C_t)
```

This is expected to be model-dependent.

The primary mechanism question is conditional:

```text
P(high internal viability | GT candidate absent)
```

The controlled Forced-miss intervention is important precisely because it lets us test this second quantity while manipulating candidate-set completeness directly.

## Reproducibility metadata

Every run should record:

- backend and serving software;
- model name/tag;
- model revision or local digest if available;
- quantization;
- candidate-generation temperature;
- candidate count N;
- reconstruction prompt version;
- epsilon;
- tau_exec;
- random seed;
- ClarifyBench commit;
- When2ask commit.

For Ollama, save the output of:

```powershell
ollama list
ollama show qwen2.5:7b
```

alongside the experiment log when a run is used in a report.

## Windows / Ollama smoke test

Install Ollama separately, then from the repository root run:

```powershell
ollama pull qwen2.5:7b
python -m pip install -e ".[dev,llm]"
python scripts/bootstrap_clarifybench.py

$env:OPENAI_BASE_URL = "http://localhost:11434/v1"
$env:OPENAI_API_KEY = "ollama"

python scripts/run_initial_candidate_probe.py --model qwen2.5:7b --output results/qwen2.5-7b/initial_probe_n5_tau08.jsonl --tau-exec 0.8 --n-candidates 5 --temperature 0.5 --limit 20 --seed 0
python scripts/summarize_initial_probe.py results/qwen2.5-7b/initial_probe_n5_tau08.jsonl
```

The `tau_exec=0.8` run is a smoke-test point only. It is not claimed to be the paper's threshold. Formal analysis uses the preregistered threshold sweep.
