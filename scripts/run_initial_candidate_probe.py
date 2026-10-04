from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from when2ask.clarifybench import load_sample
from when2ask.harness import run_probe
from when2ask.provider import OpenAIJsonGenerator
from when2ask.upstream import load_tool_specs_for_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the first-turn candidate-set diagnostic on ClarifyBench-A. "
            "This is a reconstruction probe, not a full SAGE benchmark run."
        )
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path("third_party/ClarifyBench"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Defaults to <upstream-root>/ClarifyBench/ClarifyBench_A",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=os.environ.get("WHEN2ASK_MODEL", "gpt-4o-2024-08-06"),
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument(
        "--backend",
        default=os.environ.get("WHEN2ASK_BACKEND", "openai-compatible"),
        help="Human-readable serving backend label stored in the trace.",
    )
    parser.add_argument(
        "--model-revision",
        default=os.environ.get("WHEN2ASK_MODEL_REVISION"),
        help="Optional model revision, digest, or checkpoint identifier.",
    )
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument(
        "--tau-exec",
        type=float,
        required=True,
        help="Paper value is unresolved; pass an explicit sensitivity value.",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="Sensitivity mode: exclude optional parameters from the viability product.",
    )
    args = parser.parse_args()

    dataset = args.dataset or args.upstream_root / "ClarifyBench" / "ClarifyBench_A"
    paths = sorted(dataset.glob("*.json"))
    random.Random(args.seed).shuffle(paths)
    if args.limit is not None:
        paths = paths[: args.limit]

    generator = OpenAIJsonGenerator(model=args.model, base_url=args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for path in paths:
            sample = load_sample(path)
            if not sample.ground_truth_tool_calls:
                skipped += 1
                continue

            # Phase-1 diagnostic scope: initial query aligned to the first GT call only.
            # Full multi-turn / multi-call alignment is deferred to the intervention run.
            ground_truth = sample.ground_truth_tool_calls[0]
            tools = load_tool_specs_for_sample(args.upstream_root, sample.raw)

            result = run_probe(
                generator,
                sample_id=f"{sample.sample_id}:initial:first_gt",
                user_query=sample.user_query,
                observations=(),
                tools=tools,
                n_candidates=args.n_candidates,
                tau_exec=args.tau_exec,
                epsilon=args.epsilon,
                temperature=args.temperature,
                ground_truth=ground_truth,
                include_optional=not args.required_only,
            )

            record = result.to_dict()
            record["ground_truth"] = {
                "tool_name": ground_truth.tool_name,
                "parameters": dict(ground_truth.parameters),
            }
            record["run"] = {
                "scope": "initial_query_first_ground_truth_call",
                "backend": args.backend,
                "base_url": args.base_url,
                "model": args.model,
                "model_revision": args.model_revision,
                "n_candidates": args.n_candidates,
                "temperature": args.temperature,
                "epsilon": args.epsilon,
                "tau_exec": args.tau_exec,
                "include_optional": not args.required_only,
                "seed": args.seed,
                "candidate_prompt_version": "reconstructed_v1",
                "threshold_quantity": "raw_viability",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed += 1

    print(
        json.dumps(
            {
                "dataset": str(dataset),
                "output": str(args.output),
                "completed": completed,
                "skipped": skipped,
                "scope": "initial_query_first_ground_truth_call",
                "backend": args.backend,
                "model": args.model,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
