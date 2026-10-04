"""Emit one instrumented reconstruction step from explicit decision-point input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from when2ask.models import GroundTruthCall
from when2ask.provider import OpenAIJsonGenerator
from when2ask.questioning import Aspect
from when2ask.sage import SageSettings, run_step
from when2ask.schema import load_tool_specs


class ReplayGenerator:
    """Replay model JSON objects; fixtures validate plumbing, not model quality."""
    def __init__(self, responses: list[dict]):
        self.responses = iter(responses)

    def generate_json(self, prompt, *, temperature, max_tokens):
        try:
            return next(self.responses)
        except StopIteration as exc:
            raise ValueError("replay ran out of model responses") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--replay", type=Path, help="JSON list of model responses; no network access")
    source.add_argument("--model", help="Explicit model for live OpenAI-compatible generation")
    parser.add_argument("--base-url")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    settings = SageSettings(**data["settings"])
    generator = (
        ReplayGenerator(json.loads(args.replay.read_text(encoding="utf-8")))
        if args.replay else OpenAIJsonGenerator(model=args.model, base_url=args.base_url)
    )
    truth = data.get("ground_truth")
    record = run_step(
        generator, sample_id=data["sample_id"], step=data.get("step", 0),
        user_query=data["user_query"], observations=data.get("observations", []),
        tools=load_tool_specs(data["tools"]), settings=settings,
        ground_truth=None if truth is None else GroundTruthCall(**truth),
        aspect_counts={Aspect(row["tool_name"], row["argument"]): row["count"]
                       for row in data.get("aspect_counts", [])},
    )
    record["run"] = {"scope": "explicit_decision_point", "condition": "natural",
                     "backend": "replay" if args.replay else "openai-compatible",
                     "model": args.model, "empirical": args.replay is None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of a previous trace.
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "decision": record["decision"]}, indent=2))


if __name__ == "__main__":
    main()
