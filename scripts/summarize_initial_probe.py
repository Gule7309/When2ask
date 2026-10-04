from __future__ import annotations

import argparse
import json
from pathlib import Path

from when2ask.probe_analysis import load_probe_records, summarize_initial_probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the Phase-1 initial candidate-set probe."
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
    )
    args = parser.parse_args()

    records = load_probe_records(args.trace)
    report = summarize_initial_probe(records, args.thresholds)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
