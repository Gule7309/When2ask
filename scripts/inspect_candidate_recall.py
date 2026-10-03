from __future__ import annotations

import argparse
import json

from when2ask.metrics import (
    candidate_recall,
    conditional_execution_accuracy,
    high_confidence_wrong_execution_rate,
)
from when2ask.trace_io import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute candidate-set incompleteness diagnostics from JSONL traces."
    )
    parser.add_argument("trace", help="JSONL file containing decision records")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
    )
    args = parser.parse_args()

    records = load_jsonl(args.trace)
    report = {
        "n_records": len(records),
        "candidate_recall": candidate_recall(records),
        "execution_accuracy_gt_present": conditional_execution_accuracy(
            records, gt_present=True
        ),
        "execution_accuracy_gt_absent": conditional_execution_accuracy(
            records, gt_present=False
        ),
        "hcwe": {
            str(t): high_confidence_wrong_execution_rate(records, t)
            for t in args.thresholds
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
