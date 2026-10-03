from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPOSITORY = "https://github.com/MananSuri27/ClarifyBench.git"
PINNED_COMMIT = "a85d4f9df1fc87d05c713fd408f8a57d72bcc348"


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone the pinned public ClarifyBench snapshot used by this study."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("third_party/ClarifyBench"),
    )
    args = parser.parse_args()

    destination = args.destination
    if destination.exists():
        raise SystemExit(
            f"Destination already exists: {destination}. Remove it or choose --destination."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    run("git", "clone", REPOSITORY, str(destination))
    run("git", "checkout", PINNED_COMMIT, cwd=destination)

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=destination, text=True
    ).strip()
    if head != PINNED_COMMIT:
        raise SystemExit(f"Unexpected HEAD after checkout: {head}")

    print(f"ClarifyBench pinned at {head}")


if __name__ == "__main__":
    main()
