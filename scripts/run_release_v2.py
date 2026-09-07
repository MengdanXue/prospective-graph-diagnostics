#!/usr/bin/env python3
"""Run a published v2 workflow with an explicit configuration.

The original execution files retain their historical defaults and Git blobs.
Use this entry point for the current release. Other frozen configurations
remain accessible through the original commands with an explicit --config.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "prospective_benchmark_v2.json"
COMMANDS = {
    "benchmark": "experiments/run_prospective_benchmark.py",
    "intervention": "experiments/run_degree_matched_benchmark.py",
    "assemble": "scripts/assemble_prospective_diagnostics.py",
    "degree-summary": "scripts/summarize_degree_matched_benchmark.py",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if any(value == "--config" or value.startswith("--config=") for value in args.arguments):
        parser.error("this entry point selects v2; use the original command for another configuration")
    command = [sys.executable, str(ROOT / COMMANDS[args.command]), "--config", str(CONFIG), *args.arguments]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
