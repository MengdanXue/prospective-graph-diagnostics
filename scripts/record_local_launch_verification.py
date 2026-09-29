"""Record a local launch-verification receipt for the formal input-robustness run.

The receipt replaces a GitHub Actions receipt only where the formal gate
explicitly accepts ``kind == "local_verification"``.  It runs the complete
protocol test suite on a clean checkout of the exact source commit and states
plainly that it is local.  It never claims to be GitHub CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _parse_unittest(output: str) -> dict[str, int | bool]:
    ran = re.search(r"^Ran (\d+) tests? in", output, re.MULTILINE)
    failed = re.search(r"^FAILED \(([^)]*)\)", output, re.MULTILINE)
    counts = {"failures": 0, "errors": 0, "skipped": 0}
    for key in counts:
        match = re.search(rf"{key}=(\d+)", failed.group(1) if failed else "")
        if match:
            counts[key] = int(match.group(1))
    skipped = re.search(r"skipped=(\d+)", output.splitlines()[-1] if output.splitlines() else "")
    if skipped:
        counts["skipped"] = int(skipped.group(1))
    return {"tests_run": int(ran.group(1)) if ran else 0,
            "ok": bool(re.search(r"^OK\b", output, re.MULTILINE)) and not failed, **counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite an existing receipt: {args.output}")
    if _git("status", "--porcelain"):
        raise SystemExit("source checkout is not clean (tracked or untracked changes present)")
    commit = _git("rev-parse", "HEAD")
    started = time.time()
    completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                               cwd=ROOT, capture_output=True, text=True)
    duration = time.time() - started
    output = completed.stdout + completed.stderr
    log_path = args.output.with_suffix(".log")
    log_path.write_text(output, encoding="utf-8")
    parsed = _parse_unittest(output)
    if _git("status", "--porcelain") or _git("rev-parse", "HEAD") != commit:
        raise SystemExit("source checkout changed while the suite was running")
    passed = completed.returncode == 0 and bool(parsed["ok"]) and int(parsed["tests_run"]) > 0
    conclusion = "success" if passed else "failure"
    run_id = int(started)
    receipt = {
        "record_type": "local_launch_verification",
        "kind": "local_verification",
        "note": "Complete unittest suite on a clean local checkout of the exact commit; this is not GitHub Actions.",
        "status": "passed" if passed else "failed",
        "commit": commit,
        "execution_commit": commit,
        "technical_gates_passed": passed,
        "ci": {
            "kind": "local_verification", "run_id": run_id, "status": "completed",
            "conclusion": conclusion, "commit": commit,
            "jobs": [{
                "name": "full-protocol-verification", "run_id": run_id, "status": "completed",
                "conclusion": conclusion, "commit": commit,
                "tests_run": parsed["tests_run"], "failures": parsed["failures"],
                "errors": parsed["errors"], "skipped": parsed["skipped"],
                "returncode": completed.returncode, "duration_seconds": duration,
                "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
            }],
        },
        "environment": {
            "python": sys.version.split()[0], "executable": sys.executable,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "commit": commit, **{k: parsed[k] for k in ("tests_run", "failures", "errors", "skipped")}}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
