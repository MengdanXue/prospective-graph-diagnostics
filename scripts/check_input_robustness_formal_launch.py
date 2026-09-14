"""Independent pre-launch checks for the guarded formal run.

This command produces a readiness report; it never changes the configuration
and never starts training.  The final authorization remains a separate human
decision recorded after all technical checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_formal_launch(*, root: Path, acceptance_record: Path, config_path: Path,
                        authorization_path: Path, test_receipt: Path | None = None,
                        current_commit: str | None = None) -> dict[str, Any]:
    root, acceptance_record, config_path, authorization_path = map(Path, (root, acceptance_record, config_path, authorization_path))
    acceptance = _read(acceptance_record)
    config = _read(config_path)
    authorization = _read(authorization_path)
    current = current_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    clean = not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    checks = {
        "prior_runtime_resource_acceptance_passed": acceptance.get("status") == "passed",
        "execution_sha_is_bound": acceptance.get("execution_commit") == "aef8b87a6eb7bf6ea4c3fd1482b0a6882dc176b0",
        "shared_ledger_has_no_formal_or_control_seconds": acceptance.get("shared_ledger", {}).get("charged_seconds", {}).get("formal") == 0.0 and acceptance.get("shared_ledger", {}).get("charged_seconds", {}).get("control") == 0.0,
        "no_research_scores_created": acceptance.get("research_results_created") == {"validation_evaluations": 0, "test_evaluations": 0, "formal_records": 0},
        "source_clean": clean,
        "formal_training_switch_still_off": config.get("formal_training_enabled") is False,
        "formal_launch_authorization_still_off": authorization.get("formal_launch_authorized") is False and config.get("formal_launch_authorized", False) is False,
    }
    ci = None
    if test_receipt is not None:
        ci = _read(test_receipt)
        checks.update({
            "formal_layer_test_receipt_passed": ci.get("status") == "passed" and ci.get("failed") == 0,
            "formal_layer_test_receipt_matches_current_commit": ci.get("commit") == current,
            "formal_layer_ci_receipt_passed": ci.get("ci", {}).get("conclusion") == "success" and ci.get("ci", {}).get("commit") == current,
        })
    else:
        checks.update({"formal_layer_test_receipt_passed": False, "formal_layer_test_receipt_matches_current_commit": False,
                       "formal_layer_ci_receipt_passed": False})
    technical = all(checks.values())
    return {
        "schema_version": "1.0", "status": "ready_for_formal_authorization" if technical else "blocked",
        "current_source_commit": current, "resource_execution_commit": acceptance.get("execution_commit"),
        "acceptance_record": str(acceptance_record), "acceptance_record_sha256": _sha(acceptance_record),
        "test_receipt": str(test_receipt) if test_receipt else None,
        "checks": checks, "technical_gates_passed": technical,
        "formal_training_enabled": config.get("formal_training_enabled"),
        "formal_launch_authorized": authorization.get("formal_launch_authorized"),
        "next_action": "request_one_formal_start_authorization" if technical else "resolve_failed_checks_before_requesting_authorization",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--acceptance-record", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--test-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_formal_launch(root=args.root, acceptance_record=args.acceptance_record,
                                 config_path=args.config, authorization_path=args.authorization,
                                 test_receipt=args.test_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"checks"}}, sort_keys=True))
    return 0 if result["technical_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

