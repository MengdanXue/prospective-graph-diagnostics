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

from scripts.input_robustness_formal_records import digest


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evidence(path_value: Any, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(path_value, (str, Path)) or not str(path_value):
        raise ValueError(f"{label} path is missing")
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"{label} evidence is missing: {path}")
    value = _read(path)
    if value.get("status") not in ("passed", "success"):
        raise ValueError(f"{label} evidence is not a successful independent record")
    return path, value


def check_formal_launch(*, root: Path, acceptance_record: Path, config_path: Path,
                        authorization_path: Path, test_receipt: Path | None = None,
                        current_commit: str | None = None,
                        ledger_path: Path | None = None,
                        reliability_verification: Path | None = None,
                        resource_verification: Path | None = None,
                        independent_audit: Path | None = None,
                        formal_e2e_receipt: Path | None = None) -> dict[str, Any]:
    root, acceptance_record, config_path, authorization_path = map(Path, (root, acceptance_record, config_path, authorization_path))
    acceptance = _read(acceptance_record)
    config = _read(config_path)
    authorization = _read(authorization_path)
    current = current_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    clean = not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    try:
        upstream = subprocess.check_output(["git", "rev-parse", "@{upstream}"], cwd=root, text=True).strip()
        remote_sync = upstream == current
    except (OSError, subprocess.CalledProcessError):
        upstream, remote_sync = None, False
    checks = {
        "prior_runtime_resource_acceptance_passed": acceptance.get("status") == "passed",
        "execution_sha_is_bound": acceptance.get("execution_commit") == "aef8b87a6eb7bf6ea4c3fd1482b0a6882dc176b0",
        "shared_ledger_has_no_formal_or_control_seconds": acceptance.get("shared_ledger", {}).get("charged_seconds", {}).get("formal") == 0.0 and acceptance.get("shared_ledger", {}).get("charged_seconds", {}).get("control") == 0.0,
        "no_research_scores_created": acceptance.get("research_results_created") == {"validation_evaluations": 0, "test_evaluations": 0, "formal_records": 0},
        "source_clean": clean,
        "remote_sync_verified": remote_sync,
        "formal_training_switch_still_off": config.get("formal_training_enabled") is False,
        "formal_launch_authorization_still_off": authorization.get("formal_launch_authorized") is False and config.get("formal_launch_authorized", False) is False,
    }
    evidence_errors: list[str] = []
    evidence: dict[str, Any] = {}
    for label, explicit, fallback in (
        ("reliability_verification", reliability_verification, acceptance.get("reliability", {}).get("verification_record")),
        ("resource_verification", resource_verification, acceptance.get("resource", {}).get("verification_record")),
        ("independent_audit", independent_audit, acceptance.get("resource", {}).get("independent_audit")),
    ):
        try:
            path, payload = _evidence(explicit or fallback, label)
            evidence[label] = {"path": str(path), "sha256": _sha(path), "status": payload.get("status")}
            expected_hash = None
            if label == "reliability_verification":
                expected_hash = acceptance.get("reliability", {}).get("verification_sha256")
            elif label == "resource_verification":
                expected_hash = acceptance.get("resource", {}).get("verification_sha256")
            elif label == "independent_audit":
                expected_hash = acceptance.get("resource", {}).get("independent_audit_sha256")
            if expected_hash and evidence[label]["sha256"] != expected_hash:
                raise ValueError(f"{label} SHA differs from the acceptance record")
            if payload.get("synthetic") is True or payload.get("execution_mode") == "synthetic_rehearsal":
                raise ValueError(f"{label} is synthetic rehearsal evidence")
            if label == "reliability_verification" and not (
                len(payload.get("cases", [])) == 5 and all(case.get("passed") is True for case in payload["cases"])
            ):
                raise ValueError("reliability evidence does not contain all five passed cases")
            if label == "resource_verification" and not (
                payload.get("worker_count") == 44
                and (payload.get("probe_files") == 1232 or payload.get("bitwise_matching_paired_cells") == 616)
                and payload.get("failures") == 0 and payload.get("formal_records") == 0
            ):
                raise ValueError("resource evidence does not contain the complete fixed probe scope")
            if label == "independent_audit" and not (
                payload.get("all_job_success") is True and payload.get("summary_failures") == 0
                and payload.get("formal_training_enabled") is False
            ):
                raise ValueError("independent resource audit is incomplete")
            if payload.get("source_commit") and payload.get("source_commit") != acceptance.get("execution_commit"):
                raise ValueError(f"{label} source commit differs from the accepted execution SHA")
        except Exception as exc:
            evidence_errors.append(str(exc))
    checks["independent_runtime_and_resource_evidence"] = not evidence_errors and len(evidence) == 3
    checks["independent_evidence_is_not_summary_only"] = all(
        entry.get("sha256") and entry.get("status") in ("passed", "success") for entry in evidence.values()
    ) and not acceptance.get("synthetic_only", False)

    ledger_result = None
    if ledger_path is None:
        ledger_path = Path(acceptance.get("shared_ledger", {}).get("journal", ""))
    if ledger_path.name == "budget_events.jsonl":
        ledger_path = ledger_path.parent
    try:
        if not ledger_path.is_dir() or not (ledger_path / "budget_events.jsonl").is_file():
            raise ValueError("shared ledger journal is missing")
        from scripts.input_robustness_budget import inspect_ledger, canonical_hash
        authority = {"run_id": config["run_id"],
                     "authorization_sha256": _sha(authorization_path),
                     "proposal_sha256": authorization["approved_proposal"]["sha256"],
                     "scope_sha256": canonical_hash(authorization["scope"])}
        trusted_head = acceptance.get("resource", {}).get("trusted_head")
        if not isinstance(trusted_head, dict):
            raise ValueError("trusted ledger head is missing from acceptance")
        ledger_result = inspect_ledger(ledger_path, authority=authority, expected_head=trusted_head)
        if ledger_result.get("must_stop") or ledger_result.get("open_attempts"):
            raise ValueError("trusted ledger is stopped or has open attempts")
        shared = acceptance.get("shared_ledger", {})
        if ledger_result.get("head", {}).get("event_count") != shared.get("final_event_count"):
            raise ValueError("ledger event count differs from the accepted final head")
        if ledger_result.get("charged_seconds", {}).get("formal") != 0.0 or ledger_result.get("charged_seconds", {}).get("control") != 0.0:
            raise ValueError("formal/control budget was charged before authorization")
    except Exception as exc:
        evidence_errors.append(str(exc))
    checks["trusted_shared_ledger_end_verified"] = ledger_result is not None and not evidence_errors
    resource_phase_id = f"resource_acceptance_only_{acceptance.get('execution_commit')}"
    ledger_phase = (ledger_result or {}).get("phases", {}).get(resource_phase_id, {})
    ledger_provenance = ledger_phase.get("provenance", {}) if isinstance(ledger_phase, dict) else {}
    checks["config_digest_and_bound_source_verified"] = (
        ledger_provenance.get("config_sha256") == digest(config)
        and config.get("bound_input_source", {}).get("sha256")
            == ledger_provenance.get("data_binding_sha256")
        and acceptance.get("shared_ledger", {}).get("journal_sha256") == _sha(ledger_path / "budget_events.jsonl")
        and acceptance.get("resource", {}).get("trusted_head", {}).get("event_count") == acceptance.get("shared_ledger", {}).get("final_event_count")
        and acceptance.get("shared_ledger", {}).get("formal_launch_authorized") is False
    )
    if formal_e2e_receipt is not None:
        try:
            path, payload = _evidence(formal_e2e_receipt, "formal_e2e_receipt")
            checks["actual_formal_e2e_rehearsal_verified"] = payload.get("actual_training") is True and payload.get("checkpoint_reload_verified") is True
            evidence["formal_e2e_receipt"] = {"path": str(path), "sha256": _sha(path)}
        except Exception as exc:
            evidence_errors.append(str(exc))
            checks["actual_formal_e2e_rehearsal_verified"] = False
    else:
        checks["actual_formal_e2e_rehearsal_verified"] = False
    ci = None
    if test_receipt is not None:
        ci = _read(test_receipt)
        ci_jobs = ci.get("ci", {}).get("jobs", [])
        checks.update({
            "formal_layer_test_receipt_passed": ci.get("status") == "passed" and ci.get("failed") == 0,
            "formal_layer_test_receipt_matches_current_commit": ci.get("commit") == current,
            "formal_layer_ci_receipt_passed": ci.get("ci", {}).get("conclusion") == "success" and ci.get("ci", {}).get("commit") == current
                and ci.get("ci", {}).get("status") == "completed"
                and len(ci_jobs) == 3
                and all(job.get("commit", current) == current and job.get("status") == "completed" and job.get("conclusion") == "success" for job in ci_jobs),
        })
    else:
        checks.update({"formal_layer_test_receipt_passed": False, "formal_layer_test_receipt_matches_current_commit": False,
                       "formal_layer_ci_receipt_passed": False})
    technical = all(checks.values()) and not evidence_errors
    return {
        "schema_version": "1.0", "status": "ready_for_formal_authorization" if technical else "blocked",
        "current_source_commit": current, "resource_execution_commit": acceptance.get("execution_commit"),
        "upstream_source_commit": upstream,
        "acceptance_record": str(acceptance_record), "acceptance_record_sha256": _sha(acceptance_record),
        "test_receipt": str(test_receipt) if test_receipt else None,
        "checks": checks, "technical_gates_passed": technical,
        "evidence": evidence, "evidence_errors": evidence_errors,
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
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--reliability-verification", type=Path)
    parser.add_argument("--resource-verification", type=Path)
    parser.add_argument("--independent-audit", type=Path)
    parser.add_argument("--formal-e2e-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_formal_launch(root=args.root, acceptance_record=args.acceptance_record,
                                 config_path=args.config, authorization_path=args.authorization,
                                 test_receipt=args.test_receipt, ledger_path=args.ledger,
                                 reliability_verification=args.reliability_verification,
                                 resource_verification=args.resource_verification,
                                 independent_audit=args.independent_audit,
                                 formal_e2e_receipt=args.formal_e2e_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"checks"}}, sort_keys=True))
    return 0 if result["technical_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
