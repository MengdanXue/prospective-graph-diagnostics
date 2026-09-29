"""Explicit evidence-bound historical reconciliation; dry-run is the default.

No research is dispatched. Old bytes are backed up exclusively and remain an
exact journal prefix. A conservative later real clock sample is not an observed
exit time. Review/repair software overhead is not fabricated as old experiment
time. No lock is removed, clock invented, cap changed, or failure deleted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

from scripts import input_robustness_budget as budget

SCHEMA = "review-budget-closure/1"


class ClosureError(budget.BudgetIntegrityError):
    pass


def require(condition, message):
    if not condition:
        raise ClosureError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=budget._unique_object)


def resolve(base, value):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def pointer(value, key):
    require(isinstance(key, str) and (not key or key.startswith("/")), "invalid evidence JSON pointer")
    for part in key.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        try:
            value = value[int(part)] if isinstance(value, list) else value[part]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ClosureError(f"evidence field missing: {key}") from exc
    return value


def evidence_value(reference, base):
    path = resolve(base, reference["path"])
    require(path.is_file() and budget._hex(reference.get("sha256")) and file_hash(path) == reference["sha256"], f"missing/changed evidence: {path}")
    line = reference.get("line_number")
    if line is None:
        value = read_json(path)
    else:
        require(type(line) is int and line > 0, "invalid evidence line number")
        value = None
        with path.open("rb") as stream:
            for number, raw in enumerate(stream, 1):
                if number == line:
                    value = json.loads(raw, object_pairs_hook=budget._unique_object)
                    break
        require(value is not None, "evidence line missing")
    key = reference.get("json_pointer", reference.get("terminal_sample_json_pointer"))
    actual = pointer(value, key)
    require("expected_value" in reference and actual == reference["expected_value"], f"evidence actual fields differ: {path} {key}")
    return actual


def substantive(value):
    if isinstance(value, str):
        return len(value.strip()) >= 20
    if isinstance(value, dict):
        return any(substantive(item) for item in value.values())
    if isinstance(value, list):
        return any(substantive(item) for item in value)
    return False


def _current_processes():
    """Query the operating system on each invocation, fail closed if unavailable."""
    if os.name == "nt":
        result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                                 "$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine) | ConvertTo-Json -Compress"],
                                check=True, capture_output=True, text=True, timeout=30)
        rows = json.loads(result.stdout)
        return [{"pid": row["ProcessId"], "name": row["Name"], "command": row.get("CommandLine") or ""} for row in rows]
    result = subprocess.run(["ps", "-eo", "pid=,comm=,args="], check=True, capture_output=True, text=True, timeout=30)
    return [{"pid": int(parts[0]), "name": parts[1], "command": parts[2] if len(parts) > 2 else ""}
            for row in result.stdout.splitlines() if len(parts := row.strip().split(None, 2)) >= 2]


def no_live_workers():
    # The reconciler and test runner are not experimental workers. Check all
    # research entry points, including shell wrappers, irrespective of old PID.
    needles = ("input_robustness_formal", "input_robustness_worker", "mlp_budget_extension_entry",
               "mlp_budget_extension_worker", "input_robustness_analysis", "run_input_robustness",
               "supplement_guardian")
    processes = _current_processes()
    matches = [row for row in processes if row["pid"] != os.getpid()
               and any(needle in row["command"].lower() for needle in needles)]
    require(not matches, f"live research worker/controller detected: {matches}")
    require(all(row["command"] for row in processes if "python" in row["name"].lower() and row["pid"] != os.getpid()), "cannot inspect a Python process command line")
    return {"source": "current operating-system process inventory", "enumerated_process_count": len(processes), "matching_processes": matches}


def observation(attempt_id, first, last):
    wall, mono = last["wall_ns"] - first["wall_ns"], last["monotonic_ns"] - first["monotonic_ns"]
    delta = max(0, wall, mono)
    return {"attempt_id": attempt_id, "from_sample": first, "to_sample": last, "delta_charge_ns": delta,
            "statistics": {"sample_count": 1, "minimum_wall_delta_ns": min(0, wall), "minimum_monotonic_delta_ns": min(0, mono),
                           "maximum_poll_gap_ns": delta, "maximum_clock_divergence_ns": abs(wall - mono)}}


def _review_payload(entry, state, spec_sha, report_sha, plan_sha, decision=None):
    attempt = state["attempts"][entry["attempt_id"]]
    return {"attempt_id": entry["attempt_id"], "provenance": copy.deepcopy(state["phases"][attempt["phase_id"]]["provenance"]),
            "review_receipt": {"all_existing_records_validated": True, "external_interruption_cause_confirmed": True,
                               "no_unresolved_failure_artifacts": True, "review_record_sha256": report_sha,
                               "closure_spec_sha256": spec_sha, "review_plan_sha256": plan_sha,
                               "terminal_evidence": entry["evidence"], "review_decision": decision}}


def _events(entry, state, spec_sha, report_sha="0" * 64, plan_sha="0" * 64, decision=None):
    first, last, identity = entry["old_sample"], entry["terminal_bound_sample"], entry["attempt_id"]
    zero = observation(identity, last, last)
    return [("heartbeat", {"observations": [observation(identity, first, last)]}),
            ("attempt_closed", {"attempt_id": identity, "outcome": "external_interruption", "record_sha256": None, "reason": entry["reason"]}),
            ("finalization_charge", {"observations": [zero], "settle_external": False}),
            ("finalization_charge", {"observations": [zero], "settle_external": True}),
            ("external_interruption_reviewed", _review_payload(entry, state, spec_sha, report_sha, plan_sha, decision))]


def _control_payload(entry, spec_sha, report_sha):
    return {"charge_id": entry["charge_id"], "charge_ns": entry["charge_ns"], "source_spec_sha256": spec_sha,
            "review_record_sha256": report_sha, "evidence_windows": entry["evidence_windows"], "evidence": entry["evidence"]}


def prepare(spec_path):
    """Read-only validation/projection. The returned plan does not authorize apply."""
    spec_path = Path(spec_path).resolve()
    base, spec, spec_sha = spec_path.parent, read_json(spec_path), file_hash(spec_path)
    require(spec.get("schema_version") in ("1.0", SCHEMA), "unknown reviewed closure spec schema")
    closures, controls = spec.get("closure_entries", []), spec.get("control_entries", [])
    require(isinstance(closures, list) and isinstance(controls, list) and (closures or controls), "empty reconciliation")
    identities = [entry["attempt_id"] for entry in closures] + [entry["charge_id"] for entry in controls]
    require(len(set(identities)) == len(identities), "duplicate reconciliation identity")
    ledgers = {}
    for entry in closures + controls:
        path = resolve(base, entry["ledger_path"])
        journal = path / "budget_events.jsonl"
        for lock in (path / ".writer_lock", path / ".supplement_budget_owner.lock", path.parent / ".supplement_budget_owner.lock"):
            require(not lock.exists(), f"writer/owner lock exists; preserve for explicit investigation: {lock}")
        if path not in ledgers:
            require(journal.is_file() and file_hash(journal) == entry["original_journal_sha256"], "original journal file hash differs")
            state, head = budget._read_journal(path)
            require(head == entry["old_head"], "trusted old head differs")
            require(state["unfinished_close"] is None and state["pending_finalization_id"] is None, "unresolved prior finalization")
            anchor = evidence_value(entry["trusted_head_evidence"], base)
            require(anchor == head, "external old head evidence differs")
            ledgers[path] = {"state": state, "head": head, "original_journal_sha256": entry["original_journal_sha256"], "entries": [], "controls": []}
        item = ledgers[path]
        require(item["head"] == entry["old_head"] and item["original_journal_sha256"] == entry["original_journal_sha256"], "inconsistent old journal binding")
        require(isinstance(entry.get("evidence"), list) and entry["evidence"], "missing source evidence")
        evidence = [(reference, evidence_value(reference, base)) for reference in entry["evidence"]]
        for artifact in entry.get("preserve_artifacts", []):
            artifact_path = resolve(base, artifact["path"])
            require(artifact_path.is_file() and file_hash(artifact_path) == artifact["sha256"], "preserved artifact missing or changed")
        if entry in closures:
            attempt = item["state"]["attempts"].get(entry["attempt_id"])
            require(attempt is not None and attempt["outcome"] == "open", "attempt is not the bound open attempt")
            require(all(attempt[key] == entry[key] for key in ("unit_id", "batch_id")), "old attempt unit/batch identity differs")
            require(attempt["last_sample"] == entry["old_sample"] and attempt["charged_ns"] == entry["recorded_charge_ns"], "old attempt sample/charge differs")
            last = entry["terminal_bound_sample"]
            require(budget._valid_sample(last) and all(last[key] >= entry["old_sample"][key] for key in last), "terminal bound must be later real dual clocks without rollback")
            require(any(ref.get("role") == "terminal_sample" and value == last for ref, value in evidence), "terminal sample not bound to actual evidence fields")
            require(any(ref.get("role") == "process_termination" and substantive(value) for ref, value in evidence), "substantive historical process termination proof required")
            require(entry["bound_semantics"] == "conservative_terminal_upper_bound_not_measured_exit" and entry["outcome"] == "external_interruption", "historical closure cannot claim successful/measured exit")
            delta = observation(entry["attempt_id"], entry["old_sample"], last)["delta_charge_ns"]
            require(delta == entry["additional_charge_upper_bound_ns"] and delta + attempt["charged_ns"] == entry["total_attempt_charge_upper_bound_ns"], "conservative charge does not equal max real clock delta")
            require(budget._identifier(entry["reason"]), "external interruption requires cause")
            item["entries"].append(entry)
        else:
            require(all(isinstance(window.get("evidence_indexes"), list) and window["evidence_indexes"] and all(type(index) is int and 0 <= index < len(evidence) for index in window["evidence_indexes"]) for window in entry["evidence_windows"]), "control windows require indexed source evidence")
            require(any(substantive(value) for _, value in evidence), "substantive control source evidence required")
            item["controls"].append(entry)
    projected = []
    for path, item in ledgers.items():
        state = copy.deepcopy(item["state"])
        open_ids = {identity for identity, attempt in state["attempts"].items() if attempt["outcome"] == "open"}
        require(open_ids == {entry["attempt_id"] for entry in item["entries"]}, "unresolved open attempt omitted from reconciliation")
        for entry in item["entries"]:
            for kind, payload in _events(entry, state, spec_sha):
                budget._apply_event(state, kind, payload)
        for entry in item["controls"]:
            budget._apply_event(state, "reviewed_historical_control_charge", _control_payload(entry, spec_sha, "0" * 64))
        # Review may clear only the explicitly investigated interruptions.
        unresolved = [reason for reason in budget._stop_reasons(state) if "time_cap" not in reason and not reason.startswith("insufficient cumulative admission headroom")]
        require(not unresolved, f"unresolved prior failure/clock evidence: {unresolved}")
        projected.append({"ledger_path": str(path), "original_journal_sha256": item["original_journal_sha256"], "old_head": item["head"],
                          "before_usage_ns": item["state"]["usage_ns"], "after_usage_ns": state["usage_ns"],
                          "retained_caps_ns": state["caps_ns"], "stop_reasons": budget._stop_reasons(state),
                          "closed_attempts": [entry["attempt_id"] for entry in item["entries"]],
                          "historical_control_charges": [entry["charge_id"] for entry in item["controls"]]})
    no_live_workers()
    plan = {"schema_version": SCHEMA, "spec_path": str(spec_path), "spec_sha256": spec_sha,
            "requires_explicit_review_before_apply": True, "ledgers": projected,
            "bound_semantics": "conservative_terminal_upper_bound_not_measured_exit",
            "finalization_semantics": "zero-duration settlements reuse the exact bound sample; present software repair time is not historical experiment time"}
    return plan, ledgers


def _read_review(path, plan, ledgers):
    review_path = Path(path).resolve()
    review = read_json(review_path)
    plan_sha = budget.canonical_hash(plan)
    require(review.get("schema_version") == SCHEMA and review.get("spec_sha256") == plan["spec_sha256"] and review.get("plan_sha256") == plan_sha, "review does not bind exact source spec and dry-run plan")
    report = review.get("report", {})
    report_path = resolve(review_path.parent, report["path"])
    require(report_path.is_file() and budget._hex(report.get("sha256")) and file_hash(report_path) == report["sha256"], "human-readable review report missing or changed")
    text = report_path.read_text(encoding="utf-8")
    require(plan_sha in text and plan["spec_sha256"] in text and len(text.strip()) > 200, "review report must describe and hash-bind the reviewed plan")
    decisions = review.get("decisions", [])
    require(isinstance(decisions, list) and all(isinstance(decision, dict) for decision in decisions), "review needs substantive per-entry decisions")
    indexed = {decision.get("identity"): decision for decision in decisions}
    expected = {entry["attempt_id"] for item in ledgers.values() for entry in item["entries"]} | {entry["charge_id"] for item in ledgers.values() for entry in item["controls"]}
    require(len(indexed) == len(decisions) and set(indexed) == expected, "review omitted or invented reconciliation identities")
    for identity, decision in indexed.items():
        require(identity in text and substantive(decision.get("rationale")), "review needs documented identity and substantive rationale")
        require(decision.get("records_status") == "validated" and decision.get("failure_artifacts_status") == "preserved_and_resolved", "unresolved record/failure review")
        require(decision.get("disposition") == ("external_interruption" if any(identity == entry["attempt_id"] for item in ledgers.values() for entry in item["entries"]) else "historical_control_charge"), "incorrect reviewed disposition")
    return report["sha256"], plan_sha, indexed


def reconcile(spec_path, *, apply=False, review_path=None):
    plan, ledgers = prepare(spec_path)
    result = {"mode": "dry_run", "plan": plan, "plan_sha256": budget.canonical_hash(plan)}
    if not apply:
        return result
    require(review_path is not None, "--apply requires the explicit reviewed report receipt")
    report_sha, plan_sha, decisions = _read_review(review_path, plan, ledgers)
    common = Path(os.path.commonpath([str(path.parent) for path in ledgers]))
    lock = common / ".review_budget_closure.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ClosureError("closure owner lock exists; never remove it automatically") from exc
    backups, final = [], []
    try:
        os.write(lock_fd, f"pid={os.getpid()}\nplan_sha256={plan_sha}\n".encode())
        os.fsync(lock_fd)
        # Revalidate evidence and heads after taking ownership, before any append.
        fresh, ledgers = prepare(spec_path)
        require(fresh == plan, "reviewed plan changed before apply")
        for path, item in ledgers.items():
            journal = path / "budget_events.jsonl"
            backup = path / f"budget_events.pre-reviewed-closure-{item['original_journal_sha256']}.jsonl"
            with backup.open("xb") as target, journal.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
            require(file_hash(backup) == item["original_journal_sha256"] == file_hash(journal), "immutable original backup/hash mismatch")
            backups.append({"path": str(backup), "sha256": file_hash(backup)})
        for path, item in ledgers.items():
            ledger = budget.BudgetLedger._instance(path, item["state"], item["head"], None, None)
            ledger._signature = ledger._stat_signature()
            for entry in item["entries"]:
                for kind, payload in _events(entry, ledger._state, plan["spec_sha256"], report_sha, plan_sha, decisions[entry["attempt_id"]]):
                    ledger._append(kind, payload)
            for entry in item["controls"]:
                ledger._append("reviewed_historical_control_charge", _control_payload(entry, plan["spec_sha256"], report_sha))
            with (path / "budget_events.jsonl").open("rb") as stream:
                prefix = hashlib.sha256()
                remaining = item["head"]["journal_bytes"]
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    require(block, "old journal prefix truncated")
                    prefix.update(block)
                    remaining -= len(block)
            require(prefix.hexdigest() == item["original_journal_sha256"], "old journal prefix changed")
            state, head = budget._read_journal(path)
            require(head == ledger.head and state == ledger._state, "appended hash chain replay differs")
            final.append({"ledger_path": str(path), "head": head, "journal_sha256": file_hash(path / "budget_events.jsonl"),
                          "usage_ns": state["usage_ns"], "stop_reasons": budget._stop_reasons(state)})
    finally:
        os.close(lock_fd)
        lock.unlink()
    return {**result, "mode": "applied", "backups": backups, "final_ledgers": final, "review_report_sha256": report_sha}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--apply", action="store_true", help="append only after the separate report receipt explicitly reviews the dry-run plan")
    parser.add_argument("--review", help="hash-bound explicit review receipt; required for --apply")
    args = parser.parse_args(argv)
    print(json.dumps(reconcile(args.spec, apply=args.apply, review_path=args.review), indent=2))


if __name__ == "__main__":
    main()
