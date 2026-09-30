"""Hash-bound human review of a terminal ``failed`` budget attempt.

See docs/protocol_amendment_mlp24_environmental_failure_retry_v1.md. A review
never edits the original journal, outcome or charge. It is an external record
that a cumulative handoff may use to waive one *current* blocker while the
historical stop reason stays reported. Software drafts; only a human signs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from scripts import input_robustness_budget as budget

PLAN_SCHEMA = "terminal-failure-review-plan/1"
REVIEW_SCHEMA = "terminal-failure-review/1"
DRAFT_STATUS = "DRAFT_PENDING_HUMAN_SIGNOFF_DO_NOT_CONSUME"
SIGNED_STATUS = "SIGNED"
APPROVE = "APPROVE_ENVIRONMENTAL_FAILURE_RETRY"
REJECT = "REJECT_ENVIRONMENTAL_FAILURE_RETRY"
ENVIRONMENTAL = "ENVIRONMENTAL_EXECUTION_FAILURE"
CLASSIFICATIONS = (ENVIRONMENTAL, "RESEARCH_EXECUTION_FAILURE", "UNKNOWN")
ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = "docs/protocol_amendment_mlp24_environmental_failure_retry_v1.md"
MAX_REVIEW_BYTES = 4 * 1024 * 1024
FINDING_FIELDS = {"research_computation_started", "checkpoints", "records", "test_evaluations",
                  "test_once_consumed", "blocking_stop_reasons"}
DISPOSITION_FIELDS = {"retry_allowed", "reason", "max_retries", "must_be_new_segment",
                      "new_attempt_id_required", "retry_scope", "required_environment"}
SCOPE_FIELDS = {"unit_id", "batch_id", "activity_id", "research_identity"}
IDENTITY_FIELDS = {"dataset", "condition", "model", "seed", "trials"}
STATEMENT = ("This review does not change the original failed outcome, does not remove the original charge "
             "and does not modify the original segment. It only authorizes one new retry attempt in a future "
             "new inherited segment.")


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(path) -> str:
    """Line-ending independent digest for tracked text read on either platform."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def amendment_sha256() -> str:
    return text_sha256(ROOT / AMENDMENT)


def _read_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_REVIEW_BYTES:
        raise ValueError(f"review artifact too large: {path}")
    with path.open("rb") as stream:
        return json.load(stream, object_pairs_hook=budget._unique_object)


def close_event(ledger_path, attempt_id):
    """Sequence and hash of the unique close event of ``attempt_id``."""
    found = None
    with (Path(ledger_path) / "budget_events.jsonl").open("rb") as stream:
        for line in stream:
            event = json.loads(line)
            if event["kind"] == "attempt_closed" and event["payload"]["attempt_id"] == attempt_id:
                if found is not None:
                    raise ValueError("duplicate close event")
                found = {"sequence": event["sequence"], "sha256": event["sha256"]}
    return found


def evidence_manifest(paths):
    files = [{"name": name, "path": str(Path(path).resolve()), "sha256": file_sha256(path)}
             for name, path in sorted(paths.items())]
    return {"files": files, "failure_evidence_sha256": budget.canonical_hash(
        [{"name": row["name"], "sha256": row["sha256"]} for row in files])}


def unit_identity(research_identity):
    model, seed = research_identity["model"], int(research_identity["seed"])
    return (f"unit_{research_identity['dataset']}_{research_identity['condition']}_{model}_{seed:03d}",
            f"pair_{research_identity['dataset']}_seed_{seed:03d}_{model}")


def build_review_plan(*, ledger_path, terminal_receipt_path, attempt_id, authority, research_identity,
                      evidence_paths, classification, findings, disposition):
    """Read-only: reconstruct the original identity from the preserved journal."""
    from scripts.supplement_budget_handoff import stream_snapshot
    ledger_path = Path(ledger_path).resolve()
    receipt = _read_json(terminal_receipt_path)
    snapshot = stream_snapshot(ledger_path, authority=authority, expected_head=receipt["head"])
    attempt = snapshot["attempts"][attempt_id]
    original = {"ledger_path": str(ledger_path), "journal_sha256": snapshot["journal_sha256"],
                "head": {"event_count": snapshot["head"]["event_count"], "sha256": snapshot["head"]["sha256"]},
                "terminal_receipt_sha256": file_sha256(terminal_receipt_path),
                "attempt_id": attempt_id, "close_event": close_event(ledger_path, attempt_id),
                "outcome": attempt["outcome"], "charged_ns": attempt["charged_ns"],
                "unit_id": attempt["unit_id"], "batch_id": attempt["batch_id"],
                "activity_id": attempt["activity_id"], "phase_id": attempt["phase_id"],
                "clock_stop_reasons": list(attempt["clock_stop_reasons"])}
    return {"schema_version": PLAN_SCHEMA,
            "amendment": {"path": AMENDMENT, "sha256": amendment_sha256()},
            "original": original, "research_identity": copy.deepcopy(research_identity),
            "failure_evidence": evidence_manifest(evidence_paths), "classification": classification,
            "findings": copy.deepcopy(findings), "disposition": copy.deepcopy(disposition)}


def draft_review_record(plan_path, *, source_commit):
    """Unsigned record. It never clears a blocker until a human signs it."""
    plan = _read_json(plan_path)
    plan_sha = file_sha256(plan_path)
    disposition = plan["disposition"]
    return {"schema_version": REVIEW_SCHEMA, "status": DRAFT_STATUS, "decision": None,
            "allowed_decisions": [APPROVE, REJECT], "amendment_sha256": plan["amendment"]["sha256"],
            "review_plan": {"path": str(Path(plan_path).resolve()), "sha256": plan_sha},
            "review_plan_sha256": plan_sha,
            "failure_evidence_sha256": plan["failure_evidence"]["failure_evidence_sha256"],
            "original": {key: plan["original"][key] for key in
                         ("ledger_path", "attempt_id", "outcome", "charged_ns", "terminal_receipt_sha256",
                          "journal_sha256", "head")},
            "classification": plan["classification"], "findings": plan["findings"],
            "retry_scope": disposition["retry_scope"], "max_retries": disposition["max_retries"],
            "required_environment": disposition["required_environment"],
            "source_commit": source_commit, "statement": STATEMENT,
            "signoff": {"reviewer": None, "signed_at_utc": None}}


def plan_errors(plan, *, snapshot, ledger_path, terminal_receipt_sha256):
    """Every mismatch with the preserved original; empty means consistent."""
    errors = []
    original = plan.get("original", {})
    attempt_id = original.get("attempt_id")
    attempt = snapshot["attempts"].get(attempt_id)
    if plan.get("schema_version") != PLAN_SCHEMA:
        errors.append("plan schema")
    if plan.get("amendment", {}).get("sha256") != amendment_sha256():
        errors.append("plan amendment hash")
    if attempt is None:
        return errors + ["attempt absent from the source ledger"]
    expected = {"ledger_path": str(Path(ledger_path).resolve()), "journal_sha256": snapshot["journal_sha256"],
                "head": {"event_count": snapshot["head"]["event_count"], "sha256": snapshot["head"]["sha256"]},
                "terminal_receipt_sha256": terminal_receipt_sha256, "outcome": "failed",
                "close_event": close_event(ledger_path, attempt_id), "charged_ns": attempt["charged_ns"],
                "unit_id": attempt["unit_id"], "batch_id": attempt["batch_id"],
                "activity_id": attempt["activity_id"], "phase_id": attempt["phase_id"],
                "clock_stop_reasons": list(attempt["clock_stop_reasons"])}
    errors += [f"original {key} differs" for key, value in expected.items() if original.get(key) != value]
    if attempt["outcome"] != "failed" or attempt["reviewed"]:
        errors.append("attempt is not an unreviewed failed outcome")
    identity = plan.get("research_identity", {})
    if set(identity) != IDENTITY_FIELDS or unit_identity(identity) != (attempt["unit_id"], attempt["batch_id"]):
        errors.append("research identity differs from the attempt unit and paired batch")
    manifest = plan.get("failure_evidence", {})
    files = manifest.get("files", [])
    if not files or budget.canonical_hash([{"name": r.get("name"), "sha256": r.get("sha256")} for r in files]) \
            != manifest.get("failure_evidence_sha256"):
        errors.append("failure evidence manifest hash")
    for row in files:
        if not Path(row["path"]).is_file() or file_sha256(row["path"]) != row["sha256"]:
            errors.append(f"failure evidence changed: {row.get('name')}")
    if plan.get("classification") not in CLASSIFICATIONS:
        errors.append("unknown classification")
    if set(plan.get("findings", {})) != FINDING_FIELDS or set(plan.get("disposition", {})) != DISPOSITION_FIELDS:
        errors.append("findings or disposition fields")
    return errors


def retry_errors(plan, *, attempt, retried_units):
    """Conditions under which ``retry_allowed=true`` is admissible."""
    findings, disposition = plan["findings"], plan["disposition"]
    scope = disposition.get("retry_scope", {})
    errors = []
    if plan["classification"] != ENVIRONMENTAL:
        errors.append("only an environmental execution failure may be retried")
    if (findings["research_computation_started"] is not False or findings["checkpoints"] != 0
            or findings["records"] != 0 or findings["test_evaluations"] != 0
            or findings["test_once_consumed"] is not False or findings["blocking_stop_reasons"] != []):
        errors.append("findings show research computation, outputs, test access or a blocking stop")
    if attempt["clock_stop_reasons"]:
        errors.append("clock or monitor stop reason on the failed attempt")
    if (disposition["max_retries"] != 1 or disposition["must_be_new_segment"] is not True
            or disposition["new_attempt_id_required"] is not True):
        errors.append("retry must be one new-segment attempt with a new identity")
    if set(scope) != SCOPE_FIELDS or scope != {"unit_id": attempt["unit_id"], "batch_id": attempt["batch_id"],
                                               "activity_id": attempt["activity_id"],
                                               "research_identity": plan["research_identity"]}:
        errors.append("retry scope differs from the original unit")
    if not isinstance(disposition["required_environment"], dict) or not disposition["required_environment"]:
        errors.append("retry requires a bound environment")
    if attempt["unit_id"] in retried_units:
        errors.append("this model unit already used its one retry")
    return errors


def evaluate_review(reference, *, snapshot, ledger_path, terminal_receipt_sha256, retried_units=frozenset()):
    """Decide whether a referenced record is valid and whether it allows a retry."""
    result = {"attempt_id": None, "review_record_sha256": reference.get("sha256"), "review_plan_sha256": None,
              "valid": False, "retry_allowed": False, "errors": [], "authorization": None}
    try:
        if not budget._hex(reference.get("sha256")) or file_sha256(reference["path"]) != reference["sha256"]:
            result["errors"].append("review record hash differs")
            return result
        record = _read_json(reference["path"])
        plan_ref = record.get("review_plan", {})
        result["attempt_id"] = record.get("original", {}).get("attempt_id")
        if file_sha256(plan_ref["path"]) != plan_ref.get("sha256") or record.get("review_plan_sha256") != plan_ref.get("sha256"):
            result["errors"].append("review plan hash differs")
            return result
        plan = _read_json(plan_ref["path"])
        result["review_plan_sha256"] = plan_ref["sha256"]
        errors = plan_errors(plan, snapshot=snapshot, ledger_path=ledger_path,
                             terminal_receipt_sha256=terminal_receipt_sha256)
        disposition = plan.get("disposition", {})
        signoff = record.get("signoff") or {}
        if record.get("schema_version") != REVIEW_SCHEMA:
            errors.append("review schema")
        if record.get("status") != SIGNED_STATUS:
            errors.append("review is not signed")
        if record.get("decision") not in (APPROVE, REJECT):
            errors.append("review decision")
        if not (isinstance(signoff.get("reviewer"), str) and signoff["reviewer"].strip()
                and isinstance(signoff.get("signed_at_utc"), str) and signoff["signed_at_utc"].strip()):
            errors.append("reviewer sign-off")
        if record.get("amendment_sha256") != amendment_sha256():
            errors.append("record amendment hash")
        bound = {"failure_evidence_sha256": plan.get("failure_evidence", {}).get("failure_evidence_sha256"),
                 "classification": plan.get("classification"), "findings": plan.get("findings"),
                 "retry_scope": disposition.get("retry_scope"), "max_retries": disposition.get("max_retries"),
                 "required_environment": disposition.get("required_environment")}
        errors += [f"record {key} differs from plan" for key, value in bound.items() if record.get(key) != value]
        if record.get("original", {}).get("attempt_id") != plan.get("original", {}).get("attempt_id"):
            errors.append("record attempt differs from plan")
        original_id = plan.get("original", {}).get("attempt_id")
        attempt = snapshot["attempts"].get(original_id)
        if any(row.get("consumed_by") == original_id for row in snapshot.get("retry_authorizations", {}).values()):
            errors.append("the failed attempt was itself an authorized retry")
        wants_retry = record.get("decision") == APPROVE or disposition.get("retry_allowed") is True
        if not errors and wants_retry:
            if record["decision"] != APPROVE or disposition["retry_allowed"] is not True:
                errors.append("approval and disposition disagree")
            else:
                errors += retry_errors(plan, attempt=attempt, retried_units=retried_units)
        result["errors"] = errors
        result["valid"] = not errors
        result["retry_allowed"] = result["valid"] and record["decision"] == APPROVE
        if result["retry_allowed"]:
            result["authorization"] = {
                "review_record_sha256": reference["sha256"], "review_plan_sha256": plan_ref["sha256"],
                "original": {key: plan["original"][key] for key in
                             ("ledger_path", "journal_sha256", "head", "terminal_receipt_sha256", "attempt_id",
                              "close_event", "outcome", "charged_ns")},
                "unit_id": attempt["unit_id"], "batch_id": attempt["batch_id"], "activity_id": attempt["activity_id"],
                "research_identity": copy.deepcopy(plan["research_identity"]),
                "required_environment": copy.deepcopy(disposition["required_environment"])}
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        result["errors"].append(f"unreadable or malformed review: {exc!r}")
        result["valid"] = result["retry_allowed"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description="Draft (never sign) a terminal failure review.")
    parser.add_argument("--plan-input", type=Path, required=True,
                        help="JSON with ledger_path, terminal_receipt_path, attempt_id, authority, research_identity, "
                             "evidence_paths, classification, findings and disposition")
    parser.add_argument("--plan-output", type=Path, required=True)
    parser.add_argument("--record-output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    request = _read_json(args.plan_input)
    plan = build_review_plan(**request)
    with args.plan_output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
    record = draft_review_record(args.plan_output, source_commit=args.source_commit)
    with args.record_output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({"review_plan_sha256": record["review_plan_sha256"], "status": record["status"],
                      "record": str(args.record_output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
