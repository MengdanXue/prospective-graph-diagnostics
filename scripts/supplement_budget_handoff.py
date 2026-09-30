"""Read-only handoff of the existing budget journals; never repair old attempts.

An independently preserved terminal head is required for every old journal.
Earlier receipts are chain anchors, not separate expenditure.  An open attempt
has unknown terminal time: recorded use and a through-inspection conservative
bound are reported separately.  Neither a successful model record nor a process
that is absent now closes that attempt.  Preparation cannot authorize training.

The optional execution adapter writes only a new *incremental segment*, under a
single shared owner lock, and enforces inherited totals on every admission and
poll. It never grants a fresh budget. The original ledger remains the accounting
implementation; its event replay, clocks, and cap semantics are reused here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import time

from scripts import input_robustness_budget as budget
from scripts import terminal_failure_review as failure_review

# Version 2 adds hash-bound terminal failure reviews. Every historical stop
# reason is still reported; a valid signed review only changes the current set.
SCHEMA = "supplement-budget-handoff/2"
SCHEMAS = ("supplement-budget-handoff/1", SCHEMA)
MAX_EVENT_BYTES = 2 * 1024 * 1024
MAX_RECEIPT_BYTES = 4 * 1024 * 1024
SCALARS = ("total", "resource", "control", "formal")
INCREMENT_INDEX = ".supplement_budget_segments.jsonl"


class HandoffError(budget.BudgetAdmissionError):
    pass


def _require(condition, message):
    if not condition:
        raise HandoffError(message)


def _signature(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def _file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _read_small(path):
    path = Path(path)
    _require(path.stat().st_size <= MAX_RECEIPT_BYTES,
             f"receipt too large for bounded JSON reading; extract a reviewed compact receipt first: {path}")
    with path.open("rb") as stream:
        return json.load(stream, object_pairs_hook=budget._unique_object)


def _resolve(base, path):
    value = Path(path)
    return (value if value.is_absolute() else Path(base) / value).resolve()


def _usage_seconds(usage):
    return {**{key: usage[key] / budget.NS for key in SCALARS},
            **{key: {name: amount / budget.NS for name, amount in usage[key].items()}
               for key in ("units", "batches")}}


def _sum_usage(usages):
    total = budget._empty_usage()
    for usage in usages:
        for key in SCALARS:
            total[key] += usage[key]
        for key in ("units", "batches"):
            for name, value in usage[key].items():
                total[key][name] = total[key].get(name, 0) + value
    return total


def _cap_reasons(usage):
    return budget._cap_reasons({"usage_ns": usage,
                               "caps_ns": {k: v * budget.NS for k, v in budget.CAPS_SECONDS.items()}})


def _verify_head(state, actual, expected):
    _require(isinstance(expected, dict) and all(expected.get(k) == v for k, v in actual.items()),
             "externally retained head does not match the journal prefix")
    pending = expected.get("unsettled_finalization")
    _require((pending is not None) == (state["pending_finalization_id"] is not None),
             "trusted head omitted or invented finalization charge")
    checked = copy.deepcopy(state)
    if pending is not None:
        _require(pending["attempt_id"] == state["pending_finalization_id"],
                 "trusted finalization has the wrong attempt")
        budget._apply_observation(checked, pending, allow_closed=True)
    return checked


def stream_snapshot(path, *, authority, expected_head=None, anchors=(), now_sample=None):
    """Replay a bounded initial file prefix, in O(event size + ledger state) RAM.

    A live append is reported as a blocker; it is never silently presented as a
    terminal snapshot. No .writer_lock is created and no journal is modified.
    ``anchors`` are historical external heads and are never added to usage.
    """
    path = Path(path).resolve()
    journal = path / "budget_events.jsonl"
    before = _signature(journal)
    anchor_map = {}
    for anchor in anchors:
        count = anchor.get("event_count")
        _require(type(count) is int and count > 0 and count not in anchor_map,
                 "duplicate/invalid historical anchor")
        anchor_map[count] = anchor
    state, previous, count, consumed = {}, None, 0, 0
    digest = hashlib.sha256()
    verified_anchors = []
    with journal.open("rb") as stream:
        _require(_signature(journal) == before, "journal changed before inspection")
        while consumed < before[2]:
            line = stream.readline(min(MAX_EVENT_BYTES, before[2] - consumed) + 1)
            _require(line and len(line) <= MAX_EVENT_BYTES and consumed + len(line) <= before[2]
                     and line.endswith(b"\n"), "torn or oversized budget event; preserve for review")
            consumed += len(line)
            digest.update(line)
            event = json.loads(line, object_pairs_hook=budget._unique_object)
            _require(set(event) == {"schema_version", "sequence", "previous_sha256", "kind", "payload", "sha256"},
                     "budget event fields changed")
            _require(event["schema_version"] == "1.0" and type(event["sequence"]) is int
                     and event["sequence"] == count and event["previous_sha256"] == previous,
                     "budget event sequence/hash chain gap")
            sha = event.pop("sha256")
            _require(budget._hex(sha) and budget.canonical_hash(event) == sha, "budget event hash mismatch")
            budget._apply_event(state, event["kind"], event["payload"])
            previous, count = sha, count + 1
            if count in anchor_map:
                prefix = {"event_count": count, "sha256": sha, "journal_bytes": consumed}
                _verify_head(state, prefix, anchor_map[count])
                verified_anchors.append(prefix)
    _require(count > 0 and state["authority"] == authority, "empty journal or different budget authority")
    _require(len(verified_anchors) == len(anchor_map), "historical anchor lies beyond journal tail")
    head = {"event_count": count, "sha256": previous, "journal_bytes": consumed}
    if expected_head is not None:
        state = _verify_head(state, head, expected_head)
        head = copy.deepcopy(expected_head)
    recorded = copy.deepcopy(state["usage_ns"])
    conservative_state = copy.deepcopy(state)
    now = now_sample or {"wall_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}
    _require(budget._valid_sample(now), "invalid inspection clocks")
    open_attempts = []
    for identity, attempt in state["attempts"].items():
        if attempt["outcome"] == "open":
            elapsed = max(0, *(now[k] - attempt["last_sample"][k] for k in now))
            budget._charge(conservative_state, conservative_state["attempts"][identity], elapsed)
            open_attempts.append({"attempt_id": identity, **copy.deepcopy(attempt),
                                  "additional_conservative_ns": elapsed,
                                  "terminal_time_unknown": True})
    reasons = list(budget._stop_reasons(state))
    if expected_head is None:
        reasons.append("trusted_terminal_head_missing")
    if open_attempts:
        reasons.append("open_attempt_terminal_evidence_and_review_required")
    if state["unfinished_close"] is not None:
        reasons.append("unfinished_close_requires_review")
    if (path / ".writer_lock").exists():
        reasons.append("writer_lock_present")
    after = _signature(journal)
    if after != before:
        reasons.append("journal_changed_during_inspection")
    if state["policy"]["monitor_gap_ns"] != 5 * budget.NS:
        reasons.append("monitor_policy_not_approved_5_seconds")
    return {"ledger_path": str(path), "head": head, "journal_sha256": digest.hexdigest(),
            "signature": before, "signature_after": after, "authority_sha256": state["authority_sha256"],
            "integrity_valid": True, "trusted_terminal_head_verified": expected_head is not None,
            "verified_anchors": verified_anchors, "charged_ns": recorded,
            "conservative_ns": conservative_state["usage_ns"], "inspection_sample": now,
            "open_attempts": open_attempts,
            "attempts": copy.deepcopy(state["attempts"]),
            "retry_authorizations": copy.deepcopy(state.get("retry_authorizations", {})),
            "must_stop": bool(reasons), "stop_reasons": list(dict.fromkeys(reasons)),
            "terminal_time_limitation": "An open attempt has unknown process-exit time. The conservative bound is not a measured duration or proof of actual exhaustion."}


def _head_artifact(spec, base):
    """Load a hash-bound external receipt. A self-hash of a journal is insufficient."""
    path = _resolve(base, spec["path"])
    _require(budget._hex(spec.get("sha256")) and _file_hash(path) == spec["sha256"],
             "external head receipt hash differs")
    value = _read_small(path)
    for key in spec.get("keys", ["head"]):
        _require(isinstance(value, dict) and key in value, "external receipt does not contain its declared head")
        value = value[key]
    _require(isinstance(value, dict), "external receipt head is not an object")
    return value


def discover_journals(roots):
    paths = []
    for root in roots:
        root = Path(root).resolve()
        _require(root.is_dir(), f"budget inventory root missing: {root}")
        if (root / "budget_events.jsonl").is_file():
            paths.append(root)
        for child in sorted(root.iterdir()):
            if child.is_dir() and re.fullmatch(r"segment_\d+", child.name):
                _require((child / "budget_events.jsonl").is_file(), f"incomplete segment in inventory: {child}")
                paths.append(child.resolve())
    _require(paths and len(set(paths)) == len(paths), "empty or duplicated budget inventory")
    return sorted(paths)


def _increment_index(base):
    """Small append-only inventory, not a separate accounting ledger."""
    path = Path(base) / INCREMENT_INDEX
    rows, previous = [], None
    if path.exists():
        with path.open("rb") as stream:
            for line in stream:
                _require(len(line) <= MAX_RECEIPT_BYTES and line.endswith(b"\n"), "torn incremental ledger index")
                row = json.loads(line, object_pairs_hook=budget._unique_object)
                sha = row.pop("sha256")
                _require(row["sequence"] == len(rows) and row["previous_sha256"] == previous
                         and budget.canonical_hash(row) == sha, "incremental ledger index chain differs")
                _require(budget._hex(row["handoff_spec_sha256"]) and Path(row["ledger_path"]).is_absolute(),
                         "invalid incremental ledger index binding")
                rows.append(row)
                previous = sha
    paths = [Path(row["ledger_path"]).resolve() for row in rows]
    _require(len(paths) == len(set(paths)), "duplicated incremental ledger index entry")
    consumed = [sha for row in rows for sha in row.get("consumed_failure_reviews", [])]
    _require(all(budget._hex(sha) for sha in consumed) and len(consumed) == len(set(consumed)),
             "a failure review was consumed twice in the incremental index")
    return paths, {"event_count": len(rows), "sha256": previous}


def consumed_review_map(base):
    """Review-record digest -> the one increment that consumed it."""
    path = Path(base) / INCREMENT_INDEX
    _increment_index(base)
    if not path.exists():
        return {}
    with path.open("rb") as stream:
        rows = [json.loads(line) for line in stream]
    return {sha: str(Path(row["ledger_path"]).resolve())
            for row in rows for sha in row.get("consumed_failure_reviews", [])}


def _register_increment(base, target, spec_sha, *, consumed_failure_reviews=()):
    paths, head = _increment_index(base)
    _require(target not in paths, "increment already registered")
    consumed = sorted(consumed_failure_reviews)
    already = consumed_review_map(base)
    _require(not any(sha in already for sha in consumed), "failure review already consumed by another segment")
    row = {"sequence": head["event_count"], "previous_sha256": head["sha256"],
           "ledger_path": str(target), "handoff_spec_sha256": spec_sha,
           **({"consumed_failure_reviews": consumed} if consumed else {})}
    payload = {**row, "sha256": budget.canonical_hash(row)}
    with (Path(base) / INCREMENT_INDEX).open("ab") as stream:
        stream.write(budget._canonical(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _own_launcher_chain(own, *, max_hops=6, max_create_time_gap_seconds=2.0):
    """PIDs of this invocation's own trusted Python launcher ancestors only.

    A venv ``Scripts/python.exe`` on Windows is a real parent process that
    spawns a separate base-interpreter child rather than replacing itself;
    measured on the target machine the launcher/interpreter create_time gap
    is ~0.03-0.1s, so 2s leaves wide margin without risking exempting an
    old, unrelated, coincidentally-python ancestor (see
    tests/test_active_research_processes.py, which pins this boundary with
    a 5s-old ancestor that must still be reported).

    The walk stops at the first ancestor that is not a python-named process
    or whose create_time is not close to its child's, and never continues
    past that point -- a non-python hop (shell, IDE, wsl.exe, ...) is never
    leapfrogged to exempt something further up the chain.
    """
    import psutil
    chain = set()
    current = own
    for _ in range(max_hops):
        try:
            parent = current.parent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        if parent is None:
            break
        try:
            name_is_python = "python" in (parent.name() or "").lower()
            close_enough = abs(parent.create_time() - current.create_time()) <= max_create_time_gap_seconds
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        if not (name_is_python and close_enough):
            break
        chain.add(parent.pid)
        current = parent
    return chain


def active_research_processes():
    """Read process identities; absence does not establish historical exit time."""
    import psutil
    own = psutil.Process(os.getpid())
    allowed = {own.pid, *(p.pid for p in own.children(recursive=True)), *_own_launcher_chain(own)}
    found = []
    markers = ("scripts.run_input_robustness_formal", "scripts.input_robustness_formal_worker",
               "scripts.run_mlp_budget_supplement", "scripts.run_mlp24",
               "scripts.mlp_budget_extension_entry", "scripts.mlp_budget_extension_worker")
    for process in psutil.process_iter(["pid", "ppid", "create_time", "cmdline", "name"]):
        try:
            info = process.info
            if info["pid"] in allowed or "python" not in (info["name"] or "").lower():
                continue
            command = " ".join(info["cmdline"] or [])
            if any(marker in command for marker in markers):
                found.append({"pid": info["pid"], "ppid": info["ppid"],
                              "created_at": info["create_time"], "command": command})
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise HandoffError("process inventory is incomplete; cannot rule out concurrent run") from exc
    return found


def inspect_handoff(spec, *, base_root, process_probe=None, now_sample=None):
    """Build a read-only preparation receipt from the explicitly complete inventory.

    No ``passed`` flag, supplied aggregate total, or successful record count is
    trusted. All accounting is reconstructed from journal events.
    """
    _require(spec.get("schema_version") in SCHEMAS, "unsupported budget handoff schema")
    _require(spec["schema_version"] == SCHEMA or not any(s.get("failure_reviews") for s in spec["sources"]),
             "failure reviews require the version 2 handoff schema")
    _require(budget._hex(spec.get("source_commit"), 40) and budget._hex(spec.get("config_sha256")),
             "handoff requires execution source and configuration binding")
    authority = spec["authority"]
    _require(isinstance(authority, dict) and authority, "handoff authority missing")
    roots = [_resolve(base_root, p) for p in spec["inventory_roots"]]
    inventory = discover_journals(roots)
    increments, index_head = _increment_index(base_root)
    if increments:
        _require(spec.get("increment_index_head") == index_head,
                 "handoff omits or differs from the preserved incremental index head")
    inventory = sorted(set(inventory + increments))
    paths = [_resolve(base_root, s["ledger_path"]) for s in spec["sources"]]
    _require(len(set(paths)) == len(paths) and sorted(paths) == inventory,
             "handoff must include every existing budget journal exactly once")
    snapshots = []
    for source, path in zip(spec["sources"], paths):
        trusted = _head_artifact(source["terminal_receipt"], base_root) if source.get("terminal_receipt") else None
        anchors = [_head_artifact(item, base_root) for item in source.get("anchor_receipts", [])]
        snapshots.append(stream_snapshot(path, authority=authority, expected_head=trusted,
                                         anchors=anchors, now_sample=now_sample))
    # Attempt identifiers are namespaced by journal. Repeated attempt_000000 in
    # separate runs still contributes; matching model-unit identities accumulate.
    usage = _sum_usage(item["charged_ns"] for item in snapshots)
    conservative = _sum_usage(item["conservative_ns"] for item in snapshots)
    authorizations, consumed = _apply_failure_reviews(spec["sources"], snapshots, base_root)
    historical = [f"{item['ledger_path']}: {reason}" for item in snapshots for reason in item["historical_stop_reasons"]]
    waived = [row for item in snapshots for row in item["waived_by_review"]]
    reasons = [f"{item['ledger_path']}: {reason}" for item in snapshots
               for reason in item["current_blocking_stop_reasons"]]
    global_reasons = _cap_reasons(usage)
    processes = (process_probe or active_research_processes)()
    if processes:
        global_reasons.append("existing_research_processes_active")
    after_increments, after_index_head = _increment_index(base_root)
    if sorted(set(discover_journals(roots) + after_increments)) != inventory or after_index_head != index_head:
        global_reasons.append("budget_inventory_changed_during_inspection")
    for source in snapshots:
        if _signature(Path(source["ledger_path"]) / "budget_events.jsonl") != source["signature"]:
            global_reasons.append("journal_changed_after_its_inspection")
    reasons += global_reasons
    historical += global_reasons
    seen = set()
    for source in snapshots:
        for identity, attempt in source["attempts"].items():
            key = budget.canonical_hash({"attempt_id": identity, "started_at": attempt["started_at"],
                                         "activity_id": attempt["activity_id"]})
            _require(key not in seen, "overlapping copied journals would double-count an attempt")
            seen.add(key)
    return {"schema_version": SCHEMA, "source_commit": spec["source_commit"],
            "config_sha256": spec["config_sha256"], "authority": copy.deepcopy(authority),
            "authority_sha256": budget.canonical_hash(authority), "caps_seconds": dict(budget.CAPS_SECONDS),
            "inventory_roots": [str(p) for p in roots], "source_specs": copy.deepcopy(spec["sources"]),
            "budget_base_root": str(Path(base_root).resolve()), "increment_index_head": index_head,
            "sources": snapshots, "charged_ns": usage, "charged_seconds": _usage_seconds(usage),
            "conservative_ns": conservative, "conservative_seconds": _usage_seconds(conservative),
            "remaining_recorded_ns": {key: budget.CAPS_SECONDS[key] * budget.NS - usage[key] for key in SCALARS},
            "actual_terminal_use_known": not any(item["open_attempts"] for item in snapshots),
            "active_processes": processes, "must_stop": bool(reasons),
            "stop_reasons": list(dict.fromkeys(reasons)),
            "historical_stop_reasons": list(dict.fromkeys(historical)),
            "current_blocking_stop_reasons": list(dict.fromkeys(reasons)),
            "waived_by_review": waived, "failure_retry_authorizations": authorizations,
            "consumed_failure_reviews": consumed, "formal_launch_authorized": False}


def _apply_failure_reviews(specs, snapshots, base_root):
    """Split each source's stop reasons into historical and current sets.

    Only ``unreviewed_failed:<attempt>`` can be waived, only by a valid signed
    approval, and the historical list always keeps it. Returns the approvals
    not yet consumed by an increment, and every consumed review.
    """
    consumed = consumed_review_map(base_root)
    retried_units = {unit for item in snapshots for unit in item.get("retry_authorizations", {})}
    seen_records, seen_attempts, available = set(), set(), []
    for spec, item in zip(specs, snapshots):
        historical = list(item["stop_reasons"])
        current, waived, reports = list(historical), [], []
        receipt = spec.get("terminal_receipt") or {}
        for reference in spec.get("failure_reviews", []):
            _require(reference.get("sha256") not in seen_records, "the same failure review is listed twice")
            seen_records.add(reference.get("sha256"))
            owner = consumed.get(reference.get("sha256"))
            # A review consumed by an increment is that increment's own retry;
            # it no longer counts against the unit's single retry.
            units = retried_units if owner is None else set()
            result = failure_review.evaluate_review(
                reference, snapshot=item, ledger_path=item["ledger_path"],
                terminal_receipt_sha256=receipt.get("sha256"), retried_units=units)
            attempt = result["attempt_id"] or "unknown"
            if result["attempt_id"] is not None:
                _require((item["ledger_path"], attempt) not in seen_attempts,
                         "one failed attempt cannot carry two failure reviews")
                seen_attempts.add((item["ledger_path"], attempt))
            blocker = f"unreviewed_failed:{attempt}"
            if result["retry_allowed"] and blocker in current:
                current.remove(blocker)
                row = {"ledger_path": item["ledger_path"], "attempt_id": attempt, "reason": blocker,
                       "review_record_sha256": result["review_record_sha256"],
                       "review_plan_sha256": result["review_plan_sha256"], "consumed_by": owner}
                waived.append(row)
                if owner is None:
                    available.append({**result["authorization"], "waived": dict(row)})
            elif not result["valid"] or result["retry_allowed"]:
                current.append(f"failure_review_invalid:{attempt}")
            reports.append({key: result[key] for key in ("attempt_id", "review_record_sha256", "review_plan_sha256",
                                                          "valid", "retry_allowed", "errors")})
        item.update(historical_stop_reasons=historical, current_blocking_stop_reasons=list(dict.fromkeys(current)),
                    waived_by_review=waived, failure_reviews=reports)
    return available, consumed


def validate_handoff(receipt_path, *, base_root, expected_config_sha, expected_source_commit,
                     process_probe=None, required_inventory_roots=None):
    """Recompute every source, requiring externally bound terminal heads.

    The input is a handoff *specification*, not evidence that training is
    authorized. A returned object only clears this budget check; callers still
    enforce source/CI/data/launch gates.
    """
    spec = _read_small(receipt_path)
    _require(spec.get("source_commit") == expected_source_commit
             and spec.get("config_sha256") == expected_config_sha,
             "budget handoff execution source/configuration differs")
    _require(required_inventory_roots is not None, "authoritative configuration inventory is required")
    expected_roots = sorted(str(_resolve(base_root, p)) for p in required_inventory_roots)
    actual_roots = sorted(str(_resolve(base_root, p)) for p in spec["inventory_roots"])
    _require(actual_roots == expected_roots, "handoff inventory differs from authoritative configuration")
    result = inspect_handoff(spec, base_root=base_root, process_probe=process_probe)
    _require(not result["must_stop"], "budget handoff is blocked: " + "; ".join(result["stop_reasons"]))
    result["handoff_spec_sha256"] = _file_hash(receipt_path)
    result["handoff_spec_path"] = str(Path(receipt_path).resolve())
    return result


class InheritedLedger:
    """Adapter over BudgetLedger with cumulative caps and exclusive dispatch."""

    def __init__(self, ledger, inherited, lock, *, process_probe=None):
        self._ledger, self._inherited, self._lock = ledger, inherited, lock
        self._process_probe = process_probe or active_research_processes
        self._closed = False
        registered, self._index_head = _increment_index(inherited["budget_base_root"])
        _require(self.path.resolve() in registered, "active increment is absent from shared index")
        self._segment_sequence = registered.index(self.path.resolve())

    @property
    def path(self):
        return self._ledger.path

    @property
    def segment_sequence(self):
        """Position in the shared append-only increment index; never reused."""
        return self._segment_sequence

    def known_attempt_ids(self):
        # Sources are stream_snapshot() results; boundary fixtures may carry heads only.
        inherited = {name for source in self._inherited["sources"] for name in source.get("attempts", {})}
        return inherited | set(self._ledger.snapshot()["attempts"])

    def _granted_failure_retries(self):
        """Approved reviews consumed by this increment when it was registered."""
        mine = str(self.path.resolve())
        owned = {sha for sha, owner in consumed_review_map(self._inherited["budget_base_root"]).items() if owner == mine}
        return [row for row in self._inherited.get("failure_retry_authorizations", [])
                if row["review_record_sha256"] in owned]

    def pending_failure_retries(self):
        recorded = {row["review_record_sha256"]
                    for row in self._ledger.snapshot().get("retry_authorizations", {}).values()}
        return [copy.deepcopy(row) for row in self._granted_failure_retries()
                if row["review_record_sha256"] not in recorded]

    def authorize_failure_retry(self, review_record_sha256, *, retry_attempt_id, phase_id):
        """Append the one-time authorization after phase registration, before the retry."""
        _require(not self._external_reasons(), "external inventory/ownership prevents retry authorization")
        granted = next((row for row in self.pending_failure_retries()
                        if row["review_record_sha256"] == review_record_sha256), None)
        _require(granted is not None, "failure retry was not granted to this increment by its handoff")
        _require(retry_attempt_id not in self.known_attempt_ids(),
                 "retry attempt identity already exists in the inherited or current ledgers")
        payload = {key: copy.deepcopy(granted[key]) for key in
                   ("review_record_sha256", "review_plan_sha256", "original", "unit_id", "batch_id",
                    "activity_id", "research_identity", "required_environment")}
        payload.update(phase_id=phase_id, retry_attempt_id=retry_attempt_id, retry_ordinal=1)
        return self._combine(self._ledger.authorize_failure_retry(payload))

    def _inherited_failed_units(self):
        return {attempt["unit_id"] for source in self._inherited["sources"]
                for attempt in source.get("attempts", {}).values()
                if attempt["outcome"] == "failed" and attempt["unit_id"] is not None}

    @property
    def head(self):
        return {**self._ledger.head, "inherited_spec_sha256": self._inherited["handoff_spec_sha256"]}

    def _external_reasons(self):
        reasons = []
        for source in self._inherited["sources"]:
            journal = Path(source["ledger_path"]) / "budget_events.jsonl"
            if not journal.exists() or _signature(journal) != source["signature"]:
                reasons.append("inherited_journal_changed")
        roots = self._inherited["inventory_roots"]
        increments, index_head = _increment_index(self._inherited["budget_base_root"])
        if index_head != self._index_head or self.path.resolve() not in increments:
            reasons.append("increment_index_head_changed")
        observed = {str(p) for p in discover_journals(roots) + increments} - {str(self.path.resolve())}
        if observed != {s["ledger_path"] for s in self._inherited["sources"]}:
            reasons.append("inherited_inventory_changed")
        if self._process_probe():
            reasons.append("concurrent_research_process")
        if self._closed or not self._lock.exists():
            reasons.append("budget_owner_lock_not_held")
        return reasons

    def _combine(self, snapshot):
        current = self._ledger._projected_state()["usage_ns"]
        used = _sum_usage((self._inherited["charged_ns"], current))
        result = copy.deepcopy(snapshot)
        result["charged_seconds"] = _usage_seconds(used)
        result["charged_ns"] = used
        result["remaining_seconds"] = {key: (budget.CAPS_SECONDS[key] * budget.NS - used[key]) / budget.NS
                                       for key in SCALARS}
        result["remaining_seconds"].update({key: {name: (budget.CAPS_SECONDS[cap] * budget.NS - value) / budget.NS
                                                 for name, value in used[key].items()}
                                            for key, cap in (("units", "unit"), ("batches", "batch"))})
        result["stop_reasons"] = list(dict.fromkeys(result["stop_reasons"] + _cap_reasons(used) + self._external_reasons()))
        result["must_stop"] = bool(result["stop_reasons"])
        result["head"] = self.head
        result["inherited_journal_heads"] = [{"ledger_path": s["ledger_path"], "head": s["head"]}
                                             for s in self._inherited["sources"]]
        return result

    def snapshot(self):
        return self._combine(self._ledger.snapshot())

    def poll(self, *, force=False):
        return self._combine(self._ledger.poll(force=force))

    def register_phase(self, phase_id, *, provenance, gate_receipt):
        _require(provenance["source_commit"] == self._inherited["source_commit"]
                 and provenance["config_sha256"] == self._inherited["config_sha256"],
                 "new phase differs from immutable handoff source/configuration")
        current = self.poll(force=True)
        _require(not current["must_stop"], "cannot register phase while cumulative budget is stopped")
        return self._combine(self._ledger.register_phase(phase_id, provenance=provenance, gate_receipt=gate_receipt))

    def begin_attempt(self, attempt_id, *, activity_id, phase_id, budget_group, estimated_seconds,
                      unit_id=None, batch_id=None, started_at=None):
        current = self.poll(force=True)
        _require(not current["must_stop"], "cumulative budget prevents dispatch")
        if unit_id in self._inherited_failed_units():
            _require(unit_id in current.get("retry_authorizations", {}),
                     "a unit with an inherited failed attempt requires a reviewed retry authorization")
        estimate = budget._seconds_ns(estimated_seconds)
        limits = [(budget_group, budget.CAPS_SECONDS[budget_group] * budget.NS - current["charged_ns"][budget_group]),
                  ("total", budget.CAPS_SECONDS["total"] * budget.NS - current["charged_ns"]["total"])]
        for key, name, cap in (("units", unit_id, "unit"), ("batches", batch_id, "batch")):
            if name is not None:
                limits.append((f"{cap}:{name}", budget.CAPS_SECONDS[cap] * budget.NS - current["charged_ns"][key].get(name, 0)))
        _require(all(remaining > 0 and estimate <= remaining for _, remaining in limits),
                 "insufficient cumulative admission headroom (including prior attempts)")
        return self._combine(self._ledger.begin_attempt(attempt_id, activity_id=activity_id, phase_id=phase_id,
                             budget_group=budget_group, estimated_seconds=estimated_seconds,
                             unit_id=unit_id, batch_id=batch_id, started_at=started_at))

    def close_attempt(self, attempt_id, *, outcome, record_sha256=None, reason=None):
        latest = self.poll(force=True)
        if outcome == "completed":
            _require(not latest["must_stop"], "cumulatively stopped attempt cannot claim completion")
        return self._combine(self._ledger.close_attempt(attempt_id, outcome=outcome,
                                  record_sha256=record_sha256, reason=reason))

    def review_external_interruption(self, activity_id, *, provenance, review_receipt):
        _require(not self._external_reasons(), "external inventory/ownership prevents interruption review")
        return self._combine(self._ledger.review_external_interruption(
            activity_id, provenance=provenance, review_receipt=review_receipt))

    def close(self):
        if self._closed:
            return
        final = self.snapshot()
        receipt = {"schema_version": SCHEMA, "head": self.head,
                   "handoff_spec_sha256": self._inherited["handoff_spec_sha256"],
                   "ledger_snapshot": final}
        receipt_path = self.path / "terminal_receipt.json"
        if receipt_path.exists():
            receipt_path = self.path / f"terminal_receipt_{self.head['event_count']}_{self.head['sha256'][:12]}.json"
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        # An open attempt retains the lock and all evidence for explicit review.
        if not any(a["outcome"] == "open" for a in final["attempts"].values()):
            self._lock.unlink()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def open_inherited_ledger(receipt_path, ledger_path, *, base_root, expected_config_sha,
                          expected_source_commit, process_probe=None, required_inventory_roots=None,
                          resume=False):
    """Create an increment only after a complete, closed, independently anchored handoff.

    ``resume=True`` reopens the same increment only after a trusted terminal
    receipt and explicit interruption review have entered the full inventory.
    It never closes an open attempt or clears a stale lock automatically.
    """
    base = Path(base_root).resolve()
    base.mkdir(parents=True, exist_ok=True)
    lock = base / ".supplement_budget_owner.lock"
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise HandoffError("shared budget owner lock exists; no concurrent/automatic restart") from exc
    created = False
    try:
        os.write(handle, json.dumps({"pid": os.getpid(), "created_wall_ns": time.time_ns()}).encode())
        os.fsync(handle)
        os.close(handle)
        handle = None
        inherited = validate_handoff(receipt_path, base_root=base, expected_config_sha=expected_config_sha,
                                     expected_source_commit=expected_source_commit, process_probe=process_probe,
                                     required_inventory_roots=required_inventory_roots)
        target = Path(ledger_path).resolve()
        if resume:
            registered, _ = _increment_index(base)
            _require(target in registered, "resume requires a registered existing incremental ledger")
            current = next((s for s in inherited["sources"] if Path(s["ledger_path"]) == target), None)
            _require(current is not None, "resume handoff must contain the current incremental terminal head")
            _require(not inherited.get("failure_retry_authorizations"),
                     "a reviewed failure retry must start in a new segment, never on resume")
            ledger = budget.BudgetLedger.open(target, authority=inherited["authority"], expected_head=current["head"])
            inherited["sources"] = [s for s in inherited["sources"] if Path(s["ledger_path"]) != target]
            inherited["charged_ns"] = _sum_usage(s["charged_ns"] for s in inherited["sources"])
            inherited["charged_seconds"] = _usage_seconds(inherited["charged_ns"])
        else:
            _require(not target.exists(), "increment already exists; reviewed handoff required, never reset")
            _require(all(target != Path(p) and Path(p) not in target.parents for p in inherited["inventory_roots"]),
                     "increment must not be inside immutable old inventory")
            ledger = budget.BudgetLedger.create(target, authority=inherited["authority"], monitor_gap_seconds=5)
            # Registering the increment consumes each approved review exactly once.
            _register_increment(base, target, inherited["handoff_spec_sha256"], consumed_failure_reviews=[
                row["review_record_sha256"] for row in inherited.get("failure_retry_authorizations", [])])
        created = True
        archive_name = f"inherited_budget_resume_{ledger.head['event_count']}.json" if resume else "inherited_budget.json"
        with (target / archive_name).open("x", encoding="utf-8") as stream:
            json.dump(inherited, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        return InheritedLedger(ledger, inherited, lock, process_probe=process_probe)
    except BaseException:
        if not created:
            lock.unlink()
        raise
    finally:
        if handle is not None:
            os.close(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_handoff(_read_small(args.spec), base_root=args.base_root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"must_stop": result["must_stop"], "stop_reasons": result["stop_reasons"],
                      "actual_terminal_use_known": result["actual_terminal_use_known"]}))
    return 2 if result["must_stop"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
