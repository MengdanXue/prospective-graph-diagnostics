"""Append-only approved-budget accounting without training libraries.

The caller owns process supervision and gate authorization. This module never
starts, stops, resumes, or retries a worker. Ordinary pause is handled by the
caller after the current whole model unit completes; emergencies stop promptly.

One journal spans reviewed source phases under one immutable budget authority.
Save head in an external immutable completion/review receipt. Reopening requires
that trusted head: a hash chain alone cannot detect deletion of its entire tail.
Torn records, sequence gaps, open attempts, and stale writer locks are never
repaired automatically. Inspect them read-only for human review.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

NS = 1_000_000_000
CAPS_SECONDS = {"total": 1382400, "resource": 7200, "control": 7200,
                "formal": 1368000, "unit": 28800, "batch": 57600}
GROUPS = ("resource", "control", "formal")
CI_JOBS = {"lightweight-verification", "full-protocol-verification", "manuscript-build"}


class BudgetError(RuntimeError):
    def __init__(self, message, *, inspection=None):
        super().__init__(message)
        self.inspection = inspection


class BudgetIntegrityError(BudgetError):
    pass


class BudgetWriteConflict(BudgetError):
    pass


class BudgetAdmissionError(BudgetError):
    pass


class BudgetOpenAttemptError(BudgetError):
    pass


def _require(condition, message, error=BudgetIntegrityError):
    if not condition:
        raise error(message)


def canonical_hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hex(value, length=64):
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _identifier(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 500


def _seconds_ns(value, *, nonnegative=True):
    _require(type(value) in (int, float) and math.isfinite(value), "clock/duration must be finite", BudgetError)
    _require(not nonnegative or value >= 0, "duration must be nonnegative", BudgetError)
    return round(value * NS)


def _sample(wall_clock=None, monotonic_clock=None):
    return {"wall_ns": time.time_ns() if wall_clock is None else _seconds_ns(wall_clock(), nonnegative=False),
            "monotonic_ns": time.monotonic_ns() if monotonic_clock is None else _seconds_ns(monotonic_clock(), nonnegative=False)}


def _valid_sample(sample):
    return isinstance(sample, dict) and set(sample) == {"wall_ns", "monotonic_ns"} and all(type(v) is int for v in sample.values())


def _clock_reasons(stats, policy):
    reasons = []
    if stats["minimum_wall_delta_ns"] < 0:
        reasons.append("wall_clock_rollback")
    if stats["minimum_monotonic_delta_ns"] < 0:
        reasons.append("monotonic_clock_rollback")
    if stats["maximum_clock_divergence_ns"] > policy["clock_skew_tolerance_ns"]:
        reasons.append("wall_monotonic_divergence")
    if stats["maximum_poll_gap_ns"] > policy["monitor_gap_ns"]:
        reasons.append("monitor_gap")
    return reasons


class DualClockGuard:
    """Charge max(0, wall delta, monotonic delta) on every sampled interval.

    Clock rollback never refunds an earlier charge. Callback clocks return
    seconds; defaults read integer nanoseconds. started_at uses wall/monotonic
    seconds. The owner samples frequently; journal heartbeats may be less frequent.
    """

    def __init__(self, *, wall_clock=None, monotonic_clock=None,
                 monitor_gap_seconds=5, clock_skew_tolerance_seconds=2,
                 started_at=None, initial_sample=None):
        self.wall_clock, self.monotonic_clock = wall_clock, monotonic_clock
        self.policy = {"monitor_gap_ns": _seconds_ns(monitor_gap_seconds),
                       "clock_skew_tolerance_ns": _seconds_ns(clock_skew_tolerance_seconds)}
        _require(self.policy["monitor_gap_ns"] > 0, "invalid clock guard policy")
        now = _sample(wall_clock, monotonic_clock)
        if initial_sample is not None:
            _require(started_at is None and _valid_sample(initial_sample), "invalid initial clock sample")
            first = copy.deepcopy(initial_sample)
        elif started_at is not None:
            _require(isinstance(started_at, dict) and set(started_at) == {"wall", "monotonic"}, "started_at requires both clock readings")
            first = {"wall_ns": _seconds_ns(started_at["wall"], nonnegative=False),
                     "monotonic_ns": _seconds_ns(started_at["monotonic"], nonnegative=False)}
            _require(all(first[k] <= now[k] for k in first), "started_at cannot be later than current clocks", BudgetAdmissionError)
        else:
            first = now
        self.last = first
        self.charge_ns = 0
        self.committed_charge_ns = 0
        self.committed_sample = copy.deepcopy(first)
        self._reset_stats()

    def _reset_stats(self):
        self.stats = {"sample_count": 0, "minimum_wall_delta_ns": 0,
                      "minimum_monotonic_delta_ns": 0, "maximum_poll_gap_ns": 0,
                      "maximum_clock_divergence_ns": 0}

    def sample(self):
        current = _sample(self.wall_clock, self.monotonic_clock)
        wall = current["wall_ns"] - self.last["wall_ns"]
        mono = current["monotonic_ns"] - self.last["monotonic_ns"]
        charged = max(0, wall, mono)
        self.charge_ns += charged
        self.last = current
        self.stats["sample_count"] += 1
        self.stats["minimum_wall_delta_ns"] = min(self.stats["minimum_wall_delta_ns"], wall)
        self.stats["minimum_monotonic_delta_ns"] = min(self.stats["minimum_monotonic_delta_ns"], mono)
        self.stats["maximum_poll_gap_ns"] = max(self.stats["maximum_poll_gap_ns"], charged)
        self.stats["maximum_clock_divergence_ns"] = max(self.stats["maximum_clock_divergence_ns"], abs(wall - mono))
        return {"charged_seconds": self.charge_ns / NS, "interval_charge_seconds": charged / NS,
                "stop_reasons": _clock_reasons(self.stats, self.policy), "sample": copy.deepcopy(current)}

    def observation(self, attempt_id):
        return {"attempt_id": attempt_id, "from_sample": copy.deepcopy(self.committed_sample),
                "to_sample": copy.deepcopy(self.last), "delta_charge_ns": self.charge_ns - self.committed_charge_ns,
                "statistics": dict(self.stats)}

    def commit(self):
        self.committed_sample = copy.deepcopy(self.last)
        self.committed_charge_ns = self.charge_ns
        self._reset_stats()


def _empty_usage():
    return {"total": 0, "resource": 0, "control": 0, "formal": 0, "units": {}, "batches": {}}


def _charge(state, attempt, delta):
    usage = state["usage_ns"]
    usage["total"] += delta
    usage[attempt["budget_group"]] += delta
    if attempt["unit_id"] is not None:
        key = attempt["unit_id"]
        usage["units"][key] = usage["units"].get(key, 0) + delta
    if attempt["batch_id"] is not None:
        key = attempt["batch_id"]
        usage["batches"][key] = usage["batches"].get(key, 0) + delta
    attempt["charged_ns"] += delta


def _cap_reasons(state):
    usage, caps = state["usage_ns"], state["caps_ns"]
    reasons = []
    for key, value in sorted(usage["units"].items()):
        if value >= caps["unit"]:
            reasons.append(f"unit_time_cap:{key}")
    for key, value in sorted(usage["batches"].items()):
        if value >= caps["batch"]:
            reasons.append(f"batch_time_cap:{key}")
    for group in GROUPS:
        if usage[group] >= caps[group]:
            reasons.append(f"{group}_time_cap")
    if usage["total"] >= caps["total"]:
        reasons.append("total_time_cap")
    return reasons


def _stop_reasons(state):
    reasons = list(_cap_reasons(state))
    for attempt_id, attempt in state["attempts"].items():
        if attempt["reviewed"]:
            continue
        reasons.extend(f"{reason}:{attempt_id}" for reason in attempt["clock_stop_reasons"])
        if attempt["outcome"] in ("failed", "external_interruption"):
            reasons.append(f"unreviewed_{attempt['outcome']}:{attempt_id}")
    # An attempted unreviewed resume is recorded and refused, but its later
    # explicit interruption review may authorize a restart. Insufficient fixed
    # headroom remains a stop requiring a budget amendment.
    reasons.extend(reason for reason in state["dispatch_denials"] if reason.startswith("insufficient cumulative admission headroom"))
    return list(dict.fromkeys(reasons))


def _validate_provenance(provenance, gate):
    _require(isinstance(provenance, dict) and _hex(provenance.get("source_commit"), 40), "phase requires a source commit")
    for key in ("config_sha256", "data_binding_sha256"):
        _require(_hex(provenance.get(key)), f"phase requires {key}")
    _require(isinstance(provenance.get("source_files"), dict) and provenance["source_files"] and all(_identifier(k) and _hex(v) for k, v in provenance["source_files"].items()), "phase requires executable source hashes")
    _require(isinstance(provenance.get("environment"), dict) and provenance["environment"], "phase requires environment/device binding")
    _require(isinstance(gate, dict) and _hex(gate.get("ci_receipt_file_sha256")) and _hex(gate.get("review_record_sha256")), "phase requires CI receipt and review hashes")
    receipt = gate.get("ci_receipt")
    _require(isinstance(receipt, dict) and receipt.get("commit") == provenance["source_commit"], "CI receipt must bind the phase source")
    _require(type(receipt.get("run_id")) is int and receipt["run_id"] > 0 and receipt.get("status") == "completed" and receipt.get("conclusion") == "success", "phase CI run must be complete and successful")
    jobs = receipt.get("jobs", [])
    _require(isinstance(jobs, list) and len(jobs) == 3 and all(isinstance(j, dict) for j in jobs) and {j.get("name") for j in jobs} == CI_JOBS, "phase requires all three CI jobs")
    _require(all(j.get("run_id") == receipt["run_id"] and j.get("status") == "completed" and j.get("conclusion") == "success" for j in jobs), "phase CI jobs differ or did not pass")
    _canonical(provenance)
    _canonical(gate)


def _applicable_limits(state, descriptor):
    used, caps = state["usage_ns"], state["caps_ns"]
    result = []
    if descriptor["unit_id"] is not None:
        result.append((f"unit:{descriptor['unit_id']}", caps["unit"] - used["units"].get(descriptor["unit_id"], 0)))
    if descriptor["batch_id"] is not None:
        result.append((f"batch:{descriptor['batch_id']}", caps["batch"] - used["batches"].get(descriptor["batch_id"], 0)))
    result.extend(((descriptor["budget_group"], caps[descriptor["budget_group"]] - used[descriptor["budget_group"]]), ("total", caps["total"] - used["total"])))
    return result


def _validate_start(state, payload):
    attempt_id, descriptor = payload["attempt_id"], payload["descriptor"]
    _require(_identifier(attempt_id) and attempt_id not in state["attempts"], "attempt identity already exists", BudgetWriteConflict)
    _require(set(descriptor) == {"activity_id", "phase_id", "budget_group", "unit_id", "batch_id", "estimate_ns"}, "invalid activity descriptor")
    _require(_identifier(descriptor["activity_id"]) and descriptor["phase_id"] in state["phases"] and descriptor["budget_group"] in GROUPS, "unknown activity phase/group")
    _require(type(descriptor["estimate_ns"]) is int and descriptor["estimate_ns"] >= 0, "invalid admission estimate")
    _require(all(value is None or _identifier(value) for value in (descriptor["unit_id"], descriptor["batch_id"])), "invalid unit/batch identity")
    if descriptor["budget_group"] == "formal":
        _require(descriptor["unit_id"] is not None and descriptor["batch_id"] is not None, "formal attempts require fixed unit and paired-batch identities")
    else:
        _require(descriptor["unit_id"] is None, "only formal attempts may charge a model unit")
        if descriptor["budget_group"] == "resource":
            _require(descriptor["batch_id"] is None, "resource acceptance wraps the entire phase without formal batch accounting")
    _require(_valid_sample(payload["started_at"]), "invalid attempt start clocks")
    _require(not _stop_reasons(state), "unresolved stop/failure prevents dispatch", BudgetAdmissionError)
    active = [a for a in state["attempts"].values() if a["outcome"] == "open"]
    groups = [a["budget_group"] for a in active] + [descriptor["budget_group"]]
    _require(len(groups) <= 2 and (len(groups) == 1 or sorted(groups) == ["control", "formal"]), "only one formal activity and one control activity may overlap", BudgetAdmissionError)
    activity = state["activities"].get(descriptor["activity_id"])
    if activity is not None:
        _require(activity["descriptor"] == descriptor, "activity provenance, estimate or budget identity changed", BudgetAdmissionError)
        last = state["attempts"][activity["attempt_ids"][-1]]
        _require(last["outcome"] == "external_interruption" and last["reviewed"], "completed/failed/unreviewed activity cannot restart", BudgetAdmissionError)
    if descriptor["unit_id"] in state["unit_owners"]:
        _require(state["unit_owners"][descriptor["unit_id"]] == descriptor["activity_id"], "model unit cannot be renamed to reset its budget", BudgetAdmissionError)
    if descriptor["budget_group"] == "formal" and descriptor["batch_id"] in state["batch_phases"]:
        _require(state["batch_phases"][descriptor["batch_id"]] == descriptor["phase_id"], "paired batch cannot mix reviewed source phases", BudgetAdmissionError)
    insufficient = [name for name, remaining in _applicable_limits(state, descriptor) if remaining <= 0 or descriptor["estimate_ns"] > remaining]
    _require(not insufficient, "insufficient cumulative admission headroom: " + ", ".join(insufficient), BudgetAdmissionError)


def _apply_observation(state, observation, *, allow_closed=False):
    required = {"attempt_id", "from_sample", "to_sample", "delta_charge_ns", "statistics"}
    _require(set(observation) == required and observation["attempt_id"] in state["attempts"], "invalid heartbeat identity/fields")
    attempt = state["attempts"][observation["attempt_id"]]
    _require(attempt["outcome"] == "open" or allow_closed, "heartbeat for a closed attempt")
    first, last = observation["from_sample"], observation["to_sample"]
    _require(first == attempt["last_sample"] and _valid_sample(last), "heartbeat clock continuity mismatch")
    delta, stats = observation["delta_charge_ns"], observation["statistics"]
    required_stats = {"sample_count", "minimum_wall_delta_ns", "minimum_monotonic_delta_ns", "maximum_poll_gap_ns", "maximum_clock_divergence_ns"}
    _require(set(stats) == required_stats and all(type(v) is int for v in stats.values()) and stats["sample_count"] >= 1, "invalid heartbeat sampling statistics")
    _require(type(delta) is int and delta >= max(0, last["wall_ns"] - first["wall_ns"], last["monotonic_ns"] - first["monotonic_ns"]), "heartbeat discounts elapsed clock time")
    _require(stats["maximum_poll_gap_ns"] >= 0 and stats["maximum_clock_divergence_ns"] >= 0 and delta <= stats["sample_count"] * stats["maximum_poll_gap_ns"], "heartbeat charge/statistic inconsistency")
    _charge(state, attempt, delta)
    attempt["last_sample"] = copy.deepcopy(last)
    attempt["clock_stop_reasons"] = list(dict.fromkeys(attempt["clock_stop_reasons"] + _clock_reasons(stats, state["policy"])))


def _apply_event(state, kind, payload):
    if kind == "genesis":
        _require(not state, "duplicate budget genesis")
        _require(payload["caps_seconds"] == CAPS_SECONDS and isinstance(payload["authority"], dict) and payload["authority"], "budget caps or authority changed")
        _require(canonical_hash(payload["authority"]) == payload["authority_sha256"], "authority digest mismatch")
        policy = payload["policy"]
        _require(set(policy) == {"heartbeat_ns", "monitor_gap_ns", "clock_skew_tolerance_ns"} and all(type(v) is int and v >= 0 for v in policy.values()) and policy["heartbeat_ns"] > 0 and policy["monitor_gap_ns"] > 0, "invalid durable clock policy")
        state.update(authority=copy.deepcopy(payload["authority"]), authority_sha256=payload["authority_sha256"],
                     caps_ns={k: v * NS for k, v in CAPS_SECONDS.items()}, policy=dict(policy), phases={}, attempts={}, activities={},
                     unit_owners={}, batch_phases={}, usage_ns=_empty_usage(), dispatch_denials=[],
                     unfinished_close=None, pending_finalization_id=None)
    elif kind == "phase_registered":
        phase = payload["phase_id"]
        _require(_identifier(phase) and phase not in state["phases"], "phase registration is not exclusive")
        _validate_provenance(payload["provenance"], payload["gate_receipt"])
        state["phases"][phase] = copy.deepcopy(payload)
    elif kind == "attempt_started":
        _validate_start(state, payload)
        attempt_id, descriptor = payload["attempt_id"], payload["descriptor"]
        state["attempts"][attempt_id] = {**copy.deepcopy(descriptor), "started_at": copy.deepcopy(payload["started_at"]),
                                          "last_sample": copy.deepcopy(payload["started_at"]), "charged_ns": 0,
                                          "outcome": "open", "reason": None, "record_sha256": None,
                                          "clock_stop_reasons": [], "reviewed": False}
        activity = state["activities"].setdefault(descriptor["activity_id"], {"descriptor": copy.deepcopy(descriptor), "attempt_ids": []})
        activity["attempt_ids"].append(attempt_id)
        if descriptor["unit_id"] is not None:
            state["unit_owners"][descriptor["unit_id"]] = descriptor["activity_id"]
        if descriptor["budget_group"] == "formal":
            state["batch_phases"][descriptor["batch_id"]] = descriptor["phase_id"]
    elif kind == "heartbeat":
        _require(payload["observations"] and len({o["attempt_id"] for o in payload["observations"]}) == len(payload["observations"]), "empty or duplicate heartbeat")
        for observation in payload["observations"]:
            _apply_observation(state, observation)
    elif kind == "attempt_closed":
        attempt = state["attempts"][payload["attempt_id"]]
        _require(attempt["outcome"] == "open" and payload["outcome"] in ("completed", "external_interruption", "failed"), "invalid or duplicate attempt close")
        if payload["outcome"] == "completed":
            _require(_hex(payload["record_sha256"]) and not _stop_reasons(state), "stopped attempt cannot claim validated completion")
        else:
            _require(_identifier(payload["reason"]), "interrupted/failed attempt requires an explicit cause")
        attempt.update(outcome=payload["outcome"], reason=payload["reason"], record_sha256=payload["record_sha256"])
        _require(state["unfinished_close"] is None and state["pending_finalization_id"] is None, "previous finalization remains unsettled")
        state["unfinished_close"] = payload["attempt_id"]
    elif kind == "finalization_charge":
        _require(len(payload["observations"]) == 1 and type(payload["settle_external"]) is bool, "invalid finalization settlement")
        attempt_id = payload["observations"][0]["attempt_id"]
        if payload["settle_external"]:
            _require(state["pending_finalization_id"] == attempt_id and state["unfinished_close"] is None, "unexpected external finalization settlement")
        else:
            _require(state["unfinished_close"] == attempt_id, "finalization has no matching close event")
        for observation in payload["observations"]:
            _require(state["attempts"][observation["attempt_id"]]["outcome"] != "open", "finalization charge requires a closed attempt")
            _apply_observation(state, observation, allow_closed=True)
        state["unfinished_close"] = None
        state["pending_finalization_id"] = None if payload["settle_external"] else attempt_id
    elif kind == "external_interruption_reviewed":
        attempt = state["attempts"][payload["attempt_id"]]
        _require(attempt["outcome"] == "external_interruption" and not attempt["reviewed"], "only a closed unreviewed external interruption may be reviewed")
        review = payload["review_receipt"]
        _require(isinstance(review, dict) and review.get("all_existing_records_validated") is True and review.get("external_interruption_cause_confirmed") is True and review.get("no_unresolved_failure_artifacts") is True and _hex(review.get("review_record_sha256")), "external interruption requires explicit record/cause review")
        phase = state["phases"][attempt["phase_id"]]
        _require(payload["provenance"] == phase["provenance"], "resume provenance differs from the interrupted attempt")
        attempt["reviewed"] = True
        attempt["review_receipt"] = copy.deepcopy(review)
    elif kind == "dispatch_denied":
        _require(_identifier(payload["reason"]), "dispatch denial requires a reason")
        state["dispatch_denials"].append(payload["reason"])
    else:
        raise BudgetIntegrityError(f"unknown budget journal event: {kind}")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, "duplicate JSON key in budget journal")
        value[key] = item
    return value


def _read_journal(path):
    journal = Path(path) / "budget_events.jsonl"
    raw = journal.read_bytes()
    _require(raw and raw.endswith(b"\n"), "empty or truncated budget journal")
    state, previous, count = {}, None, 0
    for line in raw.splitlines():
        _require(bool(line), "blank/gapped budget journal line")
        try:
            event = json.loads(line, object_pairs_hook=_unique_object)
            _require(set(event) == {"schema_version", "sequence", "previous_sha256", "kind", "payload", "sha256"}, "budget journal event fields changed")
            _require(event["schema_version"] == "1.0" and type(event["sequence"]) is int and event["sequence"] == count and event["previous_sha256"] == previous, "budget journal sequence/hash-chain gap")
            digest = event.pop("sha256")
            _require(_hex(digest) and canonical_hash(event) == digest, "budget journal content hash mismatch")
            _apply_event(state, event["kind"], event["payload"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise BudgetIntegrityError(f"invalid budget journal event {count}: {exc}") from exc
        previous, count = digest, count + 1
    return state, {"event_count": count, "sha256": previous, "journal_bytes": len(raw)}


def _public_snapshot(state, head):
    used = state["usage_ns"]
    charged = {key: value / NS for key, value in used.items() if key not in ("units", "batches")}
    charged.update(units={k: v / NS for k, v in used["units"].items()}, batches={k: v / NS for k, v in used["batches"].items()})
    remaining = {key: (state["caps_ns"][key] - used[key]) / NS for key in ("total", *GROUPS)}
    remaining.update(units={k: (state["caps_ns"]["unit"] - v) / NS for k, v in used["units"].items()},
                     batches={k: (state["caps_ns"]["batch"] - v) / NS for k, v in used["batches"].items()})
    attempts = {key: {**{k: copy.deepcopy(v) for k, v in value.items() if k != "charged_ns"}, "charged_seconds": value["charged_ns"] / NS} for key, value in state["attempts"].items()}
    reasons = _stop_reasons(state)
    return {"must_stop": bool(reasons), "stop_reasons": reasons, "charged_seconds": charged,
            "remaining_seconds": remaining, "attempts": attempts, "head": copy.deepcopy(head),
            "completed_activities": sorted({a["activity_id"] for a in state["attempts"].values() if a["outcome"] == "completed"}),
            "authority_sha256": state["authority_sha256"], "phases": copy.deepcopy(state["phases"]),
            "dispatch_denials": list(state["dispatch_denials"])}


def inspect_ledger(path, *, authority=None, expected_head=None, wall_clock=None, monotonic_clock=None):
    """Read-only recovery evidence. Never closes, repairs, or resumes an attempt."""
    state, head = _read_journal(path)
    if authority is not None:
        _require(state["authority"] == authority, "budget authority changed")
    if expected_head is not None:
        _require(all(expected_head.get(k) == v for k, v in head.items()), "trusted budget head differs: possible tail truncation or stale receipt")
        pending = expected_head.get("unsettled_finalization")
        _require((pending is not None) == (state["pending_finalization_id"] is not None), "trusted head omitted or invented the final journal-write charge")
        if pending is not None:
            _require(pending["attempt_id"] == state["pending_finalization_id"], "final journal-write charge has the wrong identity")
            _apply_observation(state, pending, allow_closed=True)
            head["unsettled_finalization"] = copy.deepcopy(pending)
    now = _sample(wall_clock, monotonic_clock)
    open_attempts = []
    conservative = copy.deepcopy(state)
    for attempt_id, attempt in state["attempts"].items():
        if attempt["outcome"] != "open":
            continue
        delta_wall = now["wall_ns"] - attempt["last_sample"]["wall_ns"]
        delta_mono = now["monotonic_ns"] - attempt["last_sample"]["monotonic_ns"]
        elapsed = max(0, delta_wall, delta_mono)
        _charge(conservative, conservative["attempts"][attempt_id], elapsed)
        open_attempts.append({"attempt_id": attempt_id, "recorded_charged_seconds": attempt["charged_ns"] / NS,
                              "additional_conservative_seconds_through_inspection": elapsed / NS,
                              "chargeable_seconds_through_inspection": (attempt["charged_ns"] + elapsed) / NS,
                              "last_recorded_sample": copy.deepcopy(attempt["last_sample"]), "inspection_sample": dict(now),
                              "clock_rollback_or_restart_possible": delta_wall < 0 or delta_mono < 0})
    result = _public_snapshot(state, head)
    result.update(integrity_valid=True, trusted_head_verified=expected_head is not None,
                  open_attempts=open_attempts, stale_writer_lock=(Path(path) / ".writer_lock").exists(),
                  unfinished_close=state["unfinished_close"], pending_finalization_requires_trusted_head=state["pending_finalization_id"] is not None and expected_head is None,
                  conservative_chargeable_seconds_through_inspection=_public_snapshot(conservative, head)["charged_seconds"],
                  recovery_limitation="An open attempt's actual process-exit time is unrecorded. Inspection conservatively charges through now; external evidence and a new reviewed record are required. No clock is reset and no journal is repaired.")
    if open_attempts or result["stale_writer_lock"] or result["unfinished_close"] or result["pending_finalization_requires_trusted_head"]:
        result["must_stop"] = True
        result["stop_reasons"] += (["unclosed_attempt_requires_review"] if open_attempts else []) + (["stale_writer_lock"] if result["stale_writer_lock"] else [])
        result["stop_reasons"] += (["unfinished_close_requires_review"] if result["unfinished_close"] else []) + (["finalization_requires_trusted_head"] if result["pending_finalization_requires_trusted_head"] else [])
    return result


class BudgetLedger:
    """One owner writes; a trusted external head is mandatory on reopen."""

    @classmethod
    def create(cls, path, *, authority, heartbeat_seconds=30, monitor_gap_seconds=5,
               clock_skew_tolerance_seconds=2, wall_clock=None, monotonic_clock=None):
        path = Path(path)
        _require(isinstance(authority, dict) and authority, "a bound approval/proposal/scope authority is required")
        _canonical(authority)
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise BudgetWriteConflict("budget ledger already exists; never overwrite or reset it") from exc
        instance = cls._instance(path, {}, {"event_count": 0, "sha256": None, "journal_bytes": 0}, wall_clock, monotonic_clock)
        descriptor = os.open(path / "budget_events.jsonl", os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600)
        os.close(descriptor)
        instance._signature = instance._stat_signature()
        instance._append("genesis", {"authority": copy.deepcopy(authority), "authority_sha256": canonical_hash(authority),
                                      "caps_seconds": dict(CAPS_SECONDS),
                                      "policy": {"heartbeat_ns": _seconds_ns(heartbeat_seconds), "monitor_gap_ns": _seconds_ns(monitor_gap_seconds), "clock_skew_tolerance_ns": _seconds_ns(clock_skew_tolerance_seconds)}})
        return instance

    @classmethod
    def open(cls, path, *, authority, expected_head, wall_clock=None, monotonic_clock=None):
        _require(isinstance(expected_head, dict), "reopen requires an externally preserved trusted journal head")
        inspection = inspect_ledger(path, authority=authority, expected_head=expected_head,
                                    wall_clock=wall_clock, monotonic_clock=monotonic_clock)
        if inspection["open_attempts"] or inspection["stale_writer_lock"] or inspection["unfinished_close"]:
            raise BudgetOpenAttemptError("open attempt or stale writer lock requires external review; automatic recovery is forbidden", inspection=inspection)
        state, head = _read_journal(path)
        instance = cls._instance(Path(path), state, head, wall_clock, monotonic_clock)
        instance._signature = instance._stat_signature()
        pending = expected_head.get("unsettled_finalization")
        if pending is not None:
            instance._append("finalization_charge", {"observations": [pending], "settle_external": True})
        return instance

    @classmethod
    def _instance(cls, path, state, head, wall_clock, monotonic_clock):
        instance = cls.__new__(cls)
        instance.path, instance._state, instance._head = path, state, head
        instance.wall_clock, instance.monotonic_clock = wall_clock, monotonic_clock
        instance._guards, instance._pending_finalization = {}, None
        instance._signature = None
        return instance

    def _stat_signature(self):
        stat = (self.path / "budget_events.jsonl").stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    @property
    def head(self):
        value = dict(self._head)
        if self._pending_finalization is not None:
            value["unsettled_finalization"] = copy.deepcopy(self._pending_finalization)
        return value

    def _append(self, kind, payload):
        lock = self.path / ".writer_lock"
        try:
            lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise BudgetWriteConflict("exclusive budget writer lock exists; do not repair it automatically") from exc
        try:
            os.write(lock_fd, f"pid={os.getpid()}\n".encode())
            os.fsync(lock_fd)
            _require(self._stat_signature() == self._signature, "budget journal changed outside this owner", BudgetWriteConflict)
            proposed = copy.deepcopy(self._state)
            _apply_event(proposed, kind, payload)
            event = {"schema_version": "1.0", "sequence": self._head["event_count"], "previous_sha256": self._head["sha256"], "kind": kind, "payload": copy.deepcopy(payload)}
            digest = canonical_hash(event)
            line = _canonical({**event, "sha256": digest}) + b"\n"
            fd = os.open(self.path / "budget_events.jsonl", os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0))
            try:
                written = os.write(fd, line)
                _require(written == len(line), "partial budget append; preserve journal for review", BudgetWriteConflict)
                os.fsync(fd)
            finally:
                os.close(fd)
            self._state = proposed
            self._head = {"event_count": self._head["event_count"] + 1, "sha256": digest, "journal_bytes": self._head["journal_bytes"] + len(line)}
            self._signature = self._stat_signature()
        finally:
            os.close(lock_fd)
            lock.unlink()

    def _settle_pending(self):
        if self._pending_finalization is not None:
            pending = self._pending_finalization
            self._append("finalization_charge", {"observations": [pending], "settle_external": True})
            self._pending_finalization = None

    def register_phase(self, phase_id, *, provenance, gate_receipt):
        self._settle_pending()
        _validate_provenance(provenance, gate_receipt)
        payload = {"phase_id": phase_id, "provenance": copy.deepcopy(provenance), "gate_receipt": copy.deepcopy(gate_receipt)}
        if phase_id in self._state["phases"]:
            _require(self._state["phases"][phase_id] == payload, "registered phase provenance/gate cannot change")
            return self.snapshot()
        self._append("phase_registered", payload)
        return self.snapshot()

    def begin_attempt(self, attempt_id, *, activity_id, phase_id, budget_group,
                      estimated_seconds, unit_id=None, batch_id=None, started_at=None):
        self._settle_pending()
        if self._guards:
            self.poll(force=True)
        descriptor = {"activity_id": activity_id, "phase_id": phase_id, "budget_group": budget_group,
                      "unit_id": unit_id, "batch_id": batch_id, "estimate_ns": _seconds_ns(estimated_seconds)}
        guard = DualClockGuard(wall_clock=self.wall_clock, monotonic_clock=self.monotonic_clock,
                               monitor_gap_seconds=self._state["policy"]["monitor_gap_ns"] / NS,
                               clock_skew_tolerance_seconds=self._state["policy"]["clock_skew_tolerance_ns"] / NS,
                               started_at=started_at)
        payload = {"attempt_id": attempt_id, "descriptor": descriptor, "started_at": copy.deepcopy(guard.last)}
        try:
            _validate_start(self._state, payload)
        except BudgetAdmissionError as exc:
            self._append("dispatch_denied", {"attempt_id": attempt_id, "descriptor": descriptor, "reason": str(exc)})
            raise BudgetAdmissionError(str(exc), inspection=self.snapshot()) from exc
        self._append("attempt_started", payload)
        self._guards[attempt_id] = guard
        return self.poll(force=True)

    def _projected_state(self):
        projected = copy.deepcopy(self._state)
        if self._pending_finalization is not None:
            _apply_observation(projected, self._pending_finalization, allow_closed=True)
        for attempt_id, guard in self._guards.items():
            if guard.stats["sample_count"]:
                _apply_observation(projected, guard.observation(attempt_id))
        return projected

    def snapshot(self):
        """Latest sampled state. Call poll() for a fresh live clock reading."""
        return _public_snapshot(self._projected_state(), self.head)

    def poll(self, *, force=False):
        self._settle_pending()
        observations = []
        for attempt_id, guard in self._guards.items():
            guard.sample()
            observations.append(guard.observation(attempt_id))
        projected = self._projected_state()
        due = any(o["delta_charge_ns"] >= self._state["policy"]["heartbeat_ns"] for o in observations)
        if observations and (force or due or _stop_reasons(projected)):
            self._append("heartbeat", {"observations": observations})
            for guard in self._guards.values():
                guard.commit()
        return self.snapshot()

    def close_attempt(self, attempt_id, *, outcome, record_sha256=None, reason=None):
        _require(attempt_id in self._guards, "attempt is not owned/open in this interpreter")
        self.poll(force=True)
        guard = self._guards[attempt_id]
        self._append("attempt_closed", {"attempt_id": attempt_id, "outcome": outcome, "record_sha256": record_sha256, "reason": reason})
        # Charge the first close event's fsync. Carry the final settlement write
        # in the externally bound head, avoiding infinite self-measurement.
        guard.sample()
        self._append("finalization_charge", {"observations": [guard.observation(attempt_id)], "settle_external": False})
        guard.commit()
        guard.sample()
        residual = guard.observation(attempt_id)
        self._pending_finalization = residual
        del self._guards[attempt_id]
        result = self.snapshot()
        result["final_journal_write_residual_seconds"] = residual["delta_charge_ns"] / NS
        result["final_journal_write_accounting"] = "Included conservatively in this snapshot and trusted head; settled before subsequent work or on trusted-head reopen. Paused time is not included."
        return result

    def review_external_interruption(self, activity_id, *, provenance, review_receipt):
        self._settle_pending()
        _require(not self._guards, "stop all owned activities before reviewing an interruption")
        _require(activity_id in self._state["activities"], "unknown interrupted activity")
        attempt_id = self._state["activities"][activity_id]["attempt_ids"][-1]
        self._append("external_interruption_reviewed", {"attempt_id": attempt_id, "provenance": copy.deepcopy(provenance), "review_receipt": copy.deepcopy(review_receipt)})
        return self.snapshot()
