"""Deterministic synthetic lifecycle/clock checks; no research or real sleeps."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import input_robustness_budget as budget


class FakeClock:
    def __init__(self):
        self.wall = 1_000_000.0
        self.monotonic = 1_000.0

    def wall_now(self):
        return self.wall

    def mono_now(self):
        return self.monotonic

    def advance(self, wall, monotonic=None):
        self.wall += wall
        self.monotonic += wall if monotonic is None else monotonic


def provenance(commit="a" * 40):
    return {"source_commit": commit, "config_sha256": "b" * 64,
            "data_binding_sha256": "c" * 64, "source_files": {"runner.py": "d" * 64},
            "environment": {"python": "test", "device": "cpu", "threads": 4}}


def gate(commit="a" * 40):
    return {"ci_receipt_file_sha256": "e" * 64, "review_record_sha256": "f" * 64,
            "ci_receipt": {"commit": commit, "run_id": 17, "status": "completed", "conclusion": "success",
                           "jobs": [{"name": name, "run_id": 17, "status": "completed", "conclusion": "success"}
                                    for name in sorted(budget.CI_JOBS)]}}


REVIEW = {"all_existing_records_validated": True, "external_interruption_cause_confirmed": True,
          "no_unresolved_failure_artifacts": True, "review_record_sha256": "1" * 64}
AUTHORITY = {"approval_sha256": "2" * 64, "proposal_sha256": "3" * 64, "scope_sha256": "4" * 64}


class InputRobustnessBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "ledger"
        self.clock = FakeClock()

    def ledger(self, *, long_intervals=False):
        value = budget.BudgetLedger.create(
            self.path, authority=AUTHORITY, wall_clock=self.clock.wall_now,
            monotonic_clock=self.clock.mono_now,
            monitor_gap_seconds=10_000_000 if long_intervals else 5)
        value.register_phase("resource_scheme_v2", provenance=provenance(), gate_receipt=gate())
        return value

    def begin(self, ledger, attempt="attempt000", activity="resource_acceptance_v2", group="resource",
              estimate=1, unit=None, batch=None, phase="resource_scheme_v2", **kwargs):
        return ledger.begin_attempt(attempt, activity_id=activity, phase_id=phase,
                                    budget_group=group, estimated_seconds=estimate,
                                    unit_id=unit, batch_id=batch, **kwargs)

    def finish(self, ledger, attempt="attempt000"):
        return ledger.close_attempt(attempt, outcome="completed", record_sha256="5" * 64)

    def test_dual_clock_charges_suspend_and_never_refunds_rollback(self):
        guard = budget.DualClockGuard(wall_clock=self.clock.wall_now, monotonic_clock=self.clock.mono_now)
        self.clock.advance(2)
        self.assertEqual(guard.sample()["charged_seconds"], 2)
        self.clock.advance(-10, 1)
        observed = guard.sample()
        self.assertEqual(observed["charged_seconds"], 3)
        self.assertIn("wall_clock_rollback", observed["stop_reasons"])
        self.clock.advance(100, 1)
        observed = guard.sample()
        self.assertEqual(observed["charged_seconds"], 103)
        self.assertIn("wall_monotonic_divergence", observed["stop_reasons"])
        self.assertIn("monitor_gap", observed["stop_reasons"])
        self.clock.advance(-200, -2)
        observed = guard.sample()
        self.assertEqual(observed["charged_seconds"], 103)
        self.assertIn("monotonic_clock_rollback", observed["stop_reasons"])

    def test_long_equal_clock_gap_stops_without_claiming_clock_divergence(self):
        guard = budget.DualClockGuard(wall_clock=self.clock.wall_now, monotonic_clock=self.clock.mono_now)
        self.clock.advance(6)
        observed = guard.sample()
        self.assertEqual(observed["charged_seconds"], 6)
        self.assertEqual(observed["stop_reasons"], ["monitor_gap"])
        self.clock.wall = float("nan")
        with self.assertRaises(budget.BudgetError):
            guard.sample()

    def test_frequent_polls_use_one_journal_and_bounded_heartbeat_records(self):
        ledger = self.ledger()
        self.begin(ledger)
        initial = ledger.head["event_count"]
        for _ in range(120):
            self.clock.advance(0.25)
            self.assertFalse(ledger.poll()["must_stop"])
        self.assertEqual(ledger.head["event_count"], initial + 1)
        finished = self.finish(ledger)
        self.assertEqual(finished["charged_seconds"]["resource"], 30)
        self.assertEqual(finished["charged_seconds"]["total"], 30)
        self.assertEqual({p.name for p in self.path.iterdir()}, {"budget_events.jsonl"})
        checked = budget.inspect_ledger(self.path, authority=AUTHORITY, expected_head=ledger.head,
                                         wall_clock=self.clock.wall_now, monotonic_clock=self.clock.mono_now)
        self.assertEqual(checked["charged_seconds"], finished["charged_seconds"])
        self.assertFalse(checked["must_stop"])

    def test_started_at_covers_bootstrap_and_rejects_future_start(self):
        started = {"wall": self.clock.wall, "monotonic": self.clock.monotonic}
        ledger = self.ledger()
        self.clock.advance(2)
        first = self.begin(ledger, started_at=started)
        self.assertEqual(first["charged_seconds"]["resource"], 2)
        self.clock.advance(1)
        self.assertEqual(self.finish(ledger)["charged_seconds"]["resource"], 3)
        before = ledger.head
        with self.assertRaisesRegex(budget.BudgetAdmissionError, "later"):
            self.begin(ledger, attempt="future", activity="future", started_at={"wall": self.clock.wall + 1, "monotonic": self.clock.monotonic})
        # Settling the prior final journal-write receipt is permitted, but no
        # future attempt appears and no elapsed charge is removed.
        self.assertNotIn("future", ledger.snapshot()["attempts"])
        self.assertGreaterEqual(ledger.head["event_count"], before["event_count"])

    def test_resource_overshoot_is_preserved_and_stops_dispatch(self):
        ledger = self.ledger()
        self.begin(ledger, estimate=1328)
        self.clock.advance(7300, 1)
        stopped = ledger.poll()
        self.assertEqual(stopped["charged_seconds"]["resource"], 7300)
        self.assertEqual(stopped["remaining_seconds"]["resource"], -100)
        self.assertIn("resource_time_cap", stopped["stop_reasons"])
        self.assertTrue(any(reason.startswith("monitor_gap:") for reason in stopped["stop_reasons"]))
        self.clock.advance(3)
        closed = ledger.close_attempt("attempt000", outcome="external_interruption", reason="observed suspend/monitor gap")
        self.assertEqual(closed["charged_seconds"]["resource"], 7303)
        self.assertEqual(closed["remaining_seconds"]["resource"], -103)
        with self.assertRaises(budget.BudgetAdmissionError):
            self.begin(ledger, attempt="forbidden", activity="another_phase")

    def test_formal_and_control_overlap_charge_separately_with_shared_batch(self):
        ledger = self.ledger()
        self.begin(ledger, group="formal", activity="unit_a", unit="unit_a", batch="pair")
        self.clock.advance(2)
        ledger.poll()
        self.begin(ledger, attempt="control", group="control", activity="control_a", batch="pair")
        self.clock.advance(3)
        observed = ledger.poll()
        self.assertEqual(observed["charged_seconds"]["formal"], 5)
        self.assertEqual(observed["charged_seconds"]["control"], 3)
        self.assertEqual(observed["charged_seconds"]["total"], 8)
        self.assertEqual(observed["charged_seconds"]["units"], {"unit_a": 5})
        self.assertEqual(observed["charged_seconds"]["batches"], {"pair": 8})
        self.finish(ledger, "control")
        self.finish(ledger)

    def test_two_formal_workers_cannot_overlap(self):
        ledger = self.ledger()
        self.begin(ledger, group="formal", activity="unit_a", unit="unit_a", batch="pair")
        with self.assertRaisesRegex(budget.BudgetAdmissionError, "overlap"):
            self.begin(ledger, attempt="second", group="formal", activity="unit_b", unit="unit_b", batch="pair")
        self.assertNotIn("second", ledger.snapshot()["attempts"])
        self.finish(ledger)

    def test_reviewed_external_restart_accumulates_same_unit_and_never_resets(self):
        ledger = self.ledger()
        self.begin(ledger, group="formal", activity="unit_a", unit="unit_a", batch="pair", estimate=10)
        for _ in range(4):
            self.clock.advance(2)
            ledger.poll()
        ledger.close_attempt("attempt000", outcome="external_interruption", reason="emergency user interruption")
        with self.assertRaises(budget.BudgetAdmissionError):
            self.begin(ledger, attempt="attempt001", group="formal", activity="unit_a", unit="unit_a", batch="pair", estimate=10)
        ledger.review_external_interruption("unit_a", provenance=provenance(), review_receipt=REVIEW)
        self.begin(ledger, attempt="attempt001", group="formal", activity="unit_a", unit="unit_a", batch="pair", estimate=10)
        for _ in range(3):
            self.clock.advance(2)
            ledger.poll()
        closed = self.finish(ledger, "attempt001")
        self.assertEqual(closed["charged_seconds"]["units"]["unit_a"], 14)
        self.assertEqual(closed["charged_seconds"]["batches"]["pair"], 14)
        self.assertEqual(closed["charged_seconds"]["formal"], 14)
        self.assertEqual(closed["attempts"]["attempt000"]["charged_seconds"], 8)
        self.assertEqual(closed["attempts"]["attempt001"]["charged_seconds"], 6)
        self.assertEqual(closed["completed_activities"], ["unit_a"])
        with self.assertRaises(budget.BudgetAdmissionError):
            self.begin(ledger, attempt="attempt002", group="formal", activity="unit_a", unit="unit_a", batch="pair", estimate=10)
        with self.assertRaises(budget.BudgetAdmissionError):
            self.begin(ledger, attempt="renamed", group="formal", activity="renamed_unit", unit="unit_a", batch="pair")

    def test_unit_cumulative_cap_includes_all_attempts_and_actual_overshoot(self):
        ledger = self.ledger(long_intervals=True)
        self.begin(ledger, group="formal", activity="unit_a", unit="unit_a", batch="pair")
        self.clock.advance(17000)
        ledger.poll()
        ledger.close_attempt("attempt000", outcome="external_interruption", reason="reviewed emergency")
        ledger.review_external_interruption("unit_a", provenance=provenance(), review_receipt=REVIEW)
        self.begin(ledger, attempt="attempt001", group="formal", activity="unit_a", unit="unit_a", batch="pair")
        self.clock.advance(12000)
        observed = ledger.poll()
        self.assertEqual(observed["charged_seconds"]["units"]["unit_a"], 29000)
        self.assertEqual(observed["remaining_seconds"]["units"]["unit_a"], -200)
        self.assertIn("unit_time_cap:unit_a", observed["stop_reasons"])
        closed = ledger.close_attempt("attempt001", outcome="failed", reason="unit cumulative cap")
        self.assertEqual(closed["charged_seconds"]["formal"], 29000)

    def test_paired_batch_headroom_includes_both_units_and_control(self):
        ledger = self.ledger(long_intervals=True)
        for number in range(2):
            self.begin(ledger, attempt=f"a{number}", activity=f"unit{number}", group="formal", unit=f"unit{number}", batch="pair")
            self.clock.advance(28799)
            ledger.poll()
            self.finish(ledger, f"a{number}")
        self.assertEqual(ledger.snapshot()["remaining_seconds"]["batches"]["pair"], 2)
        with self.assertRaisesRegex(budget.BudgetAdmissionError, "batch:pair"):
            self.begin(ledger, attempt="control", activity="pair_control", group="control", batch="pair", estimate=3)

    def test_subbudget_cannot_borrow_unused_reserve(self):
        ledger = self.ledger(long_intervals=True)
        self.begin(ledger)
        self.clock.advance(7199)
        ledger.poll()
        self.finish(ledger)
        self.assertEqual(ledger.snapshot()["remaining_seconds"]["control"], 7200)
        with self.assertRaisesRegex(budget.BudgetAdmissionError, "resource"):
            self.begin(ledger, attempt="next", activity="resource_second", estimate=2)
        self.assertEqual(ledger.snapshot()["charged_seconds"]["resource"], 7199)

    def test_formal_subbudget_and_total_caps_remain_independent(self):
        ledger = self.ledger(long_intervals=True)
        for number in range(48):
            self.begin(ledger, attempt=f"a{number}", activity=f"unit{number}", group="formal", unit=f"unit{number}", batch=f"pair{number}")
            self.clock.advance(28000)
            ledger.poll()
            self.finish(ledger, f"a{number}")
        self.assertEqual(ledger.snapshot()["remaining_seconds"]["formal"], 24000)
        with self.assertRaisesRegex(budget.BudgetAdmissionError, "formal"):
            self.begin(ledger, attempt="extra", activity="unit_extra", group="formal", unit="unit_extra", batch="pair_extra", estimate=24001)
        self.assertGreater(ledger.snapshot()["remaining_seconds"]["total"], 24000)

    def test_extreme_clock_jump_preserves_total_and_all_nested_overshoot(self):
        ledger = self.ledger(long_intervals=True)
        self.begin(ledger, group="formal", activity="unit_a", unit="unit_a", batch="pair")
        self.clock.advance(1400000)
        observed = ledger.poll()
        self.assertEqual(observed["charged_seconds"]["total"], 1400000)
        self.assertEqual(observed["remaining_seconds"]["total"], -17600)
        self.assertIn("total_time_cap", observed["stop_reasons"])
        self.assertIn("formal_time_cap", observed["stop_reasons"])
        self.assertIn("batch_time_cap:pair", observed["stop_reasons"])
        self.assertIn("unit_time_cap:unit_a", observed["stop_reasons"])

    def test_closed_pause_is_not_charged_and_new_reviewed_phase_may_change_source(self):
        ledger = self.ledger()
        self.begin(ledger)
        self.clock.advance(2)
        self.finish(ledger)
        head = ledger.head
        self.clock.advance(1_000_000)
        reopened = budget.BudgetLedger.open(self.path, authority=AUTHORITY, expected_head=head,
                                             wall_clock=self.clock.wall_now, monotonic_clock=self.clock.mono_now)
        self.assertEqual(reopened.snapshot()["charged_seconds"]["resource"], 2)
        reopened.register_phase("formal_record_phase", provenance=provenance("6" * 40), gate_receipt=gate("6" * 40))
        self.begin(reopened, attempt="formal_a", activity="unit_a", phase="formal_record_phase", group="formal", unit="unit_a", batch="pair")
        self.clock.advance(1)
        closed = self.finish(reopened, "formal_a")
        self.assertEqual(closed["charged_seconds"]["total"], 3)

    def test_phase_or_resume_provenance_cannot_change_silently(self):
        ledger = self.ledger()
        changed = provenance()
        changed["environment"]["threads"] = 8
        with self.assertRaisesRegex(budget.BudgetIntegrityError, "cannot change"):
            ledger.register_phase("resource_scheme_v2", provenance=changed, gate_receipt=gate())
        self.begin(ledger, group="formal", activity="unit_a", unit="unit_a", batch="pair")
        ledger.close_attempt("attempt000", outcome="external_interruption", reason="emergency")
        with self.assertRaisesRegex(budget.BudgetIntegrityError, "provenance differs"):
            ledger.review_external_interruption("unit_a", provenance=changed, review_receipt=REVIEW)
        ledger.review_external_interruption("unit_a", provenance=provenance(), review_receipt=REVIEW)
        ledger.register_phase("new_phase", provenance=provenance("6" * 40), gate_receipt=gate("6" * 40))
        with self.assertRaisesRegex(budget.BudgetAdmissionError, "provenance"):
            self.begin(ledger, attempt="changed", activity="unit_a", group="formal", unit="unit_a", batch="pair", phase="new_phase")

    def test_phase_gate_requires_real_same_source_ci_structure(self):
        ledger = self.ledger()
        for changed in ({"approved": True}, {**gate(), "ci_receipt": {**gate()["ci_receipt"], "commit": "7" * 40}},
                        {**gate(), "ci_receipt": {**gate()["ci_receipt"], "jobs": []}}):
            with self.subTest(gate=changed), self.assertRaises(budget.BudgetIntegrityError):
                ledger.register_phase("invalid", provenance=provenance(), gate_receipt=changed)
        self.assertNotIn("invalid", ledger.snapshot()["phases"])

    def test_open_attempt_refuses_reopen_and_exposes_conservative_review_charge(self):
        ledger = self.ledger(long_intervals=True)
        self.begin(ledger)
        self.clock.advance(10)
        ledger.poll(force=True)
        head = ledger.head
        before = (self.path / "budget_events.jsonl").read_bytes()
        self.clock.advance(20)
        with self.assertRaises(budget.BudgetOpenAttemptError) as caught:
            budget.BudgetLedger.open(self.path, authority=AUTHORITY, expected_head=head,
                                      wall_clock=self.clock.wall_now, monotonic_clock=self.clock.mono_now)
        self.assertEqual(caught.exception.inspection["open_attempts"][0]["chargeable_seconds_through_inspection"], 30)
        self.assertEqual(caught.exception.inspection["conservative_chargeable_seconds_through_inspection"]["resource"], 30)
        self.assertEqual((self.path / "budget_events.jsonl").read_bytes(), before)

    def test_corruption_truncation_sequence_gap_and_tail_loss_fail_closed(self):
        ledger = self.ledger()
        self.begin(ledger)
        self.clock.advance(1)
        self.finish(ledger)
        head = ledger.head
        journal = self.path / "budget_events.jsonl"
        original = journal.read_bytes()
        lines = original.splitlines(keepends=True)
        variants = (original.replace(b'"charged_ns"', b'"changed_ns"', 1) if b'"charged_ns"' in original else original.replace(b'"genesis"', b'"changed"', 1),
                    original[:-1], b"".join(lines[:1] + lines[2:]), b"".join(lines[:-1]))
        for damaged in variants:
            with self.subTest(length=len(damaged)):
                journal.write_bytes(damaged)
                with self.assertRaises(budget.BudgetError):
                    budget.BudgetLedger.open(self.path, authority=AUTHORITY, expected_head=head,
                                              wall_clock=self.clock.wall_now, monotonic_clock=self.clock.mono_now)
                self.assertEqual(journal.read_bytes(), damaged)
        journal.write_bytes(original)

    def test_exclusive_create_and_stale_writer_lock_do_not_overwrite(self):
        ledger = self.ledger()
        before = (self.path / "budget_events.jsonl").read_bytes()
        with self.assertRaises(budget.BudgetWriteConflict):
            budget.BudgetLedger.create(self.path, authority=AUTHORITY)
        lock = self.path / ".writer_lock"
        lock.write_text("previous owner", encoding="utf-8")
        with self.assertRaises(budget.BudgetWriteConflict):
            ledger.register_phase("second", provenance=provenance("6" * 40), gate_receipt=gate("6" * 40))
        self.assertEqual(lock.read_text(), "previous owner")
        self.assertEqual((self.path / "budget_events.jsonl").read_bytes(), before)

    def test_two_owners_cannot_silently_append_from_the_same_head(self):
        ledger = self.ledger()
        other = budget.BudgetLedger.open(self.path, authority=AUTHORITY, expected_head=ledger.head,
                                          wall_clock=self.clock.wall_now, monotonic_clock=self.clock.mono_now)
        ledger.register_phase("first_owner", provenance=provenance("6" * 40), gate_receipt=gate("6" * 40))
        before = (self.path / "budget_events.jsonl").read_bytes()
        with self.assertRaises(budget.BudgetWriteConflict):
            other.register_phase("stale_owner", provenance=provenance("7" * 40), gate_receipt=gate("7" * 40))
        self.assertEqual((self.path / "budget_events.jsonl").read_bytes(), before)

    def test_final_fsync_residual_is_bound_and_settled_without_charging_pause(self):
        actual_fsync = budget.os.fsync

        def timed_fsync(descriptor):
            actual_fsync(descriptor)
            self.clock.advance(0.05)

        with patch.object(budget.os, "fsync", side_effect=timed_fsync):
            ledger = self.ledger()
            self.begin(ledger)
            self.clock.advance(1)
            completed = self.finish(ledger)
            head = ledger.head
            charged = completed["charged_seconds"]["resource"]
            self.assertGreater(completed["final_journal_write_residual_seconds"], 0)
            self.assertIn("unsettled_finalization", head)
            self.clock.advance(1000)
            incomplete_head = {key: value for key, value in head.items() if key != "unsettled_finalization"}
            with self.assertRaisesRegex(budget.BudgetIntegrityError, "omitted"):
                budget.BudgetLedger.open(self.path, authority=AUTHORITY, expected_head=incomplete_head,
                                          wall_clock=self.clock.wall_now, monotonic_clock=self.clock.mono_now)
            reopened = budget.BudgetLedger.open(self.path, authority=AUTHORITY, expected_head=head,
                                                 wall_clock=self.clock.wall_now, monotonic_clock=self.clock.mono_now)
            self.assertAlmostEqual(reopened.snapshot()["charged_seconds"]["resource"], charged, places=8)
            self.assertNotIn("unsettled_finalization", reopened.head)


if __name__ == "__main__":
    unittest.main()
