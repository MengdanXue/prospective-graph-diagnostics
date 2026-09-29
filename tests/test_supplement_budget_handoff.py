"""Isolated fake-clock accounting tests; no research data or training."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import input_robustness_budget as budget
from scripts import supplement_budget_handoff as handoff
from tests.test_input_robustness_budget import FakeClock, AUTHORITY, REVIEW, gate, provenance


class BudgetHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.clock = FakeClock()

    def ledger(self, name="old"):
        value = budget.BudgetLedger.create(self.root / name, authority=AUTHORITY,
                                          wall_clock=self.clock.wall_now,
                                          monotonic_clock=self.clock.mono_now)
        value.register_phase("phase", provenance=provenance(), gate_receipt=gate())
        return value

    def run_attempt(self, ledger, *, attempt="attempt", unit="unit_Cora_normalize_features_MLP_000",
                    batch="pair_Cora_seed_000_MLP", group="formal", outcome="completed", seconds=2):
        ledger.begin_attempt(attempt, activity_id=attempt, phase_id="phase", budget_group=group,
                             estimated_seconds=1, unit_id=unit, batch_id=batch)
        self.clock.advance(seconds)
        ledger.poll(force=True)
        if outcome is not None:
            ledger.close_attempt(attempt, outcome=outcome,
                                 record_sha256="5" * 64 if outcome == "completed" else None,
                                 reason=None if outcome == "completed" else "interrupted in fixture")

    def receipt(self, ledger, name):
        path = self.root / name
        path.write_text(json.dumps({"head": ledger.head}), encoding="utf-8")
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def spec(self, ledgers):
        return {"schema_version": handoff.SCHEMA, "authority": AUTHORITY,
                "source_commit": "a" * 40, "config_sha256": "b" * 64,
                "inventory_roots": [str(l.path) for l in ledgers],
                "sources": [{"ledger_path": str(l.path),
                             "terminal_receipt": self.receipt(l, f"head-{i}.json")}
                            for i, l in enumerate(ledgers)]}

    def inspect(self, spec, **kwargs):
        return handoff.inspect_handoff(spec, base_root=self.root, process_probe=lambda: [],
                                       now_sample={"wall_ns": round(self.clock.wall * budget.NS),
                                                   "monotonic_ns": round(self.clock.monotonic * budget.NS)}, **kwargs)

    def test_streamed_replay_matches_original_with_finalization_and_never_read_bytes(self):
        ledger = self.ledger()
        self.run_attempt(ledger)
        original = budget.inspect_ledger(ledger.path, authority=AUTHORITY, expected_head=ledger.head,
                                         wall_clock=self.clock.wall_now, monotonic_clock=self.clock.mono_now)
        before = (ledger.path / "budget_events.jsonl").read_bytes()
        with patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file read forbidden")):
            result = handoff.stream_snapshot(ledger.path, authority=AUTHORITY, expected_head=ledger.head)
        self.assertEqual(handoff._usage_seconds(result["charged_ns"]), original["charged_seconds"])
        self.assertFalse(result["must_stop"])
        self.assertEqual((ledger.path / "budget_events.jsonl").read_bytes(), before)

    def test_historical_resource_anchor_is_verified_not_double_counted(self):
        ledger = self.ledger()
        self.run_attempt(ledger, attempt="resource", unit=None, batch=None, group="resource", seconds=2)
        anchor = self.receipt(ledger, "resource-head.json")
        self.run_attempt(ledger, attempt="formal", seconds=3)
        spec = self.spec([ledger])
        spec["sources"][0]["anchor_receipts"] = [anchor]
        result = self.inspect(spec)
        self.assertEqual(result["charged_seconds"]["total"], 5)
        self.assertEqual(result["charged_seconds"]["resource"], 2)
        self.assertEqual(len(result["sources"][0]["verified_anchors"]), 1)

    def test_missing_or_tampered_terminal_receipt_cannot_be_replaced_by_passed_flag(self):
        ledger = self.ledger()
        self.run_attempt(ledger)
        spec = self.spec([ledger])
        del spec["sources"][0]["terminal_receipt"]
        spec["passed"] = True
        spec["charged_seconds"] = {"total": 0}
        self.assertTrue(self.inspect(spec)["must_stop"])
        spec = self.spec([ledger])
        Path(spec["sources"][0]["terminal_receipt"]["path"]).write_text('{"head":{},"passed":true}')
        with self.assertRaisesRegex(handoff.HandoffError, "receipt hash"):
            self.inspect(spec)

    def test_truncated_valid_chain_rejected_by_trusted_terminal_head(self):
        ledger = self.ledger()
        self.run_attempt(ledger)
        trusted = ledger.head
        path = ledger.path / "budget_events.jsonl"
        path.write_bytes(b"\n".join(path.read_bytes().splitlines()[:-1]) + b"\n")
        with self.assertRaisesRegex(handoff.HandoffError, "head"):
            handoff.stream_snapshot(ledger.path, authority=AUTHORITY, expected_head=trusted)

    def test_duplicate_missing_and_extra_segments_are_rejected(self):
        ledger = self.ledger()
        spec = self.spec([ledger])
        spec["sources"].append(copy.deepcopy(spec["sources"][0]))
        with self.assertRaisesRegex(handoff.HandoffError, "exactly once"):
            self.inspect(spec)
        spec = self.spec([ledger])
        nested = budget.BudgetLedger.create(ledger.path / "segment_002", authority=AUTHORITY)
        with self.assertRaisesRegex(handoff.HandoffError, "exactly once"):
            self.inspect(spec)
        self.assertTrue((nested.path / "budget_events.jsonl").is_file())

    def test_unreviewed_failed_or_interrupted_attempts_remain_blocking(self):
        for outcome in ("failed", "external_interruption"):
            ledger = self.ledger(outcome)
            self.run_attempt(ledger, outcome=outcome)
            result = self.inspect(self.spec([ledger]))
            self.assertTrue(result["must_stop"])
            self.assertEqual(result["charged_seconds"]["formal"], 2)
            self.assertTrue(any("unreviewed_" in s for s in result["stop_reasons"]))

    def test_open_attempt_unknown_terminal_time_is_separate_from_recorded_use(self):
        ledger = self.ledger()
        self.run_attempt(ledger, outcome=None)
        self.clock.advance(100)
        result = self.inspect(self.spec([ledger]))
        self.assertEqual(result["charged_seconds"]["formal"], 2)
        self.assertEqual(result["conservative_seconds"]["formal"], 102)
        self.assertFalse(result["actual_terminal_use_known"])
        self.assertTrue(result["must_stop"])
        self.assertEqual(result["sources"][0]["open_attempts"][0]["outcome"], "open")

    def test_same_unit_pair_sum_across_segments_and_all_groups_retained(self):
        one, two = self.ledger("one"), self.ledger("two")
        self.run_attempt(one, seconds=2)
        self.run_attempt(two, seconds=3)
        self.run_attempt(one, attempt="resource", unit=None, batch=None, group="resource", seconds=1)
        self.run_attempt(two, attempt="control", unit=None, group="control", seconds=1)
        result = self.inspect(self.spec([one, two]))
        self.assertEqual(result["charged_seconds"]["formal"], 5)
        self.assertEqual(result["charged_seconds"]["resource"], 1)
        self.assertEqual(result["charged_seconds"]["control"], 1)
        self.assertEqual(result["charged_seconds"]["total"], 7)
        self.assertEqual(result["charged_seconds"]["units"]["unit_Cora_normalize_features_MLP_000"], 5)
        self.assertEqual(result["charged_seconds"]["batches"]["pair_Cora_seed_000_MLP"], 6)

    def test_reviewed_interruption_can_handoff_without_zeroing_the_attempt(self):
        ledger = self.ledger()
        self.run_attempt(ledger, outcome="external_interruption", seconds=3)
        ledger.review_external_interruption("attempt", provenance=provenance(), review_receipt=REVIEW)
        result = self.inspect(self.spec([ledger]))
        self.assertFalse(result["must_stop"])
        self.assertEqual(result["charged_seconds"]["formal"], 3)
        self.assertEqual(len(result["sources"][0]["attempts"]), 1)

    def test_live_process_blocks_handoff(self):
        ledger = self.ledger()
        spec = self.spec([ledger])
        result = handoff.inspect_handoff(spec, base_root=self.root, process_probe=lambda: [{"pid": 123}])
        self.assertTrue(result["must_stop"])
        self.assertIn("existing_research_processes_active", result["stop_reasons"])

    def test_large_snapshot_receipts_are_not_loaded(self):
        path = self.root / "huge.json"
        with path.open("wb") as stream:
            stream.seek(handoff.MAX_RECEIPT_BYTES + 10)
            stream.write(b" ")
        with patch.object(json, "load", side_effect=AssertionError("must not load huge snapshot")):
            with self.assertRaisesRegex(handoff.HandoffError, "too large"):
                handoff._read_small(path)

    def test_changed_journal_during_read_cannot_be_terminal(self):
        ledger = self.ledger()
        actual = handoff._signature(ledger.path / "budget_events.jsonl")
        changed = [*actual[:3], actual[3] + 1]
        with patch.object(handoff, "_signature", side_effect=[actual, actual, changed]):
            result = handoff.stream_snapshot(ledger.path, authority=AUTHORITY, expected_head=ledger.head)
        self.assertTrue(result["must_stop"])
        self.assertIn("journal_changed_during_inspection", result["stop_reasons"])

    def test_inherited_adapter_checks_each_poll_and_same_unit_admission(self):
        old = self.ledger("old")
        self.run_attempt(old)
        inherited = self.inspect(self.spec([old]))
        inherited["handoff_spec_sha256"] = "9" * 64
        unit, pair = "unit_Cora_normalize_features_MLP_000", "pair_Cora_seed_000_MLP"
        # Independent arithmetic boundary fixture; the already tested reader
        # supplies actual values in production, never caller-provided totals.
        inherited["charged_ns"]["units"][unit] = budget.CAPS_SECONDS["unit"] * budget.NS - 3 * budget.NS
        local = self.ledger("increment")
        lock = self.root / "owner.lock"
        lock.write_text("fixture")
        handoff._register_increment(self.root, local.path.resolve(), "9" * 64)
        adapter = handoff.InheritedLedger(local, inherited, lock, process_probe=lambda: [])
        with self.assertRaisesRegex(handoff.HandoffError, "cumulative admission"):
            adapter.begin_attempt("new", activity_id="new", phase_id="phase", budget_group="formal",
                                  estimated_seconds=4, unit_id=unit, batch_id=pair)
        adapter.begin_attempt("new", activity_id="new", phase_id="phase", budget_group="formal",
                              estimated_seconds=2, unit_id=unit, batch_id=pair)
        self.clock.advance(4)
        observed = adapter.poll(force=True)
        self.assertTrue(observed["must_stop"])
        self.assertEqual(observed["remaining_seconds"]["units"][unit], -1)
        with self.assertRaisesRegex(handoff.HandoffError, "cannot claim completion"):
            adapter.close_attempt("new", outcome="completed", record_sha256="5" * 64)
        adapter.close_attempt("new", outcome="external_interruption", reason="fixture limit")
        self.assertTrue(adapter.snapshot()["must_stop"])

    def test_open_adapter_source_binding_lock_and_immutable_segment(self):
        old = self.ledger()
        self.run_attempt(old)
        specfile = self.root / "handoff.json"
        specfile.write_text(json.dumps(self.spec([old])))
        target = self.root / "new-increment"
        with self.assertRaisesRegex(handoff.HandoffError, "differs"):
            handoff.open_inherited_ledger(specfile, target, base_root=self.root,
                                         expected_config_sha="d" * 64, expected_source_commit="a" * 40,
                                         process_probe=lambda: [])
        self.assertFalse(target.exists())
        adapter = handoff.open_inherited_ledger(specfile, target, base_root=self.root,
                                               expected_config_sha="b" * 64, expected_source_commit="a" * 40,
                                               process_probe=lambda: [], required_inventory_roots=["old"])
        self.assertEqual(adapter.snapshot()["charged_seconds"]["formal"], 2)
        with self.assertRaisesRegex(handoff.HandoffError, "owner lock"):
            handoff.open_inherited_ledger(specfile, self.root / "second", base_root=self.root,
                                         expected_config_sha="b" * 64, expected_source_commit="a" * 40,
                                         process_probe=lambda: [])
        adapter.close()
        self.assertTrue((target / "terminal_receipt.json").is_file())
        with self.assertRaisesRegex(handoff.HandoffError, "incremental index head"):
            handoff.open_inherited_ledger(specfile, target, base_root=self.root,
                                         expected_config_sha="b" * 64, expected_source_commit="a" * 40,
                                         process_probe=lambda: [], required_inventory_roots=["old"])

    def test_reviewed_same_increment_reopen_keeps_all_attempt_unit_and_pair_charges(self):
        old = self.ledger()
        self.run_attempt(old, seconds=2)
        spec = self.spec([old])
        specfile = self.root / "handoff.json"
        specfile.write_text(json.dumps(spec))
        target = self.root / "increment"
        adapter = handoff.open_inherited_ledger(specfile, target, base_root=self.root,
                                               expected_config_sha="b" * 64, expected_source_commit="a" * 40,
                                               process_probe=lambda: [], required_inventory_roots=["old"])
        adapter._ledger.wall_clock = self.clock.wall_now
        adapter._ledger.monotonic_clock = self.clock.mono_now
        adapter.register_phase("phase", provenance=provenance(), gate_receipt=gate())
        unit, pair = "unit_Cora_normalize_features_MLP_000", "pair_Cora_seed_000_MLP"
        params = dict(activity_id="added-grid", phase_id="phase", budget_group="formal",
                      estimated_seconds=1, unit_id=unit, batch_id=pair)
        adapter.begin_attempt("additional-1", **params)
        self.clock.advance(3)
        adapter.close_attempt("additional-1", outcome="external_interruption", reason="fixture emergency")
        self.assertTrue(adapter.snapshot()["must_stop"])
        adapter.review_external_interruption("added-grid", provenance=provenance(), review_receipt=REVIEW)
        adapter.close()
        spec["sources"].append({"ledger_path": str(target), "terminal_receipt": {
            "path": str(target / "terminal_receipt.json"),
            "sha256": handoff._file_hash(target / "terminal_receipt.json")}})
        spec["increment_index_head"] = handoff._increment_index(self.root)[1]
        specfile = self.root / "resume-handoff.json"
        specfile.write_text(json.dumps(spec))
        resumed = handoff.open_inherited_ledger(specfile, target, base_root=self.root,
                                               expected_config_sha="b" * 64, expected_source_commit="a" * 40,
                                               process_probe=lambda: [], resume=True,
                                               required_inventory_roots=["old"])
        resumed._ledger.wall_clock = self.clock.wall_now
        resumed._ledger.monotonic_clock = self.clock.mono_now
        self.assertEqual(resumed.snapshot()["charged_seconds"]["units"][unit], 5)
        resumed.begin_attempt("additional-2", **params)
        self.clock.advance(4)
        result = resumed.close_attempt("additional-2", outcome="completed", record_sha256="5" * 64)
        self.assertEqual(result["charged_seconds"]["units"][unit], 9)
        self.assertEqual(result["charged_seconds"]["batches"][pair], 9)
        self.assertEqual(result["charged_seconds"]["formal"], 9)
        resumed.close()
        self.assertEqual(len(list(target.glob("terminal_receipt*.json"))), 2)

    def test_every_cumulative_bucket_and_pair_cap_checked_during_projection(self):
        old = self.ledger("prior")
        original = self.inspect(self.spec([old]))
        for index, cap in enumerate(("total", "resource", "control", "formal", "batch")):
            inherited = copy.deepcopy(original)
            inherited["handoff_spec_sha256"] = "9" * 64
            local = self.ledger(f"increment-{index}")
            lock = self.root / f"owner-{index}.lock"
            lock.write_text("fixture")
            handoff._register_increment(self.root, local.path.resolve(), "9" * 64)
            # Only this adapter is running in the boundary fixture. Earlier
            # fixture increments are retained in the observed immutable set.
            for previous_index in range(index):
                prior_path = self.root / f"increment-{previous_index}"
                inherited["sources"].append({"ledger_path": str(prior_path),
                    "signature": handoff._signature(prior_path / "budget_events.jsonl"),
                    "head": budget._read_journal(prior_path)[1]})
            group = cap if cap in ("resource", "control", "formal") else "formal"
            unit = "unit_Cora_normalize_features_MLP_000" if group == "formal" else None
            pair = None if group == "resource" else "pair_Cora_seed_000_MLP"
            if cap == "batch":
                inherited["charged_ns"]["batches"][pair] = (budget.CAPS_SECONDS[cap] - 1) * budget.NS
            else:
                inherited["charged_ns"][cap] = (budget.CAPS_SECONDS[cap] - 1) * budget.NS
            adapter = handoff.InheritedLedger(local, inherited, lock, process_probe=lambda: [])
            adapter.begin_attempt("new", activity_id="new", phase_id="phase", budget_group=group,
                                  estimated_seconds=0.5, unit_id=unit, batch_id=pair)
            self.clock.advance(1)
            result = adapter.poll()  # Under 30-second durable-heartbeat interval: projected use counts.
            with self.subTest(cap=cap):
                self.assertTrue(result["must_stop"])
                self.assertTrue(any(reason.startswith(f"{cap}_time_cap") for reason in result["stop_reasons"]))

    def test_config_bound_inventory_cannot_omit_old_root(self):
        ledger = self.ledger()
        spec = self.spec([ledger])
        path = self.root / "spec.json"
        path.write_text(json.dumps(spec))
        with self.assertRaisesRegex(handoff.HandoffError, "authoritative configuration"):
            handoff.validate_handoff(path, base_root=self.root, expected_config_sha="b" * 64,
                                     expected_source_commit="a" * 40, process_probe=lambda: [],
                                     required_inventory_roots=["old", "missing-prior-run"])

    def test_copy_of_same_attempt_chain_is_rejected_not_charged_twice(self):
        one = self.ledger("one")
        self.run_attempt(one)
        two = self.ledger("two")
        (two.path / "budget_events.jsonl").write_bytes((one.path / "budget_events.jsonl").read_bytes())
        two._head = dict(one._head)
        two._pending_finalization = copy.deepcopy(one._pending_finalization)
        with self.assertRaisesRegex(handoff.HandoffError, "overlapping copied"):
            self.inspect(self.spec([one, two]))

    def test_active_increment_index_tail_loss_stops_live_poll(self):
        old = self.ledger()
        specfile = self.root / "handoff.json"
        specfile.write_text(json.dumps(self.spec([old])))
        adapter = handoff.open_inherited_ledger(specfile, self.root / "increment", base_root=self.root,
                                               expected_config_sha="b" * 64, expected_source_commit="a" * 40,
                                               process_probe=lambda: [], required_inventory_roots=["old"])
        self.assertFalse(adapter.snapshot()["must_stop"])
        (self.root / handoff.INCREMENT_INDEX).write_bytes(b"")
        stopped = adapter.poll()
        self.assertTrue(stopped["must_stop"])
        self.assertIn("increment_index_head_changed", stopped["stop_reasons"])

    def test_config_inventory_cannot_be_left_to_self_reported_spec(self):
        old = self.ledger()
        specfile = self.root / "handoff.json"
        specfile.write_text(json.dumps(self.spec([old])))
        with self.assertRaisesRegex(handoff.HandoffError, "configuration inventory is required"):
            handoff.validate_handoff(specfile, base_root=self.root, expected_config_sha="b" * 64,
                                     expected_source_commit="a" * 40, process_probe=lambda: [])


if __name__ == "__main__":
    unittest.main()
