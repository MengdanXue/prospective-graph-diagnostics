"""Reviewed environmental-failure retry: history, current blockers and one-time consumption.

Isolated fake-clock ledgers only. No research data, worker or training is used.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import input_robustness_budget as budget
from scripts import supplement_budget_handoff as handoff
from scripts import terminal_failure_review as review
from tests.test_input_robustness_budget import AUTHORITY, FakeClock, gate, provenance

ATTEMPT = "mlp24_2_000000"
UNIT, BATCH = "unit_Cora_normalize_features_MLP_000", "pair_Cora_seed_000_MLP"
ACTIVITY = f"mlp24_{UNIT}"
IDENTITY = {"dataset": "Cora", "condition": "normalize_features", "model": "MLP", "seed": 0,
            "trials": "trial_000-trial_023"}
ENVIRONMENT = {"python": "test", "device": "cpu", "threads": 4}
FINDINGS = {"research_computation_started": False, "checkpoints": 0, "records": 0, "test_evaluations": 0,
            "test_once_consumed": False, "blocking_stop_reasons": []}


def disposition(allowed=True):
    return {"retry_allowed": allowed, "reason": "wrong interpreter before research computation",
            "max_retries": 1, "must_be_new_segment": True, "new_attempt_id_required": True,
            "retry_scope": {"unit_id": UNIT, "batch_id": BATCH, "activity_id": ACTIVITY,
                            "research_identity": IDENTITY},
            "required_environment": dict(ENVIRONMENT)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FailureReviewFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.clock = FakeClock()
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()

    def ledger(self, name):
        value = budget.BudgetLedger.create(self.root / name, authority=AUTHORITY, wall_clock=self.clock.wall_now,
                                          monotonic_clock=self.clock.mono_now)
        value.register_phase("mlp24_extension", provenance=provenance(), gate_receipt=gate())
        return value

    def fail(self, ledger, attempt=ATTEMPT, unit=UNIT, batch=BATCH, seconds=5):
        ledger.begin_attempt(attempt, activity_id=f"mlp24_{unit}", phase_id="mlp24_extension",
                             budget_group="formal", estimated_seconds=1, unit_id=unit, batch_id=batch)
        self.clock.advance(seconds)
        ledger.poll(force=True)
        ledger.close_attempt(attempt, outcome="failed", reason="FormalRecordError('worker failed')")

    def terminal(self, ledger, name):
        path = self.root / f"{name}.terminal.json"
        path.write_text(json.dumps({"head": ledger.head}), encoding="utf-8")
        return {"path": str(path), "sha256": sha(path)}

    def evidence_files(self, prefix=""):
        files = {}
        for name, text in (("failure_record", '{"status": "failed"}'), ("worker_log", "Torch not compiled with CUDA enabled\n"),
                           ("supervision", '{"returncode": 1}'), ("worker_request", "request-bytes")):
            path = self.evidence / f"{prefix}{name}.txt"
            path.write_text(text, encoding="utf-8")
            files[name] = path
        return files

    def plan(self, ledger, receipt, *, attempt=ATTEMPT, classification="ENVIRONMENTAL_EXECUTION_FAILURE",
             findings=None, allowed=True, name="plan", mutate=None, identity=None):
        value = review.build_review_plan(
            ledger_path=ledger.path, terminal_receipt_path=Path(receipt["path"]), attempt_id=attempt,
            authority=AUTHORITY, research_identity=identity or IDENTITY, evidence_paths=self.evidence_files(name),
            classification=classification, findings=findings or FINDINGS, disposition=disposition(allowed))
        if mutate:
            mutate(value)
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return path

    def record(self, plan_path, *, name="record", signed=True, decision=review.APPROVE, mutate=None):
        value = review.draft_review_record(plan_path, source_commit="a" * 40)
        if signed:
            value.update(status=review.SIGNED_STATUS, decision=decision,
                         signoff={"reviewer": "fixture reviewer", "signed_at_utc": "2026-09-30T00:00:00Z"})
        if mutate:
            mutate(value)
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return {"path": str(path), "sha256": sha(path)}

    def spec(self, sources):
        return {"schema_version": handoff.SCHEMA, "authority": AUTHORITY, "source_commit": "a" * 40,
                "config_sha256": "b" * 64, "inventory_roots": [str(l.path) for l, _, _ in sources
                                                                if not l.path.name.startswith("increment")],
                "sources": [{"ledger_path": str(l.path), "terminal_receipt": receipt,
                             **({"failure_reviews": reviews} if reviews else {})}
                            for l, receipt, reviews in sources]}

    def inspect(self, spec):
        spec = copy.deepcopy(spec)
        head = handoff._increment_index(self.root)[1]
        if head["event_count"]:
            spec["increment_index_head"] = head
        return handoff.inspect_handoff(spec, base_root=self.root, process_probe=lambda: [],
                                       now_sample={"wall_ns": round(self.clock.wall * budget.NS),
                                                   "monotonic_ns": round(self.clock.monotonic * budget.NS)})

    def scenario(self, **plan_options):
        old = self.ledger("old")
        self.fail(old)
        receipt = self.terminal(old, "old")
        # A consistent non-retry review is a signed rejection.
        decision = review.APPROVE if plan_options.get("allowed", True) else review.REJECT
        record = self.record(self.plan(old, receipt, **plan_options), decision=decision)
        return old, receipt, record

    def open_segment(self, spec, name="increment_one", resume=False):
        spec = copy.deepcopy(spec)
        head = handoff._increment_index(self.root)[1]
        if head["event_count"]:
            spec["increment_index_head"] = head
        path = self.root / f"{name}-handoff.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        adapter = handoff.open_inherited_ledger(path, self.root / name, base_root=self.root,
                                               expected_config_sha="b" * 64, expected_source_commit="a" * 40,
                                               process_probe=lambda: [], resume=resume,
                                               required_inventory_roots=spec["inventory_roots"])
        adapter._ledger.wall_clock = self.clock.wall_now
        adapter._ledger.monotonic_clock = self.clock.mono_now
        return adapter


class HandoffBlockerSemanticsTests(FailureReviewFixture):
    def test_01_unreviewed_failed_blocks_and_is_historical(self):
        old = self.ledger("old")
        self.fail(old)
        result = self.inspect(self.spec([(old, self.terminal(old, "old"), None)]))
        self.assertTrue(result["must_stop"])
        self.assertTrue(any(r.endswith(f"unreviewed_failed:{ATTEMPT}") for r in result["current_blocking_stop_reasons"]))
        self.assertEqual(result["waived_by_review"], [])

    def test_02_03_valid_environmental_review_waives_only_the_current_blocker(self):
        old, receipt, record = self.scenario()
        result = self.inspect(self.spec([(old, receipt, [record])]))
        self.assertFalse(result["must_stop"], result["current_blocking_stop_reasons"])
        source = result["sources"][0]
        self.assertIn(f"unreviewed_failed:{ATTEMPT}", source["historical_stop_reasons"])
        self.assertNotIn(f"unreviewed_failed:{ATTEMPT}", source["current_blocking_stop_reasons"])
        self.assertTrue(any(r.endswith(f"unreviewed_failed:{ATTEMPT}") for r in result["historical_stop_reasons"]))
        waived = source["waived_by_review"]
        self.assertEqual([(w["attempt_id"], w["reason"], w["review_record_sha256"]) for w in waived],
                         [(ATTEMPT, f"unreviewed_failed:{ATTEMPT}", record["sha256"])])
        self.assertTrue(budget._hex(waived[0]["review_plan_sha256"]))
        self.assertEqual(len(result["failure_retry_authorizations"]), 1)

    def test_04_05_20_original_outcome_charge_and_evidence_are_unchanged(self):
        old = self.ledger("old")
        self.fail(old)
        receipt = self.terminal(old, "old")
        before = self.inspect(self.spec([(old, receipt, None)]))
        files = [old.path / "budget_events.jsonl", Path(receipt["path"])]
        hashes = {str(p): sha(p) for p in files}
        record = self.record(self.plan(old, receipt))
        files += [Path(p) for p in (self.evidence / name for name in sorted(p.name for p in self.evidence.iterdir()))]
        hashes.update({str(p): sha(p) for p in files[2:]})
        after = self.inspect(self.spec([(old, receipt, [record])]))
        adapter = self.open_segment(self.spec([(old, receipt, [record])]))
        adapter.close()
        for result in (before, after):
            attempt = result["sources"][0]["attempts"][ATTEMPT]
            self.assertEqual((attempt["outcome"], attempt["reviewed"]), ("failed", False))
        self.assertEqual(before["sources"][0]["attempts"][ATTEMPT]["charged_ns"],
                         after["sources"][0]["attempts"][ATTEMPT]["charged_ns"])
        self.assertEqual(before["charged_ns"], after["charged_ns"])
        self.assertEqual(before["sources"][0]["head"], after["sources"][0]["head"])
        self.assertEqual({str(p): sha(p) for p in files}, hashes)

    def test_06_07_research_or_unknown_failure_never_clears_the_blocker(self):
        cases = [("RESEARCH_EXECUTION_FAILURE", False, False), ("RESEARCH_EXECUTION_FAILURE", True, True),
                 ("UNKNOWN", False, False), ("UNKNOWN", True, True)]
        for classification, allowed, invalid in cases:
            with self.subTest(classification=classification, allowed=allowed):
                self.setUp()
                old, receipt, record = self.scenario(classification=classification, allowed=allowed)
                result = self.inspect(self.spec([(old, receipt, [record])]))
                self.assertTrue(result["must_stop"])
                current = result["sources"][0]["current_blocking_stop_reasons"]
                self.assertIn(f"unreviewed_failed:{ATTEMPT}", current)
                self.assertEqual(f"failure_review_invalid:{ATTEMPT}" in current, invalid)
                self.assertEqual(result["failure_retry_authorizations"], [])

    def test_08_unsigned_rejected_or_hash_mismatched_review_blocks(self):
        def tamper_plan(old, receipt):
            path = self.plan(old, receipt)
            record = self.record(path)
            text = json.loads(path.read_text())
            text["classification"] = "ENVIRONMENTAL_EXECUTION_FAILURE "
            path.write_text(json.dumps(text))
            return record

        def tamper_evidence(old, receipt):
            record = self.record(self.plan(old, receipt))
            (self.evidence / "planworker_log.txt").write_text("edited\n", encoding="utf-8")
            return record

        cases = {
            "draft": lambda old, receipt: self.record(self.plan(old, receipt), signed=False),
            "rejected": lambda old, receipt: self.record(self.plan(old, receipt), decision=review.REJECT),
            "record_hash": lambda old, receipt: {**self.record(self.plan(old, receipt)), "sha256": "0" * 64},
            "plan_hash": tamper_plan,
            "evidence_hash": tamper_evidence,
            "amendment_hash": lambda old, receipt: self.record(
                self.plan(old, receipt), mutate=lambda r: r.update(amendment_sha256="0" * 64)),
            "missing_reviewer": lambda old, receipt: self.record(
                self.plan(old, receipt), mutate=lambda r: r["signoff"].update(reviewer="")),
        }
        for label, make in cases.items():
            with self.subTest(label=label):
                self.setUp()
                old = self.ledger("old")
                self.fail(old)
                receipt = self.terminal(old, "old")
                record = make(old, receipt)
                result = self.inspect(self.spec([(old, receipt, [record])]))
                self.assertTrue(result["must_stop"])
                current = result["sources"][0]["current_blocking_stop_reasons"]
                self.assertIn(f"unreviewed_failed:{ATTEMPT}", current)
                if label != "rejected":
                    self.assertTrue(any(r.startswith("failure_review_invalid:") for r in current), current)
                self.assertEqual(result["failure_retry_authorizations"], [])

    def test_09_10_terminal_receipt_head_charge_or_attempt_mismatch_blocks(self):
        def other_receipt(value):
            value["original"]["terminal_receipt_sha256"] = "0" * 64
        mutations = {
            "terminal_receipt": other_receipt,
            "charged_ns": lambda v: v["original"].update(charged_ns=v["original"]["charged_ns"] - 1),
            "outcome": lambda v: v["original"].update(outcome="external_interruption"),
            "journal": lambda v: v["original"].update(journal_sha256="0" * 64),
            "head": lambda v: v["original"]["head"].update(event_count=v["original"]["head"]["event_count"] + 1),
            "close_event": lambda v: v["original"]["close_event"].update(sha256="0" * 64),
            "unit": lambda v: v["original"].update(unit_id="unit_Cora_normalize_features_MLP_001"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                self.setUp()
                old = self.ledger("old")
                self.fail(old)
                receipt = self.terminal(old, "old")
                record = self.record(self.plan(old, receipt, mutate=mutate))
                current = self.inspect(self.spec([(old, receipt, [record])]))["sources"][0]["current_blocking_stop_reasons"]
                self.assertIn(f"unreviewed_failed:{ATTEMPT}", current)
                self.assertTrue(any(r.startswith("failure_review_invalid:") for r in current), current)

    def test_nonzero_findings_or_open_scope_cannot_allow_retry(self):
        mutations = [{**FINDINGS, "research_computation_started": True}, {**FINDINGS, "checkpoints": 1},
                     {**FINDINGS, "records": 1}, {**FINDINGS, "test_evaluations": 1},
                     {**FINDINGS, "test_once_consumed": True}, {**FINDINGS, "blocking_stop_reasons": ["monitor_gap"]}]
        for findings in mutations:
            with self.subTest(findings=findings):
                self.setUp()
                old, receipt, record = self.scenario(findings=findings)
                self.assertTrue(self.inspect(self.spec([(old, receipt, [record])]))["must_stop"])
        for field, value in (("max_retries", 2), ("must_be_new_segment", False), ("new_attempt_id_required", False)):
            with self.subTest(field=field):
                self.setUp()
                old, receipt, record = self.scenario(mutate=lambda v, f=field, x=value: v["disposition"].update({f: x}))
                self.assertTrue(self.inspect(self.spec([(old, receipt, [record])]))["must_stop"])
        with self.subTest(field="research_identity"):
            self.setUp()
            old, receipt, record = self.scenario(identity={**IDENTITY, "seed": 1})
            self.assertTrue(self.inspect(self.spec([(old, receipt, [record])]))["must_stop"])

    def test_clock_stop_on_the_failed_attempt_is_not_waivable(self):
        old = self.ledger("old")
        old.begin_attempt(ATTEMPT, activity_id=ACTIVITY, phase_id="mlp24_extension", budget_group="formal",
                          estimated_seconds=1, unit_id=UNIT, batch_id=BATCH)
        self.clock.advance(7)  # exceeds the five-second monitor gap
        old.poll(force=True)
        old.close_attempt(ATTEMPT, outcome="failed", reason="FormalRecordError('worker failed')")
        receipt = self.terminal(old, "old")
        record = self.record(self.plan(old, receipt))
        result = self.inspect(self.spec([(old, receipt, [record])]))
        current = result["sources"][0]["current_blocking_stop_reasons"]
        self.assertIn(f"monitor_gap:{ATTEMPT}", current)
        self.assertIn(f"unreviewed_failed:{ATTEMPT}", current)

    def test_18_unrelated_unreviewed_failure_still_blocks(self):
        old, receipt, record = self.scenario()
        other = self.ledger("other")
        self.fail(other, attempt="independent", unit="unit_Cora_normalize_features_MLP_001",
                  batch="pair_Cora_seed_001_MLP")
        result = self.inspect(self.spec([(old, receipt, [record]), (other, self.terminal(other, "other"), None)]))
        self.assertTrue(result["must_stop"])
        self.assertEqual(result["sources"][0]["current_blocking_stop_reasons"], [])
        current = result["sources"][1]["current_blocking_stop_reasons"]
        self.assertEqual([r for r in current if "failed" in r], ["unreviewed_failed:independent"])

    def test_one_attempt_cannot_carry_two_reviews(self):
        old, receipt, record = self.scenario()
        second = self.record(self.plan(old, receipt, name="plan_two"), name="record_two")
        with self.assertRaisesRegex(handoff.HandoffError, "review"):
            self.inspect(self.spec([(old, receipt, [record, second])]))
        with self.assertRaisesRegex(handoff.HandoffError, "review"):
            self.inspect(self.spec([(old, receipt, [record, record])]))


class RetryAuthorizationTests(FailureReviewFixture):
    def authorized_segment(self):
        old, receipt, record = self.scenario()
        spec = self.spec([(old, receipt, [record])])
        adapter = self.open_segment(spec)
        adapter.register_phase("mlp24_extension", provenance=provenance(), gate_receipt=gate())
        return old, receipt, record, spec, adapter

    def retry(self, adapter, attempt="mlp24_s000e2_u000000_retry1", outcome="completed"):
        adapter.begin_attempt(attempt, activity_id=ACTIVITY, phase_id="mlp24_extension", budget_group="formal",
                              estimated_seconds=1, unit_id=UNIT, batch_id=BATCH)
        self.clock.advance(3)
        return adapter.close_attempt(attempt, outcome=outcome, record_sha256="5" * 64 if outcome == "completed" else None,
                                     reason=None if outcome == "completed" else "fixture failure")

    def test_02_13_new_segment_retry_requires_the_bound_new_attempt_identity(self):
        old, receipt, record, spec, adapter = self.authorized_segment()
        pending = adapter.pending_failure_retries()
        self.assertEqual([row["review_record_sha256"] for row in pending], [record["sha256"]])
        with self.assertRaisesRegex(handoff.HandoffError, "retry authorization"):
            adapter.begin_attempt("unauthorized", activity_id=ACTIVITY, phase_id="mlp24_extension",
                                  budget_group="formal", estimated_seconds=1, unit_id=UNIT, batch_id=BATCH)
        for reused in (ATTEMPT,):
            with self.assertRaisesRegex(budget.BudgetError, "attempt identity"):
                adapter.authorize_failure_retry(record["sha256"], retry_attempt_id=reused, phase_id="mlp24_extension")
        snapshot = adapter.authorize_failure_retry(record["sha256"], retry_attempt_id="mlp24_s000e2_u000000_retry1",
                                                   phase_id="mlp24_extension")
        authorization = snapshot["retry_authorizations"][UNIT]
        self.assertEqual((authorization["original"]["attempt_id"], authorization["retry_ordinal"],
                          authorization["consumed_by"]), (ATTEMPT, 1, None))
        self.assertEqual(adapter.pending_failure_retries(), [])
        with self.assertRaisesRegex(budget.BudgetAdmissionError, "bound attempt identity"):
            adapter.begin_attempt("mlp24_s000e2_u000000", activity_id=ACTIVITY, phase_id="mlp24_extension",
                                  budget_group="formal", estimated_seconds=1, unit_id=UNIT, batch_id=BATCH)
        closed = self.retry(adapter)
        self.assertEqual(closed["retry_authorizations"][UNIT]["consumed_by"], "mlp24_s000e2_u000000_retry1")
        self.assertEqual(closed["attempts"]["mlp24_s000e2_u000000_retry1"]["outcome"], "completed")
        # Charges accumulate on the same unit; the original 5 seconds remain inherited.
        self.assertEqual(closed["charged_seconds"]["units"][UNIT], 8)
        adapter.close()

    def test_authorization_requires_the_reviewed_environment_and_precedes_any_unit_attempt(self):
        old, receipt, record = self.scenario()
        adapter = self.open_segment(self.spec([(old, receipt, [record])]))
        changed = copy.deepcopy(provenance())
        changed["environment"]["device"] = "cuda-other"
        adapter.register_phase("mlp24_extension", provenance=changed, gate_receipt=gate())
        with self.assertRaisesRegex(budget.BudgetIntegrityError, "environment"):
            adapter.authorize_failure_retry(record["sha256"], retry_attempt_id="retry-id", phase_id="mlp24_extension")
        with self.assertRaisesRegex(handoff.HandoffError, "not granted"):
            adapter.authorize_failure_retry("0" * 64, retry_attempt_id="retry-id", phase_id="mlp24_extension")
        adapter.close()

    def test_11_a_review_is_consumed_once_by_one_new_segment(self):
        old, receipt, record, spec, adapter = self.authorized_segment()
        adapter.authorize_failure_retry(record["sha256"], retry_attempt_id="mlp24_s000e2_u000000_retry1",
                                        phase_id="mlp24_extension")
        adapter.close()
        segment = self.root / "increment_one"
        spec_two = copy.deepcopy(spec)
        spec_two["sources"].append({"ledger_path": str(segment), "terminal_receipt": {
            "path": str(segment / "terminal_receipt.json"), "sha256": sha(segment / "terminal_receipt.json")}})
        result = self.inspect(spec_two)
        self.assertFalse(result["must_stop"], result["current_blocking_stop_reasons"])
        waived = result["sources"][0]["waived_by_review"][0]
        self.assertEqual(waived["consumed_by"], str(segment.resolve()))
        self.assertEqual(result["failure_retry_authorizations"], [])
        second = self.open_segment(spec_two, name="increment_two")
        self.assertEqual(second.pending_failure_retries(), [])
        second.register_phase("mlp24_extension", provenance=provenance(), gate_receipt=gate())
        with self.assertRaisesRegex(handoff.HandoffError, "not granted"):
            second.authorize_failure_retry(record["sha256"], retry_attempt_id="again", phase_id="mlp24_extension")
        with self.assertRaisesRegex(handoff.HandoffError, "retry authorization"):
            second.begin_attempt("again", activity_id=ACTIVITY, phase_id="mlp24_extension", budget_group="formal",
                                 estimated_seconds=1, unit_id=UNIT, batch_id=BATCH)
        second.close()
        with self.assertRaisesRegex(handoff.HandoffError, "consumed"):
            handoff._register_increment(self.root, (self.root / "forged").resolve(), "9" * 64,
                                        consumed_failure_reviews=[record["sha256"]])

    def test_resume_of_an_existing_segment_cannot_consume_a_new_review(self):
        old = self.ledger("old")
        receipt = self.terminal(old, "old")
        adapter = self.open_segment(self.spec([(old, receipt, None)]))
        adapter.register_phase("mlp24_extension", provenance=provenance(), gate_receipt=gate())
        self.fail(adapter)
        adapter.close()
        segment = self.root / "increment_one"
        seg_receipt = {"path": str(segment / "terminal_receipt.json"), "sha256": sha(segment / "terminal_receipt.json")}
        record = self.record(self.plan(adapter._ledger, seg_receipt))
        spec = self.spec([(old, receipt, None), (adapter._ledger, seg_receipt, [record])])
        with self.assertRaisesRegex(handoff.HandoffError, "new segment"):
            self.open_segment(spec, resume=True)

    def test_12_19_one_unit_gets_at_most_one_retry_and_no_second_test_access(self):
        old, receipt, record, spec, adapter = self.authorized_segment()
        adapter.authorize_failure_retry(record["sha256"], retry_attempt_id="mlp24_s000e2_u000000_retry1",
                                        phase_id="mlp24_extension")
        self.retry(adapter)
        # A completed retry cannot run again in the same segment; a record exists once.
        with self.assertRaisesRegex(budget.BudgetAdmissionError, "cannot restart"):
            adapter.begin_attempt("mlp24_s000e2_u000000_retry2", activity_id=ACTIVITY, phase_id="mlp24_extension",
                                  budget_group="formal", estimated_seconds=1, unit_id=UNIT, batch_id=BATCH)
        adapter.close()
        segment = self.root / "increment_one"
        seg_receipt = {"path": str(segment / "terminal_receipt.json"), "sha256": sha(segment / "terminal_receipt.json")}
        # A second, independently signed review of the same original failure cannot add a retry.
        other = self.record(self.plan(old, receipt, name="plan_two"), name="record_two")
        spec_two = self.spec([(old, receipt, [other]), (adapter._ledger, seg_receipt, None)])
        spec_two["inventory_roots"] = [str(old.path)]
        result = self.inspect(spec_two)
        self.assertTrue(result["must_stop"])
        self.assertIn(f"failure_review_invalid:{ATTEMPT}", result["sources"][0]["current_blocking_stop_reasons"])
        self.assertEqual(result["failure_retry_authorizations"], [])

    def test_retry_that_fails_again_has_no_further_retry(self):
        old, receipt, record, spec, adapter = self.authorized_segment()
        retry_id = "mlp24_s000e2_u000000_retry1"
        adapter.authorize_failure_retry(record["sha256"], retry_attempt_id=retry_id, phase_id="mlp24_extension")
        self.retry(adapter, outcome="failed")
        adapter.close()
        segment = self.root / "increment_one"
        seg_receipt = {"path": str(segment / "terminal_receipt.json"), "sha256": sha(segment / "terminal_receipt.json")}
        again = self.record(self.plan(adapter._ledger, seg_receipt, attempt=retry_id, name="plan_retry"),
                            name="record_retry")
        spec_two = copy.deepcopy(spec)
        spec_two["sources"].append({"ledger_path": str(segment), "terminal_receipt": seg_receipt,
                                    "failure_reviews": [again]})
        result = self.inspect(spec_two)
        self.assertTrue(result["must_stop"])
        current = result["sources"][1]["current_blocking_stop_reasons"]
        self.assertIn(f"unreviewed_failed:{retry_id}", current)
        self.assertIn(f"failure_review_invalid:{retry_id}", current)
        self.assertEqual(result["failure_retry_authorizations"], [])

    def test_14_inherited_attempt_identities_are_known_to_the_new_segment(self):
        old, receipt, record, spec, adapter = self.authorized_segment()
        self.assertIn(ATTEMPT, adapter.known_attempt_ids())
        self.assertEqual(adapter.segment_sequence, 0)
        adapter.close()


if __name__ == "__main__":
    unittest.main()
