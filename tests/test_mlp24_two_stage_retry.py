"""Two-stage reviewed retry and stable phase provenance through the real entry main().

Real budget ledgers, handoff specifications, increment index and
open_inherited_ledger are used. Only worker results are supplied at the
controller interface, so no training runs.
"""
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import input_robustness_budget as budget
from scripts import mlp_budget_extension_entry as entry
from scripts import supplement_budget_handoff as handoff
from scripts import terminal_failure_review as review
from scripts.input_robustness_formal_records import _exclusive_json, digest, file_digest
from scripts.input_robustness_runtime import PowerEventWatcher, TaskPowerRequest
from tests.test_input_robustness_budget import AUTHORITY, gate, provenance
from tests.test_input_robustness_formal_entry import _PowerBackend
from tests.test_mlp24_retry_prerequisites import cuda_facts, write_base

COMMIT = "a" * 40
UNIT, BATCH = "unit_Cora_normalize_features_MLP_000", "pair_Cora_seed_000_MLP"
IDENTITY = {"dataset": "Cora", "condition": "normalize_features", "model": "MLP", "seed": 0,
            "trials": "trial_000-trial_023"}
BINDING = {"python": "3.12.14", "torch": "2.9.1+cu128", "device_policy": "config-bound", "torch_cuda": "12.8",
           "cuda_available": True, "cuda_device": "Fixture GPU", "cuda_driver": "1.0"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RealEntryFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {"run_id": "two-stage-fixture", "execution_mode": "research", "datasets": ["Cora"],
                       "conditions": ["normalize_features"], "seeds": [0],
                       "execution": {"model_devices": {"MLP": "cuda"}},
                       "budget": {"inventory_roots": ["original"]}}
        self.config_path = self.root / "config.json"
        _exclusive_json(self.config_path, self.config)
        _exclusive_json(self.root / "acceptance.json", {"source_commit": COMMIT, "config_sha256": digest(self.config)})
        _exclusive_json(self.root / "binding.json", {"fixture": True})
        _exclusive_json(self.root / "ci.json", {"fixture": True})
        self.base = write_base(self.root)
        self.original = budget.BudgetLedger.create(self.root / "original", authority=AUTHORITY, monitor_gap_seconds=5)
        self.original.register_phase("mlp24_extension", provenance=provenance(COMMIT), gate_receipt=gate(COMMIT))
        self.operations = []

    def fail_original(self):
        self.original.begin_attempt("mlp24_2_000000", activity_id=f"mlp24_{UNIT}", phase_id="mlp24_extension",
                                    budget_group="formal", estimated_seconds=0, unit_id=UNIT, batch_id=BATCH)
        self.original.close_attempt("mlp24_2_000000", outcome="failed", reason="FormalRecordError('worker failed')")

    def receipt(self, ledger_path, head=None):
        if head is not None:
            path = self.root / f"{Path(ledger_path).name}.terminal.json"
            path.write_text(json.dumps({"head": head}), encoding="utf-8")
        else:
            path = sorted(Path(ledger_path).glob("terminal_receipt*.json"), key=lambda p: p.stat().st_mtime_ns)[-1]
        return {"path": str(path), "sha256": sha(path)}

    def signed_review(self, original_receipt):
        evidence = {}
        for name in ("failure_record", "worker_log"):
            path = self.root / f"{name}.txt"
            path.write_text(name, encoding="utf-8")
            evidence[name] = path
        plan = review.build_review_plan(
            ledger_path=self.original.path, terminal_receipt_path=Path(original_receipt["path"]),
            attempt_id="mlp24_2_000000", authority=AUTHORITY, research_identity=IDENTITY, evidence_paths=evidence,
            classification=review.ENVIRONMENTAL,
            findings={"research_computation_started": False, "checkpoints": 0, "records": 0, "test_evaluations": 0,
                      "test_once_consumed": False, "blocking_stop_reasons": []},
            disposition={"retry_allowed": True, "reason": "fixture", "max_retries": 1, "must_be_new_segment": True,
                         "new_attempt_id_required": True,
                         "retry_scope": {"unit_id": UNIT, "batch_id": BATCH, "activity_id": f"mlp24_{UNIT}",
                                         "research_identity": IDENTITY},
                         "required_environment": dict(BINDING)})
        plan_path = self.root / "plan.json"
        _exclusive_json(plan_path, plan)
        record = {**review.draft_review_record(plan_path, source_commit=COMMIT), "status": review.SIGNED_STATUS,
                  "decision": review.APPROVE, "signoff": {"reviewer": "fixture reviewer", "signed_at_utc": "fixture"}}
        record_path = self.root / "signed.json"
        _exclusive_json(record_path, record)
        return {"path": str(record_path), "sha256": sha(record_path)}

    def spec(self, name, sources):
        value = {"schema_version": handoff.SCHEMA, "authority": AUTHORITY, "source_commit": COMMIT,
                 "config_sha256": digest(self.config), "inventory_roots": ["original"],
                 "sources": [{"ledger_path": str(path), "terminal_receipt": receipt,
                              **({"failure_reviews": reviews} if reviews else {})}
                             for path, receipt, reviews in sources]}
        head = handoff._increment_index(self.root)[1]
        if head["event_count"]:
            value["increment_index_head"] = head
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def run_main(self, output, spec_path, *flags, facts=None, binding=None, pause_in_prepare=False):
        test, evidence = self, {"unit_estimates_seconds": {"Cora/normalize_features/000": 1}}

        class Controller(entry.ExtensionExecutionController):
            def __init__(self, **kwargs):
                super().__init__(**kwargs, power_request=TaskPowerRequest(_PowerBackend()),
                                 power_watcher=PowerEventWatcher(_PowerBackend()))

            def _child(self, request, *, attempt_id, cap_seconds=28800):
                test.operations.append((request["operation"], attempt_id))
                if request["operation"] == "prepare":
                    if pause_in_prepare:
                        self.control_root.mkdir(exist_ok=True)
                        (self.control_root / entry.PAUSE_FILE).touch()
                        self._monitor()
                        (self.control_root / entry.PAUSE_FILE).rename(self.control_root / "PAUSE_ACKNOWLEDGED")
                    return {"base_evidence": evidence, "completed_keys": []}
                if request["operation"] == "unit":
                    path = output / "records" / request["condition"] / request["dataset"] / "seed_000.json"
                    _exclusive_json(path, {"interface_fixture": True})
                    return {"record_path": str(path), "record_sha256": file_digest(path)}
                _exclusive_json(output / "run_complete.json", {"interface_fixture": True})
                return {"status": "complete"}

        arguments = ["entry", "--execute", "--config", str(self.config_path), "--base-config", str(self.config_path),
                     "--base-root", str(self.base), "--binding", str(self.root / "binding.json"),
                     "--data-root", str(self.root / "data"), "--output-root", str(output),
                     "--budget-handoff", str(spec_path), "--budget-ledger", str(self.root / "segment_new"),
                     "--ci-receipt", str(self.root / "ci.json"),
                     "--acceptance-record", str(self.root / "acceptance.json"), *flags]
        with ExitStack() as stack:
            stack.enter_context(patch.object(entry, "ExtensionExecutionController", Controller))
            stack.enter_context(patch.object(entry, "_source_commit", return_value=COMMIT))
            stack.enter_context(patch.object(entry, "strict_ci_gate", return_value=gate(COMMIT)["ci_receipt"]))
            stack.enter_context(patch.object(entry, "_environment_binding", return_value=dict(binding or BINDING)))
            stack.enter_context(patch.object(entry, "runtime_environment_probe", return_value=facts or cuda_facts()))
            stack.enter_context(patch.object(sys, "argv", arguments))
            return entry.main()

    def journal(self, path):
        return [json.loads(line) for line in (Path(path) / "budget_events.jsonl").read_text().splitlines()]


class StablePhaseProvenanceTests(RealEntryFixture):
    def test_preflight_identity_ignores_artifacts_but_tracks_every_environment_fact(self):
        first = entry.formal_environment_preflight(self.config, self.base, probe=lambda d: cuda_facts())
        again = entry.formal_environment_preflight(self.config, self.base, probe=lambda d: cuda_facts())
        self.assertEqual(entry.preflight_identity(first), entry.preflight_identity(again))
        variants = [cuda_facts(sys_executable="/other/python"), cuda_facts(device_count=2)]
        for facts in variants:
            changed = entry.formal_environment_preflight(self.config, self.base, probe=lambda d, f=facts: f)
            self.assertNotEqual(entry.preflight_identity(first), entry.preflight_identity(changed))
        policy = copy.deepcopy(first)
        policy["model_devices"] = {"MLP": "cpu"}
        base = copy.deepcopy(first)
        base["base_manifest_environment"]["cuda_driver"] = "other"
        for changed in (policy, base):
            self.assertNotEqual(entry.preflight_identity(first), entry.preflight_identity(changed))

    def test_same_environment_resume_passes_with_a_new_preflight_artifact(self):
        receipt = self.receipt(self.original.path, head=self.original.head)
        output = self.root / "run"
        spec = self.spec("handoff_one", [(self.original.path, receipt, None)])
        self.assertEqual(self.run_main(output, spec, pause_in_prepare=True), 0)
        self.assertEqual([op for op, _ in self.operations], ["prepare"])
        segment = self.root / "segment_new"
        spec = self.spec("handoff_two", [(self.original.path, receipt, None), (segment, self.receipt(segment), None)])
        self.assertEqual(self.run_main(output, spec, "--resume"), 0)
        self.assertEqual([op for op, _ in self.operations], ["prepare", "prepare", "unit", "finalize"])
        artifacts = sorted((self.root / "run_environment_preflight").glob("passed_*.json"))
        self.assertEqual(len(artifacts), 2)
        phases = [e for e in self.journal(segment) if e["kind"] == "phase_registered"]
        self.assertEqual(len(phases), 1)
        environment = phases[0]["payload"]["provenance"]["environment"]
        self.assertEqual(environment["formal_environment_preflight_sha256"],
                         entry.preflight_identity(entry.read_json(artifacts[1])))
        self.assertNotIn("formal_environment_preflight", environment)

    def test_changed_environment_cannot_resume_the_registered_phase(self):
        for label, options in (("interpreter", {"facts": cuda_facts(sys_executable="/other/python")}),
                               ("driver", {"binding": {**BINDING, "cuda_driver": "2.0"}})):
            with self.subTest(label=label):
                self.setUp()  # an independent segment per variant
                receipt = self.receipt(self.original.path, head=self.original.head)
                output = self.root / "run"
                spec = self.spec("handoff_one", [(self.original.path, receipt, None)])
                self.assertEqual(self.run_main(output, spec, pause_in_prepare=True), 0)
                segment = self.root / "segment_new"
                spec = self.spec(f"handoff_{label}", [(self.original.path, receipt, None),
                                                      (segment, self.receipt(segment), None)])
                before = len(self.journal(segment))
                with self.assertRaisesRegex(budget.BudgetError, "phase provenance"):
                    self.run_main(output, spec, "--resume", **options)
                kinds = [e["kind"] for e in self.journal(segment)[before:]]
                self.assertNotIn("attempt_started", kinds)
                self.assertNotIn("phase_registered", kinds)


class TwoStageRetryTests(RealEntryFixture):
    def stage_one(self):
        self.fail_original()
        receipt = self.receipt(self.original.path, head=self.original.head)
        signed = self.signed_review(receipt)
        spec = self.spec("handoff_stage_one", [(self.original.path, receipt, [signed])])
        new_output = self.root / "retry_output"
        self.assertEqual(self.run_main(new_output, spec, "--authorize-only"), 0)
        return receipt, signed, new_output

    def test_stage_one_consumes_once_and_authorizes_without_any_attempt(self):
        receipt, signed, new_output = self.stage_one()
        segment = self.root / "segment_new"
        kinds = [e["kind"] for e in self.journal(segment)]
        self.assertEqual(kinds, ["genesis", "phase_registered", "failure_retry_authorized"])
        authorization = self.journal(segment)[2]["payload"]
        self.assertEqual((authorization["original"]["attempt_id"], authorization["review_record_sha256"],
                          authorization["retry_attempt_id"], authorization["retry_ordinal"]),
                         ("mlp24_2_000000", signed["sha256"],
                          entry.unit_attempt_id("mlp24_s000e2", 0, retry_ordinal=1), 1))
        self.assertEqual(handoff.consumed_review_map(self.root), {signed["sha256"]: str(segment.resolve())})
        self.assertEqual(self.operations, [])
        self.assertFalse(new_output.exists())
        self.assertEqual(list(self.root.glob("retry_output_preparation_*")), [])
        self.assertFalse((self.root / ".supplement_budget_owner.lock").exists())

    def test_stage_two_reopens_the_same_segment_with_a_new_output_root(self):
        receipt, signed, new_output = self.stage_one()
        segment = self.root / "segment_new"
        old_output = self.root / "old_failed_output"
        _exclusive_json(old_output / "failures/mlp24_2_000000.json", {"status": "failed"})
        old_hashes = {str(p): sha(p) for p in old_output.rglob("*") if p.is_file()}
        original_journal = sha(self.original.path / "budget_events.jsonl")
        spec = self.spec("handoff_stage_two", [(self.original.path, receipt, [signed]),
                                               (segment, self.receipt(segment), None)])
        self.assertEqual(self.run_main(new_output, spec, "--resume", "--new-output-root"), 0)
        retry_id = entry.unit_attempt_id("mlp24_s000e2", 0, retry_ordinal=1)
        self.assertEqual([op for op, _ in self.operations], ["prepare", "unit", "finalize"])
        self.assertIn(("unit", retry_id), self.operations)
        events = self.journal(segment)
        self.assertEqual([e["kind"] for e in events].count("failure_retry_authorized"), 1)
        self.assertEqual([e["kind"] for e in events].count("phase_registered"), 1)
        started = [e["payload"]["attempt_id"] for e in events if e["kind"] == "attempt_started"
                   and e["payload"]["descriptor"]["unit_id"] == UNIT]
        self.assertEqual(started, [retry_id])
        state = handoff.stream_snapshot(segment, authority=AUTHORITY)
        self.assertEqual(state["retry_authorizations"][UNIT]["consumed_by"], retry_id)
        self.assertEqual(handoff.consumed_review_map(self.root), {signed["sha256"]: str(segment.resolve())})
        self.assertEqual(handoff._increment_index(self.root)[1]["event_count"], 1)
        self.assertTrue((new_output / "manifest.json").is_file())
        self.assertEqual({str(p): sha(p) for p in old_output.rglob("*") if p.is_file()}, old_hashes)
        self.assertEqual(sha(self.original.path / "budget_events.jsonl"), original_journal)
        original = handoff.stream_snapshot(self.original.path, authority=AUTHORITY)["attempts"]["mlp24_2_000000"]
        self.assertEqual(original["outcome"], "failed")

        # A consumed authorization cannot be started again or re-authorized.
        spec = self.spec("handoff_stage_three", [(self.original.path, receipt, [signed]),
                                                 (segment, self.receipt(segment), None)])
        before = len(self.journal(segment))
        with self.assertRaisesRegex(Exception, "authorized, not yet started retry"):
            self.run_main(self.root / "retry_output_again", spec, "--resume", "--new-output-root")
        kinds = [e["kind"] for e in self.journal(segment)[before:]]
        self.assertFalse({"attempt_started", "failure_retry_authorized"} & set(kinds))
        self.assertEqual(handoff._increment_index(self.root)[1]["event_count"], 1)

    def test_existing_new_output_root_is_refused_before_any_ledger_write(self):
        receipt, signed, new_output = self.stage_one()
        segment = self.root / "segment_new"
        _exclusive_json(new_output / "manifest.json", {"previous": True})
        spec = self.spec("handoff_stage_two", [(self.original.path, receipt, [signed]),
                                               (segment, self.receipt(segment), None)])
        before = self.journal(segment)
        with self.assertRaisesRegex(Exception, "new output root already exists"):
            self.run_main(new_output, spec, "--resume", "--new-output-root")
        self.assertEqual(self.journal(segment), before)
        self.assertEqual(self.operations, [])

    def test_flag_combinations_are_refused(self):
        receipt = self.receipt(self.original.path, head=self.original.head)
        spec = self.spec("handoff", [(self.original.path, receipt, None)])
        for flags in (("--new-output-root",), ("--authorize-only", "--resume"),
                      ("--authorize-only", "--new-output-root", "--resume")):
            with self.subTest(flags=flags):
                with self.assertRaises(SystemExit):
                    self.run_main(self.root / "out", spec, *flags)
        self.assertFalse((self.root / "segment_new").exists())

    def test_authorize_only_without_a_granted_review_creates_no_authorization_or_attempt(self):
        receipt = self.receipt(self.original.path, head=self.original.head)
        spec = self.spec("handoff", [(self.original.path, receipt, None)])
        with self.assertRaisesRegex(Exception, "granted reviewed retry"):
            self.run_main(self.root / "out", spec, "--authorize-only")
        kinds = [e["kind"] for e in self.journal(self.root / "segment_new")]
        self.assertNotIn("failure_retry_authorized", kinds)
        self.assertNotIn("attempt_started", kinds)
        self.assertEqual(self.operations, [])

    def test_unrelated_unreviewed_failure_still_blocks_stage_one(self):
        self.fail_original()
        receipt = self.receipt(self.original.path, head=self.original.head)
        spec = self.spec("handoff", [(self.original.path, receipt, None)])
        with self.assertRaisesRegex(handoff.HandoffError, "unreviewed_failed"):
            self.run_main(self.root / "out", spec, "--authorize-only")
        self.assertFalse((self.root / "segment_new").exists())


if __name__ == "__main__":
    unittest.main()
