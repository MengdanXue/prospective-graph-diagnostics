"""Pause, resume and ledger-segment behaviour of the formal dispatcher."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.input_robustness_budget import BudgetError, BudgetLedger, CI_JOBS, _validate_provenance
from scripts.input_robustness_formal_entry import FormalExecutionController
from scripts.input_robustness_formal_records import (
    DuplicateRecordError, FormalRecordError, FormalRecordWriter,
)
from scripts.input_robustness_runtime import PowerEventWatcher, TaskPowerRequest
from scripts import run_input_robustness_formal as dispatch
from scripts import record_local_launch_verification as local_receipt


class _PowerBackend:
    def create_request(self, reason): return 1
    def set_request(self, handle, request_type): pass
    def clear_request(self, handle, request_type): pass
    def close_handle(self, handle): pass
    def register_notification(self, callback): self.callback = callback; return 2
    def unregister_notification(self, registration): pass


_AUTHORITY = {"run_id": "r", "authorization_sha256": "a" * 64,
              "proposal_sha256": "b" * 64, "scope_sha256": "c" * 64}


def _provenance(receipt):
    provenance = {"source_commit": "1" * 40, "config_sha256": "2" * 64, "data_binding_sha256": "3" * 64,
                  "source_files": {"scripts/x.py": "4" * 64}, "environment": {"python": "3.12"}}
    gate = {"ci_receipt_file_sha256": "5" * 64, "review_record_sha256": "6" * 64, "ci_receipt": receipt}
    return provenance, gate


class WriterReopenTests(unittest.TestCase):
    def _paths(self, base: Path):
        config, binding = base / "config.json", base / "binding.json"
        config.write_text("{}", encoding="utf-8")
        binding.write_text("{}", encoding="utf-8")
        return config, binding

    def test_reopen_requires_identical_manifest_and_incomplete_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config, binding = self._paths(base)
            manifest = {"run_id": "r", "source_commit": "1" * 40, "environment": {"torch": "2.9.1"}}
            FormalRecordWriter(base / "run", manifest=manifest, config_path=config, data_binding_path=binding)
            writer = FormalRecordWriter.reopen(base / "run", manifest=manifest,
                                               config_path=config, data_binding_path=binding)
            self.assertEqual(writer.root, base / "run")
            self.assertFalse(writer.synthetic)
            changed = {**manifest, "environment": {"torch": "2.9.2"}}
            with self.assertRaisesRegex(FormalRecordError, "environment"):
                FormalRecordWriter.reopen(base / "run", manifest=changed, config_path=config, data_binding_path=binding)
            (base / "run" / "complete.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(DuplicateRecordError):
                FormalRecordWriter.reopen(base / "run", manifest=manifest, config_path=config, data_binding_path=binding)

    def test_reopen_rejects_missing_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config, binding = self._paths(base)
            with self.assertRaisesRegex(FormalRecordError, "no formal manifest"):
                FormalRecordWriter.reopen(base / "absent", manifest={}, config_path=config, data_binding_path=binding)


class DispatchHelperTests(unittest.TestCase):
    def test_attempt_numbers_continue_after_every_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(dispatch._next_attempt_number(root), 0)
            (root / "workers").mkdir()
            (root / "workers" / "attempt_000003.json").write_text("{}", encoding="utf-8")
            checkpoint = root / "checkpoints" / "c" / "Cora" / "MLP" / "seed_000" / "attempt_000007"
            checkpoint.mkdir(parents=True)
            (root / "failures").mkdir()
            (root / "failures" / "attempt_000005.json").write_text("{}", encoding="utf-8")
            self.assertEqual(dispatch._next_attempt_number(root), 8)

    def test_segments_are_numbered_and_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "ledger"
            self.assertEqual(dispatch._next_segment(parent).name, "segment_001")
            (parent / "segment_001").mkdir(parents=True)
            (parent / "segment_002").mkdir()
            (parent / "notes").mkdir()
            self.assertEqual([p.name for p in dispatch._segments(parent)], ["segment_001", "segment_002"])
            self.assertEqual(dispatch._next_segment(parent).name, "segment_003")

    def test_prior_segment_usage_reads_every_segment_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "ledger"
            parent.mkdir()
            BudgetLedger.create(parent / "segment_001", authority=_AUTHORITY)
            before = (parent / "segment_001" / "budget_events.jsonl").read_bytes()
            usage = dispatch._prior_segment_usage(parent, _AUTHORITY)
            self.assertEqual([row["segment"] for row in usage["segments"]], ["segment_001"])
            self.assertEqual(usage["formal_seconds"], 0.0)
            self.assertEqual((parent / "segment_001" / "budget_events.jsonl").read_bytes(), before)

    def test_resume_shape_guards_fresh_and_resumed_invocations(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output, ledger = base / "run", base / "ledger"
            dispatch._check_resume_shape(resume=False, output_root=output, ledger_parent=ledger)
            with self.assertRaisesRegex(FormalRecordError, "--resume requires"):
                dispatch._check_resume_shape(resume=True, output_root=output, ledger_parent=ledger)
            output.mkdir()
            (output / "manifest.json").write_text("{}", encoding="utf-8")
            (ledger / "segment_001").mkdir(parents=True)
            with self.assertRaisesRegex(FormalRecordError, "fresh run requires"):
                dispatch._check_resume_shape(resume=False, output_root=output, ledger_parent=ledger)
            dispatch._check_resume_shape(resume=True, output_root=output, ledger_parent=ledger)

    def test_pause_file_stops_at_a_unit_boundary(self):
        class Controller:
            paused = False
            def request_normal_pause(self): self.paused = True
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = Controller()
            seen = []
            for unit in dispatch._pausable(iter([1, 2, 3]), root, lambda: controller):
                seen.append(unit)
                (root / dispatch.PAUSE_FILE).write_text("", encoding="utf-8")
            self.assertEqual(seen, [1])
            self.assertTrue(controller.paused)

    def test_segment_state_is_written_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = dispatch._write_segment_state(root, "segment_001", {"status": "paused"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "paused")
            with self.assertRaises(FileExistsError):
                dispatch._write_segment_state(root, "segment_001", {"status": "paused"})


class ControllerPauseTests(unittest.TestCase):
    def test_pause_that_ends_the_unit_source_is_not_completion(self):
        class Ledger:
            def poll(self, force=False):
                return {"must_stop": False, "stop_reasons": []}
        class Writer:
            synthetic = False
            root = Path(".")
            manifest = {}
            def finalize(self, **_):
                raise AssertionError("a paused run must never finalize")
        backend = _PowerBackend()
        controller = FormalExecutionController(writer=Writer(), ledger=Ledger(), phase_id="phase",
                                               power_request=TaskPowerRequest(backend),
                                               power_watcher=PowerEventWatcher(backend))
        def units():
            controller.request_normal_pause()
            return
            yield  # pragma: no cover
        result = controller.run_units(units(), launch_authorized=True, formal_training_enabled=True,
                                      post_stage=lambda: (_ for _ in ()).throw(AssertionError("no post stage")))
        self.assertEqual(result["status"], "paused")
        self.assertIsNone(result["complete"])


class LocalVerificationReceiptTests(unittest.TestCase):
    def test_ledger_accepts_only_a_labelled_single_job_local_receipt(self):
        local = {"kind": "local_verification", "run_id": 7, "status": "completed", "conclusion": "success",
                 "commit": "1" * 40, "jobs": [{"name": "full-protocol-verification", "run_id": 7,
                                               "status": "completed", "conclusion": "success"}]}
        _validate_provenance(*_provenance(local))
        failed = {**local, "conclusion": "failure"}
        with self.assertRaises(BudgetError):
            _validate_provenance(*_provenance(failed))
        unlabelled = {key: value for key, value in local.items() if key != "kind"}
        with self.assertRaisesRegex(BudgetError, "every verification job"):
            _validate_provenance(*_provenance(unlabelled))
        github = {**unlabelled, "jobs": [{"name": name, "run_id": 7, "status": "completed", "conclusion": "success"}
                                         for name in sorted(CI_JOBS)]}
        _validate_provenance(*_provenance(github))

    def test_dispatch_gate_accepts_local_receipt_for_the_same_commit_only(self):
        commit = "1" * 40
        receipt = {"status": "passed", "commit": commit, "ci": {
            "kind": "local_verification", "run_id": 7, "status": "completed", "conclusion": "success",
            "commit": commit, "jobs": [{"name": "full-protocol-verification", "run_id": 7, "status": "completed",
                                        "conclusion": "success", "commit": commit}]}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual(dispatch._ci_gate(path, commit)["kind"], "local_verification")
            with self.assertRaises(FormalRecordError):
                dispatch._ci_gate(path, "2" * 40)

    def test_unittest_summary_parsing(self):
        ok = local_receipt._parse_unittest("....\n----\nRan 12 tests in 1.0s\n\nOK (skipped=2)\n")
        self.assertEqual((ok["tests_run"], ok["ok"], ok["skipped"]), (12, True, 2))
        bad = local_receipt._parse_unittest("Ran 5 tests in 1.0s\n\nFAILED (failures=1, errors=2)\n")
        self.assertEqual((bad["ok"], bad["failures"], bad["errors"]), (False, 1, 2))


if __name__ == "__main__":
    unittest.main()
