"""Synthetic outer-guardian integration checks; no OS power or research work."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import accept_input_robustness_resources as guardian
from scripts import input_robustness_budget as budget


COMMIT = "a" * 40
AUTHORITY = {"run_id": "synthetic_resource_v2", "approval_sha256": "b" * 64,
             "proposal_sha256": "c" * 64, "scope_sha256": "d" * 64}


class FakeClock:
    def __init__(self):
        self.wall = 1_000_000.0
        self.monotonic = 1_000.0

    def advance(self, seconds):
        self.wall += seconds
        self.monotonic += seconds


class FakePowerRequest:
    def __init__(self, *, release_error=False):
        self.acquired = False
        self.released = False
        self.release_error = release_error
        self.cleanup_errors = []

    def acquire(self):
        self.acquired = True

    def release(self):
        self.released = True
        if self.release_error:
            self.cleanup_errors.append("synthetic release error")
            raise RuntimeError("synthetic release error")

    def snapshot(self):
        return {"acquired": self.acquired, "released": self.released,
                "cleanup_errors": list(self.cleanup_errors)}


class FakePowerWatcher:
    def __init__(self):
        self.registered = False
        self.stopped = False
        self.events = []

    def start(self):
        self.registered = True

    def stop(self):
        self.stopped = True

    def snapshot(self):
        return {"registered": self.registered, "stopped": self.stopped,
                "subscription_active": self.registered and not self.stopped,
                "cleanup_errors": [], "events": list(self.events)}


class InputRobustnessGuardianTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.clock = FakeClock()
        (self.root / "source.py").write_text("# synthetic source\n", encoding="utf-8")
        self.binding = self.root / "binding.json"
        self.authorization_path = self.root / "authorization.json"
        self.receipt_path = self.root / "ci.json"
        for path in (self.binding, self.authorization_path):
            path.write_text("{}\n", encoding="utf-8")
        self.receipt_path.write_text(json.dumps({
            "commit": COMMIT, "run_id": 17, "status": "completed", "conclusion": "success",
            "jobs": [{"name": name, "run_id": 17, "status": "completed", "conclusion": "success"}
                     for name in sorted(budget.CI_JOBS)]}), encoding="utf-8")
        self.config = {"run_id": AUTHORITY["run_id"], "analysis_status": "synthetic_resource_only",
                       "bound_input_source": {"sha256": guardian.file_hash(self.binding)}}
        self.args = SimpleNamespace(
            reliability_only=True, ledger_head_receipt=None, output_root=self.root / "output",
            budget_ledger=self.root / "ledger", ci_receipt=self.receipt_path, binding=self.binding,
            config=self.root / "config.json", data_root=self.root / "unused_data")
        self.power = FakePowerRequest()
        self.watcher = FakePowerWatcher()

    def run_synthetic(self, *, supervision_error=False, close_gap=False):
        create = budget.BudgetLedger.create
        close = budget.BudgetLedger.close_attempt
        close_calls = []

        def fake_create(path, **kwargs):
            return create(path, **kwargs, wall_clock=lambda: self.clock.wall,
                          monotonic_clock=lambda: self.clock.monotonic)

        def fake_git(command, **kwargs):
            if "status" in command:
                return ""
            return COMMIT + "\n"

        def fake_supervise(command, **kwargs):
            self.clock.advance(0.25)
            self.assertEqual(kwargs["on_poll"](), [])
            if supervision_error:
                raise RuntimeError("synthetic child-supervision failure")
            guardian.write_exclusive(self.args.output_root / "reliability_body_result.json", {
                "passed": True, "cases": [{"passed": True} for _ in range(5)],
                "model_training_steps": 0, "validation_evaluations": 0,
                "test_evaluations": 0, "formal_records": 0})
            return {"returncode": 0, "stop_reasons": [], "owned_processes_remaining": []}

        def fake_close(instance, attempt_id, **kwargs):
            close_calls.append(kwargs["outcome"])
            if close_gap and len(close_calls) == 1:
                self.clock.advance(6)
            return close(instance, attempt_id, **kwargs)

        with ExitStack() as stack:
            for target, value in (
                ("ROOT", self.root), ("DEFAULT_AUTHORIZATION", self.authorization_path),
                ("SUPERVISOR_SOURCES", ("source.py",)),
            ):
                stack.enter_context(patch.object(guardian, target, value))
            stack.enter_context(patch.object(guardian, "ledger_authority", return_value=AUTHORITY))
            stack.enter_context(patch.object(guardian, "environment_identity", return_value={"python": "synthetic", "device": "cpu"}))
            stack.enter_context(patch.object(guardian, "process_identity", return_value={"pid": 123, "create_time": self.clock.wall}))
            stack.enter_context(patch.object(guardian, "verify_manifest", return_value=None))
            stack.enter_context(patch.object(guardian.preflight, "resolved_budget", return_value={}))
            stack.enter_context(patch.object(guardian.preflight, "backend_environment", return_value={}))
            stack.enter_context(patch.object(guardian.subprocess, "check_output", side_effect=fake_git))
            stack.enter_context(patch("psutil.Process", return_value=SimpleNamespace(create_time=lambda: self.clock.wall)))
            stack.enter_context(patch.object(guardian.time, "time", side_effect=lambda: self.clock.wall))
            stack.enter_context(patch.object(guardian.time, "monotonic", side_effect=lambda: self.clock.monotonic))
            stack.enter_context(patch.object(budget.BudgetLedger, "create", side_effect=fake_create))
            stack.enter_context(patch.object(budget.BudgetLedger, "close_attempt", new=fake_close))
            stack.enter_context(patch.object(guardian, "supervise_process", side_effect=fake_supervise))
            stack.enter_context(patch("builtins.print"))
            result = guardian.run_guardian(self.args, self.config, {},
                                           power_request=self.power, power_watcher=self.watcher)
        return result, close_calls

    def read_output(self, name):
        return json.loads((self.args.output_root / name).read_text(encoding="utf-8"))

    def validate(self):
        return guardian.validate_closed_completion(
            self.args.output_root / "closed_ledger_receipt.json", self.args.budget_ledger, AUTHORITY,
            expected_phase="runtime_reliability_only")

    def test_success_requires_digest_bound_closed_head_and_native_cleanup(self):
        result, closes = self.run_synthetic()
        self.assertEqual((result, closes), (0, ["completed"]))
        accepted = self.validate()
        self.assertEqual(accepted["ledger"]["charged_seconds"]["resource"], 0.25)
        self.assertFalse(accepted["ledger"]["must_stop"])
        self.assertFalse(accepted["ledger"]["open_attempts"])
        self.assertIn("unsettled_finalization", accepted["receipt"]["head"])
        self.assertTrue(self.power.released)
        self.assertTrue(self.watcher.stopped)
        self.assertFalse(accepted["completion"]["formal_launch_authorized"])
        self.assertFalse(accepted["completion"]["formal_training_enabled"])

    def test_supervision_exception_preserves_failure_and_closes_owned_attempt(self):
        result, closes = self.run_synthetic(supervision_error=True)
        self.assertEqual((result, closes), (2, ["failed"]))
        self.assertTrue(self.power.released)
        self.assertTrue(self.watcher.stopped)
        self.assertIn("synthetic child-supervision failure", self.read_output("guardian_failure.json")["error"])
        receipt = self.read_output("closed_ledger_receipt.json")
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse(budget.inspect_ledger(self.args.budget_ledger, authority=AUTHORITY,
                                              expected_head=receipt["head"])["open_attempts"])
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self.validate()

    def test_cleanup_failure_cannot_leave_successful_acceptance(self):
        self.power.release_error = True
        result, closes = self.run_synthetic()
        self.assertEqual((result, closes), (2, ["failed"]))
        self.assertTrue(self.watcher.stopped)
        self.assertEqual(self.read_output("runtime_guardian.json")["status"], "failed")
        self.assertEqual(self.read_output("closed_ledger_receipt.json")["status"], "failed")
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self.validate()

    def test_last_close_interval_gap_vetoes_candidate_and_records_failed_close(self):
        result, closes = self.run_synthetic(close_gap=True)
        self.assertEqual((result, closes), (2, ["completed", "failed"]))
        self.assertEqual(self.read_output("acceptance_completion.json")["status"], "passed")
        self.assertTrue(self.read_output("ledger_close_failure.json")["candidate_is_not_authoritative"])
        receipt = self.read_output("closed_ledger_receipt.json")
        self.assertEqual(receipt["status"], "failed")
        inspection = budget.inspect_ledger(self.args.budget_ledger, authority=AUTHORITY, expected_head=receipt["head"])
        self.assertTrue(inspection["must_stop"])
        self.assertFalse(inspection["open_attempts"])
        self.assertEqual(inspection["charged_seconds"]["resource"], 6.25)
        self.assertTrue(any("monitor_gap" in reason for reason in inspection["stop_reasons"]))
        with self.assertRaisesRegex(ValueError, "did not pass"):
            self.validate()

    def test_post_close_completion_edit_breaks_authority(self):
        self.run_synthetic()
        path = self.args.output_root / "acceptance_completion.json"
        completion = self.read_output(path.name)
        completion["source_commit"] = "f" * 40
        path.write_text(json.dumps(completion), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "completion receipt digest mismatch"):
            self.validate()

    def test_post_close_runtime_edit_breaks_authority(self):
        self.run_synthetic()
        path = self.args.output_root / "runtime_guardian.json"
        runtime = self.read_output(path.name)
        runtime["idle_sleep_request_released"] = False
        path.write_text(json.dumps(runtime), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "runtime evidence changed"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
