"""Reject incomplete or stale CI evidence at the research entry."""
import copy
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.input_robustness_budget import BudgetLedger
from scripts.input_robustness_formal_records import FormalEmergencyStop, FormalRecordError, _exclusive_json, digest, file_digest
from scripts.input_robustness_runtime import PowerEventWatcher, TaskPowerRequest
from scripts import mlp_budget_extension_entry as entry
from scripts.mlp_budget_extension_entry import strict_ci_gate
from tests.test_input_robustness_budget import AUTHORITY, gate, provenance
from tests.test_input_robustness_formal_entry import _PowerBackend


class ExtensionCIGateTests(unittest.TestCase):
    def test_only_complete_unique_same_commit_same_run_remote_jobs_pass(self):
        commit = "a" * 40
        jobs = [{"id": index + 10, "name": name, "run_id": 123, "commit": commit,
                 "status": "completed", "conclusion": "success"}
                for index, name in enumerate(("lightweight-verification", "full-protocol-verification", "manuscript-build"))]
        valid = {"status": "passed", "commit": commit, "ci": {
            "kind": "github_actions", "commit": commit, "run_id": 123,
            "status": "completed", "conclusion": "success", "jobs": jobs}}
        invalid = []
        row = copy.deepcopy(valid)
        row["ci"]["jobs"].pop()
        invalid.append(("missing job", row))
        row = copy.deepcopy(valid)
        row["ci"]["jobs"][2] = copy.deepcopy(row["ci"]["jobs"][0])
        invalid.append(("duplicate job", row))
        for key, value in (("commit", "b" * 40), ("run_id", 124), ("conclusion", "failure")):
            row = copy.deepcopy(valid)
            row["ci"]["jobs"][1][key] = value
            invalid.append((key, row))
        row = copy.deepcopy(valid)
        row["ci"]["kind"] = "local_verification"
        invalid.append(("local evidence", row))
        row = copy.deepcopy(valid)
        row["ci"]["run_id"] = None
        for job in row["ci"]["jobs"]:
            job["run_id"] = None
        invalid.append(("null shared run", row))
        for value in (None, 10):
            row = copy.deepcopy(valid)
            row["ci"]["jobs"][2]["id"] = value
            invalid.append((f"invalid job id {value}", row))
        invalid.append(("bare success", {"status": "passed", "commit": commit}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ci.json"
            path.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(strict_ci_gate(path, commit)["run_id"], 123)
            with self.assertRaises(FormalRecordError):
                strict_ci_gate(path, "b" * 40)
            for label, receipt in invalid:
                with self.subTest(label=label):
                    path.write_text(json.dumps(receipt), encoding="utf-8")
                    with self.assertRaises(FormalRecordError):
                        strict_ci_gate(path, commit)


class ExtensionStageControlTests(unittest.TestCase):
    def controller(self, parent, *, name, control_root=None, legacy_root=None):
        ledger = BudgetLedger.create(parent / f"{name}_ledger", authority=AUTHORITY,
                                     monitor_gap_seconds=5)
        ledger.register_phase("stage", provenance=provenance(), gate_receipt=gate())
        writer = SimpleNamespace(root=parent / name, config={"execution": {}}, synthetic=False,
                                 config_path=parent / "config.json",
                                 evidence={"unit_estimates_seconds": {"Cora/normalize_features/000": 1}},
                                 manifest={"source_commit": "a" * 40})
        return entry.ExtensionExecutionController(
            writer=writer, ledger=ledger, phase_id="stage", control_root=control_root,
            legacy_control_root=legacy_root, power_request=TaskPowerRequest(_PowerBackend()),
            power_watcher=PowerEventWatcher(_PowerBackend()))

    def test_stable_and_legacy_markers_are_read_by_every_stage(self):
        for marker in (entry.PAUSE_FILE, entry.STOP_FILE):
            for location in ("stable", "legacy"):
                with self.subTest(marker=marker, location=location), tempfile.TemporaryDirectory() as temporary:
                    parent = Path(temporary)
                    output = parent / "output"
                    stable = entry.control_directory(output)
                    directory = stable if location == "stable" else output
                    directory.mkdir()
                    (directory / marker).touch()
                    for stage in ("prepare", "unit", "finalize"):
                        controller = self.controller(parent, name=stage, control_root=stable,
                                                     legacy_root=output)
                        if marker == entry.STOP_FILE:
                            with self.assertRaises(FormalEmergencyStop):
                                controller._monitor()
                        else:
                            controller._monitor()
                            self.assertTrue(controller.pause_requested)
                    if location == "stable":
                        self.assertFalse(output.exists())

    def test_preparation_pause_creates_resumable_manifest_without_starting_a_unit(self):
        """Exercise the CLI orchestration with worker results supplied at its interface."""
        class Ledger:
            segment_sequence = 0

            def __init__(self):
                self.attempts, self.events = {}, 0
            @property
            def head(self): return {"event_count": self.events}
            def register_phase(self, *args, **kwargs): self.events += 1
            def begin_attempt(self, name, **kwargs):
                self.events += 1
                self.attempts[name] = {**kwargs, "outcome": "open"}
            def close_attempt(self, name, **kwargs):
                self.events += 1
                self.attempts[name].update(kwargs)
                return self.snapshot()
            def poll(self, **kwargs): return {"must_stop": False, "stop_reasons": []}
            def snapshot(self): return {"attempts": copy.deepcopy(self.attempts)}

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            output = parent / "output"
            config_path = parent / "config.json"
            config = {"run_id": "controller-interface-fixture", "execution_mode": "research",
                      "datasets": ["Cora"], "conditions": ["normalize_features"], "seeds": [0],
                      "execution": {}, "budget": {"inventory_roots": ["original"]}}
            _exclusive_json(config_path, config)
            _exclusive_json(parent / "acceptance.json", {"source_commit": "a" * 40,
                                                       "config_sha256": digest(config)})
            _exclusive_json(parent / "binding.json", {})
            _exclusive_json(parent / "ci.json", {})
            evidence = {"unit_estimates_seconds": {"Cora/normalize_features/000": 1}}
            ledger, operations = Ledger(), []
            test = self

            class Controller(entry.ExtensionExecutionController):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs, power_request=TaskPowerRequest(_PowerBackend()),
                                     power_watcher=PowerEventWatcher(_PowerBackend()))
                def _child(self, request, *, attempt_id, cap_seconds=28800):
                    operation = request["operation"]
                    operations.append(operation)
                    if operation == "prepare":
                        test.assertEqual(self.control_root, entry.control_directory(output))
                        if not request["resume"]:
                            test.assertFalse(output.exists())
                            self.control_root.mkdir()
                            marker = self.control_root / entry.PAUSE_FILE
                            marker.touch()
                            self._monitor()
                            # Acknowledging/removing the marker must not erase the latched pause.
                            marker.rename(marker.with_suffix(".observed"))
                        return {"base_evidence": evidence, "completed_keys": []}
                    if operation == "unit":
                        path = output / "records" / request["condition"] / request["dataset"] / "seed_000.json"
                        _exclusive_json(path, {"interface_fixture": True})
                        return {"record_path": str(path), "record_sha256": file_digest(path)}
                    _exclusive_json(output / "run_complete.json", {"interface_fixture": True})
                    return {"status": "complete"}

            arguments = ["entry", "--execute", "--config", str(config_path), "--base-config", str(config_path),
                         "--base-root", str(parent / "base"), "--binding", str(parent / "binding.json"),
                         "--data-root", str(parent / "data"), "--output-root", str(output),
                         "--budget-handoff", str(parent / "handoff.json"), "--budget-ledger", str(parent / "ledger"),
                         "--ci-receipt", str(parent / "ci.json"), "--acceptance-record", str(parent / "acceptance.json")]
            with ExitStack() as stack:
                stack.enter_context(patch.object(entry, "ExtensionExecutionController", Controller))
                stack.enter_context(patch.object(entry, "_source_commit", return_value="a" * 40))
                stack.enter_context(patch.object(entry, "strict_ci_gate", return_value={}))
                stack.enter_context(patch.object(entry, "_environment_binding", return_value={}))
                # The environment gate is exercised in test_mlp24_retry_prerequisites.
                stack.enter_context(patch.object(entry, "formal_environment_preflight", return_value={
                    "schema_version": entry.PREFLIGHT_SCHEMA, "model_devices": {"MLP": "cpu"}, "runtime": {},
                    "base_manifest_environment": {}, "base_record_environments": [{}], "status": "passed"}))
                opener = stack.enter_context(patch("scripts.supplement_budget_handoff.open_inherited_ledger",
                                                   side_effect=lambda *a, **k: nullcontext(ledger)))
                with patch.object(sys, "argv", arguments):
                    self.assertEqual(entry.main(), 0)
                self.assertEqual(operations, ["prepare"])
                self.assertTrue((output / "manifest.json").is_file())
                self.assertFalse((output / "records").exists())
                summaries = list((output / "attempt_summaries").glob("*.json"))
                self.assertEqual(entry.read_json(summaries[0])["status"], "paused")
                self.assertEqual([row["budget_group"] for row in ledger.attempts.values()], ["control"])
                with patch.object(sys, "argv", arguments + ["--resume"]):
                    self.assertEqual(entry.main(), 0)
                self.assertTrue(opener.call_args.kwargs["resume"])
                self.assertEqual(operations, ["prepare", "prepare", "unit", "finalize"])
                self.assertTrue(all(row["outcome"] == "completed" for row in ledger.attempts.values()))

    def test_all_stages_emergency_close_only_after_real_owned_worker_exit(self):
        supervisor = entry.supervise_process
        for operation in ("prepare", "unit", "finalize"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temporary:
                controller = self.controller(Path(temporary), name=operation)
                def supervised_sleep(command, **kwargs):
                    poll = kwargs["on_poll"]
                    def request_stop(process, elapsed):
                        if elapsed >= .25:
                            controller.control_root.mkdir(exist_ok=True)
                            (controller.control_root / entry.STOP_FILE).touch()
                        return poll(process, elapsed)
                    kwargs["on_poll"] = request_stop
                    return supervisor([sys.executable, "-c", "import time; time.sleep(10)"], **kwargs)
                with patch.object(entry, "supervise_process", side_effect=supervised_sleep):
                    if operation == "prepare":
                        with self.assertRaises(FormalEmergencyStop):
                            controller.run_preparation({"operation": operation}, attempt_id="attempt",
                                                       activity_id="prepare")
                    else:
                        units = [{"dataset": "Cora", "condition": "normalize_features", "seed": 0,
                                  "attempt_id": "attempt"}] if operation == "unit" else []
                        result = controller.run_units(units, launch_authorized=True, formal_training_enabled=True,
                                                      final_attempt_id="attempt")
                        self.assertEqual(result["status"], "emergency_stopped")
                attempt = controller.ledger._state["attempts"]["attempt"]
                self.assertEqual(attempt["outcome"], "external_interruption")
                self.assertFalse(attempt["reviewed"])
                self.assertGreater(attempt["charged_ns"], 0)
                failure = entry.read_json(controller.writer.root / "failures/attempt.json")
                self.assertEqual(failure["operation"], operation)
                self.assertTrue(failure["worker_exit_confirmed"])
                receipt = entry.read_json(Path(failure["supervision"]["path"]))
                self.assertNotEqual(receipt["returncode"], 0)
                self.assertEqual(receipt["owned_processes_remaining"], [])
                self.assertTrue(receipt["observed_process_ids"])
                self.assertEqual(failure["supervision"]["sha256"],
                                 file_digest(Path(failure["supervision"]["path"])))
                self.assertTrue(controller.power_request.snapshot()["released"])
                self.assertTrue(controller.power_watcher.snapshot()["stopped"])

    def test_missing_or_incomplete_exit_evidence_preserves_open_attempt(self):
        receipts = [None, {"returncode": 1},
                    {"returncode": None, "owned_processes_remaining": [], "observed_process_ids": [1]},
                    {"returncode": 1, "owned_processes_remaining": [], "observed_process_ids": []},
                    {"returncode": 1, "owned_processes_remaining": [], "observed_process_ids": [None]},
                    {"returncode": 1, "owned_processes_remaining": [{"pid": 1}], "observed_process_ids": [1]}]
        for operation in ("prepare", "unit", "finalize"):
            for index, receipt in enumerate(receipts):
                with self.subTest(operation=operation, receipt=index), tempfile.TemporaryDirectory() as temporary:
                    controller = self.controller(Path(temporary), name=operation)
                    def interrupted(*args, **kwargs):
                        if receipt is not None:
                            _exclusive_json(controller.writer.root / "workers/attempt.supervision.json", receipt)
                        raise KeyboardInterrupt("injected interruption without proven cleanup")
                    with patch.object(controller, "_child", side_effect=interrupted):
                        if operation == "prepare":
                            with self.assertRaises(KeyboardInterrupt):
                                controller.run_preparation({"operation": operation}, attempt_id="attempt",
                                                           activity_id="prepare")
                        else:
                            units = [{"dataset": "Cora", "condition": "normalize_features", "seed": 0,
                                      "attempt_id": "attempt"}] if operation == "unit" else []
                            result = controller.run_units(units, launch_authorized=True, formal_training_enabled=True,
                                                          final_attempt_id="attempt")
                            self.assertEqual(result["status"], "emergency_stopped")
                    self.assertEqual(controller.ledger._state["attempts"]["attempt"]["outcome"], "open")
                    failure = entry.read_json(controller.writer.root / "failures/attempt.json")
                    self.assertFalse(failure["worker_exit_confirmed"])


if __name__ == "__main__":
    unittest.main()
