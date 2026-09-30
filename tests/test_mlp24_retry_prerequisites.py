"""Attempt identity, artifact-collision and pre-registration environment gates.

These guard the recovery path of the MLP-24 environmental-failure amendment.
Every refusal must happen before a budget attempt, charge or failed outcome.
"""
import copy
from contextlib import ExitStack, nullcontext
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.input_robustness_budget import BudgetLedger
from scripts.input_robustness_formal_records import FormalRecordError, _exclusive_json, digest, file_digest
from scripts.input_robustness_runtime import PowerEventWatcher, TaskPowerRequest
from scripts import mlp_budget_extension_entry as entry
from tests.test_input_robustness_budget import AUTHORITY, gate, provenance
from tests.test_input_robustness_formal_entry import _PowerBackend

UNIT_KEY = "Cora/normalize_features/000"
BASE_ENVIRONMENT = {"cuda_available": True, "cuda_device": "Fixture GPU", "cuda_runtime": "12.8",
                    "device": "cuda", "numpy": "2.3.5", "platform": "fixture-os", "python": "3.12.14",
                    "scikit_learn": "1.7.1", "scipy": "1.16.1", "torch": "2.9.1+cu128",
                    "torch_geometric": "2.7.0"}
BASE_MANIFEST_ENVIRONMENT = {"cuda_available": True, "cuda_device": "Fixture GPU", "device_policy": "config-bound",
                             "python": "3.12.14", "torch": "2.9.1+cu128", "torch_cuda": "12.8"}


def cuda_facts(**changes):
    facts = {"sys_executable": "/fixture/venv-gpu/python", "python": "3.12.14", "torch": "2.9.1+cu128",
             "torch_cuda": "12.8", "cuda_available": True, "device_count": 1, "cuda_device": "Fixture GPU",
             "environment_snapshot": dict(BASE_ENVIRONMENT)}
    facts.update(changes)
    return facts


def cpu_only_facts():
    return cuda_facts(sys_executable="/fixture/venv/python", torch="2.9.1+cpu", torch_cuda=None,
                      cuda_available=False, device_count=0, cuda_device=None, environment_snapshot=None)


def write_base(parent: Path, environment=None):
    base = parent / "base"
    _exclusive_json(base / "manifest.json", {"environment": dict(BASE_MANIFEST_ENVIRONMENT)})
    _exclusive_json(base / "records/normalize_features/Cora/MLP/seed_000.json",
                    {"model": "MLP", "dataset": "Cora", "condition": "normalize_features", "seed": 0,
                     "environment": dict(environment or BASE_ENVIRONMENT)})
    return base


class FakeLedger:
    """Interface-level ledger used to observe dispatch order without training."""

    def __init__(self, segment_sequence=0):
        self.segment_sequence = segment_sequence
        self.attempts, self.events, self.phases = {}, 0, []
        self.retry_authorizations = {}

    @property
    def head(self):
        return {"event_count": self.events}

    def register_phase(self, phase, **kwargs):
        self.phases.append(phase)
        self.events += 1

    def begin_attempt(self, name, **kwargs):
        self.events += 1
        self.attempts[name] = {**kwargs, "outcome": "open"}

    def close_attempt(self, name, **kwargs):
        self.events += 1
        self.attempts[name].update(kwargs)
        return self.snapshot()

    def known_attempt_ids(self):
        return set(self.attempts)

    def pending_failure_retries(self):
        return []

    def poll(self, **kwargs):
        return {"must_stop": False, "stop_reasons": []}

    def snapshot(self):
        return {"attempts": copy.deepcopy(self.attempts),
                "retry_authorizations": copy.deepcopy(self.retry_authorizations)}


class EntryHarness:
    """Drive entry.main() with worker results supplied at the controller interface."""

    def __init__(self, parent: Path, *, device="cuda"):
        self.parent = parent
        self.config = {"run_id": "prerequisite-fixture", "execution_mode": "research",
                       "datasets": ["Cora"], "conditions": ["normalize_features"], "seeds": [0],
                       "execution": {"model_devices": {"MLP": device}},
                       "budget": {"inventory_roots": ["original"]}}
        self.config_path = parent / "config.json"
        _exclusive_json(self.config_path, self.config)
        _exclusive_json(parent / "acceptance.json", {"source_commit": "a" * 40, "config_sha256": digest(self.config)})
        _exclusive_json(parent / "binding.json", {})
        _exclusive_json(parent / "ci.json", {})
        self.base = write_base(parent)
        self.operations, self.opened = [], []

    def arguments(self, output):
        p = self.parent
        return ["entry", "--execute", "--config", str(self.config_path), "--base-config", str(self.config_path),
                "--base-root", str(self.base), "--binding", str(p / "binding.json"),
                "--data-root", str(p / "data"), "--output-root", str(output),
                "--budget-handoff", str(p / "handoff.json"), "--budget-ledger", str(p / "ledger"),
                "--ci-receipt", str(p / "ci.json"), "--acceptance-record", str(p / "acceptance.json")]

    def run(self, output, ledger, *, facts=None):
        harness = self
        evidence = {"unit_estimates_seconds": {UNIT_KEY: 1}}

        class Controller(entry.ExtensionExecutionController):
            def __init__(self, **kwargs):
                super().__init__(**kwargs, power_request=TaskPowerRequest(_PowerBackend()),
                                 power_watcher=PowerEventWatcher(_PowerBackend()))

            def _child(self, request, *, attempt_id, cap_seconds=28800):
                harness.operations.append((request["operation"], attempt_id))
                if request["operation"] == "prepare":
                    return {"base_evidence": evidence, "completed_keys": []}
                if request["operation"] == "unit":
                    path = output / "records" / request["condition"] / request["dataset"] / "seed_000.json"
                    _exclusive_json(path, {"interface_fixture": True})
                    return {"record_path": str(path), "record_sha256": file_digest(path)}
                _exclusive_json(output / "run_complete.json", {"interface_fixture": True})
                return {"status": "complete"}

        def opener(*args, **kwargs):
            harness.opened.append(kwargs)
            return nullcontext(ledger)

        with ExitStack() as stack:
            stack.enter_context(patch.object(entry, "ExtensionExecutionController", Controller))
            stack.enter_context(patch.object(entry, "_source_commit", return_value="a" * 40))
            stack.enter_context(patch.object(entry, "strict_ci_gate", return_value={}))
            stack.enter_context(patch.object(entry, "_environment_binding", return_value={}))
            stack.enter_context(patch.object(entry, "runtime_environment_probe", create=True,
                                             return_value=facts or cuda_facts()))
            stack.enter_context(patch("scripts.supplement_budget_handoff.open_inherited_ledger", side_effect=opener))
            stack.enter_context(patch.object(sys, "argv", self.arguments(output)))
            return entry.main()


def stage_controller(parent: Path, name: str):
    ledger = BudgetLedger.create(parent / f"{name}_ledger", authority=AUTHORITY, monitor_gap_seconds=5)
    ledger.register_phase("stage", provenance=provenance(), gate_receipt=gate())
    writer = SimpleNamespace(root=parent / name, config={"execution": {}}, synthetic=False,
                             config_path=parent / "config.json",
                             evidence={"unit_estimates_seconds": {UNIT_KEY: 1}},
                             manifest={"source_commit": "a" * 40})
    return entry.ExtensionExecutionController(
        writer=writer, ledger=ledger, phase_id="stage", power_request=TaskPowerRequest(_PowerBackend()),
        power_watcher=PowerEventWatcher(_PowerBackend()))


class CrossSegmentAttemptIdentityTests(unittest.TestCase):
    def test_equal_event_counts_in_two_segments_never_reuse_an_attempt_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            harness = EntryHarness(parent)
            first, second = FakeLedger(segment_sequence=0), FakeLedger(segment_sequence=1)
            self.assertEqual(harness.run(parent / "run_one", first), 0)
            self.assertEqual(harness.run(parent / "run_two", second), 0)
            self.assertTrue(first.attempts and second.attempts)
            self.assertEqual(set(first.attempts) & set(second.attempts), set())
            for ledger in (first, second):
                tag = f"s{ledger.segment_sequence:03d}"
                self.assertTrue(all(tag in name for name in ledger.attempts), sorted(ledger.attempts))

    def test_unit_attempt_identity_binds_segment_event_and_unit_ordinal(self):
        ledger = FakeLedger(segment_sequence=4)
        ledger.events = 9
        self.assertEqual(entry.attempt_prefix(ledger), "mlp24_s004e9")
        self.assertEqual(entry.unit_attempt_id("mlp24_s004e9", 17), "mlp24_s004e9_u000017")
        self.assertEqual(entry.unit_attempt_id("mlp24_s004e9", 17, retry_ordinal=1), "mlp24_s004e9_u000017_retry1")
        # The historical failed identity has no segment tag and can never be produced.
        self.assertNotEqual(entry.unit_attempt_id(entry.attempt_prefix(FakeLedger(1)), 0), "mlp24_2_000000")

    def test_entry_records_a_granted_retry_before_dispatch_and_uses_its_bound_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            harness = EntryHarness(parent)
            calls = []

            class RetryLedger(FakeLedger):
                def pending_failure_retries(self):
                    if self.retry_authorizations:
                        return []
                    return [{"review_record_sha256": "7" * 64, "unit_id": "unit_Cora_normalize_features_MLP_000"}]

                def authorize_failure_retry(self, review_sha, *, retry_attempt_id, phase_id):
                    calls.append((review_sha, retry_attempt_id, phase_id, sorted(self.attempts)))
                    self.events += 1
                    self.retry_authorizations["unit_Cora_normalize_features_MLP_000"] = {
                        "retry_attempt_id": retry_attempt_id, "consumed_by": None}
                    return self.snapshot()

            ledger = RetryLedger(segment_sequence=1)
            self.assertEqual(harness.run(parent / "run", ledger), 0)
            expected = entry.unit_attempt_id("mlp24_s001e1", 0, retry_ordinal=1)
            self.assertEqual(calls, [("7" * 64, expected, "mlp24_extension", [])])
            self.assertIn(("unit", expected), harness.operations)
            self.assertEqual([name for name in ledger.attempts if "_u" in name], [expected])


class ArtifactCollisionTests(unittest.TestCase):
    def assert_refused_before_attempt(self, controller, calls, result=None):
        self.assertEqual(calls, [])
        self.assertNotIn("attempt", controller.ledger._state["attempts"])
        self.assertEqual(controller.ledger.snapshot()["charged_seconds"]["total"], 0)
        if result is not None:
            self.assertEqual(result["status"], "failed")
            self.assertIn("identity", result["failure"])

    def unit(self):
        return [{"dataset": "Cora", "condition": "normalize_features", "seed": 0, "attempt_id": "attempt"}]

    def test_existing_worker_artifact_is_refused_before_unit_attempt(self):
        for leftover in ("workers/attempt.pt", "workers/attempt.json", "workers/attempt.log",
                         "workers/attempt.supervision.json"):
            with self.subTest(leftover=leftover), tempfile.TemporaryDirectory() as temporary:
                controller = stage_controller(Path(temporary), "unit")
                _exclusive_json(controller.writer.root / leftover, {"previous": True})
                calls = []
                with patch.object(controller, "_child", side_effect=lambda *a, **k: calls.append(a)):
                    result = controller.run_units(self.unit(), launch_authorized=True,
                                                  formal_training_enabled=True, finalize=False)
                self.assert_refused_before_attempt(controller, calls, result)

    def test_existing_failure_artifact_is_refused_before_unit_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = stage_controller(Path(temporary), "unit")
            _exclusive_json(controller.writer.root / "failures/attempt.json", {"status": "failed"})
            calls = []
            with patch.object(controller, "_child", side_effect=lambda *a, **k: calls.append(a)):
                result = controller.run_units(self.unit(), launch_authorized=True,
                                              formal_training_enabled=True, finalize=False)
            self.assert_refused_before_attempt(controller, calls, result)

    def test_reused_preparation_scratch_is_refused_before_a_failed_outcome(self):
        """The actual hazard: a new segment reusing a prior `_prepare` request file."""
        with tempfile.TemporaryDirectory() as temporary:
            controller = stage_controller(Path(temporary), "scratch")
            _exclusive_json(controller.writer.root / "workers/attempt.pt", {"previous": True})
            with self.assertRaisesRegex(FormalRecordError, "identity"):
                controller.run_preparation({"operation": "prepare"}, attempt_id="attempt", activity_id="prepare")
            self.assert_refused_before_attempt(controller, [])
            self.assertFalse(any(row["outcome"] == "failed"
                                 for row in controller.ledger._state["attempts"].values()))

    def test_entry_refuses_existing_scratch_directory_before_any_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            harness = EntryHarness(parent)
            ledger = FakeLedger(segment_sequence=2)
            output = parent / "run"
            # Register the phase event first, as the entry does, to predict its prefix.
            predicted = FakeLedger(segment_sequence=2)
            predicted.events = 1
            scratch = output.parent / f"{output.name}_preparation_{entry.attempt_prefix(predicted)}"
            (scratch / "workers").mkdir(parents=True)
            with self.assertRaisesRegex(FormalRecordError, "identity"):
                harness.run(output, ledger)
            self.assertEqual(ledger.attempts, {})
            self.assertEqual(harness.operations, [])


class EnvironmentPreflightTests(unittest.TestCase):
    def config(self, device="cuda"):
        return {"execution": {"model_devices": {"MLP": device}}}

    def test_cpu_only_torch_is_rejected_for_cuda_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = write_base(Path(temporary))
            with self.assertRaisesRegex(entry.EnvironmentPreflightError, "CPU-only"):
                entry.formal_environment_preflight(self.config(), base, probe=lambda device: cpu_only_facts())

    def test_cuda_build_without_available_device_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = write_base(Path(temporary))
            facts = cuda_facts(cuda_available=False, device_count=0, cuda_device=None, environment_snapshot=None)
            with self.assertRaisesRegex(entry.EnvironmentPreflightError, "CUDA unavailable"):
                entry.formal_environment_preflight(self.config(), base, probe=lambda device: facts)

    def test_environment_identity_mismatch_with_base_is_rejected(self):
        changed = [cuda_facts(cuda_device="Other GPU", environment_snapshot={**BASE_ENVIRONMENT, "cuda_device": "Other GPU"}),
                   cuda_facts(torch="2.9.2+cu128", environment_snapshot={**BASE_ENVIRONMENT, "torch": "2.9.2+cu128"}),
                   cuda_facts(environment_snapshot={**BASE_ENVIRONMENT, "platform": "changed-os"})]
        for facts in changed:
            with self.subTest(facts=facts), tempfile.TemporaryDirectory() as temporary:
                base = write_base(Path(temporary))
                with self.assertRaisesRegex(entry.EnvironmentPreflightError, "differs"):
                    entry.formal_environment_preflight(self.config(), base, probe=lambda device, f=facts: f)

    def test_matching_cuda_environment_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = write_base(Path(temporary))
            result = entry.formal_environment_preflight(self.config(), base, probe=lambda device: cuda_facts())
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["model_devices"], {"MLP": "cuda"})
            self.assertEqual(result["runtime"]["torch_cuda"], "12.8")
            self.assertEqual(result["runtime"]["sys_executable"], "/fixture/venv-gpu/python")

    def test_gate_refuses_before_ledger_phase_or_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            harness = EntryHarness(parent)
            ledger = FakeLedger()
            with self.assertRaises((SystemExit, entry.EnvironmentPreflightError)):
                harness.run(parent / "run", ledger, facts=cpu_only_facts())
            self.assertEqual(harness.opened, [])
            self.assertEqual((ledger.phases, ledger.attempts), ([], {}))
            evidence = list((parent / "run_environment_preflight").glob("*.json"))
            self.assertEqual(len(evidence), 1)
            self.assertEqual(entry.read_json(evidence[0])["status"], "refused")
            self.assertFalse((parent / "run").exists())


if __name__ == "__main__":
    unittest.main()
