"""Acceptance boundaries for outcome-blind probes and immutable summaries.

All training in this test module is a three-step toy CPU probe. Synthetic
artifact sets exercise acceptance without touching research data or the GPU.
"""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.prospective_models import MLPModel
from scripts import preflight_input_robustness_11 as preflight


def frozen_config():
    return json.loads(preflight.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def write_fixture(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class ProbeContractTests(unittest.TestCase):
    def test_memory_monitor_includes_real_interpreter_descendants_and_windows_high_water(self):
        import psutil
        interpreter = SimpleNamespace(pid=2, memory_info=lambda: SimpleNamespace(rss=4000, peak_wset=8000))
        helper = SimpleNamespace(pid=3, memory_info=lambda: SimpleNamespace(rss=500))
        launcher = SimpleNamespace(pid=1, children=lambda recursive: [interpreter, helper],
                                   memory_info=lambda: SimpleNamespace(rss=100, peak_wset=150))
        measured = preflight.process_tree_memory(launcher, psutil)
        self.assertEqual(measured["rss_bytes"], 4600)
        self.assertEqual(measured["high_water_bytes"], 8650)
        self.assertEqual(measured["process_ids"], [1, 2, 3])

    def test_owned_subprocess_tree_is_measured_and_fully_stopped(self):
        import subprocess
        import sys
        import time
        import psutil
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "child.pid"
            child_code = "import os,time,pathlib; allocation=bytearray(32*1024**2); pathlib.Path(" + repr(str(marker)) + ").write_text(str(os.getpid())); time.sleep(30)"
            parent_code = "import subprocess,sys; subprocess.Popen([sys.executable, '-c', " + repr(child_code) + "]).wait()"
            process = subprocess.Popen([sys.executable, "-c", parent_code],
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            child_pid = None
            try:
                deadline = time.monotonic() + 10
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(marker.exists(), "owned subprocess fixture failed to start")
                child_pid = int(marker.read_text())
                measured = preflight.process_tree_memory(psutil.Process(process.pid), psutil)
                self.assertIn(child_pid, measured["process_ids"])
                self.assertGreaterEqual(measured["rss_bytes"], 32 * 1024**2)
            finally:
                preflight.terminate_process_tree(process, psutil)
                process.wait(timeout=10)
            if child_pid is not None and psutil.pid_exists(child_pid):
                self.assertEqual(psutil.Process(child_pid).status(), psutil.STATUS_ZOMBIE)

    def test_checkpoint_callback_runs_each_step_and_matches_recorded_state(self):
        snapshots = []

        def retain_checkpoint(model):
            snapshots.append({name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()})

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            torch.manual_seed(0)
            model = MLPModel(4, 8, 2, 0.5)
            # Include an unused sparse buffer to exercise full state retention.
            model.register_buffer("probe_adjacency", torch.sparse_coo_tensor(
                torch.tensor([[0, 1], [1, 0]]), torch.ones(2), (12, 12)))
            result = preflight.train_probe(
                model=model, x=torch.arange(48, dtype=torch.float32).view(12, 4) / 48,
                edge_index=torch.empty((2, 0), dtype=torch.long),
                train_indices=torch.tensor([0, 2, 4, 7, 9, 11]),
                train_labels=torch.tensor([0, 1, 0, 1, 0, 1]),
                trial={"learning_rate": 0.01, "dropout": 0.5}, training_steps=3,
                weight_decay=0.0005, torch=torch, checkpoint_sink=retain_checkpoint)
            self.assertEqual(len(snapshots), 3)
            for state, step in zip(snapshots, result["trajectory"]):
                self.assertTrue(state["probe_adjacency"].is_sparse)
                fingerprint = preflight.json_hash({name: preflight.tensor_hash(value)
                                                   for name, value in sorted(state.items())})
                self.assertEqual(fingerprint, step["state"])
            self.assertEqual(result["validation_evaluations"], 0)
            self.assertEqual(result["test_evaluations"], 0)
        finally:
            torch.set_num_threads(previous_threads)

    def test_source_binding_rejects_executable_config_and_binding_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = ("worker.py", "models.py")
            for name in sources:
                (root / name).write_text("# frozen fixture\n", encoding="utf-8")
            binding = root / "binding.json"
            binding.write_text('{"bound": true}\n', encoding="utf-8")
            config = {"scope": "preflight_only", "seed": 0}
            manifest = {"phase": "preflight_only", "config_sha256": preflight.json_hash(config),
                        "binding_sha256": preflight.file_hash(binding),
                        "source_files": {name: preflight.file_hash(root / name) for name in sources}}
            with patch.object(preflight, "ROOT", root), patch.object(preflight, "SOURCE_FILES", sources):
                preflight.verify_source_binding(manifest, config, binding)
                with self.subTest(drift="config"), self.assertRaisesRegex(ValueError, "worker/config"):
                    preflight.verify_source_binding(manifest, {**config, "seed": 9}, binding)
                for name in (*sources, "binding.json"):
                    path = root / name
                    frozen_bytes = path.read_bytes()
                    path.write_bytes(frozen_bytes + b"\n# changed\n")
                    try:
                        with self.subTest(drift=name), self.assertRaisesRegex(ValueError, "changed"):
                            preflight.verify_source_binding(manifest, config, binding)
                    finally:
                        path.write_bytes(frozen_bytes)
                incomplete = copy.deepcopy(manifest)
                del incomplete["source_files"]["models.py"]
                with self.assertRaisesRegex(ValueError, "incomplete executable-source"):
                    preflight.verify_source_binding(incomplete, config, binding)
                preflight.verify_source_binding(manifest, config, binding)

    def test_train_probe_accepts_only_training_labels_and_reports_no_heldout_metrics(self):
        parameters = inspect.signature(preflight.train_probe).parameters
        self.assertIn("train_labels", parameters)
        for name in ("y", "labels", "validation_indices", "validation_labels", "test_indices", "test_labels"):
            self.assertNotIn(name, parameters)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            x = torch.arange(48, dtype=torch.float32).view(12, 4) / 48
            indices = torch.tensor([0, 2, 4, 7, 9, 11])
            train_labels = torch.tensor([0, 1, 0, 1, 0, 1])
            edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
            arguments = dict(x=x, edge_index=edges, train_indices=indices,
                             train_labels=train_labels, trial={"learning_rate": 0.01, "dropout": 0.5},
                             training_steps=3, weight_decay=0.0005, torch=torch)
            torch.manual_seed(0)
            first = preflight.train_probe(model=MLPModel(4, 8, 2, 0.5), **arguments)
            torch.manual_seed(0)
            second = preflight.train_probe(model=MLPModel(4, 8, 2, 0.5), **arguments)
            self.assertEqual(first["initial_state"], second["initial_state"])
            self.assertEqual(first["trajectory"], second["trajectory"])
            self.assertEqual(len(first["trajectory"]), 3)
            self.assertEqual(first["validation_evaluations"], 0)
            self.assertEqual(first["test_evaluations"], 0)
            self.assertNotIn("validation_accuracy", first)
            self.assertNotIn("test_accuracy", first)
            self.assertNotIn("selected_trial_id", first)
            with self.assertRaises(TypeError):
                preflight.train_probe(model=MLPModel(4, 8, 2, 0.5), test_labels=train_labels, **arguments)
        finally:
            torch.set_num_threads(previous_threads)

    def test_nonfinite_training_loss_fails_before_success_artifact(self):
        x = torch.full((4, 2), float("nan"))
        with self.assertRaises(FloatingPointError):
            preflight.train_probe(model=MLPModel(2, 4, 2, 0.5), x=x,
                                  edge_index=torch.empty((2, 0), dtype=torch.long),
                                  train_indices=torch.tensor([0, 1]), train_labels=torch.tensor([0, 1]),
                                  trial={"learning_rate": 0.01}, training_steps=3,
                                  weight_decay=0.0005, torch=torch)

    def test_sparse_hash_is_canonical_and_sensitive_to_values_shape_and_buffers(self):
        indices = torch.tensor([[0, 2, 1], [1, 0, 2]])
        values = torch.tensor([1.0, 2.0, 3.0])
        first = torch.sparse_coo_tensor(indices, values, (3, 3))
        reordered = torch.sparse_coo_tensor(indices[:, [2, 0, 1]], values[[2, 0, 1]], (3, 3))
        self.assertEqual(preflight.tensor_hash(first), preflight.tensor_hash(reordered))
        self.assertNotEqual(preflight.tensor_hash(first), preflight.tensor_hash(
            torch.sparse_coo_tensor(indices, values + 1, (3, 3))))
        self.assertNotEqual(preflight.tensor_hash(first), preflight.tensor_hash(
            torch.sparse_coo_tensor(indices, values, (4, 4))))
        self.assertNotEqual(preflight.tensor_hash(first), preflight.tensor_hash(first.double()))
        model = torch.nn.Module()
        model.register_buffer("two_hop", first)
        before = preflight.state_hash(model)
        model.two_hop = torch.sparse_coo_tensor(indices, values + 1, (3, 3))
        self.assertNotEqual(before, preflight.state_hash(model))

    def test_exclusive_write_preserves_existing_artifact_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            preflight.write_exclusive(path, {"preserved": True})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                preflight.write_exclusive(path, {"preserved": False})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_startup_environment_is_fixed_without_mutating_parent_environment(self):
        parent = {"PYTHONHASHSEED": "9", "CUBLAS_WORKSPACE_CONFIG": ":16:8", "KEEP": "yes"}
        result = preflight.backend_environment(parent)
        self.assertEqual(parent["PYTHONHASHSEED"], "9")
        self.assertEqual(result["PYTHONHASHSEED"], "0")
        self.assertEqual(result["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(result["OMP_NUM_THREADS"], "4")
        self.assertEqual(result["KEEP"], "yes")

    def test_frozen_scope_and_fail_closed_backend_are_enforced(self):
        config = frozen_config()
        preflight.validate_config(config)
        changes = [
            ("seeds", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            ("datasets", config["datasets"][:-1]),
            ("conditions", ["normalize_features", "raw"]),
            ("formal_training_enabled", True),
        ]
        for key, value in changes:
            candidate = copy.deepcopy(config)
            candidate[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                preflight.validate_config(candidate)
        for section, field, value in [
            ("training", "trials", config["training"]["trials"][:3]),
            ("preflight", "seed", 1),
            ("preflight", "repeats", 3),
            ("execution", "deterministic_algorithms", False),
            ("execution", "deterministic_warn_only", True),
        ]:
            candidate = copy.deepcopy(config)
            candidate[section][field] = value
            with self.subTest(field=f"{section}.{field}"), self.assertRaises(ValueError):
                preflight.validate_config(candidate)
        candidate = copy.deepcopy(config)
        candidate["model_parameters"]["H2GCN"]["propagation_layers"] = 1
        with self.assertRaises(ValueError):
            preflight.validate_config(candidate)


class SummaryAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = frozen_config()
        self.expected = preflight.workers(self.config)
        self.jobs = []
        self.environment = {"torch": "fixture", "device": "cuda", "deterministic_algorithms": True,
                            "deterministic_warn_only": False}
        digest = preflight.json_hash(self.config)
        for worker in self.expected:
            folder = self.directory / "workers" / worker["worker_id"]
            common = {**worker, "phase": "preflight_only", "run_id": self.config["run_id"],
                      "analysis_status": self.config["analysis_status"], "config_sha256": digest,
                      "validation_evaluations": 0, "test_evaluations": 0}
            header = {**common, "input_sha256": "a" * 64, "transformed_sha256": "b" * 64,
                      "split_id": "c" * 64, "transform_metadata": {"fit_partition": "train_features_only"},
                      "data_provenance": {"bound": True}, "environment": self.environment,
                      "transform_seconds": 0.01}
            write_fixture(folder / "worker.json", header)
            write_fixture(folder / "worker_complete.json", {**common, "probes": 28, "failed_probes": 0,
                                                            "seconds": 0.1})
            for model in self.config["models"]:
                for index, trial in enumerate(self.config["training"]["trials"]):
                    row = {**common, "status": "success", "model": model, "seed": 0,
                           "trial_id": f"trial_{index:03d}", "configuration": trial,
                           "device": "cpu" if model == "H2GCN" else "cuda",
                           "initial_state": "d" * 64,
                           "trajectory": [{"step": step, "state": "e" * 64,
                                           "train_logits": "f" * 64, "train_loss": "1" * 64,
                                           "gradients": "2" * 64, "unscored_inference": "3" * 64}
                                          for step in range(3)],
                           "training_steps": 3, "epoch_seconds": [0.001, 0.001, 0.001],
                           "peak_cuda_reserved_bytes": 0, "peak_cuda_allocated_bytes": 0,
                           "rss_bytes": 1024, "four_dense_checkpoints_bytes": 0,
                           "four_checkpoints_bytes": 0, "retained_checkpoint_bytes": 0,
                           "retained_checkpoint_count": index + 1,
                           "h2_preparation_seconds": 0.0}
                    write_fixture(folder / "probes" / model / f"trial_{index:03d}.json", row)
            self.jobs.append({**worker, "returncode": 0, "stop_reason": None,
                              "wall_seconds": 0.1, "peak_rss_bytes": 1024,
                              "memory_scope": "owned_worker_process_tree_sum_including_windows_high_water",
                              "observed_process_ids": [123]})

    def summary(self):
        return preflight.summarize(self.config, self.directory, self.jobs)

    def probe_path(self, repeat=0):
        worker = next(w for w in self.expected if w["dataset"] == self.config["datasets"][0]
                      and w["condition"] == self.config["conditions"][0] and w["repeat"] == repeat)
        return self.directory / "workers" / worker["worker_id"] / "probes" / "MLP" / "trial_000.json"

    def mutate(self, path, **changes):
        row = json.loads(path.read_text())
        row.update(changes)
        write_fixture(path, row)

    def assert_rejected(self):
        result = self.summary()
        self.assertFalse(result["resource_and_short_repeatability_passed"])
        self.assertFalse(result["formal_launch_authorized"])
        self.assertTrue(result["failures"])

    def test_complete_matching_preflight_passes_without_authorizing_formal_training(self):
        result = self.summary()
        self.assertTrue(result["resource_and_short_repeatability_passed"], result["failures"])
        self.assertEqual(result["bitwise_matching_paired_cells"], 616)
        self.assertEqual(result["formal_records"], 0)
        self.assertEqual(result["validation_evaluations"], 0)
        self.assertEqual(result["test_evaluations"], 0)
        self.assertFalse(result["formal_launch_authorized"])

    def test_missing_probe_cannot_be_averaged_away(self):
        self.probe_path().unlink()
        self.assert_rejected()

    def test_mismatched_training_state_rejects_even_when_initial_states_match(self):
        path = self.probe_path()
        row = json.loads(path.read_text())
        row["trajectory"][0]["state"] = "9" * 64
        write_fixture(path, row)
        self.assert_rejected()

    def test_mixed_runtime_rejects_even_if_training_fingerprints_match(self):
        folder = self.directory / "workers" / self.expected[0]["worker_id"]
        self.mutate(folder / "worker.json", environment={**self.environment, "torch": "different-build"})
        self.assert_rejected()

    def test_duplicate_job_ids_cannot_replace_missing_worker_identity(self):
        self.jobs[-1] = copy.deepcopy(self.jobs[0])
        self.assert_rejected()

    def test_launcher_only_memory_record_cannot_be_accepted(self):
        self.jobs[0].pop("memory_scope")
        self.jobs[0].pop("observed_process_ids")
        self.assert_rejected()

    def test_missing_worker_completion_is_not_success(self):
        (self.directory / "workers" / self.expected[0]["worker_id"] / "worker_complete.json").unlink()
        self.assert_rejected()

    def test_both_repeats_from_wrong_config_are_rejected(self):
        for worker in self.expected[:2]:
            self.mutate(self.directory / "workers" / worker["worker_id"] / "worker.json", config_sha256="0" * 64)
        self.assert_rejected()

    def test_both_repeats_with_wrong_seed_or_trial_are_rejected(self):
        for repeat in (0, 1):
            self.mutate(self.probe_path(repeat), seed=9, trial_id="trial_003")
        self.assert_rejected()

    def test_test_evaluation_cannot_be_reported_as_zero_by_summary(self):
        for repeat in (0, 1):
            self.mutate(self.probe_path(repeat), test_evaluations=1, test_accuracy=0.9)
        self.assert_rejected()

    def test_heldout_scores_are_rejected_even_if_evaluation_counter_claims_zero(self):
        for repeat in (0, 1):
            self.mutate(self.probe_path(repeat), validation_accuracy=0.9)
        self.assert_rejected()

    def test_worker_limit_failure_blocks_acceptance_despite_complete_matching_files(self):
        self.jobs[0]["stop_reason"] = "worker RSS ceiling exceeded"
        self.assert_rejected()


if __name__ == "__main__":
    unittest.main()
