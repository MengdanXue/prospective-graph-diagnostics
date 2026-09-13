"""CPU toy and synthetic evidence checks; no research data or performance round."""
from __future__ import annotations

import copy
import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from experiments.prospective_models import H2GCNModel, prepare_h2_adjacencies
from scripts import diagnose_h2gcn_performance as diagnostic
from scripts.h2gcn_checkpoint_diagnostic import CheckpointStore


class H2PerformanceDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_fixed_scope_and_cost_rank_cannot_be_changed_after_observation(self):
        parent = diagnostic.validate_config(self.config)
        self.assertEqual(len(parent["datasets"]), 11)
        self.assertEqual(self.config["selection"]["datasets"], ["Squirrel", "Actor"])
        self.assertEqual(self.config["selection"]["trial_ids"],
                         {"Squirrel": "trial_000", "Actor": "trial_001"})
        mutations = (
            lambda c: c["selection"].update(datasets=["Squirrel", "Coauthor-CS"]),
            lambda c: c["selection"]["trial_ids"].update(Actor="trial_002"),
            lambda c: c.update(cpu_threads=[1, 2, 4, 8]),
            lambda c: c["correctness"].update(rtol=1e-3),
            lambda c: c["steps"].update(steady=list(range(4, 10))),
            lambda c: c.update(formal_training_enabled=True),
        )
        for mutate in mutations:
            altered = copy.deepcopy(self.config)
            mutate(altered)
            with self.subTest(altered=altered != self.config):
                with self.assertRaises(ValueError):
                    diagnostic.validate_config(altered)

    def test_worker_schedule_is_complete_unique_and_second_repeat_reverses_first(self):
        jobs = diagnostic.workers(self.config)
        self.assertEqual(len(jobs), 48)
        self.assertEqual(len({job["worker_id"] for job in jobs}), 48)
        setting = lambda job: (job["dataset"], job["condition"], job["cpu_threads"], job["checkpoint_mode"])
        expected = set(itertools.product(["Squirrel", "Actor"], self.config["conditions"],
                                         [1, 4, 8], self.config["checkpoint_modes"]))
        self.assertEqual({setting(job) for job in jobs[:24]}, expected)
        self.assertEqual([setting(job) for job in jobs[24:]],
                         list(reversed([setting(job) for job in jobs[:24]])))
        self.assertEqual([job["repeat"] for job in jobs], [0] * 24 + [1] * 24)
        self.assertEqual(jobs[0]["cpu_threads"], 4)
        self.assertEqual(self.config["steps"], {"total": 10, "cold": [0], "warmup": [1, 2],
                                               "steady": [3, 4, 5, 6, 7, 8, 9]})

    def test_fresh_worker_startup_enforces_cpu_only_deterministic_backend(self):
        code = """
import json
import sys
from scripts import diagnose_h2gcn_performance as diagnostic
assert 'torch' not in sys.modules
torch = diagnostic.configure_cpu(1)
x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
loss = x.square().sum()
loss.backward()
print(json.dumps({
    'threads': torch.get_num_threads(),
    'interop': torch.get_num_interop_threads(),
    'strict': torch.are_deterministic_algorithms_enabled(),
    'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
    'cuda_available': torch.cuda.is_available(),
    'cuda_initialized': torch.cuda.is_initialized(),
    'tensor_device': x.device.type,
    'gradient': x.grad.tolist(),
}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=diagnostic.ROOT,
            env=diagnostic.worker_environment(1), capture_output=True, text=True,
            timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {
            "threads": 1, "interop": 1, "strict": True, "warn_only": False,
            "cuda_available": False, "cuda_initialized": False, "tensor_device": "cpu",
            "gradient": [[2., 4.], [6., 8.]],
        })

    def test_ci_receipt_requires_exact_commit_and_complete_single_run(self):
        commit = "b" * 40
        receipt = {"commit": commit, "status": "completed", "conclusion": "success", "run_id": 123,
                   "jobs": [{"name": name, "status": "completed", "conclusion": "success", "run_id": 123}
                            for name in ("lightweight-verification", "full-protocol-verification", "manuscript-build")]}
        diagnostic.validate_ci_receipt(receipt, commit)
        altered = copy.deepcopy(receipt)
        altered["commit"] = "c" * 40
        missing_job = copy.deepcopy(receipt)
        missing_job["jobs"].pop()
        mixed_run = copy.deepcopy(receipt)
        mixed_run["jobs"][0]["run_id"] = 124
        missing_run = copy.deepcopy(receipt)
        missing_run.pop("run_id")
        for job in missing_run["jobs"]:
            job.pop("run_id")
        for corrupt in (altered, missing_job, mixed_run, missing_run):
            with self.subTest(corrupt=corrupt):
                with self.assertRaises(ValueError):
                    diagnostic.validate_ci_receipt(corrupt, commit)

    def test_exclusive_binary_publication_preserves_completed_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "evidence.npz"
            diagnostic.write_bytes_exclusive(path, b"first\x00record")
            before = (path.read_bytes(), path.stat().st_mtime_ns)
            with self.assertRaises(FileExistsError):
                diagnostic.write_bytes_exclusive(path, b"replacement")
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_numeric_snapshots_do_not_follow_later_parameter_or_adam_updates(self):
        torch.manual_seed(41)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        x = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 15
        rng_before = torch.get_rng_state().clone()
        logits = model(x)
        loss = logits.square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            inferred = model(x)
        packed = diagnostic.pack_dense_evidence(model, optimizer, logits, inferred, loss, rng_before, torch)
        frozen = {name: value.copy() for name, value in packed.items()}
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(100)
                parameter.grad.add_(200)
                for value in optimizer.state[parameter].values():
                    value.add_(300)
            logits.add_(400)
            inferred.add_(500)
            loss.add_(600)
        rng_before.zero_()
        for name, value in frozen.items():
            np.testing.assert_array_equal(packed[name], value, err_msg=f"mutable evidence: {name}")

    def test_numeric_comparison_enforces_shape_dtype_finiteness_exact_rng_and_tolerance(self):
        original = {"step000.parameters": np.array([1., 2.], dtype=np.float32),
                    "step000.train_loss": np.array(.5, dtype=np.float32),
                    "step000.rng_before": np.array([1, 7, 3], dtype=np.uint8)}
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary) / "a.npz", Path(temporary) / "b.npz"
            np.savez(left, **original)

            def compare(changed, exact=False):
                np.savez(right, **changed)
                return diagnostic.compare_numeric_files(left, right, rtol=1e-5, atol=1e-6, exact=exact)

            self.assertTrue(compare(original, exact=True)["passed"])
            close = {name: value.copy() for name, value in original.items()}
            close["step000.parameters"][0] += 5e-6
            self.assertTrue(compare(close)["passed"])
            self.assertFalse(compare(close, exact=True)["passed"])
            corruptions = []
            shape = dict(original)
            shape["step000.parameters"] = original["step000.parameters"].reshape(1, 2)
            corruptions.append(shape)
            dtype = dict(original)
            dtype["step000.parameters"] = original["step000.parameters"].astype(np.float64)
            corruptions.append(dtype)
            for invalid in (np.nan, np.inf, 1.01):
                row = {name: value.copy() for name, value in original.items()}
                row["step000.parameters"][0] = invalid
                corruptions.append(row)
            rng = {name: value.copy() for name, value in original.items()}
            rng["step000.rng_before"][1] = 8
            corruptions.append(rng)
            missing = dict(original)
            missing.pop("step000.train_loss")
            corruptions.append(missing)
            for altered in corruptions:
                with self.subTest(keys=list(altered)):
                    self.assertFalse(compare(altered)["passed"])
            np.savez(left)
            np.savez(right)
            self.assertFalse(diagnostic.compare_numeric_files(
                left, right, rtol=1e-5, atol=1e-6, exact=True)["passed"])

    def fixtures(self):
        fixtures = {}
        for dataset, condition, threads, mode in itertools.product(
                self.config["selection"]["datasets"], self.config["conditions"],
                self.config["cpu_threads"], self.config["checkpoint_modes"]):
            fixtures[dataset, condition, threads, mode] = {
                "slower_repeat_steady_median_seconds": 100. if (threads, mode) == (4, "full_state_copy") else 98.,
                "maximum_process_tree_rss_bytes": 1000 + threads,
            }
        return fixtures

    def test_numeric_artifact_validator_rejects_self_consistent_wrong_model_scope(self):
        shapes = {"parameters": (5,), "gradients": (5,), "optimizer_state": (12,),
                  "train_logits": (3, 2), "unscored_logits": (3, 2), "train_loss": (),
                  "rng_before": (3,), "rng_after": (3,)}
        arrays = {f"step{step:03d}.{name}": np.ones(shape, dtype=np.uint8 if name.startswith("rng_") else np.float32)
                  for step in range(10) for name, shape in shapes.items()}
        row = {"dense_parameter_count": 5, "parameter_tensor_count": 2, "unscored_logits_shape": [3, 2],
               "numeric_evidence_arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                           for name, value in arrays.items()},
               "steps": [{"fingerprint": {"arrays": {name: hashlib.sha256(arrays[f"step{step:03d}.{name}"].tobytes()).hexdigest()
                                                       for name in shapes}}} for step in range(10)]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "numeric.npz"
            np.savez(path, **arrays)
            diagnostic.validate_numeric_evidence(row, path)
            # Metadata and byte fingerprints are deliberately updated too: the
            # model's fixed tensor scope must still reject truncated parameters.
            for key, replacement in (("step003.parameters", np.ones(4, dtype=np.float32)),
                                     ("step004.rng_before", np.empty(0, dtype=np.uint8)),
                                     ("step005.train_logits", np.ones((3, 2), dtype=np.float64)),
                                     ("step006.train_loss", np.array(np.nan, dtype=np.float32))):
                corrupt_arrays, corrupt_row = dict(arrays), copy.deepcopy(row)
                corrupt_arrays[key] = replacement
                corrupt_row["numeric_evidence_arrays"][key] = {"shape": list(replacement.shape), "dtype": str(replacement.dtype)}
                step_name, name = key.split(".")
                corrupt_row["steps"][int(step_name[4:])]["fingerprint"]["arrays"][name] = hashlib.sha256(replacement.tobytes()).hexdigest()
                np.savez(path, **corrupt_arrays)
                with self.subTest(key=key):
                    with self.assertRaises(ValueError):
                        diagnostic.validate_numeric_evidence(corrupt_row, path)
            corrupt_arrays = dict(arrays)
            corrupt_arrays["step002.gradients"] = np.zeros(5, dtype=np.float32)
            np.savez(path, **corrupt_arrays)
            with self.assertRaisesRegex(ValueError, "fingerprints differ"):
                diagnostic.validate_numeric_evidence(row, path)

    def change_fixture_times(self, fixtures, setting, times):
        for (dataset, condition), value in zip(itertools.product(
                self.config["selection"]["datasets"], self.config["conditions"]), times):
            fixtures[dataset, condition, *setting]["slower_repeat_steady_median_seconds"] = value

    def test_qualification_needs_geometric_speed_gain_and_all_correctness_checks(self):
        fixtures, candidate = self.fixtures(), (1, "static_adjacency_once")
        self.change_fixture_times(fixtures, candidate, [80.] * 4)
        result = diagnostic.qualify_settings(self.config, fixtures, set(), True)
        recommended = result["recommended_setting"]
        self.assertEqual((recommended["cpu_threads"], recommended["checkpoint_mode"]), candidate)
        self.assertAlmostEqual(recommended["geometric_speed_ratio"], 1.25)
        self.assertFalse(result["formal_use_authorized"])
        self.assertFalse(result["budget_change_authorized"])
        for invalid in ({candidate}, {(4, "full_state_copy")}):
            self.assertIsNone(diagnostic.qualify_settings(self.config, fixtures, invalid, True)["recommended_setting"])
        self.change_fixture_times(fixtures, candidate, [92.] * 4)
        self.assertIsNone(diagnostic.qualify_settings(self.config, fixtures, set(), True)["recommended_setting"])

    def test_one_fixture_regression_or_incomplete_scope_blocks_fast_average(self):
        fixtures, candidate = self.fixtures(), (1, "static_adjacency_once")
        self.change_fixture_times(fixtures, candidate, [20., 20., 20., 111.])
        result = diagnostic.qualify_settings(self.config, fixtures, set(), True)
        row = next(row for row in result["settings"] if (row["cpu_threads"], row["checkpoint_mode"]) == candidate)
        self.assertGreater(row["geometric_speed_ratio"], 1.10)
        self.assertFalse(row["qualifies"])
        self.assertIsNone(result["recommended_setting"])
        self.change_fixture_times(fixtures, candidate, [80.] * 4)
        self.assertIsNone(diagnostic.qualify_settings(self.config, fixtures, set(), False)["recommended_setting"])
        fixtures.pop(("Actor", self.config["conditions"][1], *candidate))
        self.assertIsNone(diagnostic.qualify_settings(self.config, fixtures, set(), True)["recommended_setting"])

    def test_no_heldout_scores_or_selection_can_hide_in_nested_evidence(self):
        diagnostic.no_scores({"steps": [{"train_loss": .7}], "validation_evaluations": 0,
                              "test_evaluations": 0, "formal_records": 0})
        for forbidden in ({"test_accuracy": .5}, {"validation_loss": .8}, {"regret": .01},
                          {"selected_trial_id": "trial_000"}, {"test_evaluations": 1}, {"formal_records": 1}):
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ValueError):
                    diagnostic.no_scores({"nested": [{"deeper": forbidden}]})
        parameters = inspect.signature(diagnostic.measure_trajectory).parameters
        self.assertIn("train_labels", parameters)
        self.assertFalse({"labels", "y", "test_labels", "validation_labels", "test_indices",
                          "validation_indices"}.intersection(parameters))

    def test_original_h2_toy_ten_steps_both_modes_restore_and_timing_reconciles(self):
        edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
                              [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]])
        x = torch.arange(24, dtype=torch.float32).view(6, 4) / 24
        identity = {"worker_id": "toy", "phase": "performance_diagnostic_only",
                    "validation_evaluations": 0, "test_evaluations": 0, "formal_records": 0}
        with tempfile.TemporaryDirectory() as temporary:
            results, paths = [], []
            for mode in self.config["checkpoint_modes"]:
                torch.manual_seed(73)
                model = H2GCNModel(4, 3, 2, .5, prepare_h2_adjacencies(edges, num_nodes=6))
                destination = Path(temporary) / mode
                store = CheckpointStore(model, mode)
                result = diagnostic.measure_trajectory(
                    model=model, x=x, edge_index=edges, train_indices=torch.arange(4),
                    train_labels=torch.tensor([0, 1, 0, 1]), trial={"learning_rate": .01},
                    weight_decay=.0005, steps=10, checkpoint_store=store,
                    destination=destination, identity=identity, torch=torch)
                results.append(result)
                paths.append(destination / "numeric_evidence.npz")
                diagnostic.no_scores(result)
                self.assertEqual(len(result["steps"]), 10)
                self.assertEqual([row["retained_slots"] for row in result["steps"]], [1, 2, 3] + [4] * 7)
                self.assertEqual([row["slot"] for row in result["restoration"]], [0, 1, 2, 3])
                self.assertTrue(all(row["complete_state_and_unscored_output_exact"] for row in result["restoration"]))
                self.assertTrue(result["cpu_rng_unchanged_by_restoration"])
                self.assertEqual(len(list((destination / "steps").glob("*.json"))), 10)
                for row in result["steps"]:
                    t = row["timings"]
                    self.assertTrue(all(math.isfinite(value) and value >= 0 for value in t.values()))
                    self.assertAlmostEqual(sum(t[key] for key in diagnostic.EXCLUSIVE_STEP_COMPONENTS), t["accounted_exclusive"], places=10)
                    self.assertAlmostEqual(t["accounted_exclusive"] + t["unattributed"], t["end_to_end"], places=10)
                    self.assertAlmostEqual(sum(t[key] for key in diagnostic.COMPUTE_COMPONENTS), t["compute_only"], places=10)
                    self.assertLessEqual(t["checkpoint_parameter_copy"] + t["checkpoint_static_copy"], t["checkpoint_total"])
                with np.load(paths[-1], allow_pickle=False) as evidence:
                    self.assertEqual(len(evidence.files), 80)
                    self.assertFalse(any("one_hop" in key or "two_hop" in key for key in evidence.files))
                    self.assertTrue(all(np.isfinite(evidence[key]).all() for key in evidence.files))
                    self.assertFalse(np.array_equal(evidence["step000.parameters"], evidence["step009.parameters"]))
            self.assertTrue(diagnostic.compare_numeric_files(paths[0], paths[1], rtol=1e-5, atol=1e-6, exact=True)["passed"])
            self.assertEqual(results[0]["initial_fingerprint"], results[1]["initial_fingerprint"])
            self.assertEqual([row["fingerprint"] for row in results[0]["steps"]],
                             [row["fingerprint"] for row in results[1]["steps"]])
            self.assertEqual(results[0]["checkpoint_storage"]["static_buffer_replicas"], 4)
            self.assertEqual(results[1]["checkpoint_storage"]["static_buffer_replicas"], 1)


if __name__ == "__main__":
    unittest.main()
