import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import experiments.run_prospective_benchmark as prospective_runner

from experiments.run_prospective_benchmark import (
    _load_dataset,
    validate_resume_record,
    validate_output_root_ownership,
    run_model_unit,
    select_trial,
    write_json_exclusive,
)


class ProspectiveRunnerTests(unittest.TestCase):
    def test_output_root_rejects_artifacts_owned_by_another_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "records" / "Cora" / "MLP" / "seed_000.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text(
                json.dumps({"run_id": "route_a_prospective_v1", "status": "success"}),
                encoding="utf-8",
            )
            before = artifact.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "belongs to another run"):
                validate_output_root_ownership(root, "route_a_prospective_v2")
            self.assertEqual(artifact.read_bytes(), before)
            self.assertEqual(len(list(root.rglob("*.json"))), 1)

            validate_output_root_ownership(root, "route_a_prospective_v1")
            validate_output_root_ownership(root / "blank", "route_a_prospective_v2")

    def test_stage_failure_record_is_complete_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failures" / "Texas" / "seed_000_split.json"
            context = {
                "run_id": "route_a_prospective_v2",
                "dataset": "Texas",
                "seed": 0,
                "source_commit": "abc",
                "config_sha256": "def",
                "frozen_config": {"run_id": "route_a_prospective_v2"},
                "environment": {"python": "test"},
                "data_provenance": {"processed_files": []},
                "command": ["python", "runner.py"],
            }
            prospective_runner.write_stage_failure(
                path,
                stage="split_construction",
                exception=ValueError("infeasible split"),
                context=context,
                elapsed_seconds=1.25,
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "error")
            self.assertEqual(record["stage"], "split_construction")
            self.assertEqual(record["exception_type"], "ValueError")
            self.assertEqual(record["exception"], "infeasible split")
            self.assertIn("ValueError: infeasible split", record["traceback"])
            self.assertEqual(record["elapsed_seconds"], 1.25)
            for key, value in context.items():
                self.assertEqual(record[key], value)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                prospective_runner.write_stage_failure(
                    path,
                    stage="split_construction",
                    exception=ValueError("retry"),
                    context=context,
                    elapsed_seconds=2.0,
                )
            self.assertEqual(path.read_bytes(), before)

    def test_split_exception_is_archived_before_it_is_reraised(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failures" / "Texas" / "seed_000_split.json"
            labels = torch.tensor([0, 0, 0, 1]).numpy()
            with mock.patch.object(prospective_runner.time, "perf_counter", return_value=3.75):
                with self.assertRaisesRegex(ValueError, "fewer than three"):
                    prospective_runner.make_split_with_failure(
                        labels=labels,
                        seed=0,
                        failure_path=path,
                        context={
                            "run_id": "route_a_prospective_v2",
                            "dataset": "Texas",
                            "seed": 0,
                        },
                        started_at=1.25,
                    )
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["stage"], "split_construction")
            self.assertEqual(record["dataset"], "Texas")
            self.assertEqual(record["seed"], 0)
            self.assertEqual(record["elapsed_seconds"], 2.5)

    def test_dataset_loader_refuses_an_incomplete_cache_before_pyg_can_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                _load_dataset("Cora", root)
            self.assertEqual(list(root.rglob("*")), [])

    def test_resume_rejects_stale_success_and_existing_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            success = root / "success.json"
            success.write_text(
                json.dumps({"status": "success", "dataset": "Cora", "seed": 0, "source_commit": "old"}),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                validate_resume_record(
                    success,
                    {"dataset": "Cora", "seed": 0, "source_commit": "new"},
                )
            failure = root / "failure.json"
            failure.write_text(json.dumps({"status": "error"}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_resume_record(failure, {})

    def test_documented_direct_cli_entry_points_resolve_repository_imports(self):
        root = Path(__file__).resolve().parents[1]
        entry_points = (
            root / "experiments" / "run_prospective_benchmark.py",
            root / "experiments" / "run_degree_matched_benchmark.py",
            root / "scripts" / "assemble_prospective_diagnostics.py",
            root / "scripts" / "summarize_degree_matched_benchmark.py",
        )
        for entry_point in entry_points:
            with self.subTest(entry_point=entry_point.name):
                completed = subprocess.run(
                    [sys.executable, str(entry_point), "--help"],
                    cwd=root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    completed.stdout + completed.stderr,
                )

    def test_trial_selection_uses_validation_then_loss_then_identifier(self):
        rows = [
            {"trial_id": "trial_002", "validation_accuracy": 0.8, "validation_loss": 0.7},
            {"trial_id": "trial_001", "validation_accuracy": 0.8, "validation_loss": 0.6},
            {"trial_id": "trial_000", "validation_accuracy": 0.8, "validation_loss": 0.6},
            {"trial_id": "trial_003", "validation_accuracy": 0.7, "validation_loss": 0.1},
        ]
        self.assertEqual(select_trial(rows)["trial_id"], "trial_000")

    def test_model_unit_retains_four_trials_and_evaluates_test_once(self):
        torch.manual_seed(3)
        x = torch.randn(12, 4)
        y = torch.tensor([0, 1] * 6)
        edge_index = torch.tensor(
            [
                list(range(11)) + list(range(1, 12)),
                list(range(1, 12)) + list(range(11)),
            ],
            dtype=torch.long,
        )
        record = run_model_unit(
            run_id="route_a_prospective_v2",
            dataset="toy",
            model_id="MLP",
            seed=5,
            split_id="split-digest",
            x=x,
            y=y,
            edge_index=edge_index,
            train_indices=torch.tensor([0, 1, 2, 3, 4, 5]),
            validation_indices=torch.tensor([6, 7, 8]),
            test_indices=torch.tensor([9, 10, 11]),
            training={
                "hidden_channels": 8,
                "max_epochs": 3,
                "patience": 2,
                "weight_decay": 0.0005,
                "trials": [
                    {"learning_rate": 0.01, "dropout": 0.5},
                    {"learning_rate": 0.01, "dropout": 0.7},
                    {"learning_rate": 0.005, "dropout": 0.5},
                    {"learning_rate": 0.005, "dropout": 0.7},
                ],
            },
            source_commit="abc123",
            environment={"python": "test"},
            data_provenance={"processed_files": []},
            device=torch.device("cpu"),
        )
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["run_id"], "route_a_prospective_v2")
        self.assertEqual(record["family"], "mlp")
        self.assertEqual(len(record["trials"]), 4)
        self.assertEqual(record["test_evaluations_after_selection"], 1)
        self.assertIn(record["selected_trial_id"], {row["trial_id"] for row in record["trials"]})
        self.assertGreaterEqual(record["validation_accuracy"], 0.0)
        self.assertLessEqual(record["test_accuracy"], 1.0)
        self.assertNotIn("test_accuracy", record["trials"][0])

    def test_json_outputs_are_exclusive_and_byte_stable(self):
        payload = {"schema_version": "1.0", "status": "success", "value": 1}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            write_json_exclusive(path, payload)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_json_exclusive(path, payload)
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(json.loads(before), payload)


if __name__ == "__main__":
    unittest.main()
