import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "experiments" / "run_degree_matched_benchmark.py"
SUMMARY_PATH = ROOT / "scripts" / "summarize_degree_matched_benchmark.py"


def load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class DegreeMatchedBenchmarkTests(unittest.TestCase):
    def test_pair_uses_one_split_and_selects_each_condition_independently(self):
        runner = load_module(RUNNER_PATH, "degree_matched_runner")
        calls = []

        def fake_run_model_unit(**kwargs):
            calls.append(kwargs)
            is_randomized = len(calls) == 2
            return {
                "status": "success",
                "dataset": kwargs["dataset"],
                "model": "GCN",
                "seed": kwargs["seed"],
                "split_id": kwargs["split_id"],
                "selected_trial_id": "trial_001" if is_randomized else "trial_000",
                "validation_accuracy": 0.7 if is_randomized else 0.8,
                "validation_loss": 0.6 if is_randomized else 0.5,
                "test_accuracy": 0.65 if is_randomized else 0.75,
                "test_evaluations_after_selection": 1,
                "trials": [{"trial_id": "trial_000"}],
            }

        original_edges = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
            dtype=torch.long,
        )
        randomized_edges = torch.tensor(
            [[0, 2, 2, 1, 1, 3, 3, 0], [2, 0, 1, 2, 3, 1, 0, 3]],
            dtype=torch.long,
        )
        with mock.patch.object(runner, "run_model_unit", side_effect=fake_run_model_unit):
            record = runner.run_paired_unit(
                run_id="route_a_degree_matched_v1",
                dataset="Cora",
                seed=3,
                split_id="split-digest",
                x=torch.ones((4, 2)),
                y=torch.tensor([0, 0, 1, 1]),
                original_edge_index=original_edges,
                randomized_edge_index=randomized_edges,
                train_indices=torch.tensor([0, 2]),
                validation_indices=torch.tensor([1]),
                test_indices=torch.tensor([3]),
                training={"trials": []},
                randomization_seed=100003,
                randomization_audit={"invariants": {"degree_sequence_by_node_identical": True}},
                randomization_metadata={
                    "algorithm": "simple_undirected_double_edge_swap",
                    "status": "success",
                    "swap_status": {
                        "requested_swaps": 20,
                        "successful_swaps": 20,
                        "attempts": 24,
                    },
                },
                original_edges=[[0, 1], [0, 3], [1, 2], [2, 3]],
                randomized_edges=[[0, 2], [0, 3], [1, 2], [1, 3]],
                requested_swaps=20,
                source_commit="abc123",
                environment={"device": "cpu"},
                data_provenance={"processed_files": []},
                device=torch.device("cpu"),
            )

        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertEqual(call["run_id"], "route_a_degree_matched_v1")
            self.assertEqual(call["split_id"], "split-digest")
            self.assertTrue(torch.equal(call["train_indices"], torch.tensor([0, 2])))
            self.assertTrue(torch.equal(call["validation_indices"], torch.tensor([1])))
            self.assertTrue(torch.equal(call["test_indices"], torch.tensor([3])))
        self.assertTrue(torch.equal(calls[0]["edge_index"], original_edges))
        self.assertTrue(torch.equal(calls[1]["edge_index"], randomized_edges))
        self.assertEqual(record["randomization_seed"], 100003)
        self.assertEqual(record["run_id"], "route_a_degree_matched_v1")
        self.assertEqual(
            record["randomization_metadata"]["swap_status"]["successful_swaps"], 20
        )
        self.assertEqual(len(record["randomized_edges"]), 4)
        self.assertEqual(record["conditions"]["original"]["selected_trial_id"], "trial_000")
        self.assertEqual(record["conditions"]["randomized"]["selected_trial_id"], "trial_001")
        self.assertAlmostEqual(record["paired_test_difference"], -0.10)

    def test_structural_audit_and_randomization_seed_follow_frozen_protocol(self):
        runner = load_module(RUNNER_PATH, "degree_matched_protocol")
        config = json.loads(
            (ROOT / "configs" / "prospective_benchmark_v1.json").read_text(encoding="utf-8")
        )
        intervention = config["edge_intervention"]
        self.assertEqual(runner.randomization_seed(intervention, 7), 100007)
        self.assertEqual(runner.requested_swaps(intervention, 11), 55)

        payload = runner.randomize_graph(
            node_count=10,
            edges=[(node, (node + 1) % 10) for node in range(10)]
            + [(node, node + 5) for node in range(5)],
            labels=[node % 2 for node in range(10)],
            seed=2,
            intervention=intervention,
        )
        invariants = payload["audit"]["invariants"]
        for key in (
            "node_count_identical",
            "edge_count_identical",
            "degree_sequence_by_node_identical",
            "degree_sequence_by_node_sha256_before",
            "degree_sequence_by_node_sha256_after",
            "edge_set_sha256_before",
            "edge_set_sha256_after",
        ):
            self.assertIn(key, invariants)
        for phase in ("before", "after", "changes"):
            self.assertIn(phase, payload["audit"])

    def test_internal_numpy_edges_are_normalized_at_the_runner_boundary(self):
        runner = load_module(RUNNER_PATH, "degree_matched_numpy_boundary")
        config = json.loads(
            (ROOT / "configs" / "prospective_benchmark_v1.json").read_text(encoding="utf-8")
        )
        edges = np.asarray(
            [(node, (node + 1) % 10) for node in range(10)]
            + [(node, node + 5) for node in range(5)],
            dtype=np.int64,
        )
        payload = runner.randomize_graph(
            node_count=10,
            edges=edges,
            labels=[node % 2 for node in range(10)],
            seed=0,
            intervention=config["edge_intervention"],
        )
        self.assertEqual(payload["status"], "success")

    def test_summary_is_deterministic_and_reports_paired_inference(self):
        summary = load_module(SUMMARY_PATH, "degree_matched_summary")
        edge_tool = load_module(
            ROOT / "experiments" / "degree_preserving_edge_randomization.py",
            "degree_matched_edge_audit",
        )
        original_edges = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)}
        randomized_edges = {(0, 2), (2, 4), (1, 4), (1, 3), (3, 5), (0, 5)}
        invariants = edge_tool.invariant_audit(6, original_edges, randomized_edges)
        frozen_config = json.loads(
            (ROOT / "configs" / "prospective_benchmark_v1.json").read_text(encoding="utf-8")
        )
        expected_run_id = "route_a_degree_matched_test"
        frozen_config["edge_intervention"]["run_id"] = expected_run_id
        frozen_config_sha = hashlib.sha256(
            json.dumps(frozen_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        records = []
        for dataset_index, dataset in enumerate(("Cora", "CiteSeer", "PubMed")):
            for seed in range(10):
                original = 0.80 - dataset_index * 0.02 + seed * 0.001
                randomized = original - 0.03 + (seed % 2) * 0.002
                records.append(
                    {
                        "status": "success",
                        "run_id": expected_run_id,
                        "dataset": dataset,
                        "model": "GCN",
                        "seed": seed,
                        "split_id": f"{dataset}-{seed}",
                        "randomization_seed": 100000 + seed,
                        "requested_swaps": 30,
                        "randomization_metadata": {
                            "algorithm": "simple_undirected_double_edge_swap",
                            "status": "success",
                            "swap_status": {
                                "requested_swaps": 30,
                                "successful_swaps": 30,
                                "attempts": 60,
                            },
                        },
                        "node_count": 6,
                        "original_edges": [list(edge) for edge in sorted(original_edges)],
                        "randomized_edges": [list(edge) for edge in sorted(randomized_edges)],
                        "source_commit": "abc",
                        "config_sha256": frozen_config_sha,
                        "frozen_config": frozen_config,
                        "data_provenance": {"processed_files": []},
                        "conditions": {
                            "original": {
                                "run_id": expected_run_id,
                                "test_accuracy": original,
                                "dataset": dataset,
                                "model": "GCN",
                                "seed": seed,
                                "split_id": f"{dataset}-{seed}",
                                "source_commit": "abc",
                                "config_sha256": frozen_config_sha,
                                "frozen_config": frozen_config,
                                "data_provenance": {"processed_files": []},
                                "selected_trial_id": "trial_000",
                                "validation_accuracy": 0.8,
                                "validation_loss": 0.5,
                                "test_evaluations_after_selection": 1,
                                "trials": [
                                    {"trial_id": f"trial_{i:03d}", "configuration": frozen_config["training"]["trials"][i], "validation_accuracy": 0.8 - i * 0.01, "validation_loss": 0.5 + i * 0.01}
                                    for i in range(4)
                                ],
                            },
                            "randomized": {
                                "run_id": expected_run_id,
                                "test_accuracy": randomized,
                                "dataset": dataset,
                                "model": "GCN",
                                "seed": seed,
                                "split_id": f"{dataset}-{seed}",
                                "source_commit": "abc",
                                "config_sha256": frozen_config_sha,
                                "frozen_config": frozen_config,
                                "data_provenance": {"processed_files": []},
                                "selected_trial_id": "trial_000",
                                "validation_accuracy": 0.7,
                                "validation_loss": 0.6,
                                "test_evaluations_after_selection": 1,
                                "trials": [
                                    {"trial_id": f"trial_{i:03d}", "configuration": frozen_config["training"]["trials"][i], "validation_accuracy": 0.7 - i * 0.01, "validation_loss": 0.6 + i * 0.01}
                                    for i in range(4)
                                ],
                            },
                        },
                        "paired_test_difference": randomized - original,
                        "difference_definition": "randomized_test_accuracy_minus_original_test_accuracy",
                        "randomization_audit": {
                            "invariants": invariants
                        },
                    }
                )

        first = summary.summarize_records(
            records,
            expected_datasets=["Cora", "CiteSeer", "PubMed"],
            expected_seeds=list(range(10)),
            expected_config=frozen_config,
            bootstrap_samples=1000,
            bootstrap_seed=17,
            permutation_samples=1000,
            permutation_seed=19,
        )
        second = summary.summarize_records(
            list(reversed(records)),
            expected_datasets=["Cora", "CiteSeer", "PubMed"],
            expected_seeds=list(range(10)),
            expected_config=frozen_config,
            bootstrap_samples=1000,
            bootstrap_seed=17,
            permutation_samples=1000,
            permutation_seed=19,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["run_id"], expected_run_id)
        self.assertEqual(first["record_count"], 30)
        self.assertEqual(set(first["datasets"]), {"Cora", "CiteSeer", "PubMed"})
        for row in first["datasets"].values():
            self.assertLess(row["mean_paired_difference"], 0.0)
            self.assertLess(row["median_paired_difference"], 0.0)
            self.assertEqual(len(row["bootstrap_95_ci"]), 2)
            self.assertGreaterEqual(row["sign_flip_p_value"], 0.0)
            self.assertLessEqual(row["sign_flip_p_value"], 1.0)
            self.assertIn("holm_adjusted_p_value", row)
            self.assertEqual(len(row["paired_seed_records"]), 10)

    def test_summary_rejects_incomplete_or_invalid_pairs(self):
        summary = load_module(SUMMARY_PATH, "degree_matched_rejection")
        invalid = {
            "status": "success",
            "dataset": "Cora",
            "model": "GCN",
            "seed": 0,
            "split_id": "split",
            "randomization_seed": 100000,
            "requested_swaps": 50,
            "randomization_metadata": {
                "algorithm": "simple_undirected_double_edge_swap",
                "status": "success",
                "swap_status": {
                    "requested_swaps": 50,
                    "successful_swaps": 50,
                    "attempts": 60,
                },
            },
            "conditions": {
                "original": {"test_accuracy": 0.8},
                "randomized": {"test_accuracy": 0.7},
            },
            "paired_test_difference": -0.1,
            "randomization_audit": {
                "invariants": {"degree_sequence_by_node_identical": False}
            },
        }
        with self.assertRaises(ValueError):
            summary.summarize_records(
                [invalid],
                expected_datasets=["Cora"],
                expected_seeds=[0],
                expected_config=json.loads(
                    (ROOT / "configs" / "prospective_benchmark_v1.json").read_text(encoding="utf-8")
                ),
                bootstrap_samples=100,
                bootstrap_seed=1,
                permutation_samples=100,
                permutation_seed=2,
            )

    def test_summary_cli_refuses_to_overwrite(self):
        summary = load_module(SUMMARY_PATH, "degree_matched_cli")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "summary.json"
            output.write_text("already here", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                summary.write_json_exclusive(output, {"status": "success"})


if __name__ == "__main__":
    unittest.main()
