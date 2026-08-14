import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.evaluate_diagnostics import sign_flip_p


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
EVALUATOR = ROOT / "experiments" / "evaluate_diagnostics.py"
PREREGISTRATION = ROOT / "docs" / "preregistration_diagnostic_benchmark.md"
EXPERIMENTS_SECTION = ROOT / "sections" / "04_prospective_results.tex"
GRAPH_MODELS = ["GCN", "GAT", "GraphSAGE", "H2GCN", "LINKX", "GPR-GNN"]


class ExactSignFlipTests(unittest.TestCase):
    def test_exhaustive_branch_uses_exact_tail_probability(self):
        self.assertEqual(
            sign_flip_p(np.asarray([1.0, 1.0]), samples=1000, seed=7),
            0.5,
        )


def add_unit(
    payload,
    *,
    dataset,
    seed,
    mlp_val,
    mlp_test,
    graph_val,
    graph_test,
    homophily,
    mean_degree,
    delta_h,
    uses_test_labels=False,
):
    split_id = f"{dataset}-split-{seed}"
    payload["model_records"].append(
        {
            "dataset": dataset,
            "seed": seed,
            "split_id": split_id,
            "model": "MLP",
            "family": "mlp",
            "validation_accuracy": mlp_val,
            "test_accuracy": mlp_test,
        }
    )
    for index, model in enumerate(GRAPH_MODELS):
        payload["model_records"].append(
            {
                "dataset": dataset,
                "seed": seed,
                "split_id": split_id,
                "model": model,
                "family": "graph",
                "validation_accuracy": graph_val - 0.001 * index,
                "test_accuracy": graph_test - 0.001 * index,
            }
        )
    payload["diagnostic_records"].append(
        {
            "dataset": dataset,
            "seed": seed,
            "split_id": split_id,
            "homophily": homophily,
            "mean_degree": mean_degree,
            "delta_h": delta_h,
            "provenance": {
                "uses_test_labels": uses_test_labels,
                "homophily_label_scope": "train_only",
                "two_hop_label_scope": "train_only" if delta_h is not None else "unavailable",
            },
        }
    )


def benchmark_payload():
    return {
        "schema_version": "1.0",
        "benchmark_id": "route_a_diagnostic_v1",
        "outcome": "accuracy",
        "practical_margin": 0.01,
        "mlp_models": ["MLP"],
        "graph_models": GRAPH_MODELS,
        "bootstrap": {"samples": 500, "seed": 20260808},
        "permutation": {"samples": 2000, "seed": 20260809},
        "model_records": [],
        "diagnostic_records": [],
    }


class DiagnosticScoringTests(unittest.TestCase):
    def run_evaluator(self, payload, directory, output_name="audit.json"):
        input_path = directory / "input.json"
        output_path = directory / output_name
        input_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        result = subprocess.run(
            [str(PYTHON), str(EVALUATOR), "--input", str(input_path), "--output", str(output_path)],
            text=True,
            capture_output=True,
            check=False,
        )
        return result, output_path

    def test_margin_target_and_validation_selection_use_one_tie_policy(self):
        payload = benchmark_payload()
        add_unit(
            payload,
            dataset="A",
            seed=1,
            mlp_val=0.70,
            mlp_test=0.70,
            graph_val=0.74,
            graph_test=0.72,
            homophily=0.8,
            mean_degree=12.0,
            delta_h=0.10,
        )
        add_unit(
            payload,
            dataset="B",
            seed=1,
            mlp_val=0.70,
            mlp_test=0.70,
            graph_val=0.74,
            graph_test=0.705,
            homophily=0.8,
            mean_degree=12.0,
            delta_h=0.10,
        )
        with tempfile.TemporaryDirectory() as tmp:
            result, output = self.run_evaluator(payload, Path(tmp))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            audit = json.loads(output.read_text(encoding="utf-8"))

        targets = {row["dataset"]: row["target_action"] for row in audit["units"]}
        self.assertEqual(targets, {"A": "graph", "B": "mlp"})
        validation = audit["methods"]["validation_selection"]
        self.assertEqual(validation["covered"], 2)
        self.assertAlmostEqual(validation["selection_accuracy"], 0.5)
        self.assertAlmostEqual(audit["methods"]["random_50_50"]["selection_accuracy"], 0.5)

    def test_missing_two_hop_abstains_and_full_set_regret_uses_mlp_fallback(self):
        payload = benchmark_payload()
        add_unit(
            payload,
            dataset="A",
            seed=1,
            mlp_val=0.30,
            mlp_test=0.60,
            graph_val=0.70,
            graph_test=0.70,
            homophily=0.20,
            mean_degree=3.0,
            delta_h=None,
        )
        add_unit(
            payload,
            dataset="B",
            seed=1,
            mlp_val=0.65,
            mlp_test=0.70,
            graph_val=0.60,
            graph_test=0.60,
            homophily=0.20,
            mean_degree=3.0,
            delta_h=0.10,
        )
        with tempfile.TemporaryDirectory() as tmp:
            result, output = self.run_evaluator(payload, Path(tmp))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            audit = json.loads(output.read_text(encoding="utf-8"))

        two_hop = audit["methods"]["two_hop_only"]
        self.assertEqual(two_hop["covered"], 1)
        self.assertAlmostEqual(two_hop["coverage"], 0.5)
        self.assertAlmostEqual(two_hop["full_set_mean_regret"], 0.10)
        historical = audit["methods"]["historical_combined"]
        self.assertEqual(historical["covered"], 1)
        self.assertEqual(historical["abstained"], 1)

    def test_test_label_leakage_is_rejected_without_an_output(self):
        payload = benchmark_payload()
        add_unit(
            payload,
            dataset="A",
            seed=1,
            mlp_val=0.7,
            mlp_test=0.7,
            graph_val=0.8,
            graph_test=0.8,
            homophily=0.8,
            mean_degree=10.0,
            delta_h=0.1,
            uses_test_labels=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            result, output = self.run_evaluator(payload, Path(tmp))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("test labels", result.stderr.lower())
            self.assertFalse(output.exists())

    def test_nonmissing_diagnostic_requires_train_only_scope(self):
        payload = benchmark_payload()
        add_unit(
            payload,
            dataset="A",
            seed=1,
            mlp_val=0.7,
            mlp_test=0.7,
            graph_val=0.8,
            graph_test=0.8,
            homophily=0.8,
            mean_degree=10.0,
            delta_h=0.1,
        )
        payload["diagnostic_records"][0]["provenance"]["homophily_label_scope"] = "unavailable"
        with tempfile.TemporaryDirectory() as tmp:
            result, output = self.run_evaluator(payload, Path(tmp))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("homophily", result.stderr.lower())
            self.assertFalse(output.exists())

    def test_bootstrap_and_holm_outputs_are_byte_deterministic(self):
        payload = benchmark_payload()
        for index, dataset in enumerate(["A", "B", "C", "D"]):
            add_unit(
                payload,
                dataset=dataset,
                seed=1,
                mlp_val=0.65,
                mlp_test=0.65 + 0.01 * (index % 2),
                graph_val=0.70,
                graph_test=0.60 + 0.04 * index,
                homophily=0.2 + 0.2 * index,
                mean_degree=3.0 + 5.0 * index,
                delta_h=None if index == 0 else 0.02 * index,
            )
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first_result, first_output = self.run_evaluator(payload, Path(first_tmp))
            second_result, second_output = self.run_evaluator(payload, Path(second_tmp))
            self.assertEqual(first_result.returncode, 0, first_result.stdout + first_result.stderr)
            self.assertEqual(second_result.returncode, 0, second_result.stdout + second_result.stderr)
            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())
            audit = json.loads(first_output.read_text(encoding="utf-8"))

        comparisons = audit["paired_comparisons"]
        self.assertEqual(len(comparisons), 8)
        self.assertTrue(all(row["holm_adjusted_p"] >= row["raw_p"] for row in comparisons))
        self.assertEqual(audit["inference"]["resampling_unit"], "dataset")

    def test_manuscript_reports_the_completed_prospective_benchmark(self):
        specification = PREREGISTRATION.read_text(encoding="utf-8").lower()
        experiments = EXPERIMENTS_SECTION.read_text(encoding="utf-8").lower()
        self.assertIn("prospective analysis specification", specification)
        self.assertIn("not a claim that the historical experiments were preregistered", specification)
        self.assertIn("prospective benchmark results", experiments)
        self.assertIn("770 selected-model records", experiments)
        self.assertIn("110 diagnostic records", experiments)
        self.assertNotIn("has not yet been executed", experiments)


if __name__ == "__main__":
    unittest.main()
