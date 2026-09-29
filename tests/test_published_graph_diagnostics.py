"""Hand-calculated graph cases and isolated two-budget decision integration."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import tracemalloc
import unittest
from unittest.mock import patch

import numpy as np

from experiments.evaluate_diagnostics import score_method
from scripts.analyze_published_diagnostics import analyze_complete_extension, assemble_budget_analyses, compute_bound_metric_records
from scripts.input_robustness_analysis_adapter import _frozen_units
from scripts.input_robustness_formal_records import CONDITIONS, MODELS, FormalRecordError, digest
from scripts.published_graph_diagnostics import (
    METRICS, METHOD, adapt_published_records, candidate_thresholds, compute_train_metrics,
    evaluate_thresholds, fit_published_threshold, leave_one_dataset_out_published, make_metric_record,
)


def graph(edges, labels, *, nodes=None, num_nodes=None):
    if nodes is None:
        nodes = list(range(len(labels)))
    return compute_train_metrics(np.asarray(edges, dtype=np.int64).reshape(-1, 2).T,
                                 np.asarray(nodes, dtype=np.int64), np.asarray(labels, dtype=np.int64),
                                 num_nodes=num_nodes or len(labels))


def row(dataset, seed=0, score=0.0, graph_test=0.8, mlp_test=0.5):
    return {"dataset": dataset, "seed": seed, "split_id": dataset + str(seed), "graph_model": "GCN",
            "graph_validation": 0.7, "mlp_validation": 0.6, "graph_test": graph_test, "mlp_test": mlp_test,
            "homophily": 0.4, "mean_degree": 2.0, "delta_h": 0.0,
            "adjusted_homophily": score, "edge_label_informativeness": score}


def fixture():
    records, metrics = [], []
    for d, dataset in enumerate(("Cora", "Actor", "Cornell")):
        for seed in (0, 1):
            split_id = f"{dataset}{seed}"
            edges = [(0, 1), (2, 3)] if d == 0 else [(0, 2), (1, 3)]
            metric = graph(edges, [0, 0, 1, 1])
            metrics.append(make_metric_record(dataset, seed, split_id, metric))
            for condition in CONDITIONS:
                for model in MODELS:
                    records.append({"dataset": dataset, "seed": seed, "split_id": split_id,
                                    "condition": condition, "model": model,
                                    "validation_accuracy": 0.6 if model == "MLP" else 0.7,
                                    "validation_loss": 0.2 if model == "GCN" else 0.9,
                                    "test_accuracy": (0.5 if d == 0 else 0.8) if model == "MLP" else 0.7,
                                    "diagnostics": {"homophily": 0.4, "mean_degree": 2., "delta_h": 0.0},
                                    "transform_binding": {"condition": condition, "split_id": split_id}})
    extension = [copy.deepcopy(item) for item in records if item["model"] == "MLP"]
    for item in extension:
        item["validation_accuracy"] = 0.9
        item["test_accuracy"] = 0.95
    return records, extension, metrics


class PublishedMetricTests(unittest.TestCase):
    def test_perfect_homophily(self):
        result = graph([(0, 1), (2, 3)], [0, 0, 1, 1])
        self.assertEqual(result["metrics"]["adjusted_homophily"]["value"], 1.0)
        self.assertAlmostEqual(result["metrics"]["edge_label_informativeness"]["value"], 1.0)

    def test_perfect_bipartite_has_negative_homophily_but_full_information(self):
        result = graph([(0, 2), (0, 3), (1, 2), (1, 3)], [0, 0, 1, 1])
        self.assertEqual(result["metrics"]["adjusted_homophily"]["value"], -1.)
        self.assertAlmostEqual(result["metrics"]["edge_label_informativeness"]["value"], 1.)

    def test_independent_endpoint_labels(self):
        result = graph([(0, 1), (2, 3), (0, 2), (1, 3)], [0, 0, 1, 1])
        self.assertEqual(result["metrics"]["adjusted_homophily"]["value"], 0.)
        self.assertAlmostEqual(result["metrics"]["edge_label_informativeness"]["value"], 0.)

    def test_class_imbalance_uses_degree_mass_not_node_class_frequency(self):
        result = graph([(0, 1), (0, 2), (0, 3)], [0, 0, 0, 1])
        self.assertAlmostEqual(result["metrics"]["adjusted_homophily"]["value"], -.2)
        expected_information = (4/6)*math.log((4/6)/(25/36)) + 2*(1/6)*math.log((1/6)/(5/36))
        expected_entropy = -(5/6)*math.log(5/6) - (1/6)*math.log(1/6)
        self.assertAlmostEqual(result["metrics"]["edge_label_informativeness"]["value"],
                               expected_information/expected_entropy)

    def test_duplicates_bidirectionality_loops_and_input_immutability(self):
        original = np.array([[0, 1, 0, 2, 3, 0], [1, 0, 1, 3, 2, 0]])
        saved = original.copy()
        result = compute_train_metrics(original, np.arange(4), np.array([0, 0, 1, 1]), num_nodes=4)
        expected = graph([(0, 1), (2, 3)], [0, 0, 1, 1])
        self.assertEqual(result, expected)
        np.testing.assert_array_equal(original, saved)

    def test_relabelling_is_invariant_and_sparse_class_ids_are_safe(self):
        first = graph([(0, 1), (0, 2), (2, 3)], [0, 0, 1, 1])
        other = graph([(0, 1), (0, 2), (2, 3)], [99999999999, 99999999999, 7, 7])
        self.assertEqual(first["metrics"], other["metrics"])

    def test_training_induction_never_requires_other_labels(self):
        result = graph([(0, 1), (2, 3), (0, 4), (4, 5)], [0, 0, 1, 1], num_nodes=6)
        self.assertEqual(result, graph([(0, 1), (2, 3)], [0, 0, 1, 1], num_nodes=6))
        with self.assertRaisesRegex(FormalRecordError, "one label per training"):
            graph([(0, 1)], [0, 0, 1, 1, 2, 3], nodes=[0, 1, 2, 3], num_nodes=6)

    def test_undefined_metrics_are_retained(self):
        for edges, labels, reason in (([], [0, 1], "no_training_edges"),
                                      ([(0, 1)], [0, 0, 1], "single_endpoint_class")):
            result = graph(edges, labels)
            for metric in METRICS:
                self.assertIsNone(result["metrics"][metric]["value"])
                self.assertEqual(result["metrics"][metric]["unavailable_reason"], reason)

    def test_large_shape_few_edges_does_not_allocate_by_graph_or_class_size(self):
        tracemalloc.start()
        try:
            result = graph([(10**12-4, 10**12-3), (10**12-2, 10**12-1)],
                           [0, 0, 10**12, 10**12], nodes=list(range(10**12-4, 10**12)), num_nodes=10**12)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(result["metrics"]["adjusted_homophily"]["value"], 1.)
        self.assertLess(peak, 2_000_000)

    def test_invalid_input_rejected(self):
        for nodes in ([0, 0], [0, 99]):
            with self.assertRaises(FormalRecordError):
                graph([(0, 1)], [0, 1], nodes=nodes, num_nodes=3)
        with self.assertRaises(FormalRecordError):
            compute_train_metrics(np.array([[0.1], [1.0]]), np.arange(2), np.arange(2), num_nodes=2)


class PublishedCalibrationTests(unittest.TestCase):
    def test_negative_scores_generate_fold_midpoints_not_old_zero_one_grid(self):
        rows = [row("A", score=-.9, graph_test=.2, mlp_test=.8), row("B", score=-.3)]
        candidates = candidate_thresholds(rows, METRICS[0])
        self.assertEqual([x["kind"] for x in candidates], ["all_graph", "cutoff", "all_mlp"])
        self.assertAlmostEqual(candidates[1]["value"], -.6)
        self.assertAlmostEqual(fit_published_threshold(rows, METRICS[0])["threshold"]["value"], -.6)

    def test_equal_dataset_weight_and_smallest_threshold_tie(self):
        rows = [row("A", graph_test=.9, mlp_test=.5)]
        rows += [row("B", seed=seed, graph_test=.5, mlp_test=.6) for seed in range(9)]
        self.assertEqual(fit_published_threshold(rows, METRICS[0])["threshold"]["kind"], "all_graph")
        tied = [row("A", score=-.9, graph_test=.5, mlp_test=.5), row("B", score=.9, graph_test=.5, mlp_test=.5)]
        self.assertEqual(fit_published_threshold(tied, METRICS[0])["threshold"]["kind"], "all_graph")

    def test_heldout_scores_and_outcomes_cannot_choose_its_threshold(self):
        rows = [row("A", score=-.4), row("B", score=-.9, graph_test=.2, mlp_test=.8), row("C", score=-.3)]
        first = leave_one_dataset_out_published(rows, METRICS[0])
        changed = copy.deepcopy(rows)
        changed[0].update(adjusted_homophily=9999., graph_test=0., mlp_test=1.)
        second = leave_one_dataset_out_published(changed, METRICS[0])
        self.assertEqual(first["folds"]["A"], second["folds"]["A"])
        self.assertEqual(first["folds"]["A"]["training_datasets"], ["B", "C"])

    def test_undefined_scores_abstain_at_endpoints_and_do_not_disappear(self):
        rows = [row("A", score=None), row("B", score=.4)]
        for kind in ("all_graph", "all_mlp"):
            result = evaluate_thresholds(rows, METRICS[0], {"A": {"kind": kind}, "B": {"kind": kind}})
            self.assertEqual(result["abstained"], 1)
            self.assertEqual(result["coverage"], .5)
            self.assertEqual(result["outcomes"][0]["effective_action"], "mlp")
            self.assertAlmostEqual(result["outcomes"][0]["regret"], .3)

    def test_scoring_reuses_frozen_actions_losses_coverage_and_margin(self):
        rows = [row("A", score=.3), row("B", score=None, graph_test=.505, mlp_test=.5)]
        result = evaluate_thresholds(rows, METRICS[0], {d: {"kind": "all_graph"} for d in ("A", "B")})
        units = _frozen_units(rows, practical_margin=.01)
        for unit, action in zip(units, ("graph", "abstain")):
            unit["decisions"][METHOD] = {"action": action, "confidence": 1.0 if action == "graph" else None}
        expected = score_method(units, METHOD)
        for key in ("coverage", "abstained", "selection_accuracy", "selective_risk", "full_set_mean_regret", "covered_mean_regret"):
            self.assertEqual(result[key], expected[key])


class PublishedIntegrationTests(unittest.TestCase):
    def test_two_conditions_two_budgets_all_portfolios_and_frozen_policies(self):
        base, extension, metrics = fixture()
        result = assemble_budget_analyses(base, extension, metrics)
        for budget in ("4", "24"):
            entry = result["by_mlp_budget"][budget]
            self.assertEqual(entry["original_strategies_and_h1_lodo"]["portfolio_count"], 63)
            self.assertEqual(len(entry["original_strategies_and_h1_lodo"]["strategies"]), 9)
            for condition in CONDITIONS:
                self.assertEqual(len(entry["published_metrics"]["portfolios_by_condition"][condition]), 63)
        first = result["by_mlp_budget"]["4"]["original_strategies_and_h1_lodo"]["portfolios_by_condition"][CONDITIONS[0]]["portfolios"][-1]
        later = result["by_mlp_budget"]["24"]["original_strategies_and_h1_lodo"]["portfolios_by_condition"][CONDITIONS[0]]["portfolios"][-1]
        self.assertNotEqual(first["strategies"]["always_graph"]["mean_regret_pp"], later["strategies"]["always_graph"]["mean_regret_pp"])
        self.assertEqual(later["strategies"]["always_mlp"]["mean_regret_pp"], 0.)
        delta = result["descriptive_budget_24_minus_4"][CONDITIONS[0]][-1]["policies"]["always_graph"]
        self.assertAlmostEqual(delta["mean_regret_pp_delta"], later["strategies"]["always_graph"]["mean_regret_pp"] - first["strategies"]["always_graph"]["mean_regret_pp"])
        self.assertEqual(set(delta["dataset_regret_pp_deltas"]), {"Cora", "Actor", "Cornell"})

    def test_graph_architecture_tie_uses_model_name_never_validation_loss(self):
        base, _, metrics = fixture()
        result = adapt_published_records(base, metrics)
        full = result["portfolios_by_condition"][CONDITIONS[0]][-1]
        self.assertEqual({x["selected_graph"] for x in full["metrics"][METRICS[0]]["outcomes"]}, {"GAT"})

    def test_missing_duplicate_drifting_metric_or_model_record_rejected(self):
        base, extension, metrics = fixture()
        for malformed in (metrics[:-1], metrics + [metrics[0]]):
            with self.assertRaises(FormalRecordError):
                adapt_published_records(base, malformed)
        bad = copy.deepcopy(metrics)
        bad[0]["metrics"][METRICS[0]]["value"] = 0.123
        with self.assertRaisesRegex(FormalRecordError, "digest"):
            adapt_published_records(base, bad)
        for malformed in (extension[:-1], extension + [extension[0]]):
            with self.assertRaises(FormalRecordError):
                assemble_budget_analyses(base, malformed, metrics)
        changed = copy.deepcopy(extension)
        changed[0]["split_id"] = "other"
        with self.assertRaisesRegex(FormalRecordError, "split_id"):
            assemble_budget_analyses(base, changed, metrics)

    def test_optimistic_analysis_is_optional_and_explicitly_test_selected(self):
        base, _, metrics = fixture()
        result = adapt_published_records(base, metrics, include_optimistic=True)
        metric = result["portfolios_by_condition"][CONDITIONS[0]][-1]["metrics"][METRICS[0]]
        self.assertIn("not_deployment_estimate", metric["optimistic_test_selected_floor"]["analysis_status"])

    def test_formal_analysis_rejects_passed_marker_before_reading_research_metrics(self):
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"schema_version": "1.0", "status": "passed"}))
            (root / "complete.json").write_text(json.dumps({"status": "passed"}))
            with patch("scripts.analyze_published_diagnostics.compute_bound_metric_records") as compute:
                with self.assertRaises(FormalRecordError):
                    analyze_complete_extension(base_root=root, extension_root=root / "extension",
                                               base_config_path=root / "missing_config.json",
                                               extension_config_path=root / "missing_extension.json",
                                               binding_path=root / "missing_binding.json", data_root=root)
                compute.assert_not_called()

    def test_bound_metric_loader_selects_train_labels_only_and_reuses_conditions(self):
        import tempfile
        import json
        class Features:
            def size(self, axis):
                return 6
        class Edges:
            def numpy(self):
                return np.array([[0, 2, 0], [1, 3, 5]])
        class Labels:
            def __getitem__(self, indices):
                if set(indices) != {0, 1, 2, 3}:
                    raise AssertionError("non-training label access")
                class Selected:
                    def numpy(self):
                        return np.array([0, 0, 1, 1])
                return Selected()
        records = [{"dataset": "Cora", "seed": 0, "split_id": "split", "condition": condition} for condition in CONDITIONS]
        split = {0: ({"train": np.array([0, 1, 2, 3]), "validation": np.array([4]), "test": np.array([5])}, "split")}
        with tempfile.TemporaryDirectory() as tmp:
            binding = Path(tmp)/"binding.json"
            binding.write_text(json.dumps({"datasets": {}}))
            with patch("scripts.input_robustness_data.load_bound_dataset", return_value=(Features(), Labels(), Edges(), split, {})) as load:
                metrics = compute_bound_metric_records(records, data_root=Path(tmp), binding_path=binding)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(metrics[0]["training_edge_count"], 2)


if __name__ == "__main__":
    unittest.main()
