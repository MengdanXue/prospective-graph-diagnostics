"""Isolated numeric/action fixtures; never read the live research run."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.evaluate_diagnostics import paired_comparisons, unit_regrets
from scripts.input_robustness_analysis_adapter import (
    _frozen_units, _portfolio_rows, _units, adapt_records,
)
from scripts.input_robustness_formal_records import (
    CONDITIONS, DATASETS, GRAPH_MODELS, MODELS, FormalRecordError,
)
from scripts.input_robustness_sensitivity import (
    FALLBACKS, PORTFOLIOS, analyze_records, cross_condition, exclusions,
    resolve_combined, run, settings, summarize_case, verify_prior_outputs,
)


def units(rows):
    defaults = {"seed": 0, "graph_model": "GCN", "graph_validation": 0.8,
                "mlp_validation": 0.3, "graph_test": 0.8, "mlp_test": 0.2,
                "homophily": 0.2, "mean_degree": 7.0, "delta_h": 0.0}
    return _frozen_units([
        {**defaults, "split_id": f"{row['dataset']}-{row.get('seed', 0)}", **row}
        for row in rows], practical_margin=0.01)


def fixture_records():
    records = []
    for condition in CONDITIONS:
        for index, dataset in enumerate(DATASETS):
            for model in MODELS:
                # Validate architecture tie handling, use of validation rather
                # than test selection, and unchanged explicit decisions.
                records.append({
                    "condition": condition, "dataset": dataset, "seed": 0,
                    "split_id": f"fixture-{dataset}", "model": model,
                    "validation_accuracy": 0.3 if model == "MLP" else 0.6,
                    "test_accuracy": (0.2 if model == "MLP" else
                                      (0.7 if model == "GCN" else 0.5))
                                      + (0.05 if condition == CONDITIONS[1] else 0.0),
                    "diagnostics": {"homophily": 0.2 if index % 2 else 0.8,
                                    "mean_degree": 7.0, "delta_h": 0.0},
                })
    return records


def prior_outputs(records, samples=32):
    inference = {"paired_vs_combined": {}, "cross_condition": {}}
    frozen = {}
    portfolios = (GRAPH_MODELS,) + tuple((model,) for model in GRAPH_MODELS)
    for condition in CONDITIONS:
        inference["paired_vs_combined"][condition] = {}
        grouped = _units(records, condition)
        for portfolio in portfolios:
            label = "+".join(portfolio)
            u = _frozen_units(_portfolio_rows(grouped, portfolio), practical_margin=0.01)
            frozen[condition, label] = u
            inference["paired_vs_combined"][condition][label] = paired_comparisons(u, settings(samples))
    for portfolio in portfolios:
        label = "+".join(portfolio)
        inference["cross_condition"][label] = cross_condition(
            frozen[CONDITIONS[0], label], frozen[CONDITIONS[1], label], label, samples=samples)
    return adapt_records(records), inference


class SensitivityTests(unittest.TestCase):
    def test_fallback_changes_only_combined_abstentions_and_keeps_originals(self):
        original = units([
            {"dataset": "abstain"},
            {"dataset": "explicit_mlp", "mlp_validation": 0.6},
            {"dataset": "explicit_graph", "homophily": 0.8, "graph_test": 0.1, "mlp_test": 0.7},
        ])
        untouched = copy.deepcopy(original)
        resolved = resolve_combined(original, "graph")
        self.assertEqual(original, untouched)
        self.assertEqual([u["decisions"]["historical_combined"]["action"] for u in resolved],
                         ["graph", "mlp", "graph"])
        self.assertAlmostEqual(unit_regrets(original, "historical_combined")[0], 0.6)
        self.assertEqual(unit_regrets(resolved, "historical_combined")[0], 0.0)
        for before, after in zip(original, resolved):
            self.assertEqual(before["target_action"], after["target_action"])
            for method in before["decisions"]:
                if method != "historical_combined":
                    self.assertEqual(before["decisions"][method], after["decisions"][method])
        self.assertEqual(resolve_combined(original, "mlp"), original)

    def test_validation_fallback_uses_frozen_one_point_margin_not_test_outcome(self):
        original = units([
            {"dataset": "tie", "graph_validation": 0.25, "mlp_validation": 0.25},
            {"dataset": "submargin", "graph_validation": 0.255, "mlp_validation": 0.25},
            {"dataset": "clear", "graph_validation": 0.30, "mlp_validation": 0.25,
             "graph_test": 0.1, "mlp_test": 0.9},
        ])
        resolved = resolve_combined(original, "validation_selection")
        self.assertEqual([u["decisions"]["historical_combined"]["action"] for u in resolved],
                         ["mlp", "mlp", "graph"])
        self.assertAlmostEqual(unit_regrets(resolved, "historical_combined")[-1], 0.8)
        with self.assertRaises(FormalRecordError):
            resolve_combined(original, "oracle")

    def test_dataset_weighting_and_nonzero_resolution_are_not_seed_counts(self):
        original = units([{"dataset": "A", "seed": seed, "graph_test": 0.6}
                          for seed in range(10)] + [{"dataset": "B", "graph_test": 0.2}])
        result = summarize_case(original, original, samples=64)
        self.assertAlmostEqual(result["combined_dataset_mean_regret_pp"], 20.0)
        self.assertAlmostEqual(result["original_diagnostic_coverage"], 0.0)
        self.assertAlmostEqual(result["resolved_policy_unit_selection_accuracy"], 1 / 11)
        self.assertAlmostEqual(result["resolved_policy_dataset_selection_accuracy"], 0.5)
        row = next(r for r in result["paired_comparisons"] if r["method"] == "always_graph")
        self.assertAlmostEqual(row["mean_difference"], -0.2)
        self.assertEqual(row["raw_p"], 1.0)
        self.assertEqual(result["comparison_resolution"]["always_graph"], {
            "nonzero_datasets": 1, "negative_datasets": 1, "positive_datasets": 0,
            "sign_flip_raw_p_floor": 1.0})
        self.assertEqual(len(result["paired_comparisons"]), 8)

    def test_cross_condition_pairs_datasets_and_rejects_different_splits(self):
        control = units([{"dataset": "A", "graph_test": 0.6}, {"dataset": "B", "graph_test": 0.6}])
        candidate = units([{"dataset": "A", "graph_test": 0.4}, {"dataset": "B", "graph_test": 0.8}])
        result = cross_condition(control, candidate, "GCN", samples=64)
        self.assertAlmostEqual(result["combined_regret"]["mean"], 0.0)
        self.assertAlmostEqual(result["combined_regret"]["per_dataset"]["A"], -0.2)
        self.assertAlmostEqual(result["combined_regret"]["per_dataset"]["B"], 0.2)
        candidate[0]["split_id"] = "wrong"
        with self.assertRaises(FormalRecordError):
            cross_condition(control, candidate, "GCN", samples=64)

    def test_all_requested_scenarios_keep_primary_and_scope_boundaries(self):
        records = fixture_records()
        result = analyze_records(records, samples=32)
        self.assertEqual(len(result["excluded_datasets_by_scenario"]), 14)
        self.assertEqual(result["primary_fallback"], "mlp")
        self.assertFalse(result["within_condition_inference"]["global_multiplicity_adjustment"])
        self.assertFalse(result["cross_condition_inference"]["multiplicity_adjustment"])
        self.assertFalse(result["cross_condition_inference"]["sign_flip_test"])
        total = 0
        for condition in CONDITIONS:
            for portfolio in PORTFOLIOS:
                label = "+".join(portfolio)
                for scenario, excluded in result["excluded_datasets_by_scenario"].items():
                    for fallback in FALLBACKS:
                        case = result["cases"][condition][label][scenario][fallback]
                        self.assertEqual(set(case["included_datasets"]), set(DATASETS) - set(excluded))
                        self.assertEqual(case["units"], 11 - len(excluded))
                        total += 1
                baseline = result["cases"][condition][label]["all"]["mlp"]
                graph = result["cases"][condition][label]["all"]["graph"]
                self.assertEqual(graph["original_diagnostic_coverage"], baseline["original_diagnostic_coverage"])
                self.assertEqual(graph["combined_dataset_mean_regret_pp"], 0.0)
        self.assertEqual(total, 336)
        with self.assertRaises(FormalRecordError):
            exclusions(["A", "B"])

    def test_reconstructs_prior_results_and_rejects_tampered_or_boolean_summaries(self):
        records = fixture_records()
        adapter, inference = prior_outputs(records)
        checked = verify_prior_outputs(records, adapter, inference, samples=32)
        self.assertEqual(checked["within_condition_comparisons_reconstructed"], 112)
        tampered = copy.deepcopy(inference)
        tampered["paired_vs_combined"][CONDITIONS[0]]["GCN"][0]["raw_p"] = 0.001
        with self.assertRaisesRegex(FormalRecordError, "evidence value differs"):
            verify_prior_outputs(records, adapter, tampered, samples=32)
        with self.assertRaises(FormalRecordError):
            verify_prior_outputs(records, adapter, {"passed": True}, samples=32)
        tampered = copy.deepcopy(adapter)
        entry = next(r for r in tampered["portfolios_by_condition"][CONDITIONS[0]]["portfolios"]
                     if r["label"] == "GCN")
        entry["strategies"]["historical_combined"]["outcomes"][0]["action"] = "mlp"
        with self.assertRaisesRegex(FormalRecordError, "evidence value differs"):
            verify_prior_outputs(records, tampered, inference, samples=32)

    def test_duplicate_or_missing_model_cannot_enter_sensitivity(self):
        records = fixture_records()
        with self.assertRaisesRegex(FormalRecordError, "duplicate"):
            analyze_records(records + [copy.deepcopy(records[0])], samples=8)
        with self.assertRaisesRegex(FormalRecordError, "all seven"):
            analyze_records(records[1:], samples=8)

    def test_formal_entry_preserves_existing_output_and_requires_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output.json"
            output.write_text("old evidence", encoding="utf-8")
            with patch("scripts.input_robustness_sensitivity.validate_complete_run") as validator:
                with self.assertRaisesRegex(FormalRecordError, "refusing to replace"):
                    run(root, root / "config.json", root / "binding.json",
                        root / "adapter.json", root / "inference.json", output)
                validator.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "old evidence")
            for name in ("manifest.json", "complete.json", "config.json", "binding.json", "adapter.json", "inference.json"):
                (root / name).write_text(json.dumps({"passed": True}), encoding="utf-8")
            with self.assertRaisesRegex(FormalRecordError, "manifest schema"):
                run(root, root / "config.json", root / "binding.json",
                    root / "adapter.json", root / "inference.json", root / "new.json")
            self.assertFalse((root / "new.json").exists())


if __name__ == "__main__":
    unittest.main()
