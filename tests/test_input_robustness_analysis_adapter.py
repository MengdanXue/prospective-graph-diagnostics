"""Synthetic two-condition portfolio and calibration checks."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.input_robustness_analysis_adapter import (
    STRATEGIES, adapt_records, enumerate_graph_portfolios, fit_threshold,
    leave_one_dataset_out,
)
from scripts.input_robustness_formal_records import CONDITIONS, GRAPH_MODELS


class AnalysisAdapterTests(unittest.TestCase):
    def _records(self):
        from tests.test_input_robustness_formal_records import FormalRecordFixtureTests
        fixture = FormalRecordFixtureTests("test_four_trials_selection_checkpoint_save_and_complete_marker")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        self.addCleanup(fixture.tmp.cleanup)
        return fixture.records

    def test_enumerates_all_63_nonempty_graph_portfolios(self):
        portfolios = enumerate_graph_portfolios()
        self.assertEqual(len(portfolios), 63)
        self.assertEqual(len({tuple(portfolio) for portfolio in portfolios}), 63)
        self.assertEqual(portfolios[-1], GRAPH_MODELS)

    def test_two_conditions_have_all_strategies_and_loodo_calibration(self):
        result = adapt_records(self._records())
        self.assertEqual(result["conditions"], list(CONDITIONS))
        self.assertEqual(result["portfolio_count"], 63)
        self.assertEqual(result["strategies"], list(STRATEGIES))
        for condition in CONDITIONS:
            payload = result["portfolios_by_condition"][condition]
            self.assertEqual(payload["portfolio_count"], 63)
            self.assertEqual(payload["strategy_count"], 9)
            self.assertEqual(len(payload["leave_one_dataset_out"]), 63)
            self.assertEqual(set(payload["portfolios"][0]["strategies"]), set(STRATEGIES))
            for calibration in payload["leave_one_dataset_out"].values():
                self.assertEqual(set(calibration["folds"]), {"Cora", "Actor"})
                self.assertEqual({fold["training_datasets"] for fold in calibration["folds"].values()}, {1})
        self.assertEqual(len(result["cross_condition"]), 63)

    def test_threshold_uses_training_datasets_only_and_has_smallest_tie(self):
        rows = [
            {"dataset": "A", "homophily": 0.2, "graph_test": 0.2, "mlp_test": 0.8},
            {"dataset": "B", "homophily": 0.8, "graph_test": 0.8, "mlp_test": 0.2},
        ]
        result = leave_one_dataset_out(rows)
        self.assertEqual(set(result["folds"]), {"A", "B"})
        self.assertEqual(result["folds"]["A"]["training_datasets"], 1)
        self.assertEqual(fit_threshold(rows), 0.21)

    def test_adapter_does_not_accept_missing_models(self):
        records = self._records()
        records = [record for record in records if record["model"] != "H2GCN"]
        with self.assertRaises(ValueError):
            adapt_records(records)
