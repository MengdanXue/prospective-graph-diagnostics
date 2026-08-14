import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "prospective_benchmark_v1.json"
CONFIG_V2 = ROOT / "configs" / "prospective_benchmark_v2.json"


class ProspectiveProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_scope_is_frozen_before_outcomes(self):
        self.assertEqual(self.config["run_id"], "route_a_prospective_v1")
        self.assertEqual(
            self.config["datasets"],
            [
                "Cora",
                "CiteSeer",
                "PubMed",
                "Texas",
                "Wisconsin",
                "Cornell",
                "Chameleon",
                "Squirrel",
                "Actor",
                "Roman-empire",
                "Amazon-ratings",
                "Coauthor-CS",
            ],
        )
        self.assertEqual(self.config["seeds"], list(range(10)))
        self.assertEqual(
            self.config["models"],
            ["MLP", "GCN", "GAT", "GraphSAGE", "H2GCN", "LINKX", "GPR-GNN"],
        )

    def test_every_model_gets_four_equal_budget_trials(self):
        training = self.config["training"]
        self.assertEqual(training["hidden_channels"], 64)
        self.assertEqual(training["max_epochs"], 500)
        self.assertEqual(training["patience"], 100)
        self.assertEqual(training["weight_decay"], 5e-4)
        expected = [
            {"learning_rate": lr, "dropout": dropout}
            for lr in (0.01, 0.005)
            for dropout in (0.5, 0.7)
        ]
        self.assertEqual(training["trials"], expected)

    def test_split_diagnostic_and_intervention_rules_are_frozen(self):
        self.assertEqual(self.config["split"], {"train": 0.6, "validation": 0.2, "test": 0.2})
        self.assertEqual(self.config["diagnostics"]["two_hop_walks"], 50_000)
        self.assertEqual(self.config["diagnostics"]["label_scope"], "train_only")
        edge = self.config["edge_intervention"]
        self.assertEqual(edge["datasets"], ["Cora", "CiteSeer", "PubMed"])
        self.assertEqual(edge["successful_swaps_per_edge"], 5)
        self.assertEqual(edge["randomization_seed_offset"], 100_000)
        self.assertEqual(edge["model"], "GCN")


class ProspectiveProtocolV2Tests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_V2.read_text(encoding="utf-8"))

    def test_v2_freezes_eligibility_amendment_before_new_outcomes(self):
        self.assertEqual(self.config["run_id"], "route_a_prospective_v2")
        self.assertEqual(self.config["parent_run_id"], "route_a_prospective_v1")
        self.assertEqual(self.config["amendment_date"], "2026-08-11")
        self.assertEqual(
            self.config["amendment_reason"],
            "Texas has a singleton class and cannot satisfy the per-class three-way split.",
        )
        self.assertEqual(
            self.config["eligibility"],
            {
                "minimum_class_count": 3,
                "excluded_datasets": ["Texas"],
                "evaluated_before_v2_outcomes": True,
            },
        )
        self.assertEqual(
            self.config["datasets"],
            [
                "Cora",
                "CiteSeer",
                "PubMed",
                "Wisconsin",
                "Cornell",
                "Chameleon",
                "Squirrel",
                "Actor",
                "Roman-empire",
                "Amazon-ratings",
                "Coauthor-CS",
            ],
        )
        self.assertNotIn("Texas", self.config["datasets"])

    def test_v2_changes_no_scientific_setting_beyond_scope_and_eligibility(self):
        v1 = json.loads(CONFIG.read_text(encoding="utf-8"))
        for key in (
            "seeds",
            "models",
            "split",
            "training",
            "model_parameters",
            "diagnostics",
            "evaluator",
            "edge_intervention",
        ):
            self.assertEqual(self.config[key], v1[key])


if __name__ == "__main__":
    unittest.main()
