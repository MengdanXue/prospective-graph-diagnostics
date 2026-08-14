import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (ROOT / "sections" / "03_decision_protocol.tex").read_text(
    encoding="utf-8"
)
RESULTS = (ROOT / "sections" / "04_prospective_results.tex").read_text(
    encoding="utf-8"
)
INTERVENTION = (ROOT / "sections" / "05_edge_intervention.tex").read_text(
    encoding="utf-8"
)


class DiagnosticLanguageTests(unittest.TestCase):
    def test_label_dependent_diagnostics_are_train_only(self):
        self.assertIn("training labels only", PROTOCOL)
        self.assertIn("Test labels", PROTOCOL)
        self.assertIn("forbidden diagnostic inputs", PROTOCOL)

    def test_negative_result_is_reported_against_trivial_baselines(self):
        lower = RESULTS.lower()
        self.assertIn("always-graph", lower)
        self.assertIn("no stable incremental decision value", lower)
        self.assertIn("regret", lower)

    def test_intervention_does_not_overclaim_a_single_mechanism(self):
        lower = INTERVENTION.lower()
        self.assertIn("does not isolate a single changed property", lower)
        self.assertIn("do not show that homophily alone causes the change", lower)
        self.assertIn("do not", lower)


if __name__ == "__main__":
    unittest.main()
