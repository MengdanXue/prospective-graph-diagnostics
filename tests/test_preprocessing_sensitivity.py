import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = json.loads((ROOT / "results/diagnostic/route_a_prospective_v2/analysis/preprocessing_sensitivity.json").read_text())
SECTION = (ROOT / "sections_tmlr/04a_equal_budget_sensitivity.tex").read_text()


class PreprocessingSensitivityTests(unittest.TestCase):
    def test_complete_scope_and_environment_are_recorded(self):
        self.assertTrue(SUMMARY["source_run"]["scope_verified"])
        self.assertEqual(SUMMARY["source_run"]["raw_record_count"], 280)
        self.assertEqual(set(SUMMARY["datasets"]), {"Roman-empire", "Amazon-ratings"})
        for dataset in SUMMARY["datasets"].values():
            self.assertEqual(set(dataset), {"normalize_features", "raw"})
            for condition in dataset.values():
                self.assertEqual(len(condition["units"]), 10)

    def test_reported_table_values_bind_to_summary(self):
        rows = re.findall(
            r"^(Roman-empire|Amazon-ratings) & (NormalizeFeatures|Raw) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & (\d+/\d+) \\\\",
            SECTION,
            re.M,
        )
        self.assertEqual(len(rows), 4)
        for dataset, label, mlp, graph, gap, regret, targets in rows:
            condition = "normalize_features" if label == "NormalizeFeatures" else "raw"
            entry = SUMMARY["datasets"][dataset][condition]
            self.assertEqual(mlp, f"{100 * entry['mean_mlp_test']:.2f}")
            self.assertEqual(graph, f"{100 * entry['mean_graph_test']:.2f}")
            self.assertEqual(gap, f"{100 * entry['mean_test_gap']:.2f}")
            self.assertEqual(regret, f"{entry['mean_regret_pp']['historical_combined']:.2f}")
            self.assertEqual(targets, f"{entry['target_graph']}/10")

    def test_preprocessing_change_is_reported_as_paired_descriptive_evidence(self):
        self.assertAlmostEqual(SUMMARY["paired_differences"]["Roman-empire"]["mlp_test"]["mean_raw_minus_normalized"], .4948108971)
        self.assertAlmostEqual(SUMMARY["paired_differences"]["Amazon-ratings"]["mlp_test"]["mean_raw_minus_normalized"], .0458792329)
        self.assertIn("post-hoc two-dataset sensitivity", SECTION)
        self.assertIn("does not show that normalization caused the full benchmark's negative result", SECTION)


if __name__ == "__main__":
    unittest.main()
