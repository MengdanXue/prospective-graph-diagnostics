import copy
import json
from pathlib import Path
import re
import unittest
from scripts.summarize_portfolio_robustness import fit_threshold, leave_one_dataset_out
from scripts.check_latex_log import issues


class RobustnessTests(unittest.TestCase):
    def test_manuscript_robustness_tables_match_released_values(self):
        root = Path(__file__).resolve().parents[1]
        result = json.loads((root / "results/diagnostic/route_a_prospective_v2/analysis/portfolio_robustness.json").read_text())
        source = (root / "sections_tmlr/04a_equal_budget_sensitivity.tex").read_text()
        subset_rows = re.findall(r"^(\d+) & (\d+) & (\d+) & (-?[\d.]+) & (-?[\d.]+) \\\\", source, re.M)
        self.assertEqual(len(subset_rows), 6)
        for stated, expected in zip(subset_rows, result["by_size"]):
            self.assertEqual(tuple(map(int, stated[:3])), tuple(expected[k] for k in ("size", "count", "combined_better")))
            self.assertEqual(stated[3:], (f"{expected['min_delta_pp']:.2f}", f"{expected['max_delta_pp']:.2f}"))
        calibrated_rows = re.findall(r"^(Six architectures|GAT|GCN|GPR-GNN|GraphSAGE|H2GCN|LINKX) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\", source, re.M)
        self.assertEqual(len(calibrated_rows), 7)
        for name, frozen, calibrated, always in calibrated_rows:
            key = "full" if name == "Six architectures" else name
            entry = next(r for r in result["subsets"] if (len(r["portfolio"]) == 6 if key == "full" else r["portfolio"] == [name]))
            self.assertEqual(frozen, f"{entry['mean_regret_pp']['combined']:.2f}")
            self.assertEqual(always, f"{entry['mean_regret_pp']['always_graph']:.2f}")
            self.assertEqual(calibrated, f"{result['calibrated_threshold'][key]['mean_regret_pp']:.2f}")

    def test_heldout_outcomes_cannot_change_its_selected_threshold(self):
        rows = [{"dataset": d, "homophily": h, "graph_test": g, "mlp_test": m}
                for d, h, g, m in [("A", .2, .1, .9), ("B", .8, .9, .1), ("C", .6, .8, .2)]]
        initial = leave_one_dataset_out(rows)["folds"]["A"]["threshold"]
        altered = copy.deepcopy(rows)
        altered[0].update(graph_test=1., mlp_test=0.)
        self.assertEqual(initial, leave_one_dataset_out(altered)["folds"]["A"]["threshold"])

    def test_dataset_weighting_is_not_seed_weighting(self):
        a = {"dataset": "A", "homophily": .5, "graph_test": .8, "mlp_test": .1}
        b = {"dataset": "B", "homophily": .5, "graph_test": .2, "mlp_test": .8}
        self.assertEqual(fit_threshold([a, b]), fit_threshold([a] + [b] * 100))
        self.assertEqual(fit_threshold([a, b]), -1.)

    def test_constant_policies_and_ties_are_defined(self):
        self.assertEqual(fit_threshold([{"dataset": "A", "homophily": .5, "graph_test": .5, "mlp_test": .5}]), -1.)
        with self.assertRaises(ValueError):
            leave_one_dataset_out([])

    def test_latex_errors_cannot_pass_as_successful_compilation(self):
        for line in [r"Overfull \hbox (1.0pt too wide)", "LaTeX Warning: Reference `missing' on page 1 undefined.",
                     "LaTeX Warning: Citation `missing' on page 1 undefined.", "LaTeX Warning: Label `a' multiply defined."]:
            self.assertTrue(issues(line), line)
        self.assertEqual(issues(r"Underfull \hbox (badness 1000)"), [])


if __name__ == "__main__":
    unittest.main()
