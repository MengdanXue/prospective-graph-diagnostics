"""Bind the repositioned manuscript's new claims to the frozen audit.

The repositioned entry point (main_tmlr.tex) reuses the frozen experimental
record and adds a structural-mismatch analysis. These tests assert that every
new quantity it states is reproduced by the deterministic re-aggregation in
scripts/summarize_winning_architectures.py, and that the repositioned sources
remain free of the claims forbidden by the Route A design.
"""

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from experiments.evaluate_diagnostics import holm_adjust, sign_flip_p
from scripts.audit_route_a_claims import audit
from scripts.summarize_winning_architectures import summarize

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main_tmlr.tex"
MAIN_TEXT = MAIN.read_text(encoding="utf-8")
INTRODUCTION = (ROOT / "sections_tmlr" / "01_introduction.tex").read_text(encoding="utf-8")
RESULTS = (ROOT / "sections_tmlr" / "04_prospective_results.tex").read_text(encoding="utf-8")
DISCUSSION = (ROOT / "sections_tmlr" / "07_discussion.tex").read_text(encoding="utf-8")
CONCLUSION = (ROOT / "sections_tmlr" / "08_conclusion.tex").read_text(encoding="utf-8")
APPENDIX = (ROOT / "sections_tmlr" / "09_reproducibility_appendix.tex").read_text(encoding="utf-8")
EQUAL_BUDGET_SECTION = (
    ROOT / "sections_tmlr" / "04a_equal_budget_sensitivity.tex"
).read_text(encoding="utf-8")
EQUAL_BUDGET = json.loads(
    (
        ROOT / "results" / "diagnostic" / "route_a_prospective_v2" / "analysis" / "equal_budget_sensitivity.json"
    ).read_text(encoding="utf-8")
)
AUDIT = json.loads(
    (
        ROOT / "results" / "diagnostic" / "route_a_prospective_v2" / "analysis" / "diagnostic_audit.json"
    ).read_text(encoding="utf-8")
)
PUBLISHED = json.loads(
    (
        ROOT / "results" / "diagnostic" / "route_a_prospective_v2" / "analysis" / "winning_architectures.json"
    ).read_text(encoding="utf-8")
)


class RepositionedManuscriptTests(unittest.TestCase):
    def test_published_summary_matches_regenerated_summary(self):
        self.assertEqual(PUBLISHED, summarize(AUDIT))

    def test_latex_inputs_resolve(self):
        missing = [
            target
            for target in re.findall(r"\\input\{([^}]+)\}", MAIN_TEXT)
            if not (ROOT / f"{target}.tex").is_file()
        ]
        self.assertEqual(missing, [])

    def test_claim_audit_passes_on_repositioned_entry(self):
        self.assertEqual(audit(ROOT, "main_tmlr.tex"), [])

    def test_submitted_manuscript_is_untouched_by_repositioning(self):
        submitted = (ROOT / "main_neurocomputing.tex").read_text(encoding="utf-8")
        self.assertIn("\\journal{Neurocomputing}", submitted)
        self.assertIn("A Prospective Evaluation of Simple Graph Diagnostics", submitted)
        for target in re.findall(r"\\input\{([^}]+)\}", submitted):
            self.assertTrue(target.startswith("sections/"), target)

    def test_heterophily_aware_win_count_is_reported_accurately(self):
        totals = PUBLISHED["totals"]
        self.assertEqual(totals["heterophily_aware_wins"], 95)
        self.assertEqual(totals["units"], 110)
        for text in (INTRODUCTION, DISCUSSION):
            self.assertIn("95 of the 110 units", text)

    def test_regret_concentration_is_reported_accurately(self):
        concentration = PUBLISHED["regret_concentration"]
        self.assertEqual(concentration["top_four_units"], 40)
        self.assertEqual(concentration["top_four_heterophily_aware_wins"], 35)
        self.assertEqual(concentration["top_four_combined_graph_actions"], 0)
        self.assertLess(concentration["top_four_max_homophily"], 0.55)
        self.assertAlmostEqual(
            concentration["top_four_share_of_combined_regret"], 0.951, places=3
        )
        for text in (INTRODUCTION, DISCUSSION):
            self.assertIn("95.1\\%", text)
            # The 40 declined units are stated in every venue-facing summary,
            # with wording that varies between "all 40 units" and
            # "all 40 of those units".
            self.assertRegex(text, r"all 40 (?:of those )?units")

    def test_attainable_resolution_limitation_is_disclosed(self):
        self.assertIn("$2^{1-k}$", RESULTS)
        self.assertNotIn("could not have rejected under any realization", RESULTS)
        self.assertIn("conditional resolution limitation", RESULTS)
        self.assertIn("Holm's first step", RESULTS)
        self.assertIn("$p=0.015625$", RESULTS)
        always_graph = next(
            c for c in AUDIT["paired_comparisons"] if c["method"] == "always_graph"
        )
        self.assertAlmostEqual(always_graph["raw_p"], 2 / 2**7)
        self.assertGreater(always_graph["holm_adjusted_p"], 0.05)

    def test_eleven_datasets_do_not_imply_zero_attainable_power(self):
        observed_pattern = np.array([-1.0] * 7 + [0.0] * 4)
        self.assertEqual(sign_flip_p(observed_pattern, samples=10000, seed=0), 2 / 2**7)
        possible_pattern = np.ones(11)
        floor = sign_flip_p(possible_pattern, samples=10000, seed=0)
        family = [{"raw_p": floor} for _ in range(8)]
        holm_adjust(family)
        self.assertLess(family[0]["holm_adjusted_p"], .05)
        self.assertIn("0.0009765625", RESULTS)

    def test_equal_budget_table_matches_the_frozen_summary(self):
        rows = re.findall(
            r"^([A-Za-z0-9-]+) & 4 & (\d+)/110 & ([\d.]+) & ([\d.]+) & ([\d.]+) & (\d)/(\d)/(\d) \\\\",
            EQUAL_BUDGET_SECTION,
            re.M,
        )
        self.assertEqual(len(rows), 6)
        for name, targets, always, combined, validation, wins, losses, ties in rows:
            entry = EQUAL_BUDGET["equal_budget_single_architecture"][name]
            self.assertEqual(entry["total_trials"], 4)
            self.assertEqual(entry["graph_targets"], int(targets))
            for policy, stated in (
                ("always_graph", always),
                ("combined", combined),
                ("validation", validation),
            ):
                self.assertAlmostEqual(
                    entry["mean_regret_pp"][policy], float(stated), places=2
                )
            comparison = entry["combined_vs_always_graph"]
            self.assertEqual(comparison["dataset_wins"], int(wins))
            self.assertEqual(comparison["dataset_losses"], int(losses))
            self.assertEqual(comparison["dataset_ties"], int(ties))

    def test_equal_budget_summary_reproduces_the_published_portfolio(self):
        full = EQUAL_BUDGET["full_portfolio"]
        self.assertEqual(full["total_trials"], 24)
        self.assertEqual(full["graph_targets"], 89)
        for policy, expected in (
            ("always_graph", 0.26),
            ("combined", 7.46),
            ("validation", 0.22),
        ):
            self.assertAlmostEqual(full["mean_regret_pp"][policy], expected, places=2)

    def test_equal_budget_claim_is_bounded_not_absolute(self):
        # The matched-budget result shows the rule beating always-graph on mean
        # regret for GCN and GAT, so the manuscript must not claim the
        # diagnostics carry no information under any portfolio.
        for name in ("GCN", "GAT"):
            entry = EQUAL_BUDGET["equal_budget_single_architecture"][name]
            self.assertLess(
                entry["mean_regret_pp"]["combined"],
                entry["mean_regret_pp"]["always_graph"],
            )
        for text in (INTRODUCTION, DISCUSSION, CONCLUSION, MAIN_TEXT, EQUAL_BUDGET_SECTION):
            self.assertIn("post-hoc", text.lower())
            self.assertNotIn("No threshold on a scalar one-hop summary repairs this", text)
            self.assertNotIn("Why Homophily Statistics Cannot", text)
        self.assertIn("It does not equalize training time", EQUAL_BUDGET_SECTION)
        self.assertNotIn("Equal-Total-Compute Sensitivity", EQUAL_BUDGET_SECTION)
        self.assertIn("not a proof that no scalar threshold can improve it", DISCUSSION)

    def test_portfolio_interpretation_preserves_the_gpr_counterexample(self):
        single = EQUAL_BUDGET["equal_budget_single_architecture"]
        for other in ("GCN", "GAT"):
            self.assertLess(single["GPR-GNN"]["mean_regret_pp"]["combined"], single[other]["mean_regret_pp"]["combined"])
        self.assertIn("lowest absolute Combined regret (1.50 points)", EQUAL_BUDGET_SECTION)
        self.assertIn("not the number of units in which graph beats MLP", DISCUSSION)
        self.assertIn("does not guarantee a better selected test result", EQUAL_BUDGET_SECTION)

    def test_appendix_does_not_reuse_body_table_numbers(self):
        self.assertNotIn(r"\setcounter{table}{0}", MAIN_TEXT)

    def test_negative_result_language_is_retained(self):
        self.assertIn("do not demonstrate incremental value in this benchmark", RESULTS.lower())
        self.assertNotIn("no stable incremental decision value", RESULTS.lower())

    def test_mismatch_claim_is_scoped_to_the_paired_portfolio(self):
        self.assertIn("heterophily-aware portfolio", CONCLUSION)
        self.assertIn("does not imply that richer diagnostics", DISCUSSION)
        self.assertNotIn("regardless of architecture", DISCUSSION)

    def test_appendix_table_rows_match_the_frozen_summary(self):
        rows = re.findall(
            r"^([A-Za-z-]+) & ([\d.]+) & .+? & (\d+)/10 & (\d+) & ([\d.]+) \\\\",
            APPENDIX,
            re.M,
        )
        self.assertEqual(len(rows), 11)
        for name, homophily, aware, graph_actions, regret in rows:
            entry = PUBLISHED["datasets"][name]
            self.assertAlmostEqual(
                entry["mean_train_label_homophily"], float(homophily), places=3
            )
            self.assertEqual(entry["heterophily_aware_wins"], int(aware))
            self.assertEqual(entry["combined_selects_graph"], int(graph_actions))
            self.assertAlmostEqual(
                100 * entry["mean_combined_regret"], float(regret), places=2
            )

    def test_summary_script_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"summary-{i}.json" for i in range(2)]
            for path in paths:
                completed = subprocess.run(
                    [sys.executable, "scripts/summarize_winning_architectures.py", "--output", str(path)],
                    cwd=ROOT, capture_output=True, text=True, check=True,
                )
                self.assertIn("95/110", completed.stdout)
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            self.assertEqual(json.loads(paths[0].read_text()), PUBLISHED)
            original = paths[0].read_bytes()
            retry = subprocess.run(
                [sys.executable, "scripts/summarize_winning_architectures.py", "--output", str(paths[0])],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(paths[0].read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
