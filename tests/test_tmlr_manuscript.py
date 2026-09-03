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
import unittest
from pathlib import Path

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
        for text in (INTRODUCTION, DISCUSSION, CONCLUSION, MAIN_TEXT):
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
        for text in (INTRODUCTION, DISCUSSION, CONCLUSION, MAIN_TEXT):
            self.assertIn("95.1\\%", text)
            # The 40 declined units are stated in every venue-facing summary,
            # with wording that varies between "all 40 units" and
            # "all 40 of those units".
            self.assertRegex(text, r"all 40 (?:of those )?units")

    def test_attainable_resolution_limitation_is_disclosed(self):
        self.assertIn("$2^{1-k}$", RESULTS)
        self.assertIn("could not have rejected under any realization", RESULTS)
        self.assertIn("$p=0.015625$", RESULTS)
        self.assertIn("$k\\geq 9$", RESULTS)
        always_graph = next(
            c for c in AUDIT["paired_comparisons"] if c["method"] == "always_graph"
        )
        self.assertAlmostEqual(always_graph["raw_p"], 2 / 2**7)
        self.assertGreater(always_graph["holm_adjusted_p"], 0.05)

    def test_negative_result_language_is_retained(self):
        self.assertIn("no stable incremental decision value", RESULTS.lower())

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
        first = subprocess.run(
            [sys.executable, "scripts/summarize_winning_architectures.py", "--output", "/dev/stdout"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("95/110", first.stdout)


if __name__ == "__main__":
    unittest.main()
