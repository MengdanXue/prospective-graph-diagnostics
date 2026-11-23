"""Bind the repositioned manuscript's new claims to the frozen audit.

The repositioned entry point (main_tmlr.tex) reuses the frozen experimental
record and adds a structural-mismatch analysis. These tests assert that every
new quantity it states is reproduced by the deterministic re-aggregation in
scripts/summarize_winning_architectures.py, and that the repositioned sources
remain free of the claims forbidden by the Route A design.
"""

import copy
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from experiments.evaluate_diagnostics import (
    action_regret,
    dataset_mean_regret,
    fixed_decisions,
    holm_adjust,
    sign_flip_p,
)
from scripts.audit_route_a_claims import audit
from scripts.summarize_winning_architectures import summarize

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main_tmlr.tex"
MAIN_TEXT = MAIN.read_text(encoding="utf-8")
INTRODUCTION = (ROOT / "sections_tmlr" / "01_introduction.tex").read_text(encoding="utf-8")
PROTOCOL = (ROOT / "sections_tmlr" / "03_decision_protocol.tex").read_text(encoding="utf-8")
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
        # Bind the detailed claim where it is reported; concise summaries need
        # not repeat every appendix statistic.
        self.assertIn("in 95 of 110 units", APPENDIX)

    def test_regret_concentration_is_reported_accurately(self):
        concentration = PUBLISHED["regret_concentration"]
        self.assertEqual(concentration["top_four_units"], 40)
        self.assertEqual(concentration["top_four_heterophily_aware_wins"], 35)
        self.assertEqual(concentration["top_four_combined_graph_actions"], 0)
        self.assertLess(concentration["top_four_max_homophily"], 0.55)
        self.assertAlmostEqual(
            concentration["top_four_share_of_combined_regret"], 0.951, places=3
        )
        self.assertIn("95.1\\%", APPENDIX)
        # The dataset table below independently binds the graph-action counts
        # and regrets; the introduction and discussion may summarize them.

    def test_attainable_resolution_limitation_is_disclosed(self):
        self.assertIn("$2^{1-k}$", RESULTS)
        self.assertNotIn("could not have rejected under any realization", RESULTS)
        self.assertIn("Structural resolution limitation of the key comparison", RESULTS)
        self.assertIn("identically zero, irrespective of the learned test accuracies", RESULTS)
        self.assertIn("unattainable for this key comparison", RESULTS)
        self.assertIn("Holm's first step", RESULTS)
        self.assertIn("$p=0.015625$", RESULTS)
        self.assertIn("Holm's running maximum", RESULTS)
        self.assertIn("$6p_0=0.09375$", RESULTS)
        self.assertIn("fixed eight-comparison family and realized dataset composition", RESULTS)
        always_graph = next(
            c for c in AUDIT["paired_comparisons"] if c["method"] == "always_graph"
        )
        self.assertAlmostEqual(always_graph["raw_p"], 2 / 2**7)
        self.assertGreater(always_graph["holm_adjusted_p"], 0.05)
        self.assertEqual(always_graph["holm_adjusted_p"], 0.125)

    def test_fixed_actions_impose_structural_zeros_and_holm_resolution_bound(self):
        def effective_action(unit, method):
            action = unit["decisions"][method]["action"]
            return "mlp" if action == "abstain" else action

        units = AUDIT["units"]
        for unit in units:
            self.assertEqual(fixed_decisions(unit), unit["decisions"])
        datasets = {unit["dataset"] for unit in units}
        self.assertEqual(len(units), 110)
        self.assertEqual(len(datasets), 11)
        methods = [row["method"] for row in AUDIT["paired_comparisons"]]
        self.assertEqual(len(methods), 8)
        differing_datasets = {
            method: {
                unit["dataset"] for unit in units
                if effective_action(unit, method)
                != effective_action(unit, "historical_combined")
            }
            for method in methods
        }
        self.assertEqual(
            {method: len(names) for method, names in differing_datasets.items()},
            {
                "always_graph": 7, "always_mlp": 4, "degree_only": 6,
                "homophily_only": 0, "homophily_plus_degree": 5,
                "random_50_50": 11, "two_hop_only": 7,
                "validation_selection": 6,
            },
        )
        structural_zeros = datasets - differing_datasets["always_graph"]
        self.assertEqual(structural_zeros, {"Cora", "CiteSeer", "PubMed", "Coauthor-CS"})
        self.assertEqual(sum(unit["dataset"] in structural_zeros for unit in units), 40)
        reference = dataset_mean_regret(units, "historical_combined")
        candidate = dataset_mean_regret(units, "always_graph")
        observed_pattern = np.array([candidate[name] - reference[name] for name in sorted(datasets)])
        self.assertEqual(np.count_nonzero(observed_pattern), 7)
        self.assertEqual(sign_flip_p(observed_pattern, samples=10000, seed=0), 2 / 2**7)

        # Fixed diagnostic actions constrain nonzero dataset differences even
        # when trained accuracies change. Validation may change on all eleven
        # datasets, so give it the most permissive possible raw-p floor.
        floor_family = []
        for method in methods:
            maximum_nonzero = (
                len(datasets) if method == "validation_selection"
                else len(differing_datasets[method])
            )
            floor_family.append({
                "method": method,
                "raw_p": 2 ** (1 - maximum_nonzero) if maximum_nonzero else 1.0,
            })
        self.assertEqual(
            {row["method"] for row in floor_family if row["raw_p"] < 2 / 2**7},
            {"random_50_50", "validation_selection"},
        )
        holm_adjust(floor_family)
        core_bound = next(row for row in floor_family if row["method"] == "always_graph")
        self.assertEqual(core_bound["holm_adjusted_p"], 6 * (2 / 2**7))
        self.assertGreater(core_bound["holm_adjusted_p"], .05)

        # A tied two-hop floor can put always-graph fourth, while Holm's
        # running maximum retains the third-position factor of six.
        tied_family = [row.copy() for row in floor_family if row["method"] != "always_graph"]
        tied_family.append(core_bound.copy())
        ordered = sorted(tied_family, key=lambda row: row["raw_p"])
        self.assertEqual([row["method"] for row in ordered].index("always_graph"), 3)
        holm_adjust(tied_family)
        tied_core = next(row for row in tied_family if row["method"] == "always_graph")
        self.assertEqual(tied_core["holm_adjusted_p"], 6 * (2 / 2**7))

    def test_paper_level_rule_preserves_regret_coverage_and_interval_requirements(self):
        specification = (ROOT / "docs" / "preregistration_diagnostic_benchmark.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("full-set regret relative to simple baselines without relying on lower coverage", specification)
        self.assertIn("the uncertainty interval excludes zero for the predeclared comparison", specification)
        self.assertIn("reduces full-set regret relative to simple baselines without relying on lower coverage", PROTOCOL)
        self.assertIn("uncertainty interval excludes zero in the favorable direction", PROTOCOL)
        self.assertIn("baseline-minus-Combined difference, the favorable direction is positive", PROTOCOL)
        self.assertIn("a significance result alone does not replace the regret and coverage conditions", PROTOCOL)
        self.assertGreater(
            AUDIT["methods"]["historical_combined"]["full_set_mean_regret"],
            AUDIT["methods"]["always_graph"]["full_set_mean_regret"],
        )
        comparison = next(row for row in AUDIT["paired_comparisons"] if row["method"] == "always_graph")
        self.assertLess(comparison["bootstrap_95_ci"][1], 0)
        self.assertIn("independently of the sign-flip resolution limit", RESULTS)

    def test_policy_origin_remains_disclosed_without_private_source_identifiers(self):
        self.assertIn("8 August 2026", PROTOCOL)
        self.assertIn("after inspection of legacy aggregate outputs", PROTOCOL)
        self.assertIn("exploratory-origin rule", PROTOCOL)
        self.assertIn("source provenance is retained in the author archive", PROTOCOL)
        self.assertNotIn(r"route\_a\_diagnostic\_v1", PROTOCOL)
        self.assertNotIn("dca835a", PROTOCOL)
        self.assertNotIn("confirmatory", PROTOCOL.lower())
        self.assertNotIn("confirmatory", RESULTS.lower())

    def test_fallback_decomposition_table_matches_frozen_unit_regrets(self):
        units = AUDIT["units"]
        groups = {
            "Covered decisions": [u for u in units if u["decisions"]["historical_combined"]["action"] != "abstain"],
            "Abstentions with MLP fallback": [u for u in units if u["decisions"]["historical_combined"]["action"] == "abstain"],
            "Total": units,
        }
        rows = re.findall(
            r"^(Covered decisions|Abstentions with MLP fallback|Total) & (\d+) & ([\d.]+) & ([\d.]+) \\\\",
            RESULTS, re.M,
        )
        self.assertEqual(len(rows), 3)
        group_sums = {}
        for label, count, mean, contribution in rows:
            selected = groups[label]
            total = sum(action_regret(
                unit,
                "mlp" if unit["decisions"]["historical_combined"]["action"] == "abstain"
                else unit["decisions"]["historical_combined"]["action"],
            ) for unit in selected)
            group_sums[label] = total
            self.assertEqual(len(selected), int(count))
            self.assertEqual(f"{100 * total / len(selected):.3f}", mean)
            self.assertEqual(f"{100 * total / len(units):.3f}", contribution)
        self.assertEqual(len(groups["Covered decisions"]), 75)
        self.assertEqual(len(groups["Abstentions with MLP fallback"]), 35)
        self.assertAlmostEqual(
            group_sums["Covered decisions"] + group_sums["Abstentions with MLP fallback"],
            group_sums["Total"], places=12,
        )
        self.assertAlmostEqual(
            group_sums["Total"] / len(units),
            AUDIT["methods"]["historical_combined"]["full_set_mean_regret"], places=12,
        )
        share = 100 * group_sums["Abstentions with MLP fallback"] / group_sums["Total"]
        self.assertIn(f"{share:.3f}\\% of the total regret", RESULTS)

    def test_random_policy_can_reject_in_the_same_family_with_fixed_diagnostic_actions(self):
        # This is a counterexample fixture, not a new research result. Keep
        # every actual diagnostic and validation input, but give each unit's
        # Combined action the lower synthetic test accuracy.
        units = copy.deepcopy(AUDIT["units"])
        for unit in units:
            combined = unit["decisions"]["historical_combined"]["action"]
            combined = "mlp" if combined == "abstain" else combined
            unit["selected_graph_test"] = .25 if combined == "graph" else .75
            unit["selected_mlp_test"] = .75 if combined == "graph" else .25
            unit["test_gap"] = unit["selected_graph_test"] - unit["selected_mlp_test"]
            unit["target_action"] = "graph" if unit["test_gap"] > .01 else "mlp"
            self.assertEqual(fixed_decisions(unit), unit["decisions"])
        reference = dataset_mean_regret(units, "historical_combined")
        family = []
        for comparison in AUDIT["paired_comparisons"]:
            method = comparison["method"]
            candidate = dataset_mean_regret(units, method)
            differences = np.array([
                candidate[name] - reference[name] for name in sorted(reference)
            ])
            if method == "random_50_50":
                self.assertEqual(len(differences), 11)
                self.assertTrue(np.all(differences == -.25))
            family.append({
                "method": method,
                "raw_p": sign_flip_p(differences, samples=10000, seed=0),
            })
        self.assertEqual(len(family), 8)
        holm_adjust(family)
        random = next(row for row in family if row["method"] == "random_50_50")
        self.assertEqual(random["raw_p"], 2 / 2**11)
        self.assertEqual(random["holm_adjusted_p"], 8 * (2 / 2**11))
        self.assertLess(random["holm_adjusted_p"], .05)
        self.assertIn("does not imply that every comparison in the family has zero power", RESULTS)

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
        self.assertIn("does not prove a universal mechanism or exclude other threshold choices", EQUAL_BUDGET_SECTION)

    def test_portfolio_interpretation_preserves_the_gpr_counterexample(self):
        single = EQUAL_BUDGET["equal_budget_single_architecture"]
        for other in ("GCN", "GAT"):
            self.assertLess(single["GPR-GNN"]["mean_regret_pp"]["combined"], single[other]["mean_regret_pp"]["combined"])
        self.assertIn("lowest absolute Combined regret (1.50 points)", EQUAL_BUDGET_SECTION)
        self.assertIn("not wins over MLP", APPENDIX)
        self.assertIn("does not guarantee a better selected test result", EQUAL_BUDGET_SECTION)

    def test_appendix_does_not_reuse_body_table_numbers(self):
        self.assertNotIn(r"\setcounter{table}{0}", MAIN_TEXT)

    def test_negative_result_language_is_retained(self):
        self.assertIn("do not demonstrate incremental value in this benchmark", RESULTS.lower())
        self.assertNotIn("no stable incremental decision value", RESULTS.lower())

    def test_mismatch_claim_is_scoped_to_the_paired_portfolio(self):
        self.assertIn("not a representative comparison with all published diagnostic methods", EQUAL_BUDGET_SECTION)
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
