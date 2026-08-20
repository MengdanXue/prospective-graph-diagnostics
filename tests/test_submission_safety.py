import json
import re
import unittest
from pathlib import Path

from scripts.audit_route_a_claims import resolve_active_sources, strip_latex_comment
import scripts.plot_prospective_regret_coverage as regret_coverage_plot


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MAIN = ROOT / "main_neurocomputing.tex"
MAIN = ACTIVE_MAIN.read_text(encoding="utf-8")
INTRODUCTION = (ROOT / "sections" / "01_introduction.tex").read_text(encoding="utf-8")
RELATED = (ROOT / "sections" / "02_related_work.tex").read_text(encoding="utf-8")
PROTOCOL = (ROOT / "sections" / "03_decision_protocol.tex").read_text(encoding="utf-8")
PROSPECTIVE_RESULTS = (ROOT / "sections" / "04_prospective_results.tex").read_text(encoding="utf-8")
THEORY = (ROOT / "sections" / "06_fixed_degree_analysis.tex").read_text(encoding="utf-8")
EDGE_INTERVENTION = (ROOT / "sections" / "05_edge_intervention.tex").read_text(encoding="utf-8")
DISCUSSION = (ROOT / "sections" / "07_discussion.tex").read_text(encoding="utf-8")
CONCLUSION = (ROOT / "sections" / "08_conclusion.tex").read_text(encoding="utf-8")
CLAIM_MAP = (ROOT / "docs" / "claim_evidence_map.md").read_text(encoding="utf-8")
SUMMARY = json.loads(
    (ROOT / "results" / "discriminability" / "route_a_grid_v1" / "summary" / "summary.json").read_text(
        encoding="utf-8"
    )
)
PROSPECTIVE = json.loads(
    (ROOT / "results" / "diagnostic" / "route_a_prospective_v2" / "analysis" / "diagnostic_audit.json").read_text(
        encoding="utf-8"
    )
)
DEGREE_MATCHED = json.loads(
    (ROOT / "results" / "diagnostic" / "route_a_degree_matched_v1" / "summary" / "summary.json").read_text(
        encoding="utf-8"
    )
)
ACTIVE_SOURCES = resolve_active_sources(ROOT, ACTIVE_MAIN)
ACTIVE_TEXT = "\n".join(
    strip_latex_comment(line)
    for path in ACTIVE_SOURCES
    for line in path.read_text(encoding="utf-8").splitlines()
)


class SubmissionSafetyTests(unittest.TestCase):
    def test_active_entry_uses_elsevier_front_matter(self):
        self.assertIn("\\documentclass[preprint,12pt]{elsarticle}", MAIN)
        self.assertIn("\\journal{Neurocomputing}", MAIN)
        self.assertIn("\\begin{frontmatter}", MAIN)
        self.assertIn("\\begin{abstract}", MAIN)
        self.assertIn("\\begin{keyword}", MAIN)
        self.assertNotIn("IEEEtran", MAIN)
        self.assertNotIn("\\maketitle", MAIN)
        self.assertNotIn("IEEEkeywords", MAIN)

    def test_abstract_is_within_the_250_word_submission_target(self):
        abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", MAIN, re.DOTALL)
        self.assertIsNotNone(abstract)
        words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", abstract.group(1))
        self.assertLessEqual(len(words), 250)

    def test_active_entry_uses_benchmark_centered_section_order(self):
        expected_inputs = [
            "sections/01_introduction",
            "sections/02_related_work",
            "sections/03_decision_protocol",
            "sections/04_prospective_results",
            "sections/05_edge_intervention",
            "sections/06_fixed_degree_analysis",
            "sections/07_discussion",
            "sections/08_conclusion",
        ]
        positions = [MAIN.index(f"\\input{{{path}}}") for path in expected_inputs]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(MAIN.count("\\input{"), len(expected_inputs))

    def test_submission_highlights_are_concise(self):
        highlights = [
            line[2:].strip()
            for line in (ROOT / "highlights.txt").read_text(encoding="utf-8").splitlines()
            if line.startswith("- ")
        ]
        self.assertGreaterEqual(len(highlights), 3)
        self.assertLessEqual(len(highlights), 5)
        self.assertTrue(all(len(line) <= 85 for line in highlights))

    def test_highlights_are_supplied_only_as_a_separate_file(self):
        self.assertNotIn("\\begin{highlights}", MAIN)
        self.assertNotIn("\\end{highlights}", MAIN)

    def test_title_and_abstract_match_the_scoped_contribution(self):
        self.assertNotIn("SNR Dynamics", MAIN)
        self.assertIn("A Prospective Evaluation of Graph Diagnostics", MAIN)
        self.assertIn("55.5\\%", MAIN)
        self.assertIn("7.46", MAIN)
        self.assertIn("80.9\\%", MAIN)
        self.assertIn("91.8\\%", MAIN)
        self.assertIn(
            "diagnostics whose label-dependent graph statistics use training labels only",
            MAIN,
        )

    def test_neighbor_only_approximation_uses_the_exact_closed_form(self):
        self.assertIn(
            "\\kappa=\\frac{\\rho^2d}{1+\\eta}\\leq\\rho^2d",
            THEORY.replace(" ", "").replace("\n", ""),
        )
        self.assertNotIn(
            "\\frac{\\rho^2d}{1+\\eta}\\leq\\kappa",
            THEORY.replace(" ", "").replace("\n", ""),
        )
        self.assertIn("For $\\rho\\neq 0$", THEORY)

    def test_combined_threshold_provenance_is_explicit(self):
        self.assertIn("predecessor commit \\texttt{dca835a}", PROTOCOL)
        self.assertIn("after inspection of legacy aggregate outputs", PROTOCOL)
        self.assertIn("before any confirmatory unit-level record was generated", PROTOCOL)
        self.assertIn("not derived through a prospective threshold sweep", PROTOCOL)
        self.assertIn("not adjusted using the final 110 units", PROTOCOL)
        self.assertIn("rather than claim whole-project preregistration", PROTOCOL)

    def test_introduction_reports_prospective_evaluation_as_completed(self):
        self.assertIn("770 model records", INTRODUCTION)
        self.assertIn(
            "110 diagnostic records whose label-dependent graph statistics use training labels only",
            INTRODUCTION,
        )
        self.assertNotIn("train-only diagnostic records", INTRODUCTION)
        self.assertNotIn("prospective diagnostic benchmark has not yet been executed", INTRODUCTION)

    def test_related_work_uses_the_precise_label_scope(self):
        self.assertIn(
            "transparent rules whose label-dependent graph statistics use training labels only",
            RELATED,
        )
        self.assertNotIn("transparent train-only rules", RELATED)

    def test_protocol_defines_the_decision_boundary_and_forbids_test_labels(self):
        self.assertIn("\\delta=0.01", PROTOCOL)
        self.assertIn("Test labels, test accuracy", PROTOCOL)
        self.assertIn("oracle-portfolio regret", PROTOCOL)
        self.assertIn("Holm correction", PROTOCOL)
        self.assertNotIn("Information Budget", PROTOCOL)

    def test_protocol_reports_the_frozen_training_setup(self):
        for phrase in (
            "Adam",
            "cross-entropy",
            "64",
            "500",
            "100",
            "5\\times 10^{-4}",
            "0.01",
            "0.005",
            "0.5",
            "0.7",
            "one held-out test evaluation",
        ):
            self.assertIn(phrase, PROTOCOL)

    def test_active_text_discloses_asymmetric_family_search_budget(self):
        self.assertIn("four-trial tuning budget", PROTOCOL)
        self.assertIn("24 trial configurations", PROTOCOL)
        self.assertIn("four trial configurations", PROTOCOL)
        self.assertIn("asymmetric portfolios", PROTOCOL)
        self.assertNotIn("Each family receives the same four-trial", RELATED)
        self.assertNotIn("no model family obtains a larger search budget", PROTOCOL)

    def test_target_accuracy_and_regret_objectives_are_distinguished(self):
        self.assertIn("intentionally measure different objectives", PROTOCOL)
        self.assertIn("sub-margin graph gain", PROTOCOL)
        self.assertIn("raw predictive-accuracy opportunity loss", PROTOCOL)

    def test_scope_does_not_claim_graph_acquisition_savings(self):
        self.assertNotIn("acquisition is costly", INTRODUCTION)
        self.assertIn("training and tuning the graph-model portfolio", INTRODUCTION)

    def test_prospective_claim_is_limited_to_new_confirmatory_records(self):
        self.assertIn("newly generated confirmatory records", INTRODUCTION)
        prospective_lower = PROSPECTIVE_RESULTS.lower()
        self.assertIn("legacy aggregate results", prospective_lower)
        self.assertIn("private predecessor history", prospective_lower)

    def test_regret_coverage_display_is_defined_and_included(self):
        self.assertIn("covered-set raw-accuracy regret", PROTOCOL)
        self.assertIn("fig:regret_coverage", PROSPECTIVE_RESULTS)
        self.assertTrue(
            (
                ROOT
                / "results"
                / "diagnostic"
                / "route_a_prospective_v2"
                / "analysis"
                / "prospective_regret_coverage.pdf"
            ).exists()
        )
        self.assertIn("sustained regret region comparable", PROSPECTIVE_RESULTS)
        self.assertNotIn("a different fixed threshold rescues", PROSPECTIVE_RESULTS)
        self.assertIn(
            "degree-only and homophily-plus-degree reduce descriptive full-set regret",
            PROSPECTIVE_RESULTS,
        )
        self.assertIn(
            "none of the evaluated low-dimensional rules improves on always-graph",
            PROSPECTIVE_RESULTS,
        )
        self.assertIn("unadjusted 95\\% bootstrap interval", PROSPECTIVE_RESULTS)
        self.assertIn("with an unadjusted interval", PROSPECTIVE_RESULTS)

    def test_regret_coverage_confidence_scores_are_reproducible_from_the_text(self):
        self.assertIn(
            "We denote the resulting one-hop and path-weighted length-two endpoint agreement rates by $h_1$ and $h_2$",
            PROTOCOL,
        )
        self.assertIn(
            "are the validation accuracies of the selected graph and MLP portfolios",
            PROTOCOL,
        )
        self.assertIn("c(v;s)=\\min\\{1,|v|/s\\}", PROTOCOL)
        self.assertIn(
            "a_{\\mathrm{val}}^{\\mathrm{G}}-a_{\\mathrm{val}}^{\\mathrm{M}}-\\delta",
            PROTOCOL,
        )
        for phrase in ("s=0.25", "h_1-0.55", "s=0.45", "s=0.60"):
            self.assertIn(phrase, PROTOCOL)
        self.assertIn(
            "\\max\\{0.55-h_1,a_{\\mathrm{val}}^{\\mathrm{M}}-0.40\\}",
            PROTOCOL,
        )
        self.assertIn("Abstentions receive no confidence score", PROTOCOL)

    def test_statistical_claims_do_not_overstate_non_significance(self):
        self.assertIn(
            "paired comparisons do not establish an advantage for the combined diagnostic",
            MAIN,
        )
        self.assertNotIn("paired comparisons establish no advantage", MAIN)
        self.assertIn("does not demonstrate incremental value", PROSPECTIVE_RESULTS)
        self.assertIn("does not reduce regret relative to always-graph", PROSPECTIVE_RESULTS)
        self.assertNotIn("predeclared incremental-value criterion", PROSPECTIVE_RESULTS)

    def test_data_availability_prints_the_repository_and_release_urls(self):
        self.assertRegex(
            MAIN,
            r"\\section\*\{Data and code availability\}\s*\\begin\{sloppypar\}",
        )
        self.assertIn(
            "\\url{https://github.com/MengdanXue/prospective-graph-diagnostics}",
            MAIN,
        )
        self.assertIn(
            "\\url{https://github.com/MengdanXue/prospective-graph-diagnostics/releases/tag/v0.1.0}",
            MAIN,
        )

    def test_constant_confidence_policies_plot_only_the_full_set_endpoint(self):
        self.assertTrue(hasattr(regret_coverage_plot, "_display_curve"))
        for method in ("always_graph", "always_mlp"):
            coverage, regret = regret_coverage_plot._display_curve(PROSPECTIVE, method)
            self.assertEqual([100.0], coverage)
            self.assertEqual(1, len(regret))

        for method in ("historical_combined", "validation_selection"):
            coverage, regret = regret_coverage_plot._display_curve(PROSPECTIVE, method)
            self.assertGreater(len(coverage), 1)
            self.assertEqual(len(coverage), len(regret))

    def test_discussion_and_conclusion_retain_theory_and_experiment_limits(self):
        self.assertIn("does not imply that every possible graph diagnostic must fail", DISCUSSION)
        self.assertIn("three independent dataset clusters", DISCUSSION)
        self.assertNotIn("binary or balanced multi-class", CONCLUSION)
        self.assertNotIn("two-hop recovery ratio", CONCLUSION)
        self.assertIn("no stable incremental decision value", CONCLUSION)
        self.assertIn("does not isolate homophily", CONCLUSION)

    def test_primary_results_table_contains_every_frozen_policy(self):
        for policy in PROSPECTIVE["methods"]:
            display_name = {
                "always_graph": "Always-graph",
                "always_mlp": "Always-MLP",
                "degree_only": "Degree-only",
                "historical_combined": "Combined",
                "homophily_only": "Homophily-only",
                "homophily_plus_degree": "Homophily+degree",
                "random_50_50": "Random 50/50",
                "two_hop_only": "Two-hop-only",
                "validation_selection": "Validation selection",
            }[policy]
            self.assertIn(display_name, PROSPECTIVE_RESULTS)

    def test_primary_results_table_matches_frozen_audit(self):
        rows = {
            "historical_combined": "Combined & 55.5 & 68.2 & 7.46",
            "always_graph": "Always-graph & 80.9 & 100.0 & 0.26",
            "always_mlp": "Always-MLP & 19.1 & 100.0 & 9.69",
            "random_50_50": "Random 50/50 & 50.0 & 100.0 & 4.97",
            "homophily_only": "Homophily-only & 55.5 & 100.0 & 7.46",
            "degree_only": "Degree-only & 37.3 & 100.0 & 6.24",
            "homophily_plus_degree": "Homophily+degree & 46.4 & 100.0 & 5.80",
            "two_hop_only": "Two-hop-only & 28.2 & 100.0 & 8.27",
            "validation_selection": "Validation selection & 91.8 & 100.0 & 0.22",
        }
        for method, row in rows.items():
            metrics = PROSPECTIVE["methods"][method]
            self.assertIn(row, PROSPECTIVE_RESULTS)
            self.assertAlmostEqual(float(row.split(" & ")[1]), metrics["selection_accuracy"] * 100, places=1)
            self.assertAlmostEqual(float(row.split(" & ")[2]), metrics["coverage"] * 100, places=1)
            self.assertAlmostEqual(float(row.split(" & ")[3]), metrics["full_set_mean_regret"] * 100, places=2)

    def test_edge_intervention_table_matches_frozen_summary(self):
        rows = {
            "Cora": "Cora & 88.0 & 42.4 & $-45.5$",
            "CiteSeer": "CiteSeer & 77.2 & 47.4 & $-29.9$",
            "PubMed": "PubMed & 87.3 & 69.9 & $-17.3$",
        }
        for dataset, row in rows.items():
            self.assertIn(row, EDGE_INTERVENTION)
            effect = DEGREE_MATCHED["datasets"][dataset]["mean_paired_difference"] * 100
            displayed_effect = float(row.split("$")[1])
            self.assertAlmostEqual(displayed_effect, effect, places=1)

    def test_claim_map_and_abstract_match_the_frozen_summary(self):
        self.assertNotIn("pending direct validation", CLAIM_MAP.lower())
        self.assertIn("exact expression matches empirical moments across a frozen parameter grid", CLAIM_MAP.lower())
        self.assertIn("| supported for the prescribed surrogate grid |", CLAIM_MAP.lower())
        self.assertEqual(SUMMARY["record_counts"]["total"], 2000)
        self.assertEqual(SUMMARY["record_counts"]["primary_relative_error_defined"], 1920)
        self.assertEqual(SUMMARY["record_counts"]["approximation_defined"], 400)
        self.assertEqual(SUMMARY["record_counts"]["error"], 0)
        self.assertAlmostEqual(SUMMARY["primary_estimand"]["median"] * 100, 0.94, places=2)
        self.assertAlmostEqual(SUMMARY["secondary_estimand"]["median_absolute_error"], 0.293, places=3)

    def test_all_active_sources_use_scoped_terms_and_keep_pending_results_pending(self):
        self.assertNotIn("Budget", ACTIVE_TEXT)
        self.assertNotIn("recovery-ratio analysis", ACTIVE_TEXT)
        self.assertNotIn("prospective diagnostic benchmark was executed", ACTIVE_TEXT)
        self.assertNotIn("verified degree-matched results", ACTIVE_TEXT)

    def test_prospective_abstract_numbers_match_frozen_audit(self):
        self.assertEqual(PROSPECTIVE["record_counts"]["datasets"], 11)
        self.assertEqual(PROSPECTIVE["record_counts"]["evaluation_units"], 110)
        self.assertEqual(PROSPECTIVE["record_counts"]["model_records"], 770)
        self.assertEqual(PROSPECTIVE["record_counts"]["diagnostic_records"], 110)
        combined = PROSPECTIVE["methods"]["historical_combined"]
        graph = PROSPECTIVE["methods"]["always_graph"]
        validation = PROSPECTIVE["methods"]["validation_selection"]
        self.assertAlmostEqual(combined["selection_accuracy"] * 100, 55.5, places=1)
        self.assertAlmostEqual(combined["coverage"] * 100, 68.2, places=1)
        self.assertAlmostEqual(combined["full_set_mean_regret"] * 100, 7.46, places=2)
        self.assertAlmostEqual(graph["selection_accuracy"] * 100, 80.9, places=1)
        self.assertAlmostEqual(graph["full_set_mean_regret"] * 100, 0.26, places=2)
        self.assertAlmostEqual(validation["selection_accuracy"] * 100, 91.8, places=1)
        self.assertAlmostEqual(validation["full_set_mean_regret"] * 100, 0.22, places=2)

    def test_degree_matched_abstract_numbers_match_frozen_summary(self):
        self.assertEqual(DEGREE_MATCHED["record_count"], 30)
        expected = {"Cora": -45.5, "CiteSeer": -29.9, "PubMed": -17.3}
        for dataset, rounded_points in expected.items():
            observed = DEGREE_MATCHED["datasets"][dataset]["mean_paired_difference"] * 100
            self.assertAlmostEqual(observed, rounded_points, places=1)
        self.assertEqual(DEGREE_MATCHED["dataset_clustered_overall"]["sign_flip_p_value"], 0.25)


if __name__ == "__main__":
    unittest.main()
