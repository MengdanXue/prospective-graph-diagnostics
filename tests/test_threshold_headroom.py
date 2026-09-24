"""Boundary, scoring, and provenance checks for retained-record reanalysis."""

import copy
import json
import math
import re
import unittest

from experiments.evaluate_diagnostics import fixed_decisions
from scripts.summarize_fallback_sensitivity import ANALYSIS, COMBINED, load_json
from scripts.summarize_threshold_headroom import (
    ROOT, confusion_summary, dataset_mean, exact_threshold_curve, read_verified_inputs,
    summarize_portfolio, validation_fallback_summary,
)
from tests import test_equal_budget_sensitivity as integrity_fixture


def unit(dataset, h, graph, mlp, mlp_validation=.6, graph_validation=.7):
    row = {"dataset": dataset, "homophily": h, "mean_degree": 4., "delta_h": .1,
           "selected_graph_test": graph, "selected_mlp_test": mlp,
           "selected_graph_validation": graph_validation, "selected_mlp_validation": mlp_validation,
           "test_gap": graph - mlp, "target_action": "graph" if graph - mlp > .01 else "mlp"}
    row["decisions"] = fixed_decisions(row)
    return row


class ExactThresholdTests(unittest.TestCase):
    def test_closed_upper_endpoints_include_zero_and_one_correctly(self):
        rows = [unit("A", h, .9, .1) for h in (0., .5, 1.)]
        curve = exact_threshold_curve(rows)
        self.assertEqual([(x["lower"], x["upper"], x["lower_inclusive"], x["upper_inclusive"])
                          for x in curve], [(0., 0., True, True), (0., .5, False, True), (.5, 1., False, True)])
        self.assertEqual([x["graph_actions"] for x in curve], [3, 2, 1])
        self.assertEqual(curve[0]["mean_regret_pp"], 0.)
        # h=1 stays graph throughout the permitted domain: all-MLP is unavailable.
        self.assertAlmostEqual(curve[-1]["mean_regret_pp"], 160 / 3)

    def test_exact_search_finds_a_narrow_optimum_missed_by_point_zero_one_grid(self):
        rows = [unit("A", .22156150203722058, .2, .8),
                unit("A", .22196910468282424, .9, .1)]
        curve = exact_threshold_curve(rows)
        minimum = min(row["mean_regret_pp"] for row in curve)
        optimal = [row for row in curve if row["mean_regret_pp"] == minimum]
        self.assertEqual(minimum, 0.)
        self.assertEqual(len(optimal), 1)
        self.assertEqual((optimal[0]["lower"], optimal[0]["upper"], optimal[0]["lower_inclusive"]),
                         (.22156150203722058, .22196910468282424, False))
        grid_losses = [100 * sum(max(row["selected_graph_test"], row["selected_mlp_test"])
                               - row["selected_graph_test" if row["homophily"] >= i / 100 else "selected_mlp_test"]
                               for row in rows) / len(rows) for i in range(101)]
        self.assertGreater(min(grid_losses), minimum)

    def test_duplicate_h_values_form_one_boundary_and_every_sample_matches_direct_actions(self):
        rows = [unit("A", .2, .1, .7), unit("A", .2, .9, .2), unit("B", .8, .4, .6)]
        curve = exact_threshold_curve(rows)
        self.assertEqual(len(curve), 3)
        samples = [0., 1., .2, .8, math.nextafter(.2, 1), math.nextafter(.8, 1)] + [i / 1000 for i in range(1001)]
        for t in samples:
            intervals = [r for r in curve if (t >= r["lower"] if r["lower_inclusive"] else t > r["lower"])
                         and t <= r["upper"]]
            self.assertEqual(len(intervals), 1)
            expected = dataset_mean((r["dataset"], max(r["selected_graph_test"], r["selected_mlp_test"])
                                     - r["selected_graph_test" if r["homophily"] >= t else "selected_mlp_test"])
                                    for r in rows)
            self.assertAlmostEqual(intervals[0]["mean_regret_pp"], 100 * expected, places=12)

    def test_dataset_weights_do_not_become_seed_weights(self):
        a, b = unit("A", .5, .1, .9), unit("B", .5, .9, .1)
        original = exact_threshold_curve([a, b])
        duplicated_seed_weight = exact_threshold_curve([a] * 30 + [b])
        self.assertEqual([r["mean_regret_pp"] for r in original],
                         [r["mean_regret_pp"] for r in duplicated_seed_weight])

    def test_zero_headroom_is_undefined_fraction_and_negative_closure_is_preserved(self):
        result = summarize_portfolio([unit("A", .2, .9, .1)], ["GCN"])
        self.assertEqual(result["headroom_pp"], 0.)
        self.assertTrue(all(value is None for value in result["headroom_closed_fraction"].values()))
        other = summarize_portfolio([unit("A", .2, .9, .1), unit("B", .8, .4, .5)], ["GCN"])
        self.assertLess(other["headroom_closed_fraction"]["combined"], 0)

    def test_bad_homophily_cannot_enter_threshold_search(self):
        for h in (None, float("nan"), float("inf"), -.001, 1.001, True):
            broken = unit("A", .2, .6, .4)
            broken["homophily"] = h
            with self.subTest(h=h), self.assertRaises(ValueError):
                exact_threshold_curve([broken])


class ScoringSemanticsTests(unittest.TestCase):
    def test_practical_target_and_positive_raw_regret_are_distinct(self):
        rows = [unit("A", .2, .5078125, .5), unit("A", .2, .515625, .5),
                unit("B", .2, .5078125, .5, mlp_validation=.3),
                unit("B", .8, .9, .8)]
        before = copy.deepcopy(rows)
        result = confusion_summary(rows)["groups"]
        self.assertEqual((result["mlp"]["units"], result["mlp"]["practical_margin_target_graph"],
                          result["mlp"]["positive_raw_regret_units"]), (2, 1, 2))
        self.assertEqual((result["abstain"]["practical_margin_target_mlp"],
                          result["abstain"]["positive_raw_regret_units"]), (1, 1))
        self.assertEqual(result["graph"]["positive_raw_regret_units"], 0)
        self.assertEqual(rows, before)

    def test_validation_fallback_changes_only_abstentions_and_keeps_original_margin(self):
        # Explicit MLP remains MLP although validation favours graph; explicit
        # graph remains graph although validation favours MLP. The .0078125
        # validation advantage of graph at an abstention is below the 1pp rule.
        rows = [unit("A", .2, .8, .2, .6, .9), unit("A", .8, .8, .2, .9, .1),
                unit("A", .2, .8, .2, .25, .2578125), unit("A", .2, .8, .2, .25, .5)]
        result = validation_fallback_summary(rows)
        self.assertAlmostEqual(result["mean_regret_pp"], 30.)
        self.assertEqual(result["abstentions_assigned_by_validation"], 2)
        self.assertEqual(result["coverage_before_fallback"], .5)

    def test_saved_reanalysis_contains_all_portfolios_and_independent_regression_values(self):
        result = load_json(ANALYSIS / "threshold_headroom.json")
        expected = {
            "full": (.2558052810755643, 7.457189234820278, .2188813144510443, .2558052810755643),
            "GCN": (5.29237067157572, 1.744283938949758, .005068453875455, .8115341311151331),
            "GAT": (6.345880857922815, 2.083435194058852, .006887912750244, .6777744672515176),
            "GPR-GNN": (.8892461386593905, 1.496835093606602, .3078804368322546, .4519137875600295),
            "GraphSAGE": (.376353765075857, 2.848237807100469, .1202497969974171, .376353765075857),
            "H2GCN": (.1335001262751493, 2.358126748691906, .2285275676033714, .1335001262751493),
            "LINKX": (2.132917005907406, 7.991380136121403, .315311767838218, 2.132917005907406),
        }
        self.assertEqual(set(result["portfolios"]), set(expected))
        for name, values in expected.items():
            row = result["portfolios"][name]
            for policy, value in zip(("always_graph", "combined", "validation_selection", "test_oracle_h1_cutoff"), values):
                self.assertAlmostEqual(row["policy_regret_pp"][policy], value, places=11)
            self.assertEqual(row["units"], 110)
            self.assertEqual(row["datasets"], 11)
            self.assertAlmostEqual(min(item["mean_regret_pp"] for item in row["exact_threshold_curve"]), values[-1], places=11)
        groups = result["full_portfolio_confusion"]["groups"]
        for name, counts in (("graph", (40, 40, 0)), ("mlp", (35, 19, 23)), ("abstain", (35, 30, 32))):
            self.assertEqual(tuple(groups[name][key] for key in ("units", "practical_margin_target_graph", "positive_raw_regret_units")), counts)
        self.assertAlmostEqual(sum(row["full_set_regret_contribution_pp"] for row in groups.values()),
                               expected["full"][1], places=12)
        sensitivities = result["existing_sensitivities"]
        self.assertAlmostEqual(sensitivities["validation_fallback"]["all_11"]["mean_regret_pp"], 1.8055274540727788, places=12)
        self.assertAlmostEqual(sensitivities["validation_fallback"]["without_chameleon_squirrel"]["mean_regret_pp"], .44204990069071454, places=12)
        self.assertEqual(result["source_verification"]["verified_model_records"], 770)
        self.assertEqual(result["source_verification"]["verified_diagnostic_records"], 110)

    def test_manuscript_tables_are_bound_to_computed_values(self):
        result = load_json(ANALYSIS / "threshold_headroom.json")
        source = (ROOT / "sections_tmlr/09a_threshold_headroom.tex").read_text(encoding="utf-8")
        rows = re.findall(r"^(Six architectures|GAT|GCN|GPR-GNN|GraphSAGE|H2GCN|LINKX) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\", source, re.M)
        self.assertEqual(len(rows), 7)
        for name, headroom, minimum, fraction in rows:
            entry = result["portfolios"]["full" if name == "Six architectures" else name]
            self.assertEqual(headroom, f"{entry['headroom_pp']:.3f}")
            self.assertEqual(minimum, f"{entry['policy_regret_pp']['test_oracle_h1_cutoff']:.3f}")
            self.assertEqual(fraction, f"{100 * entry['headroom_closed_fraction']['test_oracle_h1_cutoff']:.2f}")
        groups = result["full_portfolio_confusion"]["groups"]
        for label, key in (("Graph", "graph"), ("MLP", "mlp"), (r"Abstain $\to$ MLP", "abstain")):
            line = next(line for line in source.splitlines() if line.startswith(label + " &"))
            fields = [field.strip().rstrip("\\").strip() for field in line.split("&")[1:]]
            expected = groups[key]
            self.assertEqual(tuple(map(int, fields[:4])), tuple(expected[field] for field in
                             ("units", "practical_margin_target_graph", "practical_margin_target_mlp", "positive_raw_regret_units")))
            self.assertEqual(fields[4], f"{expected['full_set_regret_contribution_pp']:.3f}")


class RetainedInputIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = integrity_fixture.EqualTrialSensitivityTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.audit_path = self.fixture.base / "audit.json"
        self.config_path = self.fixture.base / "config.json"
        self.audit_path.write_text(json.dumps(self.fixture.audit), encoding="utf-8")
        self.config_path.write_text(json.dumps(self.fixture.config), encoding="utf-8")

    def read(self):
        return read_verified_inputs(self.fixture.records, self.audit_path, self.config_path)

    def test_valid_inputs_are_bound_to_hashes_and_scope_without_mutation(self):
        files = list(self.fixture.base.rglob("*.json"))
        before = {path: path.read_bytes() for path in files}
        _, records, provenance = self.read()
        self.assertEqual(len(records), 1)
        self.assertEqual(provenance["verified_model_records"], 7)
        self.assertEqual(provenance["verified_diagnostic_records"], 1)
        self.assertEqual(set(provenance["input_sources"]), {"audit", "config", "manifest"})
        for source in provenance["input_sources"].values():
            self.assertEqual(len(source["sha256"]), 64)
            self.assertGreater(source["bytes"], 0)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_missing_duplicate_and_tampered_model_records_are_rejected(self):
        path = self.fixture.records / "toy/GCN/seed_000.json"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.read()
        path.write_bytes(original)
        extra = path.with_name("seed_000_copy.json")
        extra.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.read()
        extra.unlink()
        self.fixture.mutate(path, "test_accuracy", .999)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.read()

    def test_duplicate_json_keys_cannot_be_hidden_by_checksums(self):
        path = self.fixture.records / "toy/GCN/seed_000.json"
        content = path.read_text()
        path.write_text('{"test_accuracy": 0.99,' + content[1:], encoding="utf-8")
        self.fixture.write_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self.read()

    def test_modified_audit_is_not_accepted_as_an_independent_source(self):
        audit = load_json(self.audit_path)
        audit["units"][0]["selected_graph_test"] = .999
        self.audit_path.write_text(json.dumps(audit), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "audit value mismatch"):
            self.read()


if __name__ == "__main__":
    unittest.main()
