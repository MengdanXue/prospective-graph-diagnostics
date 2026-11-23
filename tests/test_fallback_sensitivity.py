import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_fallback_sensitivity import ANALYSIS, load_json, main, summarize


class FallbackSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = load_json(ANALYSIS / "diagnostic_audit.json")
        cls.preprocessing = load_json(ANALYSIS / "preprocessing_sensitivity.json")
        cls.result = summarize(cls.audit, cls.preprocessing)

    def test_frozen_decomposition_recovers_original_regret_and_coverage(self):
        row = self.result["frozen_four_way"]["all_11"]["fallbacks"]["mlp"]
        covered, abstained = (row["decomposition"][key] for key in ("covered", "abstained"))
        self.assertEqual((covered["units"], abstained["units"]), (75, 35))
        self.assertAlmostEqual(abstained["share_of_total_regret"], .7587620353792078, places=12)
        self.assertAlmostEqual(covered["subset_mean_regret_pp"], 2.6384704907735186, places=12)
        self.assertAlmostEqual(abstained["subset_mean_regret_pp"], 17.78301511492048, places=12)
        for scope in self.result["frozen_four_way"].values():
            for item in scope["fallbacks"].values():
                groups = item["decomposition"].values()
                self.assertEqual(sum(g["units"] for g in groups), item["units"])
                self.assertAlmostEqual(sum(g["full_set_contribution_pp"] for g in groups), item["mean_regret_pp"], places=12)
                self.assertAlmostEqual(sum(g["sum_regret_pp_units"] for g in groups) / item["units"], item["mean_regret_pp"], places=12)
                self.assertAlmostEqual(sum(g["share_of_total_regret"] for g in groups), 1, places=12)
        self.assertAlmostEqual(row["mean_regret_pp"] / 100, self.audit["methods"]["historical_combined"]["full_set_mean_regret"], places=14)

    def test_four_way_reference_differences_use_fixed_scope(self):
        expected = {("all_11", "mlp"): (7.457189234820278, 7.201383953744715),
                    ("all_11", "graph"): (1.8150842731649224, 1.5592789920893584),
                    ("without_chameleon_squirrel", "mlp"): (4.898154371314579, 4.585503472222222),
                    ("without_chameleon_squirrel", "graph"): (.4537304573588901, .14107955826653376)}
        for (scope, fallback), (regret, excess) in expected.items():
            with self.subTest(scope=scope, fallback=fallback):
                row = self.result["frozen_four_way"][scope]["fallbacks"][fallback]
                self.assertAlmostEqual(row["mean_regret_pp"], regret, places=12)
                self.assertAlmostEqual(row["excess_over_always_graph_pp"], excess, places=12)
        self.assertIn("not a range over all published exclusions", self.result["frozen_four_way_scope"])

    def test_same_run_preprocessing_uses_per_seed_validation_actions(self):
        grid = self.result["preprocessing_two_by_two"]
        normalized, raw = (grid[key]["fallbacks"] for key in ("normalize_features", "raw"))
        self.assertEqual(normalized["mlp"]["pre_fallback_actions"], {"abstain": 20})
        self.assertEqual(raw["mlp"]["pre_fallback_actions"], {"mlp": 20})
        self.assertAlmostEqual(normalized["mlp"]["mean_regret_pp"], 20.876048654317856, places=12)
        self.assertEqual(normalized["graph"]["mean_regret_pp"], 0)
        for fallback in ("mlp", "graph"):
            self.assertAlmostEqual(raw[fallback]["mean_regret_pp"], 13.29863578081131, places=12)
        for dataset in self.preprocessing["datasets"].values():
            for row in dataset["raw"]["units"]:
                self.assertGreaterEqual(row["mlp_validation"], .4)
        self.assertIn("no mixed-run aggregation", self.result["scope"])

    def test_summary_binding_does_not_claim_absent_split_ids_verified(self):
        binding = self.result["preprocessing_binding"]
        self.assertEqual(binding["split_ids_checked"], 0)
        self.assertEqual(binding["split_ids_absent_from_preprocessing_summary"], 40)
        self.assertIn("not proof of split identity", binding["limitation"])
        modified = copy.deepcopy(self.preprocessing)
        row = modified["datasets"]["Amazon-ratings"]["raw"]["units"][0]
        original = next(u for u in self.audit["units"] if (u["dataset"], u["seed"]) == (row["dataset"], row["seed"]))
        row["split_id"] = original["split_id"]
        checked = summarize(self.audit, modified)["preprocessing_binding"]
        self.assertEqual((checked["split_ids_checked"], checked["split_ids_absent_from_preprocessing_summary"]), (1, 39))
        row["split_id"] = "wrong-split"
        with self.assertRaisesRegex(ValueError, "split_id mismatch"):
            summarize(self.audit, modified)

    def test_duplicate_or_missing_frozen_keys_are_rejected(self):
        for mode in ("duplicate", "missing", "missing-field"):
            broken = copy.deepcopy(self.audit)
            if mode == "duplicate":
                broken["units"][-1] = copy.deepcopy(broken["units"][0])
            elif mode == "missing":
                broken["units"].pop()
            else:
                del broken["units"][0]["selected_mlp_validation"]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                summarize(broken, self.preprocessing)

    def test_duplicate_missing_and_unknown_preprocessing_keys_are_rejected(self):
        for mode in ("duplicate", "missing", "unknown-seed", "missing-validation", "missing-condition", "missing-actions"):
            broken = copy.deepcopy(self.preprocessing)
            rows = broken["datasets"]["Roman-empire"]["raw"]["units"]
            if mode == "duplicate":
                rows[-1] = copy.deepcopy(rows[0])
            elif mode == "missing":
                rows.pop()
            elif mode == "unknown-seed":
                rows[0]["seed"] = 10
            elif mode == "missing-validation":
                del rows[0]["mlp_validation"]
            elif mode == "missing-condition":
                del broken["datasets"]["Roman-empire"]["raw"]
            else:
                del rows[0]["actions"]["historical_combined"]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                summarize(self.audit, broken)

    def test_saved_actions_and_original_scores_cannot_be_silently_rewritten(self):
        broken = copy.deepcopy(self.audit)
        broken["units"][0]["decisions"]["historical_combined"]["action"] = "graph"
        with self.assertRaisesRegex(ValueError, "saved decisions differ"):
            summarize(broken, self.preprocessing)
        broken = copy.deepcopy(self.audit)
        broken["methods"]["historical_combined"]["full_set_mean_regret"] += .01
        with self.assertRaisesRegex(ValueError, "saved value does not match"):
            summarize(broken, self.preprocessing)
        for field in ("actions", "regret"):
            broken = copy.deepcopy(self.preprocessing)
            row = broken["datasets"]["Roman-empire"]["raw"]["units"][0]
            row[field]["historical_combined"] = "graph" if field == "actions" else 0
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize(self.audit, broken)
        for group in ("dataset", "condition"):
            broken = copy.deepcopy(self.preprocessing)
            row = (broken["datasets"]["Roman-empire"]["raw"] if group == "dataset"
                   else broken["conditions"]["raw"])
            row["mean_regret_pp"]["historical_combined"] += .01
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, "saved value does not match"):
                summarize(self.audit, broken)
        before = copy.deepcopy((self.audit, self.preprocessing))
        summarize(self.audit, self.preprocessing)
        self.assertEqual((self.audit, self.preprocessing), before)

    def test_cli_defaults_to_stdout_and_explicit_output_cannot_overwrite(self):
        inputs = [ANALYSIS / "diagnostic_audit.json", ANALYSIS / "preprocessing_sensitivity.json"]
        before = [p.read_bytes() for p in inputs]
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            main([])
        self.assertEqual(json.loads(stream.getvalue())["frozen_four_way"], self.result["frozen_four_way"])
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "reanalysis.json"
            main(["--output", str(destination)])
            self.assertEqual(json.loads(destination.read_text())["preprocessing_two_by_two"], self.result["preprocessing_two_by_two"])
            with self.assertRaises(FileExistsError):
                main(["--output", str(destination)])
            duplicate = Path(tmp) / "duplicate.json"
            duplicate.write_text('{"units": [], "units": []}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_json(duplicate)
        for path, original in zip(inputs, before):
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
