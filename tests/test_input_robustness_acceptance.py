"""Verify the published resource evidence without training or scoring models."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "results/diagnostic/posthoc_input_robustness_11_v1/preflight"


class PublishedInputRobustnessAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/input_robustness_11_v1.json").read_text())
        cls.profile = json.loads((PREFLIGHT / "resource_profile.json").read_text())
        cls.report = json.loads((PREFLIGHT / "acceptance.json").read_text())

    def test_binding_and_execution_provenance_are_preserved(self):
        config_digest = hashlib.sha256(json.dumps(
            self.config, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest()
        self.assertEqual(self.profile["config_sha256"], config_digest)
        self.assertEqual(self.report["config_sha256"], config_digest)
        self.assertEqual(self.report["base_preserved_commit"], self.config["parent_source_commit"])
        self.assertEqual(self.report["measured_execution_commit"], self.profile["source_commit"])
        self.assertEqual(self.report["resource_profile_sha256"], hashlib.sha256(
            (PREFLIGHT / "resource_profile.json").read_bytes()).hexdigest())
        binding_digest = hashlib.sha256((PREFLIGHT / "data_binding.json").read_bytes()).hexdigest()
        self.assertEqual(self.report["data_binding_sha256"], binding_digest)
        self.assertEqual(self.profile["binding_sha256"], binding_digest)

    def test_every_fixed_cell_and_worker_is_present_once(self):
        expected = set(itertools.product(self.config["datasets"], self.config["conditions"],
                                        self.config["models"], [f"trial_{n:03d}" for n in range(4)]))
        rows = self.profile["comparisons"]
        actual = [(r["dataset"], r["condition"], r["model"], r["trial_id"]) for r in rows]
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(set(actual), expected)
        self.assertTrue(all(r["bitwise_repeat_match"] and r["successful_repeats"] == 2 for r in rows))
        expected_jobs = set(itertools.product(self.config["datasets"], self.config["conditions"], range(2)))
        jobs = self.profile["jobs"]
        self.assertEqual(len(jobs), len(expected_jobs))
        self.assertEqual({(j["dataset"], j["condition"], j["repeat"]) for j in jobs}, expected_jobs)
        self.assertTrue(all(j["returncode"] == 0 and j["stop_reason"] is None for j in jobs))
        self.assertEqual(self.report["counts"]["probes"], len(rows) * 2)
        self.assertEqual(self.report["counts"]["training_steps"], len(rows) * 2 * 3)

    def test_projection_recomputation_explains_all_failed_gates(self):
        config, profile = self.config, self.profile
        estimate = 2 * (sum(r["measured_seconds_per_epoch"] for r in profile["comparisons"])
                        * config["training"]["max_epochs"] * len(config["seeds"])
                        + profile["runtime_estimate"]["setup_allowance_before_safety_factor_seconds"])
        self.assertTrue(math.isclose(estimate, profile["runtime_estimate"]["conservative_seconds"], rel_tol=1e-12))
        self.assertTrue(math.isclose(estimate, self.report["time_projection"]["conservative_seconds"], rel_tol=1e-12))
        self.assertGreater(estimate, config["resource_budget"]["formal_total_worker_wall_seconds"])
        failed_units = set()
        for dataset, condition, model in itertools.product(config["datasets"], config["conditions"], config["models"]):
            unit_time = 2 * config["training"]["max_epochs"] * sum(
                r["measured_seconds_per_epoch"] for r in profile["comparisons"]
                if (r["dataset"], r["condition"], r["model"]) == (dataset, condition, model))
            if unit_time > config["resource_budget"]["formal_model_unit_wall_seconds"]:
                failed_units.add((dataset, condition, model))
        self.assertEqual(failed_units, {("Squirrel", c, "H2GCN") for c in config["conditions"]})
        self.assertEqual({(f["dataset"], f["condition"], f["model"]) for f in profile["failures"] if "model" in f}, failed_units)
        self.assertEqual(len(profile["failures"]), len(failed_units) + 1)
        self.assertEqual(self.report["time_projection"]["failed_gates"], profile["failures"])

    def test_memory_is_complete_process_tree_and_within_caps(self):
        budget = self.config["resource_budget"]
        for job in self.profile["jobs"]:
            self.assertEqual(job["memory_scope"], "owned_worker_process_tree_sum_including_windows_high_water")
            self.assertGreaterEqual(len(job["observed_process_ids"]), 2)
            self.assertLessEqual(job["peak_rss_bytes"], budget["max_worker_rss_gib"] * 1024**3)
        for row in self.profile["comparisons"]:
            self.assertLessEqual(row["peak_cuda_reserved_bytes"], budget["max_cuda_reserved_gib"] * 1024**3)
            self.assertLessEqual(row["peak_cuda_allocated_bytes"], budget["max_cuda_allocated_gib"] * 1024**3)
        self.assertEqual(self.report["memory"]["peak_process_tree_bytes"], max(j["peak_rss_bytes"] for j in self.profile["jobs"]))
        self.assertEqual(self.report["memory"]["peak_cuda_reserved_bytes"], max(r["peak_cuda_reserved_bytes"] for r in self.profile["comparisons"]))
        self.assertEqual(self.report["memory"]["peak_cuda_allocated_bytes"], max(r["peak_cuda_allocated_bytes"] for r in self.profile["comparisons"]))

    def test_preflight_cannot_be_reported_as_formal_or_scientific_acceptance(self):
        self.assertFalse(self.config["formal_training_enabled"])
        self.assertFalse(self.profile["formal_launch_authorized"])
        self.assertFalse(self.profile["resource_and_short_repeatability_passed"])
        self.assertFalse(self.report["checks"]["formal_launch_authorized"])
        self.assertEqual(self.report["status"], "blocked_by_formal_time_budget")
        self.assertEqual(self.profile["phase"], "preflight_only")
        self.assertEqual(self.report["analysis_status"], "post_hoc_training_recipe_robustness")
        self.assertEqual(set(self.report["scientific_claim_status"].values()), {"not_yet_evaluated_on_new_formal_records"})
        for key in ("validation_evaluations", "test_evaluations", "formal_records"):
            self.assertEqual(self.profile[key], 0)
            self.assertEqual(self.report["counts"][key], 0)
        forbidden = {"validation_accuracy", "validation_loss", "test_accuracy", "test_loss", "selected_trial_id", "regret", "regret_pp"}
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
        check(self.profile)
        check(self.report)


if __name__ == "__main__":
    unittest.main()
