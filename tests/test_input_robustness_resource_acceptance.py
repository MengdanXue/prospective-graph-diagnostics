"""Synthetic checks for the approved resource stage; no research models run."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import accept_input_robustness_resources as acceptance
from scripts import preflight_input_robustness_11 as preflight
from tests import test_input_robustness_preflight as fixtures


def approved_config():
    return json.loads(Path(acceptance.DEFAULT_CONFIG).read_text(encoding="utf-8"))


def authorization():
    return json.loads(Path(acceptance.DEFAULT_AUTHORIZATION).read_text(encoding="utf-8"))


def passing_monitor():
    return {"status": "passed", "stop_reasons": [], "worker_count": 44,
            "idle_sleep_request_acquired": True, "idle_sleep_request_released": True}


def passing_ledger():
    return {"integrity_passed": True, "must_stop": False, "stop_reasons": [],
            "usage_seconds": {"total": 5.0, "resource": 5.0, "formal": 0.0, "control": 0.0},
            "remaining_seconds": {"total": 1382395.0, "resource": 7195.0,
                                  "formal": 1368000.0, "control": 7200.0}}


class ApprovedConfigTests(unittest.TestCase):
    def test_approved_budget_preserves_scientific_recipe_and_formal_lock(self):
        config = approved_config()
        acceptance.validate_approved_config(config, authorization())
        original = fixtures.frozen_config()
        for key in ("datasets", "conditions", "seeds", "models", "training", "split",
                    "model_parameters", "diagnostics", "transform",
                    "expected_records", "expected_trials"):
            with self.subTest(field=key):
                self.assertEqual(config[key], original[key])
        self.assertEqual(config["execution"], {**original["execution"],
                         "h2gcn_checkpoint_mode": "original_full_state_copy"})
        self.assertIs(config["formal_training_enabled"], False)
        self.assertEqual(config["resource_budget"]["formal_model_unit_wall_seconds"], 28800)
        self.assertEqual(config["resource_budget"]["formal_total_worker_wall_seconds"], 1368000)
        self.assertEqual(len(preflight.workers(config)), 44)
        self.assertTrue(config["preflight"]["retain_all_four_trial_states"])

    def test_scientific_or_execution_changes_are_not_budget_authorization(self):
        original = approved_config()
        for path, value in [
            (("datasets",), original["datasets"][:-1]),
            (("seeds",), list(range(1, 11))),
            (("conditions",), ["normalize_features", "raw"]),
            (("formal_training_enabled",), True),
            (("training", "max_epochs"), 499),
            (("training", "trials"), original["training"]["trials"][:3]),
            (("execution", "torch_num_threads"), 8),
            (("execution", "deterministic_warn_only"), True),
            (("execution", "model_devices", "H2GCN"), "cuda"),
            (("execution", "h2gcn_checkpoint_mode"), "static_adjacency_once"),
            (("preflight", "retain_all_four_trial_states"), False),
            (("preflight", "repeats"), 1),
            (("preflight", "training_steps"), 10),
        ]:
            candidate = copy.deepcopy(original)
            node = candidate
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            with self.subTest(field=".".join(path)), self.assertRaises(ValueError):
                acceptance.validate_approved_config(candidate, authorization())

    def test_authorization_must_open_only_the_resource_stage(self):
        for key, value in (("budget_change_authorized", False),
                           ("resource_acceptance_authorized", False),
                           ("formal_launch_authorized", True),
                           ("formal_training_enabled", True)):
            candidate = authorization()
            candidate[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                acceptance.validate_approved_config(approved_config(), candidate)

    def test_formal_reserves_cannot_be_spent_by_changing_legacy_budget_fields(self):
        for key, value in (("formal_total_worker_wall_seconds", 1382400),
                           ("formal_model_unit_wall_seconds", 57600),
                           ("preflight_wall_seconds", 7201)):
            candidate = approved_config()
            candidate["resource_budget"][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                acceptance.validate_approved_config(candidate, authorization())


class SourceCIGateTests(unittest.TestCase):
    def setUp(self):
        self.commit = "a" * 40
        self.receipt = {"commit": self.commit, "run_id": 41, "status": "completed",
                        "conclusion": "success",
                        "url": "https://github.com/MengdanXue/prospective-graph-diagnostics/actions/runs/41",
                        "jobs": [{"name": name, "status": "completed", "conclusion": "success", "run_id": 41}
                                 for name in ("full-protocol-verification", "lightweight-verification", "manuscript-build")]}

    def test_only_the_same_completed_successful_commit_is_admissible(self):
        acceptance.validate_ci_receipt(self.receipt, self.commit)
        for changes in ({"commit": "b" * 40}, {"status": "in_progress"},
                        {"conclusion": "failure"}, {"run_id": 0}, {"run_id": True}):
            candidate = {**self.receipt, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                acceptance.validate_ci_receipt(candidate, self.commit)

    def test_missing_duplicate_skipped_or_other_run_jobs_reject(self):
        candidates = []
        missing = copy.deepcopy(self.receipt)
        missing["jobs"].pop()
        candidates.append(missing)
        duplicate = copy.deepcopy(self.receipt)
        duplicate["jobs"][-1] = copy.deepcopy(duplicate["jobs"][0])
        candidates.append(duplicate)
        for changes in ({"status": "queued"}, {"conclusion": "skipped"},
                        {"conclusion": "failure"}, {"run_id": 42}):
            candidate = copy.deepcopy(self.receipt)
            candidate["jobs"][0].update(changes)
            candidates.append(candidate)
        for candidate in candidates:
            with self.subTest(jobs=candidate["jobs"]), self.assertRaises(ValueError):
                acceptance.validate_ci_receipt(candidate, self.commit)


class ResourceSummaryTests(unittest.TestCase):
    def setUp(self):
        self.config = approved_config()
        fixture = fixtures.SummaryAcceptanceTests()
        with patch.object(fixtures, "frozen_config", return_value=self.config):
            fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.directory, self.jobs = fixture.directory, fixture.jobs
        self.expected = preflight.workers(self.config)
        self.monitor, self.ledger = passing_monitor(), passing_ledger()

    def summary(self):
        return acceptance.summarize_acceptance(
            self.config, self.directory, self.jobs, self.monitor, self.ledger)

    def assert_rejected(self):
        result = self.summary()
        self.assertFalse(result["resource_and_short_repeatability_passed"])
        self.assertFalse(result["formal_launch_authorized"])
        self.assertTrue(result["failures"])
        return result

    def path(self, worker_index=0, model="MLP", trial=0):
        return (self.directory / "workers" / self.expected[worker_index]["worker_id"] /
                "probes" / model / f"trial_{trial:03d}.json")

    def mutate(self, path, **changes):
        row = json.loads(path.read_text(encoding="utf-8"))
        row.update(changes)
        fixtures.write_fixture(path, row)

    def test_complete_fresh_evidence_passes_but_never_authorizes_formal_training(self):
        result = self.summary()
        self.assertTrue(result["resource_and_short_repeatability_passed"], result["failures"])
        self.assertEqual(result["observed_workers"], 44)
        self.assertEqual(result["bitwise_matching_paired_cells"], 616)
        self.assertEqual(len(list((self.directory / "workers").glob("*/probes/*/*.json"))), 1232)
        for key in ("validation_evaluations", "test_evaluations", "formal_records"):
            self.assertEqual(result[key], 0)
        self.assertIs(result["formal_launch_authorized"], False)

    def test_a_failed_worker_vetoes_complete_matching_probe_files(self):
        self.jobs[0]["returncode"] = 2
        self.jobs[0]["stop_reason"] = "first failed probe"
        self.assert_rejected()

    def test_missing_and_duplicate_worker_identity_cannot_supply_full_scope(self):
        self.jobs[-1] = copy.deepcopy(self.jobs[0])
        self.assert_rejected()

    def test_missing_probe_or_completion_cannot_be_accepted(self):
        self.path().unlink()
        (self.directory / "workers" / self.expected[0]["worker_id"] / "worker_complete.json").unlink()
        self.assert_rejected()

    def test_matching_but_wrong_probe_identity_is_rejected(self):
        for index in (0, 1):
            self.mutate(self.path(index), seed=9, config_sha256="0" * 64)
        self.assert_rejected()

    def test_heldout_scores_or_selected_models_never_enter_acceptance(self):
        for index in (0, 1):
            self.mutate(self.path(index), test_accuracy=0.9, selected_trial_id="trial_000")
        self.assert_rejected()

    def test_monitor_failure_after_complete_workers_cannot_leave_a_pass(self):
        self.monitor.update(status="stopped", stop_reasons=["monitor gap during inventory"])
        self.assert_rejected()

    def test_idle_sleep_request_must_be_acquired_and_released(self):
        for key in ("idle_sleep_request_acquired", "idle_sleep_request_released"):
            original = self.monitor[key]
            self.monitor[key] = False
            with self.subTest(field=key):
                self.assert_rejected()
            self.monitor[key] = original

    def test_monitor_worker_coverage_and_stop_reasons_are_independent_vetoes(self):
        self.monitor["worker_count"] = 43
        self.assert_rejected()
        self.monitor["worker_count"] = 44
        self.monitor["stop_reasons"] = ["suspend detected"]
        self.assert_rejected()

    def test_ledger_integrity_or_stop_state_vetoes_complete_evidence(self):
        for changes in ({"integrity_passed": False}, {"must_stop": True},
                        {"stop_reasons": ["resource budget exceeded"]}):
            self.ledger = {**passing_ledger(), **changes}
            with self.subTest(changes=changes):
                self.assert_rejected()

    def test_resource_and_total_overshoot_is_preserved_and_rejected(self):
        self.ledger["usage_seconds"].update(resource=7201.0, total=7201.0)
        self.ledger["remaining_seconds"].update(resource=-1.0, total=1375199.0)
        self.assert_rejected()

    def test_formal_or_control_work_cannot_be_hidden_in_resource_stage(self):
        self.ledger["usage_seconds"]["formal"] = 1.0
        self.assert_rejected()

    def test_full_unit_projection_includes_setup_and_equals_original_global_formula(self):
        # Training contributes 2 seconds/unit; per-worker nontraining residual
        # is 0.1 - (28 probes * 3 epochs * 0.001) = 0.016 seconds.
        rows = acceptance.derive_unit_projections(self.config, self.directory, self.jobs)
        self.assertEqual(len(rows), 154)
        identities = {(row["dataset"], row["condition"], row["model"]) for row in rows}
        self.assertEqual(len(identities), 154)
        expected = 2 * (4 * 500 * 0.001 + 0.01 + (0.1 - 28 * 3 * 0.001))
        for row in rows:
            self.assertAlmostEqual(row["four_trial_unit_seconds"], expected, places=12)
        projected = sum(row["four_trial_unit_seconds"] for row in rows) * 10
        original = preflight.summarize(self.config, self.directory, self.jobs)
        self.assertTrue(math.isclose(projected, original["runtime_estimate"]["conservative_seconds"], rel_tol=1e-12))

    def test_invalid_raw_measurement_cannot_be_masked_by_other_repeat_or_zero_floor(self):
        original_job = copy.deepcopy(self.jobs[0])
        for key in ("wall_seconds", "peak_rss_bytes"):
            for value in (-1.0, float("nan")):
                self.jobs[0] = {**original_job, key: value}
                with self.subTest(kind="job", field=key, value=value), self.assertRaises(ValueError):
                    acceptance.derive_unit_projections(self.config, self.directory, self.jobs)
        self.jobs[0] = original_job
        header = self.directory / "workers" / self.expected[0]["worker_id"] / "worker.json"
        original_header = json.loads(header.read_text())
        for value in (-1.0, float("nan")):
            fixtures.write_fixture(header, {**original_header, "transform_seconds": value})
            with self.subTest(kind="header", value=value), self.assertRaises(ValueError):
                acceptance.derive_unit_projections(self.config, self.directory, self.jobs)
        fixtures.write_fixture(header, original_header)
        path = self.path()
        original_probe = json.loads(path.read_text())
        for key in ("h2_preparation_seconds", "peak_cuda_reserved_bytes", "peak_cuda_allocated_bytes",
                    "rss_bytes", "four_checkpoints_bytes"):
            for value in (-1.0, float("nan")):
                fixtures.write_fixture(path, {**original_probe, key: value})
                with self.subTest(kind="probe", field=key, value=value), self.assertRaises(ValueError):
                    acceptance.derive_unit_projections(self.config, self.directory, self.jobs)

    def test_setup_can_reject_unit_whose_training_alone_is_below_eight_hours(self):
        for index in (0, 1):
            folder = self.directory / "workers" / self.expected[index]["worker_id"]
            self.mutate(folder / "worker.json", transform_seconds=15.0)
            self.jobs[index]["wall_seconds"] = 90.0
            for trial in range(4):
                self.mutate(self.path(index, trial=trial), epoch_seconds=[7.195] * 3)
        rows = acceptance.derive_unit_projections(self.config, self.directory, self.jobs)
        affected = [row for row in rows if row["dataset"] == self.config["datasets"][0]
                    and row["condition"] == self.config["conditions"][0] and row["model"] == "MLP"]
        self.assertEqual(len(affected), 1)
        self.assertLess(2 * 500 * 4 * 7.195, 28800)
        self.assertGreater(affected[0]["four_trial_unit_seconds"], 28800)
        self.assert_rejected()

    def test_global_projection_uses_380_hour_allocation_not_384_hour_total(self):
        # Choose equal, finite step times so every unit fits eight hours while
        # the original fixed formula produces 382 hours for all formal units.
        target = 382 * 3600
        per_epoch = (target / 2 - 22 * 0.01 * 70) / (616 * 500 * 10)
        for path in (self.directory / "workers").glob("*/probes/*/*.json"):
            self.mutate(path, epoch_seconds=[per_epoch] * 3)
        rows = acceptance.derive_unit_projections(self.config, self.directory, self.jobs)
        projection = sum(row["four_trial_unit_seconds"] for row in rows) * 10
        self.assertAlmostEqual(projection, target, places=6)
        self.assertTrue(all(row["four_trial_unit_seconds"] < 28800 for row in rows))
        self.assertGreater(projection, 1368000)
        self.assertLess(projection, 1382400)
        self.assert_rejected()


class FailedProbeStopTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_root = Path(temporary.name)
        self.config = approved_config()
        self.worker = preflight.workers(self.config)[0]
        self.directory = self.run_root / "workers" / self.worker["worker_id"]
        self.directory.mkdir(parents=True)

    def invoke(self):
        return acceptance.run_probe_worker(self.config, self.worker, self.run_root, {}, self.directory)

    def request_pause(self):
        fixtures.write_fixture(self.run_root / "control" / "pause_request.json", {"request": "normal_pause"})

    def successful_trial(self, model, trial):
        row = {"status": "success", "model": model, "trial_id": f"trial_{trial:03d}",
               "phase": "preflight_only", "validation_evaluations": 0, "test_evaluations": 0}
        preflight.train_probe(trial=self.config["training"]["trials"][trial])
        preflight.write_exclusive(self.directory / "probes" / model / f"trial_{trial:03d}.json", row)

    def test_failed_probe_is_saved_before_exit_and_no_following_probe_runs(self):
        writer = preflight.write_exclusive
        reached = []

        def fake_worker(*args, **kwargs):
            preflight.write_exclusive(self.directory / "failed.json", {"status": "failed", "exception": "synthetic failure"})
            reached.append("unrequested next probe")
            preflight.write_exclusive(self.directory / "next.json", {"status": "success"})

        with patch.object(preflight, "run_worker", side_effect=fake_worker):
            with self.assertRaises(RuntimeError):
                self.invoke()
        self.assertEqual(json.loads((self.directory / "failed.json").read_text())["status"], "failed")
        self.assertEqual(reached, [])
        self.assertFalse((self.directory / "next.json").exists())
        self.assertIs(preflight.write_exclusive, writer)

    def test_successful_worker_retains_original_writer_bytes(self):
        writer = preflight.write_exclusive
        row = {"status": "success", "training_steps": 3, "trajectory": [{"step": 0}]}
        reference = self.directory / "reference.json"
        writer(reference, row)

        def fake_worker(*args, **kwargs):
            preflight.write_exclusive(self.directory / "success.json", row)

        with patch.object(preflight, "run_worker", side_effect=fake_worker):
            self.invoke()
        self.assertEqual((self.directory / "success.json").read_bytes(), reference.read_bytes())
        self.assertIs(preflight.write_exclusive, writer)

    def test_unexpected_worker_exception_restores_writer(self):
        writer = preflight.write_exclusive
        trainer = preflight.train_probe
        with patch.object(preflight, "run_worker", side_effect=OSError("synthetic write failure")):
            with self.assertRaises(OSError):
                self.invoke()
        self.assertIs(preflight.write_exclusive, writer)
        self.assertIs(preflight.train_probe, trainer)

    def test_normal_pause_finishes_four_trials_and_saves_unit_before_exiting(self):
        writer, trainer = preflight.write_exclusive, preflight.train_probe
        reached = []
        model = self.config["models"][0]

        def fake_worker(*args, **kwargs):
            for trial in range(4):
                self.successful_trial(model, trial)
                if trial == 0:
                    self.request_pause()
            reached.append("next model")

        with patch.object(preflight, "train_probe", return_value={}) as probe:
            with patch.object(preflight, "run_worker", side_effect=fake_worker):
                with self.assertRaises(acceptance.GracefulPause):
                    self.invoke()
            self.assertEqual(probe.call_count, 4)
            self.assertIs(preflight.train_probe, probe)
        self.assertEqual(reached, [])
        self.assertEqual(len(list((self.directory / "probes" / model).glob("trial_*.json"))), 4)
        marker = json.loads((self.directory / "unit_completions" / f"{model}.json").read_text())
        self.assertEqual(marker["trials"], 4)
        self.assertEqual(marker["probe_sha256"], [preflight.file_hash(
            self.directory / "probes" / model / f"trial_{trial:03d}.json") for trial in range(4)])
        self.assertIs(preflight.write_exclusive, writer)
        self.assertIs(preflight.train_probe, trainer)
        self.assertFalse(issubclass(acceptance.GracefulPause, Exception),
                         "ordinary pause must bypass the unchanged worker's numeric-failure handler")

    def test_pause_arriving_between_models_cannot_start_the_next_training_probe(self):
        first, second = self.config["models"][:2]
        reached = []

        def fake_worker(*args, **kwargs):
            for trial in range(4):
                self.successful_trial(first, trial)
            # This arrives after the previous writer checked for a pause.
            self.request_pause()
            self.successful_trial(second, 0)
            reached.append("unrequested next training")

        with patch.object(preflight, "train_probe", return_value={}) as probe:
            with patch.object(preflight, "run_worker", side_effect=fake_worker):
                with self.assertRaises(acceptance.GracefulPause):
                    self.invoke()
            self.assertEqual(probe.call_count, 4)
            self.assertIs(preflight.train_probe, probe)
        self.assertEqual(reached, [])
        self.assertTrue((self.directory / "unit_completions" / f"{first}.json").exists())
        self.assertFalse((self.directory / "probes" / second / "trial_000.json").exists())

    def test_existing_normal_pause_does_not_start_any_worker_probe(self):
        self.request_pause()
        writer, trainer = preflight.write_exclusive, preflight.train_probe
        with patch.object(preflight, "run_worker") as worker:
            with self.assertRaises(acceptance.GracefulPause):
                self.invoke()
        worker.assert_not_called()
        self.assertIs(preflight.write_exclusive, writer)
        self.assertIs(preflight.train_probe, trainer)


if __name__ == "__main__":
    unittest.main()
