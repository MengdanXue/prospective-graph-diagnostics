import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_preprocessing_records import validate_preprocessing_run


class PreprocessingRecordValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "run"
        self.config = {
            "schema_version": "1.0",
            "run_id": "toy_preprocessing",
            "analysis_status": "post_hoc_paired_sensitivity",
            "datasets": {"Toy": {"filename": "toy.npz", "sha256": "a" * 64, "shape": [4, 2]}},
            "conditions": ["raw"],
            "seeds": [0],
            "models": ["MLP", "GCN"],
            "training": {
                "hidden_channels": 4,
                "max_epochs": 2,
                "patience": 1,
                "trials": [{"learning_rate": 0.01, "dropout": 0.5}, {"learning_rate": 0.01, "dropout": 0.7}],
                "test_evaluations_after_selection": 1,
            },
        }
        self.audit = {"units": [{"dataset": "Toy", "seed": 0, "split_id": "toy-split-0"}]}
        environment = {"device": "cpu", "python": "3.12"}
        manifest = {
            "run_id": self.config["run_id"],
            "config_sha256": self._digest(self.config),
            "source_commit": "a" * 40,
            "environment": environment,
            "config": self.config,
            "analysis_status": self.config["analysis_status"],
            "preflight": False,
        }
        self.root.mkdir(parents=True)
        self._write(self.root / "run_manifest.json", manifest)
        self._write(self.root / "complete.json", {"status": "complete", "expected_records": 2, "config_sha256": manifest["config_sha256"]})
        for model in self.config["models"]:
            self._write(self._record_path(model), self._record(model, environment))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _digest(value):
        import hashlib

        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

    def _record_path(self, model):
        return self.root / "records" / "raw" / "Toy" / model / "seed_000.json"

    def _record(self, model, environment):
        return {
            "schema_version": "1.0",
            "run_id": self.config["run_id"],
            "status": "success",
            "dataset": "Toy",
            "model": model,
            "seed": 0,
            "preprocessing": "raw",
            "split_id": "toy-split-0",
            "source_commit": "a" * 40,
            "environment": environment,
            "config_sha256": self._digest(self.config),
            "frozen_config": copy.deepcopy(self.config),
            "training_configuration": copy.deepcopy(self.config["training"]),
            "data_provenance": {
                "raw_filename": "toy.npz",
                "raw_sha256": "a" * 64,
                "preprocessing": "raw",
                "transformed_feature_sha256": "b" * 64,
            },
            "trials": [
                {"trial_id": "trial_000", "configuration": self.config["training"]["trials"][0], "validation_accuracy": 0.7, "validation_loss": 0.3, "best_epoch": 1, "epochs_completed": 2, "duration_seconds": 1.0, "history": [{"epoch": 0, "train_loss": 0.5, "validation_accuracy": 0.6, "validation_loss": 0.4}, {"epoch": 1, "train_loss": 0.4, "validation_accuracy": 0.7, "validation_loss": 0.3}]},
                {"trial_id": "trial_001", "configuration": self.config["training"]["trials"][1], "validation_accuracy": 0.6, "validation_loss": 0.4, "best_epoch": 1, "epochs_completed": 2, "duration_seconds": 1.0, "history": [{"epoch": 0, "train_loss": 0.6, "validation_accuracy": 0.5, "validation_loss": 0.5}, {"epoch": 1, "train_loss": 0.5, "validation_accuracy": 0.6, "validation_loss": 0.4}]},
            ],
            "selected_trial_id": "trial_000",
            "validation_accuracy": 0.7,
            "validation_loss": 0.3,
            "test_accuracy": 0.5,
            "test_evaluations_after_selection": 1,
            "duration_seconds": 1.0,
        }

    def validate(self):
        return validate_preprocessing_run(self.root, config=self.config, audit=self.audit)

    def reset_fixture(self):
        self.temp.cleanup()
        self.setUp()

    def mutate_record(self, model="MLP", **changes):
        path = self._record_path(model)
        row = json.loads(path.read_text(encoding="utf-8"))
        for key, value in changes.items():
            if isinstance(value, tuple):
                target, nested_key, nested_value = value
                row[target][nested_key] = nested_value
            else:
                row[key] = value
        self._write(path, row)

    def test_valid_fixture_returns_mapping_and_compact_metadata(self):
        result = self.validate()
        self.assertEqual(result["metadata"]["record_count"], 2)
        self.assertEqual(set(result["records"]), {("raw", "Toy", 0)})
        self.assertEqual(set(result["records"][("raw", "Toy", 0)]), {"MLP", "GCN"})
        self.assertFalse(result["metadata"]["raw_npz_hashes_checked"])

    def test_extra_semantic_json_record_is_rejected(self):
        duplicate = self._record_path("MLP").with_name("seed_000_copy.json")
        self._write(duplicate, json.loads(self._record_path("MLP").read_text(encoding="utf-8")))
        with self.assertRaisesRegex(ValueError, "extra"):
            self.validate()

    def test_missing_record_is_rejected(self):
        self._record_path("GCN").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.validate()

    def test_provenance_and_split_mutations_are_rejected(self):
        for changes, message in (
            ({"source_commit": "other"}, "source_commit"),
            ({"environment": {"device": "other"}}, "environment"),
            ({"split_id": "other"}, "split_id"),
            ({"data_provenance": ("data_provenance", "raw_sha256", "c" * 64)}, "raw_sha256"),
        ):
            with self.subTest(message=message):
                self.reset_fixture()
                self.mutate_record(**changes)
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()

    def test_transformed_digest_must_be_uniform_per_dataset_and_condition(self):
        self.mutate_record(model="GCN", data_provenance=("data_provenance", "transformed_feature_sha256", "c" * 64))
        with self.assertRaisesRegex(ValueError, "inconsistent transformed"):
            self.validate()

    def test_nonfinite_and_out_of_range_metrics_are_rejected(self):
        self.mutate_record(test_accuracy=float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.validate()
        self.reset_fixture()
        self.mutate_record(test_accuracy=1.1)
        with self.assertRaisesRegex(ValueError, r"outside \[0, 1\]"):
            self.validate()

    def test_selected_trial_fidelity_and_test_count_are_rejected(self):
        self.mutate_record(selected_trial_id="trial_001")
        with self.assertRaisesRegex(ValueError, "selected trial"):
            self.validate()
        self.reset_fixture()
        self.mutate_record(test_evaluations_after_selection=2)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.validate()

    def test_manifest_or_frozen_config_digest_mismatch_is_rejected(self):
        manifest_path = self.root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["config_sha256"] = "d" * 64
        self._write(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "digest"):
            self.validate()
        self.reset_fixture()
        self.mutate_record(frozen_config={"changed": True})
        with self.assertRaisesRegex(ValueError, "frozen_config"):
            self.validate()

    def test_trial_checkpoint_must_be_supported_by_history(self):
        path = self._record_path("MLP")
        row = json.loads(path.read_text(encoding="utf-8"))
        row["trials"][0]["best_epoch"] = 0
        self._write(path, row)
        with self.assertRaisesRegex(ValueError, "best_epoch"):
            self.validate()
        self.reset_fixture()
        path = self._record_path("MLP")
        row = json.loads(path.read_text(encoding="utf-8"))
        row["trials"][0]["history"][0]["epoch"] = 9
        self._write(path, row)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.validate()

    def test_preflight_and_failure_artifacts_cannot_be_complete_results(self):
        manifest_path = self.root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["preflight"] = True
        self._write(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "preflight"):
            self.validate()
        self.reset_fixture()
        self._write(self.root / "failure_1.json", {"status": "error"})
        with self.assertRaisesRegex(ValueError, "failure artifact"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
