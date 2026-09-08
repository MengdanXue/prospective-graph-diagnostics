import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from scripts import run_mlp_optimization_diagnostic as runner
from scripts import summarize_mlp_optimization_diagnostic as summary


class MlpOptimizationSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "diagnostic"
        self.config = {
            "schema_version": "1.0", "run_id": "toy_mlp_diag", "analysis_status": "post_hoc_validation_only_mechanism_diagnostic",
            "datasets": {"Toy": {"filename": "toy.npz", "sha256": "a" * 64, "shape": [4, 2]}},
            "conditions": ["raw", "normalize_features", "normalize_scaled", "normalize_centered", "normalize_centered_scaled"], "regularization": {"wd_5e-4": 0.0005, "wd_0": 0.0},
            "seeds": [0, 1], "models": ["MLP"], "expected_records": 20, "expected_trials": 20,
            "training": {"hidden_channels": 2, "max_epochs": 1, "patience": 1, "trials": [{"learning_rate": 0.01, "dropout": 0.5}], "selection_order": ["validation_accuracy_desc", "validation_loss_asc", "trial_id_asc"], "test_evaluations_after_selection": 0},
        }
        self.audit = {"units": [{"dataset": "Toy", "seed": seed, "split_id": f"toy-split-{seed}"} for seed in self.config["seeds"]]}
        self.environment = {"device": "cpu", "python": "3.12"}
        self.fingerprint = {"files": {"runner": "b" * 64}, "config_sha256": summary.config_sha256(self.config) if hasattr(summary, "config_sha256") else self._digest(self.config)}
        manifest = {"schema_version": "1.0", "run_id": self.config["run_id"], "config_sha256": self._digest(self.config), "source_commit": "c" * 40, "source_fingerprint": self.fingerprint, "environment": self.environment, "analysis_status": self.config["analysis_status"], "expected_records": 20, "expected_trials": 20, "test_evaluations_after_selection": 0, "config": self.config}
        self._write(self.root / "run_manifest.json", manifest)
        self._write(self.root / "complete.json", {"status": "complete", "run_id": self.config["run_id"], "expected_records": 20, "expected_trials": 20, "config_sha256": manifest["config_sha256"], "source_fingerprint": self.fingerprint, "test_evaluations_after_selection": 0})
        for condition in self.config["conditions"]:
            for seed in self.config["seeds"]:
                for regularization in self.config["regularization"]:
                    self._write(self._path(condition, seed, regularization), self._row(condition, seed, regularization))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def _digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _path(self, condition, seed, regularization):
        return self.root / "records" / condition / "Toy" / regularization / f"seed_{seed:03d}.json"

    def _metadata(self, condition):
        return {"condition": condition, "fit_partition": "train_features_only", "statistics_dtype": "float64", "output_dtype": "float32", "epsilon": 1e-12, "fitted_normalized_mean": [0.0, 0.0], "fitted_scale": 1.0, "feature_statistics": {"raw_train_centered_rms": [1.0, 1.0], "normalized_train_centered_rms": [1.0, 1.0]}}

    def _row(self, condition, seed, regularization):
        training = copy.deepcopy(self.config["training"])
        training["weight_decay"] = self.config["regularization"][regularization]
        x = torch.arange(8, dtype=torch.float32).reshape(4, 2) + (0.0 if condition == "raw" else 1.0)
        feature_hash = hashlib.sha256(x.numpy().tobytes()).hexdigest()
        selected_accuracy, selected_loss = 0.5, 0.5
        trial = {"trial_id": "trial_000", "configuration": self.config["training"]["trials"][0], "validation_accuracy": selected_accuracy, "validation_loss": selected_loss, "best_epoch": 0, "epochs_completed": 1, "duration_seconds": 0.01, "history": [{"epoch": 0, "train_loss": 0.5, "validation_accuracy": selected_accuracy, "validation_loss": selected_loss}]}
        return {"schema_version": "1.0", "run_id": self.config["run_id"], "status": "success", "dataset": "Toy", "model": "MLP", "family": "mlp", "seed": seed, "split_id": f"toy-split-{seed}", "condition": condition, "preprocessing": condition, "regularization": regularization, "weight_decay": self.config["regularization"][regularization], "validation_accuracy": selected_accuracy, "validation_loss": selected_loss, "selected_trial_id": "trial_000", "trials": [trial], "partitions": {"train": self._metrics(2), "validation": self._metrics(2, selected_accuracy)}, "training_configuration": training, "test_evaluations_after_selection": 0, "source_commit": "c" * 40, "source_fingerprint": self.fingerprint, "environment": self.environment, "data_provenance": {"raw_filename": "toy.npz", "raw_sha256": "a" * 64, "split_id": f"toy-split-{seed}", "transformed_feature_sha256": feature_hash, "source_fingerprint": self.fingerprint}, "transform_metadata": self._metadata(condition), "config_sha256": self._digest(self.config), "frozen_config": self.config, "duration_seconds": 0.1}

    @staticmethod
    def _metrics(n, accuracy=0.5):
        return {"loss": 0.5, "accuracy": accuracy, "label_counts": {"0": 1, "1": 1}, "prediction_counts": {"0": 2, "1": 0}, "recall_by_class": {"0": 1.0, "1": 2 * accuracy - 1.0}, "balanced_accuracy": accuracy, "training_majority_class": 0, "training_majority_accuracy": 0.5, "dominant_prediction_class": 0, "dominant_prediction_fraction": 1.0, "n": n}

    def fake_inputs(self, data_root, dataset, specification, audit):
        x = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        y = torch.tensor([0, 1, 0, 1])
        edge = torch.empty((2, 0), dtype=torch.long)
        splits = {seed: ({"train": [0, 1], "validation": [2, 3], "test": []}, f"toy-split-{seed}") for seed in self.config["seeds"]}
        return x, y, edge, splits

    def fake_transform(self, x, train_indices, condition):
        transformed = x.clone() + (0.0 if condition == "raw" else 1.0)
        return transformed, self._metadata(condition)

    def reconstruct(self):
        return mock.patch.object(runner, "load_inputs", self.fake_inputs), mock.patch.object(runner, "fit_transform_features", self.fake_transform)

    def test_paired_contrasts_preserve_seed_signs_and_counts(self):
        contrast_config = copy.deepcopy(self.config)
        contrast_config["seeds"] = [0, 1, 2]
        records = {}
        for condition in contrast_config["conditions"]:
            for regularization in contrast_config["regularization"]:
                for seed in contrast_config["seeds"]:
                    row = self._row(condition, seed, regularization)
                    if condition == "normalize_features":
                        delta = (0.1, 0.0, -0.1)[seed]
                        row["validation_accuracy"] += delta
                    if condition == "normalize_scaled" and regularization == "wd_0":
                        row["validation_accuracy"] += 0.1
                    records[(condition, "Toy", regularization, seed)] = row
        result = summary._contrasts(records, contrast_config)
        contrast = result["Toy"]["by_decay"]["wd_5e-4"]["normalized_minus_raw"]["validation_accuracy"]
        self.assertEqual(contrast["n"], 3)
        self.assertEqual((contrast["positive_count"], contrast["zero_count"], contrast["negative_count"]), (1, 1, 1))
        self.assertEqual([row["seed"] for row in contrast["differences"]], [0, 1, 2])
        interaction = result["Toy"]["scale_interactions"]["uncentered_scale_effect_zero_minus_nonzero_validation_accuracy"]
        self.assertAlmostEqual(interaction["mean_difference"], 0.1)
        contrast_config["regularization"] = dict(reversed(list(contrast_config["regularization"].items())))
        self.assertEqual(summary._contrasts(records, contrast_config), result)

    def test_valid_toy_run_returns_all_units_and_aggregates(self):
        with self.reconstruct()[0], self.reconstruct()[1], mock.patch.object(summary, "_controls", return_value={}):
            result = summary.summarize(self.root, Path(self.temp.name), Path(self.temp.name), self.config, self.audit)
        self.assertEqual(result["record_count"], 20)
        self.assertEqual(len(result["units"]), 20)
        self.assertEqual(result["aggregates"]["Toy"]["raw"]["wd_0"]["validation"]["accuracy"]["mean"], 0.5)

    def test_summary_rejects_missing_and_extra_records(self):
        with self.reconstruct()[0], self.reconstruct()[1]:
            self._path("raw", 0, "wd_5e-4").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                summary._reconstruct_inputs(self.config, Path(self.temp.name), self.audit, self.root, _load_manifest(self.root))
        self.setUp()
        with self.reconstruct()[0], self.reconstruct()[1]:
            extra = self._path("raw", 0, "wd_5e-4").with_name("seed_000_copy.json")
            self._write(extra, {})
            with self.assertRaisesRegex(ValueError, "extra"):
                summary._reconstruct_inputs(self.config, Path(self.temp.name), self.audit, self.root, _load_manifest(self.root))

    def test_summary_rejects_provenance_counts_and_test_fields(self):
        mutations = (("data_provenance", "raw_sha256", "d" * 64, "provenance"), ("partitions", "validation", {"loss": 0.5, "accuracy": 0.5, "label_counts": {"0": 2}, "prediction_counts": {"0": 1}, "recall_by_class": {"0": 1.0}, "balanced_accuracy": 1.0, "training_majority_class": 0, "training_majority_accuracy": 1.0, "dominant_prediction_class": 0, "dominant_prediction_fraction": 1.0, "n": 2}, "count"), ("test_accuracy", None, 0.5, "test"))
        for key, nested, value, message in mutations:
            with self.subTest(message=message):
                self.setUp()
                path = self._path("raw", 0, "wd_5e-4")
                row = json.loads(path.read_text(encoding="utf-8"))
                if nested is None:
                    row[key] = value
                elif key == "partitions":
                    row[key][nested] = value
                else:
                    row[key][nested] = value
                self._write(path, row)
                with self.reconstruct()[0], self.reconstruct()[1]:
                    with self.assertRaisesRegex(ValueError, message):
                        summary._reconstruct_inputs(self.config, Path(self.temp.name), self.audit, self.root, _load_manifest(self.root))


def _load_manifest(root):
    return json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
