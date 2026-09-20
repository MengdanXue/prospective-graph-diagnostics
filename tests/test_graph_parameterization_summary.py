import copy
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import scripts as scripts_package

from scripts import run_mlp_optimization_diagnostic as transform_runner
from scripts import summarize_graph_parameterization_diagnostic as summary
from scripts.validation_diagnostic_common import validate_record as generic_validate_record


class GraphParameterizationSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "graph-run"
        self.config = {
            "schema_version": "1.0", "run_id": "toy_graph_parameterization", "analysis_status": "post_hoc_validation_only_transfer_diagnostic",
            "datasets": {"ToyA": {"filename": "a.npz", "sha256": "a" * 64, "shape": [4, 2]}, "ToyB": {"filename": "b.npz", "sha256": "b" * 64, "shape": [4, 2]}},
            "conditions": ["raw", "normalize_features", "normalize_centered_scaled"], "regularization": {"wd_5e-4": 0.0005},
            "seeds": list(range(10)), "models": ["MLP", "GCN", "GAT", "GraphSAGE", "H2GCN", "LINKX", "GPR-GNN"], "expected_records": 420, "expected_trials": 1680,
            "training": {"hidden_channels": 2, "max_epochs": 1, "patience": 1, "weight_decay": 0.0005, "trials": [{"learning_rate": 0.01, "dropout": 0.5}, {"learning_rate": 0.01, "dropout": 0.7}, {"learning_rate": 0.005, "dropout": 0.5}, {"learning_rate": 0.005, "dropout": 0.7}], "selection_order": ["validation_accuracy_desc", "validation_loss_asc", "trial_id_asc"], "test_evaluations_after_selection": 0},
        }
        self.audit = {"units": [{"dataset": dataset, "seed": seed, "split_id": f"{dataset}-split-{seed}"} for dataset in self.config["datasets"] for seed in self.config["seeds"]]}
        self.environment = {"device": "cpu", "python": "3.12"}
        self.fingerprint = {"files": {"runner": "c" * 64}, "config_sha256": self._digest(self.config)}
        manifest = {"schema_version": "1.0", "run_id": self.config["run_id"], "config_sha256": self._digest(self.config), "source_commit": "d" * 40, "source_fingerprint": self.fingerprint, "environment": self.environment, "analysis_status": self.config["analysis_status"], "expected_records": 420, "expected_trials": 1680, "test_evaluations_after_selection": 0, "config": self.config}
        self.root.mkdir(parents=True)
        self._write(self.root / "run_manifest.json", manifest)
        self._write(self.root / "complete.json", {"status": "complete", "run_id": self.config["run_id"], "expected_records": 420, "expected_trials": 1680, "config_sha256": manifest["config_sha256"], "source_fingerprint": self.fingerprint, "test_evaluations_after_selection": 0})
        for condition in self.config["conditions"]:
            for dataset in self.config["datasets"]:
                for model in self.config["models"]:
                    for seed in self.config["seeds"]:
                        self._write(self._path(condition, dataset, model, seed), self._row(condition, dataset, model, seed))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def _write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def _path(self, condition, dataset, model, seed):
        return self.root / "records" / condition / dataset / model / f"seed_{seed:03d}.json"

    def _metadata(self, condition):
        return {"condition": condition, "fit_partition": "train_features_only", "statistics_dtype": "float64", "output_dtype": "float32", "epsilon": 1e-12, "fitted_normalized_mean": [0.0, 0.0], "fitted_scale": 1.0, "feature_statistics": {"raw_train_centered_rms": 1.0, "normalized_train_centered_rms": 1.0}}

    def _row(self, condition, dataset, model, seed):
        x = torch.arange(8, dtype=torch.float32).reshape(4, 2) + (0.0 if condition == "raw" else 1.0)
        feature_hash = hashlib.sha256(x.numpy().tobytes()).hexdigest()
        trials = [{"trial_id": f"trial_{index:03d}", "configuration": configuration, "validation_accuracy": 0.5, "validation_loss": 0.5, "best_epoch": 0, "epochs_completed": 1, "duration_seconds": 0.01, "history": [{"epoch": 0, "train_loss": 0.5, "validation_accuracy": 0.5, "validation_loss": 0.5}]} for index, configuration in enumerate(self.config["training"]["trials"])]
        family = "mlp" if model == "MLP" else "graph"
        return {"schema_version": "1.0", "run_id": self.config["run_id"], "status": "success", "dataset": dataset, "model": model, "family": family, "seed": seed, "split_id": f"{dataset}-split-{seed}", "condition": condition, "preprocessing": condition, "regularization": "wd_5e-4", "weight_decay": 0.0005, "validation_accuracy": 0.5, "validation_loss": 0.5, "selected_trial_id": "trial_000", "trials": trials, "partitions": {"train": self._metrics(), "validation": self._metrics()}, "training_configuration": self.config["training"], "test_evaluations_after_selection": 0, "source_commit": "d" * 40, "source_fingerprint": self.fingerprint, "environment": self.environment, "data_provenance": {"raw_filename": "a.npz" if dataset == "ToyA" else "b.npz", "raw_sha256": "a" * 64 if dataset == "ToyA" else "b" * 64, "split_id": f"{dataset}-split-{seed}", "transformed_feature_sha256": feature_hash, "source_fingerprint": self.fingerprint}, "transform_metadata": self._metadata(condition), "config_sha256": self._digest(self.config), "frozen_config": self.config, "duration_seconds": 0.1}

    @staticmethod
    def _metrics():
        return {"loss": 0.5, "accuracy": 0.5, "label_counts": {"0": 1, "1": 1}, "prediction_counts": {"0": 2, "1": 0}, "recall_by_class": {"0": 1.0, "1": 0.0}, "balanced_accuracy": 0.5, "training_majority_class": 0, "training_majority_accuracy": 0.5, "dominant_prediction_class": 0, "dominant_prediction_fraction": 1.0, "n": 2}

    def fake_inputs(self, data_root, dataset, specification, audit):
        x = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        y = torch.tensor([0, 1, 0, 1])
        edge = torch.empty((2, 0), dtype=torch.long)
        splits = {seed: ({"train": [0, 1], "validation": [2, 3], "test": []}, f"{dataset}-split-{seed}") for seed in self.config["seeds"]}
        return x, y, edge, splits

    def fake_transform(self, x, train_indices, condition):
        return x.clone() + (0.0 if condition == "raw" else 1.0), self._metadata(condition)

    def patches(self):
        fake_module = types.SimpleNamespace(source_fingerprint=lambda path, digest: self.fingerprint, validate_record=generic_validate_record)
        return (mock.patch.dict(sys.modules, {"scripts.run_graph_parameterization_diagnostic": fake_module}), mock.patch.object(scripts_package, "run_graph_parameterization_diagnostic", fake_module, create=True), mock.patch.object(transform_runner, "load_inputs", self.fake_inputs), mock.patch.object(transform_runner, "fit_transform_features", self.fake_transform))

    def test_full_toy_scope_reconstructs_all_units(self):
        patches = self.patches()
        with patches[0], patches[1], patches[2], patches[3]:
            manifest, _, _ = summary._validate_manifest(self.root, self.config, Path(self.temp.name) / "config.json")
            records, split_ids, hashes = summary._reconstruct_records(self.root, Path(self.temp.name), self.config, self.audit, manifest)
        self.assertEqual(len(records), 420)
        self.assertEqual(len(split_ids), 20)
        self.assertEqual(len(hashes), 60)

    def test_missing_and_duplicate_records_are_rejected(self):
        path = self._path("raw", "ToyA", "MLP", 0)
        path.unlink()
        patches = self.patches()
        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaisesRegex(ValueError, "missing"):
                summary._reconstruct_records(self.root, Path(self.temp.name), self.config, self.audit, _load_manifest(self.root))
        self.setUp()
        self._write(self._path("raw", "ToyA", "MLP", 0).with_name("seed_000_copy.json"), {})
        patches = self.patches()
        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaisesRegex(ValueError, "extra"):
                summary._reconstruct_records(self.root, Path(self.temp.name), self.config, self.audit, _load_manifest(self.root))

    def test_provenance_counts_and_test_fields_fail_for_the_target_reason(self):
        mutations = (("provenance",), ("counts",), ("test",))
        for (kind,) in mutations:
            with self.subTest(kind=kind):
                self.setUp()
                path = self._path("raw", "ToyA", "MLP", 0)
                row = json.loads(path.read_text(encoding="utf-8"))
                if kind == "provenance":
                    row["data_provenance"]["raw_sha256"] = "e" * 64
                elif kind == "counts":
                    row["partitions"]["validation"]["prediction_counts"]["0"] = 1
                else:
                    row["test_accuracy"] = 0.5
                self._write(path, row)
                patches = self.patches()
                with patches[0], patches[1], patches[2], patches[3]:
                    expected_message = {"provenance": "provenance", "counts": "count total", "test": "test metrics"}[kind]
                    with self.assertRaisesRegex(ValueError, expected_message):
                        summary._reconstruct_records(self.root, Path(self.temp.name), self.config, self.audit, _load_manifest(self.root))

    def test_graph_selection_ties_break_by_model_id(self):
        records = {}
        for dataset in self.config["datasets"]:
            for model in self.config["models"]:
                for seed in self.config["seeds"]:
                    for condition in self.config["conditions"]:
                        records[(condition, dataset, model, seed)] = self._row(condition, dataset, model, seed)
        selected = summary._selection(records, self.config)
        self.assertEqual(selected["ToyA"]["raw"]["selection_counts"]["GAT"], 10)
        self.assertEqual(sum(selected["ToyA"]["normalize_features"]["selection_counts"].values()), 10)


def _load_manifest(root):
    return json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
