import copy
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts import run_mlp_optimization_diagnostic as runner


class _FakeModel(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index):
        return self.bias.expand(x.size(0), -1)


def _fake_builder(model_id, *, out_channels, **kwargs):
    return _FakeModel(out_channels)


class MlpOptimizationDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor(
            [[1.0, 2.0, 4.0], [2.0, 1.0, 3.0], [3.0, 4.0, 2.0], [4.0, 3.0, 1.0],
             [5.0, 6.0, 2.0], [6.0, 5.0, 1.0]], dtype=torch.float32
        )
        self.train = torch.tensor([0, 1, 2, 3])

    def test_statistics_are_fit_on_train_rows_only(self):
        # The held-out perturbation is positive, so it preserves NormalizeFeatures'
        # transductive global minimum while testing only the added affine fit.
        _, metadata = runner.fit_transform_features(self.x, self.train, "normalize_centered_scaled")
        changed = self.x.clone()
        changed[4:] += 1000
        _, changed_metadata = runner.fit_transform_features(changed, self.train, "normalize_centered_scaled")
        self.assertEqual(metadata["fitted_normalized_mean"], changed_metadata["fitted_normalized_mean"])
        self.assertEqual(metadata["fitted_scale"], changed_metadata["fitted_scale"])

    def test_normalize_condition_matches_existing_transform(self):
        transformed, _ = runner.fit_transform_features(self.x, self.train, "normalize_features")
        expected = runner.transform_features(self.x, "normalize_features")
        torch.testing.assert_close(transformed, expected)

    def test_affine_conditions_have_expected_relationship(self):
        normalized, metadata = runner.fit_transform_features(self.x, self.train, "normalize_features")
        centered, _ = runner.fit_transform_features(self.x, self.train, "normalize_centered")
        scaled, _ = runner.fit_transform_features(self.x, self.train, "normalize_scaled")
        centered_scaled, _ = runner.fit_transform_features(self.x, self.train, "normalize_centered_scaled")
        mean = torch.tensor(metadata["fitted_normalized_mean"])
        scale = torch.tensor(metadata["fitted_scale"])
        torch.testing.assert_close(centered, normalized - mean, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(scaled, normalized * scale, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(centered_scaled, (normalized - mean) * scale, rtol=1e-5, atol=1e-6)
        self.assertIsInstance(metadata["fitted_scale"], float)
        raw_train = self.x[self.train].numpy().astype(np.float64)
        norm_train = normalized[self.train].numpy().astype(np.float64)
        raw_mean = raw_train.mean(axis=0)
        norm_mean = norm_train.mean(axis=0)
        expected = np.sqrt(np.mean((raw_train - raw_mean) ** 2)) / np.sqrt(np.mean((norm_train - norm_mean) ** 2))
        self.assertAlmostEqual(metadata["fitted_scale"], float(expected), places=14)

    def test_zero_rms_is_rejected(self):
        x = torch.full_like(self.x, 7)
        with self.assertRaises(ValueError):
            runner.fit_transform_features(x, self.train, "raw")

    def _training(self):
        return {
            "hidden_channels": 3, "max_epochs": 2, "patience": 1,
            "weight_decay": 0.0,
            "trials": [{"learning_rate": 0.01, "dropout": 0.5}, {"learning_rate": 0.005, "dropout": 0.7}],
            "selection_order": ["validation_accuracy_desc", "validation_loss_asc", "trial_id_asc"],
            "test_evaluations_after_selection": 0,
        }

    def test_selection_and_checkpoint_metrics_never_access_test(self):
        calls = []

        def fake_train(**kwargs):
            self.assertNotIn("test_indices", kwargs)
            calls.append(kwargs["trial_id"])
            trial_number = int(kwargs["trial_id"][-3:])
            state = {"bias": torch.tensor([0.0, 1.0]) if trial_number == 0 else torch.tensor([1.0, 0.0])}
            return ({
                "trial_id": kwargs["trial_id"], "configuration": dict(kwargs["trial"]),
                "validation_accuracy": 0.2 if trial_number == 0 else 0.8,
                "validation_loss": 0.9 if trial_number == 0 else 0.4,
                "best_epoch": 0, "epochs_completed": 1, "history": [], "duration_seconds": 0.0,
            }, state)

        x, metadata = runner.fit_transform_features(self.x, self.train, "raw")
        row = runner.run_validation_unit(
            run_id="test", dataset="toy", model_id="MLP", seed=0, split_id="split",
            condition="raw", regularization="wd_0", weight_decay=0.0, x=x,
            y=torch.tensor([0, 1, 0, 1, 0, 1]), edge_index=torch.empty((2, 0), dtype=torch.long),
            train_indices=self.train, validation_indices=torch.tensor([4, 5]), training=self._training(),
            source_commit="commit", environment={"python": "test"},
            data_provenance={"source_fingerprint": {"files": {}}}, transform_metadata=metadata,
            device=torch.device("cpu"), train_trial_fn=fake_train, model_builder=_fake_builder,
        )
        self.assertEqual(calls, ["trial_000", "trial_001"])
        self.assertEqual(row["selected_trial_id"], "trial_001")
        self.assertEqual(row["test_evaluations_after_selection"], 0)
        self.assertNotIn("test_accuracy", row)
        self.assertEqual(row["partitions"]["train"]["n"], 4)

    def test_validate_record_rejects_changed_resume_provenance(self):
        training = self._training()
        row = {
            "status": "success", "run_id": "test", "dataset": "toy", "model": "MLP", "seed": 0,
            "split_id": "split", "condition": "raw", "regularization": "wd_0", "weight_decay": 0.0,
            "selected_trial_id": "trial_000", "validation_accuracy": 0.8, "validation_loss": 0.4,
            "trials": [
                {"trial_id": "trial_000", "configuration": training["trials"][0], "validation_accuracy": 0.8, "validation_loss": 0.4},
                {"trial_id": "trial_001", "configuration": training["trials"][1], "validation_accuracy": 0.2, "validation_loss": 0.9},
            ],
            "partitions": {"train": self._metrics(4), "validation": self._metrics(2)},
            "training_configuration": training, "test_evaluations_after_selection": 0,
            "transform_metadata": {"fit_partition": "train_features_only", "fitted_normalized_mean": [0.0], "fitted_scale": 1.0},
        }
        expected = {"source_fingerprint": {"files": {"runner": "new"}}}
        with self.assertRaises(ValueError):
            runner.validate_record(row, expected, training)

    def test_real_cpu_training_record_passes_and_mutations_fail(self):
        x = torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, 4.0, 1.0], [3.0, 1.0, 4.0], [4.0, 3.0, 2.0],
             [5.0, 2.0, 6.0], [6.0, 5.0, 1.0], [7.0, 4.0, 3.0], [8.0, 1.0, 5.0]]
        )
        y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        train = torch.arange(6)
        validation = torch.arange(6, 8)
        training = {
            "hidden_channels": 4, "max_epochs": 3, "patience": 1, "weight_decay": 0.0,
            "trials": [{"learning_rate": 0.01, "dropout": 0.5}, {"learning_rate": 0.005, "dropout": 0.7}],
            "selection_order": ["validation_accuracy_desc", "validation_loss_asc", "trial_id_asc"],
            "test_evaluations_after_selection": 0,
        }
        transformed, metadata = runner.fit_transform_features(x, train, "raw")
        provenance = {"source_fingerprint": {"files": {"runner": "fixture"}}}
        row = runner.run_validation_unit(
            run_id="test", dataset="toy", model_id="MLP", seed=0, split_id="split",
            condition="raw", regularization="wd_0", weight_decay=0.0, x=transformed, y=y,
            edge_index=torch.empty((2, 0), dtype=torch.long), train_indices=train,
            validation_indices=validation, training=training, source_commit="commit",
            environment={"python": "test"}, data_provenance=provenance,
            transform_metadata=metadata, device=torch.device("cpu"),
        )
        expected = {key: row[key] for key in (
            "run_id", "dataset", "model", "seed", "split_id", "condition", "regularization",
            "weight_decay", "source_commit", "source_fingerprint", "environment", "config_sha256",
            "frozen_config", "data_provenance")}
        runner.validate_record(row, expected, training)
        changed = copy.deepcopy(row)
        changed["source_fingerprint"] = {"files": {"runner": "changed"}}
        with self.assertRaises(ValueError):
            runner.validate_record(changed, expected, training)
        changed = copy.deepcopy(row)
        changed["partitions"]["validation"]["n"] += 1
        with self.assertRaises(ValueError):
            runner.validate_record(changed, expected, training)
        changed = copy.deepcopy(row)
        changed["test_accuracy"] = 0.5
        with self.assertRaises(ValueError):
            runner.validate_record(changed, expected, training)

    @staticmethod
    def _metrics(n):
        return {
            "loss": 0.5, "accuracy": 0.5, "label_counts": {"0": n}, "prediction_counts": {"0": n},
            "recall_by_class": {"0": 1.0}, "balanced_accuracy": 1.0,
            "training_majority_class": 0, "training_majority_accuracy": 1.0,
            "dominant_prediction_class": 0, "dominant_prediction_fraction": 1.0, "n": n,
        }


if __name__ == "__main__":
    unittest.main()
