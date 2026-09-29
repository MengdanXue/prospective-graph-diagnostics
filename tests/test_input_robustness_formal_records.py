"""Synthetic end-to-end checks for the guarded formal record contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.input_robustness_formal_records import (
    CONDITIONS, GRAPH_MODELS, MODELS, DuplicateRecordError, FormalRecordError,
    FormalRecordWriter, build_record, expected_keys,
)
from scripts.input_robustness_checkpoint_store import save_checkpoint
from scripts.validate_input_robustness_formal_records import validate_complete_run, validate_run


class FormalRecordFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "synthetic-formal"
        self.scope = expected_keys(datasets=("Cora", "Actor"), conditions=CONDITIONS, models=MODELS, seeds=(0,))
        self.manifest = {
            "schema_version": "1.0", "run_id": "synthetic_formal_fixture_v1",
            "source_commit": "a" * 40, "config_sha256": "b" * 64,
            "data_binding_sha256": "c" * 64, "environment": {"python": "test", "device": "cpu"},
            "scope": {"datasets": ["Cora", "Actor"], "conditions": list(CONDITIONS),
                      "models": list(MODELS), "seeds": [0]},
            "expected_records": len(self.scope), "expected_trials": len(self.scope) * 4,
        }
        self.writer = FormalRecordWriter(self.root, manifest=self.manifest, synthetic=True)
        self.records = []
        training = {"trials": [
            {"learning_rate": 0.01, "dropout": 0.5}, {"learning_rate": 0.01, "dropout": 0.7},
            {"learning_rate": 0.005, "dropout": 0.5}, {"learning_rate": 0.005, "dropout": 0.7},
        ]}
        for dataset in ("Cora", "Actor"):
            for condition in CONDITIONS:
                binding = {"dataset": dataset, "condition": condition, "seed": 0,
                           "split_id": f"{dataset}-split-0", "transformed_feature_sha256": ("d" if condition == CONDITIONS[0] else "e") * 64,
                           "fit_statistics_sha256": ("f" if condition == CONDITIONS[0] else "0") * 64}
                diagnostics = {"homophily": 0.4 if dataset == "Cora" else 0.7,
                               "mean_degree": 3.0 if dataset == "Cora" else 1.0,
                               "delta_h": 0.1 if condition == CONDITIONS[0] else -0.1,
                               "historical_action": "graph" if dataset == "Actor" else "mlp"}
                for model in MODELS:
                    trials = []
                    for index, configuration in enumerate(training["trials"]):
                        trials.append({"trial_id": f"trial_{index:03d}", "configuration": configuration,
                                       "validation_accuracy": 0.55 + 0.02 * index + (0.01 if model != "MLP" else 0),
                                       "validation_loss": 0.8 - 0.01 * index})
                    checkpoint_root = self.root / "checkpoints" / condition / dataset / model / "seed_000"
                    checkpoints = []
                    for index in range(4):
                        checkpoint = save_checkpoint(
                            checkpoint_root, f"trial_{index:03d}",
                            {"weight": torch.tensor([float(index + 1)])},
                        )
                        checkpoint["path"] = (Path("checkpoints") / condition / dataset / model /
                                               "seed_000" / checkpoint["path"]).as_posix()
                        checkpoints.append(checkpoint)
                    self.records.append(build_record(
                        run_id=self.manifest["run_id"], dataset=dataset, condition=condition,
                        model=model, seed=0, split_id=binding["split_id"], source_commit="a" * 40,
                        config_sha256="b" * 64, data_binding_sha256="c" * 64,
                        environment=self.manifest["environment"], transform_binding=binding,
                        training_configuration=training, trials=trials,
                        test_accuracy=0.5 + (0.1 if model in GRAPH_MODELS else 0),
                        checkpoint_manifest=checkpoints, diagnostics=diagnostics, synthetic=True))
        for record in self.records:
            self.writer.write_record(record)

    def test_four_trials_selection_checkpoint_save_and_complete_marker(self):
        complete = self.writer.finalize(expected=self.scope)
        self.assertEqual(complete["expected_records"], 28)
        result = validate_complete_run(self.root, expected_keys=self.scope, synthetic=True)
        self.assertEqual(result["record_count"], 28)
        self.assertEqual(result["trial_count"], 112)
        self.assertEqual(result["validation_evaluations"], 112)
        self.assertEqual(result["test_evaluations"], 28)
        self.assertEqual(result["formal_training_enabled"], False)

    def test_duplicate_record_identity_is_rejected(self):
        path = self.root / "records" / CONDITIONS[0] / "Cora" / "MLP" / "duplicate.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.records[0]), encoding="utf-8")
        with self.assertRaises(FormalRecordError):
            validate_run(self.root, expected_keys=self.scope, synthetic=True, require_complete=False)

    def test_missing_record_is_rejected(self):
        target = self.root / "records" / CONDITIONS[0] / "Cora" / "MLP" / "seed_000.json"
        target.unlink()
        with self.assertRaises(FormalRecordError):
            validate_run(self.root, expected_keys=self.scope, synthetic=True, require_complete=False)

    def test_writer_never_overwrites_an_existing_unit(self):
        with self.assertRaises(DuplicateRecordError):
            self.writer.write_record(self.records[0])

    def test_formal_execution_gate_stays_closed_until_both_flags_are_true(self):
        from scripts.input_robustness_formal_records import run_formal_model_unit
        with self.assertRaises(FormalRecordError):
            run_formal_model_unit(writer=self.writer, core_runner=lambda **_: self.records[0],
                                  launch_authorized=False, formal_training_enabled=False,
                                  checkpoint_manifest=self.records[0]["checkpoint_manifest"])
