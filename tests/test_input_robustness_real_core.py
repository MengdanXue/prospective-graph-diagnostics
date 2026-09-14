"""End-to-end rehearsal of the real training core and checkpoint contract."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from scripts.input_robustness_formal_records import (
    CONDITIONS, FormalRecordWriter, expected_keys, run_rehearsal_model_unit,
)
from scripts.validate_input_robustness_formal_records import validate_complete_run


class RealCoreRehearsalTests(unittest.TestCase):
    def test_real_core_runs_four_trials_reloads_checkpoints_and_validates_summary(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "rehearsal"
            scope = expected_keys(datasets=("Cora",), conditions=(CONDITIONS[0],), models=("MLP",), seeds=(0,))
            manifest = {
                "schema_version": "1.0", "run_id": "real_core_rehearsal_v1",
                "source_commit": "a" * 40, "config_sha256": "b" * 64,
                "data_binding_sha256": "c" * 64, "environment": {"device": "cpu"},
                "scope": {"datasets": ["Cora"], "conditions": [CONDITIONS[0]], "models": ["MLP"], "seeds": [0]},
                "expected_records": 1, "expected_trials": 4,
            }
            writer = FormalRecordWriter(root, manifest=manifest, synthetic=True)
            x = torch.tensor([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.], [0., 0., 1.],
                              [1., 0., 1.], [0., 1., 1.], [1., 1., 1.], [0.5, 0.5, 0.]])
            y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
            indices = lambda values: torch.tensor(values, dtype=torch.long)
            training = {
                "hidden_channels": 4, "max_epochs": 1, "patience": 1, "weight_decay": 0.0005,
                "trials": [
                    {"learning_rate": 0.01, "dropout": 0.5}, {"learning_rate": 0.01, "dropout": 0.7},
                    {"learning_rate": 0.005, "dropout": 0.5}, {"learning_rate": 0.005, "dropout": 0.7},
                ],
            }
            record = run_rehearsal_model_unit(
                writer=writer, condition=CONDITIONS[0], run_id=manifest["run_id"], dataset="Cora",
                model_id="MLP", seed=0, split_id="synthetic-split-0", x=x, y=y,
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                train_indices=indices([0, 1, 2, 3]), validation_indices=indices([4, 5]),
                test_indices=indices([6, 7]), training=training, source_commit="a" * 40,
                environment={"device": "cpu"}, data_provenance={"synthetic": True},
                device=torch.device("cpu"), config_sha256="b" * 64, data_binding_sha256="c" * 64,
                frozen_config=training, diagnostics={"homophily": 0.4, "mean_degree": 3.0, "delta_h": 0.1},
            )
            self.assertEqual(len(record["trials"]), 4)
            self.assertEqual(record["validation_evaluations"], 4)
            self.assertEqual(record["test_evaluation"]["count"], 1)
            self.assertTrue(record["checkpoint_complete"])
            self.assertEqual(len(record["checkpoint_manifest"]), 4)
            complete = writer.finalize(expected=scope)
            self.assertEqual(complete["expected_records"], 1)
            result = validate_complete_run(root, expected_keys=scope, synthetic=True)
            self.assertEqual(result["record_count"], 1)
            self.assertEqual(result["test_evaluations"], 1)

