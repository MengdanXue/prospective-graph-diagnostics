import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.prospective_models import build_model
from experiments.run_prospective_benchmark import _train_trial, seed_everything
from scripts import audit_training_reproducibility as audit


class TrainingReproducibilityAuditTests(unittest.TestCase):
    def test_frozen_scope_is_twenty_two_independent_workers(self):
        config = json.loads((audit.ROOT / "configs/training_reproducibility_audit_v1.json").read_text())
        workers = audit.expand_workers(config)
        self.assertEqual(len(workers), 22)
        self.assertEqual(sum(w["backend"] == "cpu" for w in workers), 2)
        self.assertTrue(all(w["max_epochs"] == 12 for w in workers if w["backend"] == "cpu"))

    def test_state_hash_includes_buffers_dtype_and_shape(self):
        a = {"weight": torch.ones(2), "running_mean": torch.zeros(2)}
        b = {**a, "running_mean": torch.ones(2)}
        self.assertNotEqual(audit.state_fingerprint(a), audit.state_fingerprint(b))
        self.assertNotEqual(audit.tensor_hash(torch.ones(2)), audit.tensor_hash(torch.ones(1, 2)))
        self.assertNotEqual(audit.tensor_hash(torch.ones(2)), audit.tensor_hash(torch.ones(2).double()))

    def test_artifact_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "record.json"
            audit.write_exclusive(path, {"original": True})
            with self.assertRaises(FileExistsError):
                audit.write_exclusive(path, {"original": False})
            self.assertEqual(json.loads(path.read_text()), {"original": True})

    def test_instrumentation_preserves_frozen_cpu_training_and_selection(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(17)
        x = torch.randn(32, 8, generator=generator)
        y = torch.randint(3, (32,), generator=generator)
        edges = torch.stack([torch.arange(32), torch.arange(32).roll(1)])
        train, validation = torch.arange(20), torch.arange(20, 28)
        trial = {"learning_rate": 0.01, "dropout": 0.7}
        training = {"max_epochs": 8, "patience": 3, "weight_decay": 0.0005, "hidden_channels": 8}
        for model_id in ("MLP", "LINKX"):
            reference, expected_state = _train_trial(model_id=model_id, seed=4, trial_id="trial_001", trial=trial, x=x, y=y, edge_index=edges, train_indices=train, validation_indices=validation, hidden_channels=8, max_epochs=8, patience=3, weight_decay=0.0005, device=torch.device("cpu"), h2_adjacencies=None)
            seed_everything(4)
            model = build_model(model_id, num_nodes=32, in_channels=8, hidden_channels=8, out_channels=3, dropout=0.7, edge_index=edges)
            actual, actual_state = audit.train_instrumented(model=model, x=x, y=y, edges=edges, train=train, validation=validation, training=training, trial=trial, early_epochs=[0, 1, 2, 3, 4], torch=torch)
            self.assertEqual(reference["history"], actual["history"])
            self.assertEqual(reference["best_epoch"], actual["best_epoch"])
            self.assertEqual(audit.state_fingerprint(expected_state), audit.state_fingerprint(actual_state))


if __name__ == "__main__":
    unittest.main()
