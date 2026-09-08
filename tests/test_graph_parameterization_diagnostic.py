import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.prospective_models import prepare_h2_adjacencies
from scripts import run_graph_parameterization_diagnostic as runner


class GraphParameterizationDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.models = ("MLP", "GCN", "GAT", "GraphSAGE", "H2GCN", "LINKX", "GPR-GNN")
        self.x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 1.0, 3.0], [3.0, 1.0, 4.0, 2.0], [4.0, 3.0, 2.0, 1.0],
             [5.0, 2.0, 6.0, 1.0], [6.0, 5.0, 1.0, 2.0], [7.0, 4.0, 3.0, 5.0], [8.0, 1.0, 5.0, 6.0]], dtype=torch.float32
        )
        self.y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        self.edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
        self.train = torch.arange(6)
        self.validation = torch.arange(6, 8)
        self.training = {
            "hidden_channels": 4, "max_epochs": 2, "patience": 1, "weight_decay": 0.0005,
            "trials": [{"learning_rate": 0.01, "dropout": 0.5}],
            "selection_order": ["validation_accuracy_desc", "validation_loss_asc", "trial_id_asc"],
            "test_evaluations_after_selection": 0,
        }

    def _expected(self, row, provenance, metadata):
        return {
            "run_id": row["run_id"], "dataset": row["dataset"], "model": row["model"], "family": row["family"],
            "seed": row["seed"], "split_id": row["split_id"], "condition": row["condition"], "preprocessing": row["preprocessing"],
            "regularization": row["regularization"], "weight_decay": row["weight_decay"], "source_commit": row["source_commit"],
            "source_fingerprint": row["source_fingerprint"], "environment": row["environment"], "config_sha256": row["config_sha256"],
            "frozen_config": row["frozen_config"], "data_provenance": provenance, "transform_metadata": metadata,
        }

    def test_actual_cpu_training_validates_all_seven_model_ids(self):
        h2 = prepare_h2_adjacencies(self.edges, num_nodes=len(self.y))
        transformed, metadata = runner.fit_transform_features(self.x, self.train, "raw")
        provenance = {"raw_filename": "toy.npz", "raw_sha256": "a" * 64, "split_id": "toy", "transformed_feature_sha256": hashlib.sha256(transformed.numpy().tobytes()).hexdigest(), "source_fingerprint": {"files": {"toy": "b" * 64}}}
        rows = []
        for model in self.models:
            row = runner.run_validation_unit(
                run_id="toy_graph", dataset="Toy", model_id=model, seed=0, split_id="toy", condition="raw",
                regularization="wd_5e-4", weight_decay=0.0005, x=transformed, y=self.y, edge_index=self.edges,
                train_indices=self.train, validation_indices=self.validation, training=self.training,
                source_commit="c" * 40, environment={"python": "test"}, data_provenance=provenance,
                transform_metadata=metadata, device=torch.device("cpu"), h2_adjacencies=h2,
            )
            row["family"] = runner._family(model)
            expected = self._expected(row, provenance, metadata)
            runner.validate_record(row, expected, self.training, model=model, family=runner._family(model))
            rows.append(row)
        self.assertEqual([row["model"] for row in rows], list(self.models))
        self.assertEqual({row["family"] for row in rows}, {"mlp", "graph"})
        self.assertTrue(all("test_accuracy" not in row for row in rows))

    def test_validator_rejects_wrong_family_metadata_history_counts_test_and_provenance(self):
        h2 = prepare_h2_adjacencies(self.edges, num_nodes=len(self.y))
        transformed, metadata = runner.fit_transform_features(self.x, self.train, "raw")
        provenance = {"raw_filename": "toy.npz", "raw_sha256": "a" * 64, "split_id": "toy", "transformed_feature_sha256": hashlib.sha256(transformed.numpy().tobytes()).hexdigest(), "source_fingerprint": {"files": {"toy": "b" * 64}}}
        row = runner.run_validation_unit(
            run_id="toy_graph", dataset="Toy", model_id="GCN", seed=0, split_id="toy", condition="raw",
            regularization="wd_5e-4", weight_decay=0.0005, x=transformed, y=self.y, edge_index=self.edges,
            train_indices=self.train, validation_indices=self.validation, training=self.training,
            source_commit="c" * 40, environment={"python": "test"}, data_provenance=provenance,
            transform_metadata=metadata, device=torch.device("cpu"), h2_adjacencies=h2,
        )
        row["family"] = "graph"
        expected = self._expected(row, provenance, metadata)
        runner.validate_record(row, expected, self.training, model="GCN", family="graph")
        mutations = []
        wrong = copy.deepcopy(row); wrong["family"] = "mlp"; mutations.append(wrong)
        wrong = copy.deepcopy(row); wrong["transform_metadata"]["fitted_scale"] += 1; mutations.append(wrong)
        wrong = copy.deepcopy(row); wrong["trials"][0]["history"][0]["epoch"] = 1; mutations.append(wrong)
        wrong = copy.deepcopy(row); wrong["partitions"]["validation"]["n"] += 1; mutations.append(wrong)
        wrong = copy.deepcopy(row); wrong["test_loss"] = 0.1; mutations.append(wrong)
        wrong = copy.deepcopy(row); wrong["data_provenance"]["split_id"] = "changed"; mutations.append(wrong)
        for mutated in mutations:
            with self.assertRaises(ValueError):
                runner.validate_record(mutated, expected, self.training, model="GCN", family="graph")

    def test_extra_scope_json_is_rejected(self):
        config = {"conditions": ["raw"], "datasets": {"Toy": {}}, "models": ["GCN"], "seeds": [0]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "records" / "raw" / "Toy" / "GCN").mkdir(parents=True)
            (root / "records" / "raw" / "Toy" / "GCN" / "seed_000.json").write_text("{}", encoding="utf-8")
            extra = root / "records" / "unexpected.json"
            extra.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                runner._check_output_contents(root, config)

    def test_completed_resume_is_read_only_and_incomplete_scope_is_rejected_before_training(self):
        config_path = runner.ROOT / "configs/graph_parameterization_diagnostic_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        digest = runner.config_sha256(config) if hasattr(runner, "config_sha256") else __import__("experiments.run_prospective_benchmark", fromlist=["config_sha256"]).config_sha256(config)
        environment = {"device": "cpu", "python": "test"}
        source = "c" * 40
        fingerprint = runner.source_fingerprint(config_path, digest)
        manifest = {"schema_version": "1.0", "run_id": config["run_id"], "config_sha256": digest, "source_commit": source, "source_fingerprint": fingerprint, "environment": environment, "analysis_status": config["analysis_status"], "expected_records": 420, "expected_trials": 1680, "test_evaluations_after_selection": 0, "config": config}
        complete = {"status": "complete", "run_id": config["run_id"], "expected_records": 420, "expected_trials": 1680, "config_sha256": digest, "source_fingerprint": fingerprint, "test_evaluations_after_selection": 0}

        def make_root(root, missing=False):
            (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
            expected = list(runner._expected_relpaths(config))
            if missing:
                expected = expected[:-1]
            for relative in expected:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_root(root)
            argv = ["runner", "--data-root", str(root), "--output-root", str(root), "--device", "cpu", "--resume"]
            with mock.patch.object(runner, "_source_commit", return_value=source), mock.patch.object(runner, "environment_snapshot", return_value=environment), mock.patch.object(runner, "_prevalidate_existing") as prevalidate, mock.patch.object(runner, "run_validation_unit", side_effect=AssertionError("training was attempted")):
                before = {path: path.stat().st_mtime_ns for path in root.rglob("*.json")}
                with mock.patch("sys.argv", argv):
                    runner.main()
                after = {path: path.stat().st_mtime_ns for path in root.rglob("*.json")}
                self.assertEqual(before, after)
                prevalidate.assert_called_once()
            incomplete = root / "incomplete"
            incomplete.mkdir()
            make_root(incomplete, missing=True)
            with mock.patch.object(runner, "_source_commit", return_value=source), mock.patch.object(runner, "environment_snapshot", return_value=environment), mock.patch.object(runner, "run_validation_unit", side_effect=AssertionError("training was attempted")):
                with mock.patch("sys.argv", ["runner", "--data-root", str(root), "--output-root", str(incomplete), "--device", "cpu", "--resume"]):
                    with self.assertRaises(ValueError):
                        runner.main()
            self.assertFalse(list(incomplete.glob("failure_*.json")))
            wrong = root / "wrong"
            wrong.mkdir()
            (wrong / "run_manifest.json").write_text(json.dumps({"run_id": "other"}), encoding="utf-8")
            with mock.patch.object(runner, "_source_commit", return_value=source), mock.patch.object(runner, "environment_snapshot", return_value=environment):
                with mock.patch("sys.argv", ["runner", "--data-root", str(root), "--output-root", str(wrong), "--device", "cpu", "--resume"]):
                    with self.assertRaises(ValueError):
                        runner.main()
            self.assertFalse(list(wrong.glob("failure_*.json")))


if __name__ == "__main__":
    unittest.main()
