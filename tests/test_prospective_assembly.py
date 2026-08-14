import json
import tempfile
import unittest
import hashlib
from pathlib import Path

from scripts.assemble_prospective_diagnostics import assemble_payload


MODELS = ["MLP", "GCN", "GAT", "GraphSAGE", "H2GCN", "LINKX", "GPR-GNN"]


def write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ProspectiveAssemblyTests(unittest.TestCase):
    def make_fixture(self, root: Path, *, run_id: str = "route_a_prospective_v1"):
        config = {
            "datasets": ["toy"],
            "seeds": [0],
            "models": MODELS,
            "run_id": run_id,
            "training": {
                "trials": [
                    {"learning_rate": 0.01, "dropout": 0.5},
                    {"learning_rate": 0.01, "dropout": 0.7},
                    {"learning_rate": 0.005, "dropout": 0.5},
                    {"learning_rate": 0.005, "dropout": 0.7},
                ]
            },
            "evaluator": {
                "specification_version": "route_a_diagnostic_v1",
                "practical_margin": 0.01,
                "bootstrap": {"samples": 10, "seed": 1},
                "permutation": {"samples": 10, "seed": 2},
            },
        }
        config_sha = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for model in MODELS:
            write(
                root / "records" / "toy" / model / "seed_000.json",
                {
                    "status": "success",
                    "dataset": "toy",
                    "seed": 0,
                    "split_id": "same-split",
                    "model": model,
                    "family": "mlp" if model == "MLP" else "graph",
                    "validation_accuracy": 0.6,
                    "validation_loss": 0.5,
                    "test_accuracy": 0.55,
                    "source_commit": "abc",
                    "environment": {"python": "test"},
                    "data_provenance": {"processed_files": []},
                    "trials": [
                        {
                            "trial_id": f"trial_{index:03d}",
                            "configuration": config["training"]["trials"][index],
                            "validation_accuracy": 0.6 - index * 0.01,
                            "validation_loss": 0.5 + index * 0.01,
                        }
                        for index in range(4)
                    ],
                    "test_evaluations_after_selection": 1,
                    "selected_trial_id": "trial_000",
                    "config_sha256": config_sha,
                    "frozen_config": config,
                    "run_id": run_id,
                },
            )
        write(
            root / "diagnostics" / "toy" / "seed_000.json",
            {
                "status": "success",
                "dataset": "toy",
                "seed": 0,
                "split_id": "same-split",
                "homophily": 0.5,
                "mean_degree": 4.0,
                "delta_h": 0.1,
                "provenance": {
                    "uses_test_labels": False,
                    "homophily_label_scope": "train_only",
                    "two_hop_label_scope": "train_only",
                    "source_commit": "abc",
                    "environment": {"python": "test"},
                    "data": {"processed_files": []},
                    "config_sha256": config_sha,
                    "frozen_config": config,
                },
                "run_id": run_id,
            },
        )
        return config

    def test_complete_records_assemble_to_the_frozen_evaluator_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.make_fixture(root)
            payload = assemble_payload(root, config)
        self.assertEqual(payload["benchmark_id"], "route_a_diagnostic_v1")
        self.assertEqual(len(payload["model_records"]), 7)
        self.assertEqual(len(payload["diagnostic_records"]), 1)
        self.assertEqual(payload["graph_models"], MODELS[1:])

    def test_v2_records_assemble_only_under_the_matching_v2_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v2_config = self.make_fixture(root, run_id="route_a_prospective_v2")
            payload = assemble_payload(root, v2_config)
            self.assertEqual(len(payload["model_records"]), 7)

            mismatched = {**v2_config, "run_id": "route_a_prospective_v1"}
            with self.assertRaises(ValueError):
                assemble_payload(root, mismatched)

    def test_missing_failed_inconsistent_or_leaky_records_are_rejected(self):
        cases = ("missing", "failed", "split", "leakage")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = self.make_fixture(root)
                gcn = root / "records" / "toy" / "GCN" / "seed_000.json"
                diagnostic = root / "diagnostics" / "toy" / "seed_000.json"
                if case == "missing":
                    gcn.unlink()
                elif case == "failed":
                    payload = json.loads(gcn.read_text())
                    payload["status"] = "error"
                    write(gcn, payload)
                elif case == "split":
                    payload = json.loads(gcn.read_text())
                    payload["split_id"] = "wrong"
                    write(gcn, payload)
                else:
                    payload = json.loads(diagnostic.read_text())
                    payload["provenance"]["uses_test_labels"] = True
                    write(diagnostic, payload)
                with self.assertRaises((FileNotFoundError, ValueError)):
                    assemble_payload(root, config)

    def test_mixed_or_incomplete_provenance_is_rejected(self):
        cases = (
            "run_id",
            "diagnostic_run_id",
            "commit",
            "trial_count",
            "test_count",
            "config",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = self.make_fixture(root)
                gcn = root / "records" / "toy" / "GCN" / "seed_000.json"
                payload = json.loads(gcn.read_text())
                if case == "run_id":
                    payload["run_id"] = "wrong"
                elif case == "diagnostic_run_id":
                    diagnostic = root / "diagnostics" / "toy" / "seed_000.json"
                    diagnostic_payload = json.loads(diagnostic.read_text())
                    diagnostic_payload["run_id"] = "wrong"
                    write(diagnostic, diagnostic_payload)
                elif case == "commit":
                    payload["source_commit"] = "different"
                elif case == "trial_count":
                    payload["trials"] = []
                elif case == "test_count":
                    payload["test_evaluations_after_selection"] = 999
                else:
                    payload["config_sha256"] = "bad"
                write(gcn, payload)
                with self.assertRaises(ValueError):
                    assemble_payload(root, config)


if __name__ == "__main__":
    unittest.main()
