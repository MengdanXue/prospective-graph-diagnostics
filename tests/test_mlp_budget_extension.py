"""Real CPU training/checkpoint tests for the isolated 4-to-24 extension."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.run_prospective_benchmark import environment_snapshot
from experiments.prospective_models import prepare_h2_adjacencies
from scripts.input_robustness_data import array_sha256, fit_transform_features_bounded
from scripts.input_robustness_formal_records import (
    CONDITIONS, MODELS, FormalRecordError, FormalRecordWriter, digest, file_digest, run_formal_model_unit,
)
from scripts.mlp_budget_extension import (
    ExtensionRecordWriter, base_record, build_trial_grid, load_base_evidence,
    read_json, run_mlp24_unit, validate_extension_config, validate_extension_run,
)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def build_fixture(parent: Path, *, full_portfolio=False):
    torch.set_num_threads(1)
    tensors = {
        "x": torch.tensor([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.], [0., 0., 1.],
                           [1., 0., 1.], [0., 1., 1.], [1., 1., 1.], [.5, .5, 0.]]),
        "y": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "train_indices": torch.tensor([0, 1, 2, 3]),
        "validation_indices": torch.tensor([4, 5]), "test_indices": torch.tensor([6, 7]),
    }
    datasets = ["Cora", "CiteSeer"] if full_portfolio else ["Cora"]
    models = list(MODELS) if full_portfolio else ["MLP"]
    transformed, fitted = {}, {}
    for condition in CONDITIONS:
        values, metadata = fit_transform_features_bounded(tensors["x"], tensors["train_indices"], condition)
        transformed[condition], fitted[condition] = {**tensors, "x": values}, metadata
    binding = {"bound_split_count": len(datasets), "datasets": {}}
    for dataset in datasets:
        fixture_tensors = {}
        for condition in CONDITIONS:
            tensor_path = parent / f"{dataset}_{condition}.pt"
            torch.save(transformed[condition], tensor_path)
            fixture_tensors[f"{condition}/0"] = {"path": tensor_path.name, "sha256": file_digest(tensor_path),
                "tensor_sha256": {key: array_sha256(value) for key, value in transformed[condition].items()}}
        binding["datasets"][dataset] = {
            "split_checks": [{"seed": 0, "split_id": "fixture-split"}],
            "full_diagnostic_checks": [{"seed": 0, "diagnostics": {"homophily": .4, "mean_degree": 3., "delta_h": .1}}],
            "candidate_checks": [{"seed": 0, "fit_metadata": fitted[condition],
                                  "transformed_feature_sha256": array_sha256(transformed[condition]["x"])} for condition in CONDITIONS],
            "normalize_features_sha256": array_sha256(transformed[CONDITIONS[0]]["x"]),
            "fixture_tensors": fixture_tensors,
        }
    binding_path = parent / "binding.json"
    write(binding_path, binding)
    training = {"hidden_channels": 4, "max_epochs": 1, "patience": 1, "weight_decay": .0005,
                "trials": [{k: row[k] for k in ("learning_rate", "dropout")} for row in build_trial_grid()[:4]]}
    execution = {"device": "cpu", "model_devices": {model: "cpu" for model in MODELS},
                 "workers": 1, "torch_num_threads": 4, "torch_num_interop_threads": 1,
                 "deterministic_algorithms": True, "deterministic_warn_only": False,
                 "cublas_workspace_config": ":4096:8", "allow_tf32": False,
                 "cudnn_benchmark": False, "cudnn_deterministic": True,
                 "h2gcn_checkpoint_mode": "original_full_state_copy"}
    record_count = len(datasets) * len(models) * 2
    config = {"formal_training_enabled": False, "datasets": datasets, "conditions": list(CONDITIONS),
              "models": models, "seeds": [0], "expected_records": record_count, "expected_trials": record_count * 4,
              "training": training, "execution": execution,
              "bound_input_source": {"path": str(binding_path), "sha256": file_digest(binding_path)}}
    config_path = parent / "base_config.json"
    write(config_path, config)
    environment = environment_snapshot(torch.device("cpu"))
    writer = FormalRecordWriter(parent / "base", manifest={
        "run_id": "isolated-base", "source_commit": "a" * 40, "config_sha256": digest(config),
        "data_binding_sha256": file_digest(binding_path), "environment": environment,
        "scope": {"datasets": datasets, "conditions": list(CONDITIONS), "models": models, "seeds": [0]},
        "expected_records": record_count, "expected_trials": record_count * 4,
    }, config_path=config_path, data_binding_path=binding_path)
    for dataset in datasets:
        for model in models:
            h2 = prepare_h2_adjacencies(tensors["edge_index"], num_nodes=len(tensors["y"])) if model == "H2GCN" else None
            for condition in CONDITIONS:
                run_formal_model_unit(writer=writer, launch_authorized=True, formal_training_enabled=True,
                    condition=condition, run_id="isolated-base", dataset=dataset, model_id=model, seed=0,
                    split_id="fixture-split", **transformed[condition], training=training, source_commit="a" * 40,
                    environment=environment, data_provenance={"source": "isolated generated tensors"},
                    device=torch.device("cpu"), config_sha256=digest(config), frozen_config=config,
                    h2_adjacencies=h2,
                    data_binding_sha256=file_digest(binding_path), diagnostics=binding["datasets"][dataset]["full_diagnostic_checks"][0]["diagnostics"],
                    transform_binding={"dataset": dataset, "condition": condition, "seed": 0,
                                       "split_id": "fixture-split", "transformed_feature_sha256": array_sha256(transformed[condition]["x"]),
                                       "fit_statistics_sha256": digest(fitted[condition])})
    writer.finalize()
    extension = {"run_id": "isolated-mlp24", "execution_mode": "isolated_fixture",
                 "formal_training_enabled": False, "datasets": datasets, "conditions": list(CONDITIONS),
                 "seeds": [0], "execution": execution,
                 "training": {k: training[k] for k in ("hidden_channels", "max_epochs", "patience")},
                 "base": {"source_commit": "a" * 40, "config_sha256": digest(config),
                          "data_binding_sha256": file_digest(binding_path)}}
    extension["training"]["trials"] = build_trial_grid()
    extension_path = parent / "extension_config.json"
    write(extension_path, extension)
    evidence = load_base_evidence(writer.root, config_path=config_path, binding_path=binding_path,
                                  extension_config=extension)
    return {"parent": parent, "base_writer": writer, "base_root": writer.root,
            "base_config_path": config_path, "binding_path": binding_path,
            "config": extension, "config_path": extension_path, "evidence": evidence,
            "tensors": transformed, "environment": environment}


def make_extension_writer(fixture, *, name="extension", resume=False):
    return ExtensionRecordWriter(fixture["parent"] / name, config_path=fixture["config_path"],
                                  base_evidence=fixture["evidence"], source_commit="b" * 40, resume=resume)


def run_fixture_unit(fixture, writer, condition):
    old = base_record(fixture["evidence"], "Cora", condition, 0)
    return run_mlp24_unit(base_evidence=fixture["evidence"], config=fixture["config"], source_commit="b" * 40,
        dataset="Cora", condition=condition, seed=0, output_root=writer.root,
        checkpoint_dir=writer.root / "checkpoints" / condition, **fixture["tensors"][condition],
        split_id=old["split_id"], transform_binding=old["transform_binding"],
        environment=fixture["environment"], device=torch.device("cpu"))


def check_fixture(fixture, writer, **kwargs):
    return validate_extension_run(writer.root, config_path=fixture["config_path"],
        base_root=fixture["base_root"], base_config_path=fixture["base_config_path"],
        binding_path=fixture["binding_path"], **kwargs)


class MLPBudgetExtensionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = build_fixture(Path(self.directory.name))

    def test_real_twenty_new_trials_preserve_four_checkpoints_and_test_selected_once(self):
        import scripts.mlp_budget_extension as extension_module
        before = {str(p): file_digest(p) for p in self.fixture["base_root"].rglob("*") if p.is_file()}
        writer = make_extension_writer(self.fixture)
        with patch.object(extension_module, "_train_trial", wraps=extension_module._train_trial) as train, \
                patch.object(extension_module, "_accuracy", wraps=extension_module._accuracy) as test:
            for condition in CONDITIONS:
                row = run_fixture_unit(self.fixture, writer, condition)
                writer.write_record(row)
            self.assertEqual(train.call_count, 40)
            self.assertEqual(test.call_count, 2)
            self.assertEqual({call.kwargs["weight_decay"] for call in train.call_args_list}, {0, .0005})
        complete = writer.finalize()
        result = check_fixture(self.fixture, writer)
        self.assertEqual(complete["new_training_trials"], 40)
        self.assertEqual(result["candidate_count"], 48)
        self.assertEqual(len(list((writer.root / "checkpoints").rglob("*.pt"))), 40)
        self.assertEqual(before, {str(p): file_digest(p) for p in self.fixture["base_root"].rglob("*") if p.is_file()})
        for row in result["records"]:
            self.assertEqual(row["checkpoint_manifest"][:4],
                             [{**entry, "origin": "base"} for entry in base_record(
                                 self.fixture["evidence"], "Cora", row["condition"], 0)["checkpoint_manifest"]])

    def test_missing_duplicate_and_tampered_checkpoint_are_rejected(self):
        writer = make_extension_writer(self.fixture)
        for condition in CONDITIONS:
            writer.write_record(run_fixture_unit(self.fixture, writer, condition))
        writer.finalize()
        path = writer.root / "records" / CONDITIONS[0] / "Cora" / "seed_000.json"
        duplicate = path.with_name("duplicate.json")
        shutil.copyfile(path, duplicate)
        with self.assertRaises(FormalRecordError):
            check_fixture(self.fixture, writer)
        duplicate.unlink()
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(FormalRecordError, "incomplete"):
            check_fixture(self.fixture, writer)
        path.write_bytes(original)
        checkpoint = next((writer.root / "checkpoints").rglob("*.pt"))
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            check_fixture(self.fixture, writer)

    def test_base_completion_and_grid_and_bound_features_cannot_be_faked(self):
        changed = copy.deepcopy(self.fixture["config"])
        changed["training"]["trials"][4]["learning_rate"] = .1
        with self.assertRaisesRegex(FormalRecordError, "grid"):
            validate_extension_config(changed, read_json(self.fixture["base_config_path"]))
        marker = self.fixture["base_root"] / "complete.json"
        write(marker, {"status": "passed"})
        with self.assertRaises(FormalRecordError):
            load_base_evidence(self.fixture["base_root"], config_path=self.fixture["base_config_path"],
                binding_path=self.fixture["binding_path"], extension_config=self.fixture["config"])


if __name__ == "__main__":
    unittest.main()
