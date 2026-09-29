"""MLP 24-candidate extension of a completed, immutable four-trial run.

The old validator remains unchanged.  Four checkpoint references are inherited
read-only, twenty checkpoints are trained by the frozen trainer, and a fresh
post-selection test evaluation is recorded for the validation-selected winner.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import torch

from experiments.prospective_models import build_model
from experiments.run_prospective_benchmark import _accuracy, _train_trial, seed_everything
from scripts.input_robustness_checkpoint_store import load_checkpoint, save_checkpoint, verify_checkpoint
from scripts.input_robustness_data import array_sha256
from scripts.input_robustness_formal_records import (
    CONDITIONS, DATASETS, MODELS, FormalRecordError, _exclusive_json,
    digest, file_digest, record_key, validate_record,
)
from scripts.validate_input_robustness_formal_records import validate_complete_run

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mlp-budget-extension/1.0"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FormalRecordError(f"JSON object required: {path}")
    return value


def build_trial_grid() -> list[dict[str, Any]]:
    inherited = [(0.01, 0.5, 0.0005), (0.01, 0.7, 0.0005),
                 (0.005, 0.5, 0.0005), (0.005, 0.7, 0.0005)]
    additional = [row for row in itertools.product((0.001, 0.005, 0.01),
                  (0.0, 0.3, 0.5, 0.7), (0.0, 0.0005)) if row not in inherited]
    return [{"trial_id": f"trial_{i:03d}", "learning_rate": lr,
             "dropout": dropout, "weight_decay": wd}
            for i, (lr, dropout, wd) in enumerate(inherited + additional)]


def validate_extension_config(config: Mapping[str, Any], base_config: Mapping[str, Any]) -> None:
    if config.get("formal_training_enabled") is not False:
        raise FormalRecordError("extension config must keep formal training disabled")
    mode = config.get("execution_mode", "formal")
    if mode not in {"formal", "research", "isolated_fixture"}:
        raise FormalRecordError("unknown extension execution mode")
    for name in ("datasets", "conditions", "seeds"):
        if config.get(name) != base_config.get(name) or not config.get(name):
            raise FormalRecordError(f"extension {name} must equal the complete base scope")
    if mode != "isolated_fixture" and (config["datasets"] != list(DATASETS)
            or config["conditions"] != list(CONDITIONS) or config["seeds"] != list(range(10))
            or base_config.get("models") != list(MODELS)):
        raise FormalRecordError("research extension requires all eleven datasets and paired seeds")
    training = config["training"]
    if training.get("trials") != build_trial_grid():
        raise FormalRecordError("MLP extension grid or stable trial identifiers differ")
    for name in ("hidden_channels", "max_epochs", "patience"):
        if training.get(name) != base_config["training"].get(name):
            raise FormalRecordError(f"extension changed the inherited {name}")
    if mode != "isolated_fixture" and any(training[name] != expected for name, expected in
                                  (("hidden_channels", 64), ("max_epochs", 500), ("patience", 100))):
        raise FormalRecordError("research extension training recipe differs")
    for field in ("device", "workers", "torch_num_threads", "torch_num_interop_threads",
                  "deterministic_algorithms", "deterministic_warn_only", "cublas_workspace_config",
                  "allow_tf32", "cudnn_benchmark", "cudnn_deterministic", "h2gcn_checkpoint_mode"):
        if config.get("execution", {}).get(field) != base_config.get("execution", {}).get(field):
            raise FormalRecordError(f"extension runtime differs from the base runtime: {field}")
    for model in ("MLP", "H2GCN"):
        if config["execution"].get("model_devices", {}).get(model) != base_config["execution"].get("model_devices", {}).get(model):
            raise FormalRecordError(f"extension device differs from base: {model}")
    if mode == "isolated_fixture" and config["execution"].get("model_devices", {}).get("MLP") != "cpu":
        raise FormalRecordError("isolated fixtures must use the CPU")
    old = base_config["training"]
    if old.get("weight_decay") != 0.0005 or old.get("trials") != [
            {k: row[k] for k in ("learning_rate", "dropout")} for row in build_trial_grid()[:4]]:
        raise FormalRecordError("base four-trial grid is not the approved inherited grid")


def load_base_evidence(base_root: Path, *, config_path: Path, binding_path: Path,
                       extension_config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the actual complete old tree, not a supplied success boolean."""
    base_root, config_path, binding_path = map(Path, (base_root, config_path, binding_path))
    base_config = read_json(config_path)
    validate_extension_config(extension_config, base_config)
    summary = validate_complete_run(base_root, config_path=config_path,
                                    data_binding_path=binding_path, synthetic=False)
    manifest = read_json(base_root / "manifest.json")
    declared = extension_config.get("base", {})
    actual = {"source_commit": manifest["source_commit"], "config_sha256": digest(base_config),
              "data_binding_sha256": file_digest(binding_path)}
    for field, value in actual.items():
        if declared.get(field) != value:
            raise FormalRecordError(f"base {field} does not match the extension authority")
    source_files = {}
    if extension_config.get("execution_mode", "formal") != "isolated_fixture":
        for relative in ("experiments/run_prospective_benchmark.py", "experiments/prospective_models.py"):
            historical = subprocess.check_output(["git", "show", f"{actual['source_commit']}:{relative}"], cwd=ROOT)
            # Git normalizes text newlines; compare canonical working content.
            current = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
            if historical.replace(b"\r\n", b"\n") != current:
                raise FormalRecordError(f"frozen trainer/model source changed: {relative}")
            source_files[relative] = hashlib.sha256(current).hexdigest()
    records, estimates = {}, {}
    for path in sorted((base_root / "records").rglob("*.json")):
        row = read_json(path)
        if row["model"] == "MLP":
            key = identity(row["dataset"], row["condition"], row["seed"])
            records[key] = {"path": path.relative_to(base_root).as_posix(), "sha256": file_digest(path)}
            # Conservative bounded estimate, not a guarantee; all actual
            # attempts are still stopped by the cumulative ledger caps.
            epoch_max = max(float(t["duration_seconds"]) / int(t["epochs_completed"]) for t in row["trials"])
            overhead = max(0.0, row["duration_seconds"] - sum(t["duration_seconds"] for t in row["trials"]))
            estimates[key] = 2 * (20 * epoch_max * extension_config["training"]["max_epochs"] + overhead)
    return {"root": str(base_root.resolve()), "config_path": str(config_path.resolve()),
            "binding_path": str(binding_path.resolve()), **actual,
            "manifest_sha256": file_digest(base_root / "manifest.json"),
            "complete_sha256": file_digest(base_root / "complete.json"),
            "record_digest": summary["record_digest"], "record_count": summary["record_count"],
            "mlp_records": records, "unit_estimates_seconds": estimates, "training_source_files": source_files}


def identity(dataset: str, condition: str, seed: int) -> str:
    return f"{dataset}/{condition}/{int(seed):03d}"


def base_record(evidence: Mapping[str, Any], dataset: str, condition: str, seed: int) -> dict[str, Any]:
    root = Path(evidence["root"])
    for filename, field in (("manifest.json", "manifest_sha256"), ("complete.json", "complete_sha256")):
        if file_digest(root / filename) != evidence[field]:
            raise FormalRecordError(f"base {filename} changed after validation")
    ref = evidence["mlp_records"][identity(dataset, condition, seed)]
    path = (root / ref["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or file_digest(path) != ref["sha256"]:
        raise FormalRecordError("base MLP record changed or escapes its root")
    row = read_json(path)
    validate_record(row, expected=read_json(root / "manifest.json"), synthetic=False)
    if record_key(row) != (dataset, condition, "MLP", int(seed)):
        raise FormalRecordError("base MLP record identity mismatch")
    for item in row["checkpoint_manifest"]:
        verify_checkpoint(root, item)
    return row


def load_fixture_tensors(evidence: Mapping[str, Any], dataset: str, condition: str, seed: int) -> dict[str, torch.Tensor]:
    """Read an isolated fixture from its authoritative, hash-bound data file."""
    binding_path = Path(evidence["binding_path"])
    entry = read_json(binding_path)["datasets"][dataset]
    specification = entry["fixture_tensors"][f"{condition}/{seed}"]
    path = (binding_path.parent / specification["path"]).resolve()
    if not path.is_relative_to(binding_path.parent.resolve()) or file_digest(path) != specification["sha256"]:
        raise FormalRecordError("fixture tensor file path/hash differs from binding")
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    if set(tensors) != set(specification["tensor_sha256"]) or any(
            array_sha256(value) != specification["tensor_sha256"][key] for key, value in tensors.items()):
        raise FormalRecordError("fixture tensor values differ from binding")
    return tensors


def verify_unit_inputs(evidence: Mapping[str, Any], *, dataset: str, condition: str, seed: int,
                       tensors: Mapping[str, torch.Tensor], fixture: bool) -> None:
    binding_path = Path(evidence["binding_path"])
    if file_digest(binding_path) != evidence["data_binding_sha256"]:
        raise FormalRecordError("data binding changed after base validation")
    entry = read_json(binding_path)["datasets"][dataset]
    if fixture:
        expected = entry["fixture_tensors"][f"{condition}/{seed}"]["tensor_sha256"]
    else:
        expected = {"y": entry["tensor_sha256"]["y"], "edge_index": entry["tensor_sha256"]["edge_index"],
                    **{f"{part}_indices": entry["tensor_sha256"][f"seed_{seed}_{part}"]
                       for part in ("train", "validation", "test")}}
    if any(key not in tensors or array_sha256(tensors[key]) != value for key, value in expected.items()):
        raise FormalRecordError("actual labels, graph, or partitions differ from the authoritative binding")


def select_extension_trial(rows: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(rows) != 24 or [row["trial_id"] for row in rows] != [row["trial_id"] for row in build_trial_grid()]:
        raise FormalRecordError("exactly 24 ordered unique trials are required")
    return min(rows, key=lambda row: (-float(row["validation_accuracy"]),
                                     float(row["validation_loss"]), row["trial_id"]))


def run_mlp24_unit(*, base_evidence: Mapping[str, Any], config: Mapping[str, Any],
                   source_commit: str, dataset: str, condition: str, seed: int,
                   output_root: Path, checkpoint_dir: Path, x: torch.Tensor,
                   y: torch.Tensor, edge_index: torch.Tensor, train_indices: torch.Tensor,
                   validation_indices: torch.Tensor, test_indices: torch.Tensor,
                   split_id: str, transform_binding: Mapping[str, Any],
                   environment: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    old = base_record(base_evidence, dataset, condition, seed)
    validate_extension_config(config, read_json(Path(base_evidence["config_path"])))
    if old["environment"] != dict(environment) or str(device) != environment.get("device"):
        raise FormalRecordError("runtime environment differs from inherited checkpoints")
    if old["split_id"] != split_id or old["transform_binding"] != dict(transform_binding):
        raise FormalRecordError("split or transform differs from inherited checkpoints")
    if array_sha256(x) != transform_binding["transformed_feature_sha256"]:
        raise FormalRecordError("actual feature tensor differs from the bound transform")
    verify_unit_inputs(base_evidence, dataset=dataset, condition=condition, seed=seed,
        tensors={"x": x, "y": y, "edge_index": edge_index, "train_indices": train_indices,
                 "validation_indices": validation_indices, "test_indices": test_indices},
        fixture=config.get("execution_mode") == "isolated_fixture")
    checkpoint_dir, output_root = Path(checkpoint_dir), Path(output_root)
    if not checkpoint_dir.resolve().is_relative_to(output_root.resolve()):
        raise FormalRecordError("new checkpoints must stay inside the extension output")
    rows, manifests = [], []
    for index, specification in enumerate(build_trial_grid()):
        trial_id = specification["trial_id"]
        if index < 4:
            row = copy.deepcopy(old["trials"][index])
            checkpoint = copy.deepcopy(old["checkpoint_manifest"][index])
            row["configuration"] = {k: specification[k] for k in ("learning_rate", "dropout", "weight_decay")}
            row["origin"] = "base"
            checkpoint["origin"] = "base"
        else:
            candidate = {k: specification[k] for k in ("learning_rate", "dropout", "weight_decay")}
            row, state = _train_trial(
                model_id="MLP", seed=int(seed), trial_id=trial_id, trial=candidate,
                x=x, y=y, edge_index=edge_index, train_indices=train_indices,
                validation_indices=validation_indices, device=device, h2_adjacencies=None,
                hidden_channels=config["training"]["hidden_channels"],
                max_epochs=config["training"]["max_epochs"], patience=config["training"]["patience"],
                weight_decay=specification["weight_decay"],
            )
            row["origin"] = "extension"
            checkpoint = save_checkpoint(checkpoint_dir, trial_id, state)
            checkpoint["path"] = (checkpoint_dir / checkpoint["path"]).relative_to(output_root).as_posix()
            checkpoint["origin"] = "extension"
        rows.append(row)
        manifests.append(checkpoint)
    selected = select_extension_trial(rows)
    checkpoint = next(item for item in manifests if item["trial_id"] == selected["trial_id"])
    checkpoint_root = Path(base_evidence["root"]) if checkpoint["origin"] == "base" else output_root
    verify_checkpoint(checkpoint_root, checkpoint)
    state = load_checkpoint(checkpoint_root / checkpoint["path"])
    seed_everything(int(seed))
    model = build_model("MLP", num_nodes=x.size(0), in_channels=x.size(1),
                        hidden_channels=config["training"]["hidden_channels"],
                        out_channels=int(y.max().item()) + 1,
                        dropout=selected["configuration"]["dropout"], edge_index=edge_index,
                        h2_adjacencies=None).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        test_accuracy = _accuracy(model(x.to(device), edge_index.to(device)),
                                  y.to(device), test_indices.to(device))
    return {"schema_version": SCHEMA, "status": "success", "run_id": config["run_id"],
            "dataset": dataset, "condition": condition, "model": "MLP", "seed": int(seed),
            "source_commit": source_commit, "config_sha256": digest(config),
            "base_evidence_sha256": digest(base_evidence), "base_record_sha256":
                base_evidence["mlp_records"][identity(dataset, condition, seed)]["sha256"],
            "split_id": split_id, "transform_binding": dict(transform_binding),
            "environment": dict(environment), "training_configuration": dict(config["training"]),
            "diagnostics": old["diagnostics"], "data_provenance": old["data_provenance"],
            "data_binding_sha256": base_evidence["data_binding_sha256"],
            "trials": rows, "checkpoint_manifest": manifests,
            "selected_trial_id": selected["trial_id"],
            "validation_accuracy": selected["validation_accuracy"],
            "validation_loss": selected["validation_loss"], "test_accuracy": test_accuracy,
            "test_evaluation": {"selected_trial_id": selected["trial_id"], "count": 1,
                                "kind": "fresh_post_selection", "checkpoint_sha256": checkpoint["sha256"]},
            "inherited_trials": 4, "new_training_trials": 20, "candidate_count": 24,
            "test_evaluations_after_selection": 1, "duration_seconds": time.perf_counter() - started}


def validate_extension_record(record: Mapping[str, Any], *, config: Mapping[str, Any],
                              evidence: Mapping[str, Any], root: Path, source_commit: str) -> None:
    if record.get("schema_version") != SCHEMA or record.get("status") != "success":
        raise FormalRecordError("invalid extension record schema/status")
    dataset, condition, model, seed = record_key(record)
    if (dataset not in config["datasets"] or condition not in config["conditions"] or model != "MLP"
            or seed not in config["seeds"]):
        raise FormalRecordError("unexpected extension record identity")
    old = base_record(evidence, dataset, condition, seed)
    bindings = {"run_id": config["run_id"], "source_commit": source_commit,
                "config_sha256": digest(config), "base_evidence_sha256": digest(evidence),
                "base_record_sha256": evidence["mlp_records"][identity(dataset, condition, seed)]["sha256"],
                "data_binding_sha256": evidence["data_binding_sha256"],
                "training_configuration": config["training"], "inherited_trials": 4,
                "new_training_trials": 20, "candidate_count": 24, "test_evaluations_after_selection": 1}
    for field in ("environment", "split_id", "transform_binding", "diagnostics", "data_provenance"):
        bindings[field] = old[field]
    for field, expected in bindings.items():
        if record.get(field) != expected:
            raise FormalRecordError(f"extension record binding mismatch: {field}")
    rows = record.get("trials", [])
    selected = select_extension_trial(rows)
    checkpoints = record.get("checkpoint_manifest", [])
    if len(checkpoints) != 24 or [item.get("trial_id") for item in checkpoints] != [row["trial_id"] for row in build_trial_grid()]:
        raise FormalRecordError("all 24 ordered checkpoint manifests are required")
    for index, (row, checkpoint, specification) in enumerate(zip(rows, checkpoints, build_trial_grid())):
        expected_config = {k: specification[k] for k in ("learning_rate", "dropout", "weight_decay")}
        origin = "base" if index < 4 else "extension"
        if row.get("configuration") != expected_config or row.get("origin") != origin or checkpoint.get("origin") != origin:
            raise FormalRecordError("trial grid/origin mismatch")
        if any(name.startswith("test_") for name in row):
            raise FormalRecordError("test results are forbidden in tuning trials")
        for field in ("validation_accuracy", "validation_loss", "duration_seconds"):
            if not isinstance(row.get(field), (float, int)) or not math.isfinite(row[field]):
                raise FormalRecordError(f"nonfinite/missing trial {field}")
        if not 0 <= row["validation_accuracy"] <= 1 or row["duration_seconds"] < 0:
            raise FormalRecordError("invalid trial metric range")
        history = row.get("history", [])
        if (not history or row.get("epochs_completed") != len(history)
                or len(history) > config["training"]["max_epochs"]):
            raise FormalRecordError("missing or invalid full trial history")
        for epoch, item in enumerate(history):
            if item.get("epoch") != epoch or any(
                    not isinstance(item.get(field), (int, float)) or not math.isfinite(item[field])
                    for field in ("train_loss", "validation_loss", "validation_accuracy")):
                raise FormalRecordError("nonfinite or nonsequential full trial history")
        best = min(history, key=lambda item: (-item["validation_accuracy"], item["validation_loss"], item["epoch"]))
        if any(row.get(field) != best[key] for field, key in
               (("best_epoch", "epoch"), ("validation_accuracy", "validation_accuracy"), ("validation_loss", "validation_loss"))):
            raise FormalRecordError("trial winner does not match its full validation history")
        if index < 4:
            inherited = copy.deepcopy(old["trials"][index])
            inherited.update(configuration=expected_config, origin="base")
            if row != inherited or checkpoint != {**old["checkpoint_manifest"][index], "origin": "base"}:
                raise FormalRecordError("inherited trial/checkpoint was modified")
        verify_checkpoint(Path(evidence["root"]) if index < 4 else root, checkpoint)
    if record.get("selected_trial_id") != selected["trial_id"] or any(
            record.get(field) != selected[field] for field in ("validation_accuracy", "validation_loss")):
        raise FormalRecordError("24-candidate selection rule mismatch")
    winner = next(item for item in checkpoints if item["trial_id"] == selected["trial_id"])
    if record.get("test_evaluation") != {"selected_trial_id": selected["trial_id"], "count": 1,
                                         "kind": "fresh_post_selection", "checkpoint_sha256": winner["sha256"]}:
        raise FormalRecordError("test evaluation is not bound to the selected real checkpoint")
    if not isinstance(record.get("test_accuracy"), (float, int)) or not 0 <= record["test_accuracy"] <= 1:
        raise FormalRecordError("invalid selected test accuracy")


def validate_extension_run(root: Path, *, config_path: Path, base_root: Path,
                           base_config_path: Path, binding_path: Path,
                           require_complete: bool = True, allow_partial: bool = False) -> dict[str, Any]:
    root = Path(root)
    config = read_json(config_path)
    evidence = load_base_evidence(base_root, config_path=base_config_path,
                                 binding_path=binding_path, extension_config=config)
    manifest = read_json(root / "manifest.json")
    if (manifest.get("schema_version") != SCHEMA or manifest.get("config_sha256") != digest(config)
            or manifest.get("base_evidence") != evidence or manifest.get("run_id") != config["run_id"]):
        raise FormalRecordError("extension manifest differs from authoritative evidence")
    source = manifest.get("source_commit", "")
    if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
        raise FormalRecordError("extension source commit is missing")
    expected = {(d, c, "MLP", int(s)) for d in config["datasets"] for c in config["conditions"] for s in config["seeds"]}
    entries, records, seen = [], [], set()
    for path in sorted((root / "records").rglob("*.json")):
        row = read_json(path)
        key = record_key(row)
        if key in seen:
            raise FormalRecordError("duplicate extension record")
        canonical_path = root / "records" / key[1] / key[0] / f"seed_{key[3]:03d}.json"
        if path.resolve() != canonical_path.resolve():
            raise FormalRecordError("extension record path/identity mismatch")
        validate_extension_record(row, config=config, evidence=evidence, root=root, source_commit=source)
        seen.add(key)
        entries.append({"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)})
        records.append(row)
    if (seen - expected) or (not allow_partial and seen != expected):
        raise FormalRecordError(f"extension scope incomplete or unexpected: {len(seen)}/{len(expected)}")
    summary = {"schema_version": SCHEMA, "status": "complete", "run_id": config["run_id"],
               "record_count": len(records), "candidate_count": 24 * len(records),
               "new_training_trials": 20 * len(records), "test_evaluations": len(records),
               "record_digest": digest(entries), "config_sha256": digest(config),
               "base_evidence_sha256": digest(evidence), "source_commit": source}
    if require_complete and read_json(root / "complete.json") != summary:
        raise FormalRecordError("complete marker does not match the validated records")
    return {**summary, "records": records}


class ExtensionRecordWriter:
    """Append-only extension writer; inherited records and checkpoints stay untouched."""
    synthetic = False

    def __init__(self, root: Path, *, config_path: Path, base_evidence: Mapping[str, Any],
                 source_commit: str, resume: bool = False):
        self.root, self.config_path = Path(root), Path(config_path)
        self.config = read_json(config_path)
        self.evidence = dict(base_evidence)
        self.manifest = {"schema_version": SCHEMA, "run_id": self.config["run_id"],
                         "source_commit": source_commit, "config_sha256": digest(self.config),
                         "base_evidence": self.evidence}
        if resume:
            if (self.root / "run_complete.json").exists() or read_json(self.root / "manifest.json") != self.manifest:
                raise FormalRecordError("resume differs from immutable extension manifest or is complete")
        else:
            if self.root.exists():
                raise FormalRecordError("extension output must be a new directory")
            _exclusive_json(self.root / "manifest.json", self.manifest)

    def write_record(self, row: Mapping[str, Any]) -> Path:
        validate_extension_record(row, config=self.config, evidence=self.evidence, root=self.root,
                                  source_commit=self.manifest["source_commit"])
        path = self.root / "records" / row["condition"] / row["dataset"] / f"seed_{int(row['seed']):03d}.json"
        _exclusive_json(path, row)
        return path

    def write_failure(self, row: Mapping[str, Any], *, identity: str) -> Path:
        path = self.root / "failures" / f"{identity}.json"
        _exclusive_json(path, row)
        return path

    def finalize(self) -> dict[str, Any]:
        summary = validate_extension_run(self.root, config_path=self.config_path,
            base_root=Path(self.evidence["root"]), base_config_path=Path(self.evidence["config_path"]),
            binding_path=Path(self.evidence["binding_path"]), require_complete=False)
        summary.pop("records")
        if (self.root / "complete.json").exists():
            if read_json(self.root / "complete.json") != summary:
                raise FormalRecordError("existing complete marker differs from revalidated records")
        else:
            _exclusive_json(self.root / "complete.json", summary)
        return summary
