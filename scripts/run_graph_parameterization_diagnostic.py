"""Run the fixed, validation-only graph input parameterization diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.prospective_models import prepare_h2_adjacencies
from experiments.run_prospective_benchmark import (
    _source_commit,
    config_sha256,
    environment_snapshot,
    write_json_exclusive,
)
from scripts.run_mlp_optimization_diagnostic import (
    AUDIT_PATH,
    fit_transform_features,
    load_inputs,
    run_validation_unit,
)
from scripts.validation_diagnostic_common import validate_record


SOURCE_FILES = (
    "scripts/run_graph_parameterization_diagnostic.py",
    "scripts/validation_diagnostic_common.py",
    "scripts/run_mlp_optimization_diagnostic.py",
    "scripts/run_preprocessing_sensitivity.py",
    "experiments/run_prospective_benchmark.py",
    "experiments/prospective_data.py",
    "experiments/prospective_models.py",
)
GRAPH_MODELS = ("MLP", "GCN", "GAT", "GraphSAGE", "H2GCN", "LINKX", "GPR-GNN")
CONDITIONS = {"raw", "normalize_features", "normalize_centered_scaled"}
ORDERED_CONDITIONS = ("raw", "normalize_features", "normalize_centered_scaled")
_FAILURE_OUTPUT: Path | None = None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint(config_path: Path, digest: str) -> dict[str, Any]:
    files = {relative: _sha256_file(ROOT / relative) for relative in SOURCE_FILES}
    files["config"] = _sha256_file(config_path)
    return {"files": files, "config_sha256": digest}


def _family(model: str) -> str:
    return "mlp" if model == "MLP" else "graph"


def _expected_relpaths(config: dict[str, Any]) -> dict[str, tuple[str, str, str, int]]:
    return {
        str(Path("records") / condition / dataset / model / f"seed_{int(seed):03d}.json"): (condition, dataset, model, int(seed))
        for condition in config["conditions"]
        for dataset in config["datasets"]
        for model in config["models"]
        for seed in config["seeds"]
    }


def _check_output_contents(output_root: Path, config: dict[str, Any]) -> None:
    expected = _expected_relpaths(config)
    for path in output_root.rglob("*.json"):
        relative = str(path.relative_to(output_root))
        if path.name.startswith("failure_"):
            raise ValueError(f"failure artifact requires a fresh output root: {path}")
        if relative in {"run_manifest.json", "complete.json"}:
            continue
        if relative not in expected:
            raise ValueError(f"unexpected JSON artifact in output root: {path}")


def _effective_training(config: dict[str, Any]) -> dict[str, Any]:
    training = dict(config["training"])
    training["weight_decay"] = float(config["regularization"]["wd_5e-4"])
    return training


def _expected_row(
    *, config: dict[str, Any], dataset: str, model: str, seed: int, split_id: str,
    condition: str, source: str, fingerprint: dict[str, Any], environment: dict[str, Any],
    digest: str, frozen_config: dict[str, Any], provenance: dict[str, Any], metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": config["run_id"], "dataset": dataset, "model": model, "family": _family(model),
        "seed": int(seed), "split_id": split_id, "condition": condition, "preprocessing": condition,
        "regularization": "wd_5e-4", "weight_decay": 0.0005, "source_commit": source,
        "source_fingerprint": fingerprint, "environment": environment, "config_sha256": digest,
        "frozen_config": frozen_config, "data_provenance": provenance, "transform_metadata": metadata,
    }


def _prevalidate_existing(
    *, config: dict[str, Any], data_root: Path, output_root: Path, audit: dict[str, Any],
    source: str, fingerprint: dict[str, Any], environment: dict[str, Any], digest: str,
) -> None:
    """Validate every existing record and its reconstructed input before training starts."""
    training = _effective_training(config)
    for dataset, specification in config["datasets"].items():
        x, y, edge_index, splits = load_inputs(data_root, dataset, specification, audit)
        del edge_index
        for seed in config["seeds"]:
            split, split_id = splits[int(seed)]
            train_indices = torch.as_tensor(split["train"], dtype=torch.long)
            for condition in config["conditions"]:
                transformed, metadata = fit_transform_features(x, train_indices, condition)
                transformed_hash = hashlib.sha256(transformed.numpy().tobytes()).hexdigest()
                provenance = {
                    "raw_filename": specification["filename"], "raw_sha256": specification["sha256"],
                    "split_id": split_id, "transformed_feature_sha256": transformed_hash,
                    "source_fingerprint": fingerprint,
                }
                for model in config["models"]:
                    path = output_root / "records" / condition / dataset / model / f"seed_{int(seed):03d}.json"
                    if not path.exists():
                        continue
                    row = json.loads(path.read_text(encoding="utf-8"))
                    expected = _expected_row(config=config, dataset=dataset, model=model, seed=int(seed), split_id=split_id, condition=condition, source=source, fingerprint=fingerprint, environment=environment, digest=digest, frozen_config=config, provenance=provenance, metadata=metadata)
                    validate_record(row, expected, training, model=model, family=_family(model))
                    expected_sizes = {"train": len(split["train"]), "validation": len(split["validation"])}
                    for partition, indices in (("train", split["train"]), ("validation", split["validation"])):
                        metrics = row["partitions"][partition]
                        if metrics["n"] != expected_sizes[partition]:
                            raise ValueError(f"{path}: {partition} size does not match frozen split")
                        labels = y[torch.as_tensor(indices, dtype=torch.long)]
                        counts = torch.bincount(labels, minlength=int(y.max().item()) + 1)
                        expected_counts = {str(index): int(counts[index].item()) for index in range(len(counts))}
                        if metrics["label_counts"] != expected_counts:
                            raise ValueError(f"{path}: {partition} label counts do not match frozen labels")
        del x, y


def main() -> None:
    global _FAILURE_OUTPUT
    _FAILURE_OUTPUT = None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/graph_parameterization_diagnostic_v1.json")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if set(config.get("datasets", {})) != {"Roman-empire", "Amazon-ratings"} or tuple(config.get("seeds", ())) != tuple(range(10)):
        raise ValueError("graph diagnostic dataset/seed scope does not match frozen design")
    if tuple(config["models"]) != GRAPH_MODELS or tuple(config["conditions"]) != ORDERED_CONDITIONS:
        raise ValueError("graph diagnostic scope does not match frozen model/condition set")
    if config.get("regularization") != {"wd_5e-4": 0.0005}:
        raise ValueError("graph diagnostic regularization scope does not match frozen design")
    expected_records = len(config["datasets"]) * len(config["seeds"]) * len(config["models"]) * len(config["conditions"])
    expected_trials = expected_records * len(config["training"]["trials"])
    if config.get("expected_records") != expected_records or config.get("expected_trials") != expected_trials:
        raise ValueError("graph diagnostic expected scope counts are inconsistent")
    expected_trials_grid = [
        {"learning_rate": 0.01, "dropout": 0.5}, {"learning_rate": 0.01, "dropout": 0.7},
        {"learning_rate": 0.005, "dropout": 0.5}, {"learning_rate": 0.005, "dropout": 0.7},
    ]
    training_config = config.get("training", {})
    if (training_config.get("hidden_channels") != 64 or training_config.get("max_epochs") != 500 or
            training_config.get("patience") != 100 or training_config.get("weight_decay") != 0.0005 or
            training_config.get("trials") != expected_trials_grid or
            training_config.get("selection_order") != ["validation_accuracy_desc", "validation_loss_asc", "trial_id_asc"] or
            training_config.get("test_evaluations_after_selection") != 0):
        raise ValueError("graph diagnostic training recipe does not match frozen design")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    source = _source_commit()
    environment = environment_snapshot(device)
    digest = config_sha256(config)
    fingerprint = source_fingerprint(args.config.resolve(), digest)
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "1.0", "run_id": config["run_id"], "config_sha256": digest,
        "source_commit": source, "source_fingerprint": fingerprint, "environment": environment,
        "analysis_status": config["analysis_status"], "expected_records": config["expected_records"],
        "expected_trials": config["expected_trials"], "test_evaluations_after_selection": 0,
        "config": config,
    }
    manifest_path = args.output_root / "run_manifest.json"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("output root belongs to another run or resume was not requested")
    else:
        if args.output_root.exists() and any(args.output_root.iterdir()):
            raise ValueError("nonempty output root has no matching manifest")
        write_json_exclusive(manifest_path, manifest)
    _check_output_contents(args.output_root, config)
    complete = {
        "status": "complete", "run_id": config["run_id"], "expected_records": config["expected_records"],
        "expected_trials": config["expected_trials"], "config_sha256": digest,
        "source_fingerprint": fingerprint, "test_evaluations_after_selection": 0,
    }
    complete_path = args.output_root / "complete.json"
    if complete_path.exists() and (not args.resume or json.loads(complete_path.read_text(encoding="utf-8")) != complete):
        raise ValueError("complete output exists; resume requires exact completed artifact")
    if complete_path.exists():
        expected_paths = set(_expected_relpaths(config))
        actual_paths = {str(path.relative_to(args.output_root)) for path in (args.output_root / "records").rglob("*.json")}
        if actual_paths != expected_paths:
            raise ValueError("completed output does not have exact record scope")

    # This pass performs all input/statistics/provenance validation before any new training.
    _prevalidate_existing(config=config, data_root=args.data_root, output_root=args.output_root, audit=audit, source=source, fingerprint=fingerprint, environment=environment, digest=digest)

    _FAILURE_OUTPUT = args.output_root
    if complete_path.exists():
        return

    training = _effective_training(config)
    for dataset, specification in config["datasets"].items():
        x, y, edge_index, splits = load_inputs(args.data_root, dataset, specification, audit)
        h2 = prepare_h2_adjacencies(edge_index, num_nodes=len(y))
        for seed in config["seeds"]:
            split, split_id = splits[int(seed)]
            train_indices = torch.as_tensor(split["train"], dtype=torch.long)
            validation_indices = torch.as_tensor(split["validation"], dtype=torch.long)
            for condition in config["conditions"]:
                transformed, metadata = fit_transform_features(x, train_indices, condition)
                transformed_hash = hashlib.sha256(transformed.numpy().tobytes()).hexdigest()
                provenance = {
                    "raw_filename": specification["filename"], "raw_sha256": specification["sha256"],
                    "split_id": split_id, "transformed_feature_sha256": transformed_hash,
                    "source_fingerprint": fingerprint,
                }
                for model in config["models"]:
                    path = args.output_root / "records" / condition / dataset / model / f"seed_{int(seed):03d}.json"
                    expected = _expected_row(config=config, dataset=dataset, model=model, seed=int(seed), split_id=split_id, condition=condition, source=source, fingerprint=fingerprint, environment=environment, digest=digest, frozen_config=config, provenance=provenance, metadata=metadata)
                    if path.exists():
                        if not args.resume:
                            raise ValueError(f"record already exists; explicit resume required: {path}")
                        continue
                    row = run_validation_unit(
                        run_id=config["run_id"], dataset=dataset, model_id=model, seed=int(seed),
                        split_id=split_id, condition=condition, regularization="wd_5e-4", weight_decay=0.0005,
                        x=transformed, y=y, edge_index=edge_index, train_indices=train_indices,
                        validation_indices=validation_indices, training=training, source_commit=source,
                        environment=environment, data_provenance=provenance, transform_metadata=metadata,
                        device=device, h2_adjacencies=h2, config_sha256_value=digest, frozen_config=config,
                    )
                    row["family"] = _family(model)
                    validate_record(row, expected, training, model=model, family=_family(model))
                    write_json_exclusive(path, row)
                    print(json.dumps({"completed": str(path.relative_to(args.output_root)), "seconds": row["duration_seconds"]}), flush=True)
        del x, y, edge_index, h2
    if complete_path.exists():
        if json.loads(complete_path.read_text(encoding="utf-8")) != complete:
            raise ValueError("existing complete artifact does not match current run")
    else:
        write_json_exclusive(complete_path, complete)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Preserve a visible failure marker after manifest ownership is established.
        try:
            if _FAILURE_OUTPUT is not None:
                write_json_exclusive(_FAILURE_OUTPUT / f"failure_{time.time_ns()}.json", {
                    "status": "error", "type": type(exc).__name__, "message": str(exc),
                })
        except Exception:
            pass
        raise
