"""Run the fixed, validation-only MLP optimization diagnostic.

This runner deliberately does not accept test indices.  It reuses the frozen
training recipe and trial implementation, but records only train and
validation predictions from the selected checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from experiments.prospective_models import build_model
from experiments.run_prospective_benchmark import (
    _source_commit,
    _train_trial,
    config_sha256,
    environment_snapshot,
    select_trial,
    seed_everything,
    write_json_exclusive,
)
from scripts.run_preprocessing_sensitivity import load_inputs, transform_features


AUDIT_PATH = ROOT / "results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json"
SOURCE_FILES = (
    "scripts/run_mlp_optimization_diagnostic.py",
    "scripts/run_preprocessing_sensitivity.py",
    "experiments/run_prospective_benchmark.py",
    "experiments/prospective_data.py",
    "experiments/prospective_models.py",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint(config_path: Path, config_digest: str) -> dict[str, Any]:
    """Return executable/config hashes used to bind manifests and records."""
    files = {
        relative: _sha256_file(ROOT / relative)
        for relative in SOURCE_FILES
    }
    files["config"] = _sha256_file(config_path)
    return {"files": files, "config_sha256": config_digest}


def _as_float64_rows(x: torch.Tensor, indices: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float64, copy=False)[
        indices.detach().cpu().numpy()
    ]


def fit_transform_features(
    x: torch.Tensor,
    train_indices: torch.Tensor,
    condition: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply unchanged PyG NormalizeFeatures, then fit added affine stats on train rows."""
    valid = {"raw", "normalize_features", "normalize_scaled", "normalize_centered", "normalize_centered_scaled"}
    if condition not in valid:
        raise ValueError(f"unknown preprocessing condition: {condition}")
    normalized = transform_features(x, "normalize_features")
    train_raw = _as_float64_rows(x, train_indices)
    train_normalized = _as_float64_rows(normalized, train_indices)
    raw_mean = train_raw.mean(axis=0, dtype=np.float64)
    normalized_mean = train_normalized.mean(axis=0, dtype=np.float64)
    raw_rms = float(np.sqrt(np.mean((train_raw - raw_mean) ** 2, dtype=np.float64)))
    normalized_rms = float(np.sqrt(np.mean((train_normalized - normalized_mean) ** 2, dtype=np.float64)))
    if not (
        np.isfinite(raw_rms)
        and np.isfinite(normalized_rms)
        and raw_rms > 1e-12
        and normalized_rms > 1e-12
    ):
        raise ValueError("nonfinite or near-zero train feature RMS")
    scale = raw_rms / normalized_rms
    if not np.isfinite(scale):
        raise ValueError("nonfinite fitted feature scale")

    raw64 = x.detach().cpu().numpy().astype(np.float64, copy=False)
    normalized64 = normalized.detach().cpu().numpy().astype(np.float64, copy=False)
    if condition == "raw":
        output = raw64
    elif condition == "normalize_features":
        output = normalized64
    elif condition == "normalize_scaled":
        output = normalized64 * scale
    elif condition == "normalize_centered":
        output = normalized64 - normalized_mean
    else:
        output = (normalized64 - normalized_mean) * scale
    output_tensor = torch.from_numpy(np.asarray(output, dtype=np.float32).copy())
    metadata = {
        "condition": condition,
        "fit_partition": "train_features_only",
        "statistics_dtype": "float64",
        "output_dtype": "float32",
        "epsilon": 1e-12,
        "fitted_normalized_mean": normalized_mean.tolist(),
        "fitted_scale": float(scale),
        "feature_statistics": {
            "raw_train_centered_rms": raw_rms,
            "normalized_train_centered_rms": normalized_rms,
        },
    }
    return output_tensor, metadata


def _count_dict(values: torch.Tensor, n_classes: int) -> dict[str, int]:
    counts = torch.bincount(values.detach().cpu().long(), minlength=n_classes)
    return {str(index): int(counts[index].item()) for index in range(n_classes)}


def partition_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    training_majority_class: int,
    n_classes: int,
) -> dict[str, Any]:
    """Compute diagnostics for one non-test partition."""
    local_logits = logits[indices]
    local_labels = labels[indices]
    predictions = local_logits.argmax(dim=1)
    loss = float(F.cross_entropy(local_logits, local_labels).item())
    accuracy = float((predictions == local_labels).float().mean().item())
    label_counts = _count_dict(local_labels, n_classes)
    prediction_counts = _count_dict(predictions, n_classes)
    recalls: dict[str, float | None] = {}
    supported_recalls: list[float] = []
    for class_id in range(n_classes):
        support = int((local_labels == class_id).sum().item())
        if support == 0:
            recalls[str(class_id)] = None
        else:
            recall = float(((predictions == class_id) & (local_labels == class_id)).sum().item() / support)
            recalls[str(class_id)] = recall
            supported_recalls.append(recall)
    dominant_class = max(range(n_classes), key=lambda cls: (prediction_counts[str(cls)], -cls))
    n = int(local_labels.numel())
    return {
        "loss": loss,
        "accuracy": accuracy,
        "label_counts": label_counts,
        "prediction_counts": prediction_counts,
        "recall_by_class": recalls,
        "balanced_accuracy": float(np.mean(supported_recalls)) if supported_recalls else None,
        "training_majority_class": int(training_majority_class),
        "training_majority_accuracy": float((local_labels == training_majority_class).float().mean().item()),
        "dominant_prediction_class": int(dominant_class),
        "dominant_prediction_fraction": float(prediction_counts[str(dominant_class)] / n) if n else None,
        "n": n,
    }


def _effective_training(config: dict[str, Any], weight_decay: float) -> dict[str, Any]:
    training = dict(config["training"])
    training["weight_decay"] = float(weight_decay)
    return training


def run_validation_unit(
    *,
    run_id: str,
    dataset: str,
    model_id: str,
    seed: int,
    split_id: str,
    condition: str,
    regularization: str,
    weight_decay: float,
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    training: dict[str, Any],
    source_commit: str,
    environment: dict[str, Any],
    data_provenance: dict[str, Any],
    transform_metadata: dict[str, Any],
    device: torch.device,
    h2_adjacencies: tuple[torch.Tensor, torch.Tensor] | None = None,
    config_sha256_value: str = "test-only",
    frozen_config: dict[str, Any] | None = None,
    train_trial_fn: Callable[..., tuple[dict[str, Any], dict[str, torch.Tensor]]] | None = None,
    model_builder: Callable[..., torch.nn.Module] | None = None,
) -> dict[str, Any]:
    """Train/select/evaluate one condition and decay without test access."""
    train_trial_fn = train_trial_fn or _train_trial
    model_builder = model_builder or build_model
    started = time.perf_counter()
    trial_rows: list[dict[str, Any]] = []
    states: dict[str, dict[str, torch.Tensor]] = {}
    for index, trial in enumerate(training["trials"]):
        trial_id = f"trial_{index:03d}"
        row, state = train_trial_fn(
            model_id=model_id, seed=seed, trial_id=trial_id, trial=trial,
            x=x, y=y, edge_index=edge_index, train_indices=train_indices,
            validation_indices=validation_indices,
            hidden_channels=int(training["hidden_channels"]),
            max_epochs=int(training["max_epochs"]), patience=int(training["patience"]),
            weight_decay=float(weight_decay), device=device,
            h2_adjacencies=h2_adjacencies,
        )
        trial_rows.append(row)
        states[trial_id] = state
    selected = select_trial(trial_rows)
    selected_configuration = selected["configuration"]
    seed_everything(seed)
    model = model_builder(
        model_id, num_nodes=x.size(0), in_channels=x.size(1),
        hidden_channels=int(training["hidden_channels"]),
        out_channels=int(y.max().item()) + 1,
        dropout=float(selected_configuration["dropout"]), edge_index=edge_index,
        h2_adjacencies=h2_adjacencies,
    ).to(device)
    model.load_state_dict(states[selected["trial_id"]])
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device), edge_index.to(device)).detach().cpu()
    labels = y.detach().cpu()
    train_cpu = train_indices.detach().cpu()
    validation_cpu = validation_indices.detach().cpu()
    train_labels = labels[train_cpu]
    n_classes = int(labels.max().item()) + 1
    majority_counts = torch.bincount(train_labels, minlength=n_classes)
    training_majority_class = int(majority_counts.argmax().item())
    partitions = {
        "train": partition_metrics(logits, labels, train_cpu, training_majority_class, n_classes),
        "validation": partition_metrics(logits, labels, validation_cpu, training_majority_class, n_classes),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "success",
        "dataset": dataset,
        "model": model_id,
        "family": "mlp",
        "seed": int(seed),
        "split_id": split_id,
        "condition": condition,
        "preprocessing": condition,
        "regularization": regularization,
        "weight_decay": float(weight_decay),
        "validation_accuracy": float(selected["validation_accuracy"]),
        "validation_loss": float(selected["validation_loss"]),
        "selected_trial_id": selected["trial_id"],
        "trials": trial_rows,
        "partitions": partitions,
        "training_configuration": dict(training),
        "test_evaluations_after_selection": 0,
        "source_commit": source_commit,
        "source_fingerprint": data_provenance.get("source_fingerprint"),
        "environment": environment,
        "data_provenance": data_provenance,
        "transform_metadata": transform_metadata,
        "config_sha256": config_sha256_value,
        "frozen_config": frozen_config,
        "duration_seconds": time.perf_counter() - started,
    }


def _finite(value: Any, *, allow_none: bool = False) -> bool:
    if value is None:
        return allow_none
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_record(
    row: dict[str, Any],
    expected: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one completed row; return a compact identity for callers."""
    required = ("schema_version", "run_id", "dataset", "model", "family", "seed", "split_id",
                "condition", "preprocessing", "regularization", "weight_decay", "source_commit",
                "source_fingerprint", "environment", "data_provenance", "config_sha256",
                "frozen_config", "transform_metadata")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"missing record fields: {missing}")
    if row.get("schema_version") != "1.0" or row.get("model") != "MLP" or row.get("family") != "mlp":
        raise ValueError("record schema/model identity mismatch")
    if row.get("preprocessing") != row.get("condition") or row.get("condition") not in {
        "raw", "normalize_features", "normalize_scaled", "normalize_centered", "normalize_centered_scaled"
    }:
        raise ValueError("record condition identity mismatch")
    if row.get("status") != "success":
        raise ValueError("record is not successful")
    if row["schema_version"] != "1.0" or row["model"] != "MLP" or row["family"] != "mlp":
        raise ValueError("record schema or model family mismatch")
    if row["preprocessing"] != row["condition"]:
        raise ValueError("preprocessing condition mismatch")
    if expected:
        mismatches = [key for key, value in expected.items() if row.get(key) != value]
        if mismatches:
            raise ValueError(f"record provenance mismatch: {mismatches}")
    if row.get("test_evaluations_after_selection") != 0:
        raise ValueError("test evaluations must be zero")
    forbidden_test_fields = [key for key in row if key.startswith("test_") and key != "test_evaluations_after_selection"]
    if forbidden_test_fields:
        raise ValueError(f"test metrics must not be recorded: {forbidden_test_fields}")
    if not _finite(row.get("weight_decay")) or float(row["weight_decay"]) < 0 or not _finite(row.get("duration_seconds")):
        raise ValueError("invalid training scalar")
    if training is None:
        training = row.get("training_configuration")
    if not isinstance(training, dict):
        raise ValueError("missing training configuration")
    if row.get("training_configuration") != training:
        raise ValueError("training configuration mismatch")
    trials = row.get("trials")
    if not isinstance(trials, list) or len(trials) != len(training["trials"]):
        raise ValueError("trial count mismatch")
    for index, (trial, configuration) in enumerate(zip(trials, training["trials"])):
        if trial.get("trial_id") != f"trial_{index:03d}" or trial.get("configuration") != configuration:
            raise ValueError("trial grid mismatch")
        history = trial.get("history")
        if not isinstance(history, list) or not history:
            raise ValueError("missing trial history")
        if len(history) != trial.get("epochs_completed"):
            raise ValueError("history length mismatch")
        if len(history) > int(training["max_epochs"]):
            raise ValueError("history exceeds max epochs")
        for epoch, entry in enumerate(history):
            if not isinstance(entry, dict) or not _is_nonnegative_int(entry.get("epoch")):
                raise ValueError("invalid history epoch")
            if entry["epoch"] != epoch:
                raise ValueError("history epochs are not contiguous")
            for key in ("train_loss", "validation_loss", "validation_accuracy"):
                if not _finite(entry.get(key)):
                    raise ValueError("nonfinite history metric")
            if not 0.0 <= float(entry["validation_accuracy"]) <= 1.0:
                raise ValueError("history accuracy out of range")
            if float(entry["train_loss"]) < 0 or float(entry["validation_loss"]) < 0:
                raise ValueError("history loss must be nonnegative")
        for key in ("validation_accuracy", "validation_loss", "best_epoch", "epochs_completed", "duration_seconds"):
            if not _finite(trial.get(key)):
                raise ValueError("nonfinite trial metric")
        if not _is_nonnegative_int(trial.get("best_epoch")) or not _is_nonnegative_int(trial.get("epochs_completed")):
            raise ValueError("invalid trial epoch audit")
        if trial["best_epoch"] >= len(history):
            raise ValueError("best epoch outside history")
        best_history = min(history, key=lambda entry: (-float(entry["validation_accuracy"]), float(entry["validation_loss"]), int(entry["epoch"])))
        if trial["best_epoch"] != best_history["epoch"]:
            raise ValueError("best checkpoint does not match history selection")
        if not math.isclose(float(trial["validation_accuracy"]), float(best_history["validation_accuracy"]), abs_tol=1e-12):
            raise ValueError("trial accuracy does not match history")
        if not math.isclose(float(trial["validation_loss"]), float(best_history["validation_loss"]), abs_tol=1e-12):
            raise ValueError("trial loss does not match history")
        if len(history) < int(training["max_epochs"]):
            if len(history) - 1 - int(trial["best_epoch"]) < int(training["patience"]):
                raise ValueError("early stopping does not satisfy patience")
    selected = select_trial(trials)
    if row.get("selected_trial_id") != selected["trial_id"]:
        raise ValueError("selected trial mismatch")
    for key in ("validation_accuracy", "validation_loss"):
        if not math.isclose(float(row.get(key)), float(selected[key]), abs_tol=1e-12):
            raise ValueError(f"selected {key} mismatch")
    partitions = row.get("partitions")
    if set(partitions or {}) != {"train", "validation"}:
        raise ValueError("partition scope mismatch")
    for name, metrics in partitions.items():
        if not isinstance(metrics, dict) or not _finite(metrics.get("loss")) or not _finite(metrics.get("accuracy")):
            raise ValueError(f"invalid {name} metrics")
        if not 0.0 <= float(metrics["accuracy"]) <= 1.0 or float(metrics["loss"]) < 0:
            raise ValueError(f"{name} metric out of range")
        n = int(metrics.get("n", -1))
        if n <= 0 or not _is_nonnegative_int(metrics.get("n")):
            raise ValueError(f"invalid {name} size")
        label_counts = metrics.get("label_counts")
        prediction_counts = metrics.get("prediction_counts")
        if not isinstance(label_counts, dict) or not isinstance(prediction_counts, dict) or set(label_counts) != set(prediction_counts):
            raise ValueError(f"{name} count keys mismatch")
        for key in ("label_counts", "prediction_counts"):
            counts = metrics.get(key)
            if (not isinstance(counts, dict) or
                    not all(_is_nonnegative_int(v) for v in counts.values()) or
                    sum(counts.values()) != n):
                raise ValueError(f"{name} count total mismatch")
        recalls = metrics.get("recall_by_class")
        if not isinstance(recalls, dict) or set(recalls) != set(label_counts):
            raise ValueError(f"invalid {name} recalls")
        supported = []
        for class_id, value in recalls.items():
            if label_counts[class_id] == 0:
                if value is not None:
                    raise ValueError(f"unsupported class recall must be null: {name}")
            else:
                if not _finite(value) or not 0.0 <= float(value) <= 1.0:
                    raise ValueError(f"invalid {name} recall")
                supported.append(float(value))
        for key in ("balanced_accuracy", "training_majority_accuracy", "dominant_prediction_fraction"):
            if not _finite(metrics.get(key)) or not 0.0 <= float(metrics[key]) <= 1.0:
                raise ValueError(f"invalid {name} {key}")
        if not math.isclose(float(metrics["balanced_accuracy"]), float(np.mean(supported)), abs_tol=1e-7):
            raise ValueError(f"{name} balanced accuracy mismatch")
        weighted_recall = sum(label_counts[class_id] * float(recalls[class_id]) for class_id in recalls if recalls[class_id] is not None) / n
        if not math.isclose(float(metrics["accuracy"]), weighted_recall, abs_tol=1e-7):
            raise ValueError(f"{name} accuracy/recall mismatch")
        majority = int(metrics.get("training_majority_class", -1))
        if not _is_nonnegative_int(majority) or str(majority) not in label_counts:
            raise ValueError(f"invalid {name} majority class")
        train_metrics = partitions["train"]
        if majority != int(train_metrics.get("training_majority_class", -1)):
            raise ValueError("partition majority class mismatch")
        if not math.isclose(float(metrics["training_majority_accuracy"]), label_counts[str(majority)] / n, abs_tol=1e-7):
            raise ValueError(f"{name} majority baseline mismatch")
        dominant = int(metrics.get("dominant_prediction_class", -1))
        if not _is_nonnegative_int(dominant) or str(dominant) not in prediction_counts:
            raise ValueError(f"invalid {name} dominant class")
        if prediction_counts[str(dominant)] != max(prediction_counts.values()):
            raise ValueError(f"{name} dominant class mismatch")
        if not math.isclose(float(metrics["dominant_prediction_fraction"]), prediction_counts[str(dominant)] / n, abs_tol=1e-7):
            raise ValueError(f"{name} dominant fraction mismatch")
    if not math.isclose(float(partitions["validation"]["accuracy"]), float(row["validation_accuracy"]), abs_tol=1e-7):
        raise ValueError("selected validation accuracy differs from partition")
    if abs(float(partitions["validation"]["loss"]) - float(row["validation_loss"])) > 1e-6:
        raise ValueError("selected validation loss differs from partition")
    train_counts = partitions["train"]["label_counts"]
    expected_majority = max((int(class_id) for class_id in train_counts), key=lambda class_id: (train_counts[str(class_id)], -class_id))
    if int(partitions["train"]["training_majority_class"]) != expected_majority:
        raise ValueError("training majority class is not derived from train labels")
    if not isinstance(row["transform_metadata"].get("fitted_scale"), (int, float)):
        raise ValueError("fitted scale must be scalar")
    if not _finite(row["transform_metadata"]["fitted_scale"]):
        raise ValueError("invalid fitted scale")
    metadata = row.get("transform_metadata")
    if not isinstance(metadata, dict) or metadata.get("fit_partition") != "train_features_only":
        raise ValueError("missing train-fitted transform metadata")
    for key in ("fitted_normalized_mean",):
        values = metadata.get(key)
        if not isinstance(values, list) or not values or not all(_finite(v) for v in values):
            raise ValueError("invalid fitted transform vector")
    return {"dataset": row.get("dataset"), "seed": row.get("seed"), "condition": row.get("condition"), "regularization": row.get("regularization")}


def _load_audit() -> dict[str, Any]:
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def _expected_record_relpaths(config: dict[str, Any]) -> set[str]:
    return {
        str(Path("records") / condition / dataset / regularization / f"seed_{seed:03d}.json")
        for dataset in config["datasets"]
        for seed in config["seeds"]
        for condition in config["conditions"]
        for regularization in config["regularization"]
    }


def _check_output_contents(output_root: Path, config: dict[str, Any]) -> None:
    """Reject stale failure artifacts or semantic JSON outside the fixed scope."""
    expected = _expected_record_relpaths(config)
    for path in output_root.rglob("*.json"):
        relative = str(path.relative_to(output_root))
        if path.name.startswith("failure_"):
            raise ValueError(f"failure artifact requires a fresh output root: {path}")
        if relative in {"run_manifest.json", "complete.json"}:
            continue
        if relative not in expected:
            raise ValueError(f"unexpected JSON artifact in output root: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mlp_optimization_diagnostic_v1.json")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    source = _source_commit()
    environment = environment_snapshot(device)
    digest = config_sha256(config)
    fingerprint = source_fingerprint(args.config, digest)
    audit = _load_audit()
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
    complete_path = args.output_root / "complete.json"
    if complete_path.exists() and (
        not args.resume or json.loads(complete_path.read_text(encoding="utf-8")) != {
            "status": "complete", "run_id": config["run_id"], "expected_records": config["expected_records"],
            "expected_trials": config["expected_trials"], "config_sha256": digest,
            "source_fingerprint": fingerprint, "test_evaluations_after_selection": 0,
        }
    ):
        raise ValueError("complete output exists; resume requires an exact completed artifact")
    try:
        for name, specification in config["datasets"].items():
            x, y, edge_index, splits = load_inputs(args.data_root, name, specification, audit)
            h2 = None
            for seed in config["seeds"]:
                split, split_id = splits[seed]
                train_indices = torch.as_tensor(split["train"], dtype=torch.long)
                validation_indices = torch.as_tensor(split["validation"], dtype=torch.long)
                for condition in config["conditions"]:
                    transformed, metadata = fit_transform_features(x, train_indices, condition)
                    transformed_digest = hashlib.sha256(transformed.numpy().tobytes()).hexdigest()
                    for regularization, weight_decay in config["regularization"].items():
                        path = args.output_root / "records" / condition / name / regularization / f"seed_{seed:03d}.json"
                        source_data = {
                            "raw_filename": specification["filename"],
                            "raw_sha256": specification["sha256"],
                            "split_id": split_id,
                            "transformed_feature_sha256": transformed_digest,
                            "source_fingerprint": fingerprint,
                        }
                        expected = {
                            "run_id": config["run_id"], "dataset": name, "model": "MLP", "seed": seed,
                            "split_id": split_id, "condition": condition, "regularization": regularization,
                            "weight_decay": float(weight_decay), "source_commit": source,
                            "source_fingerprint": fingerprint, "environment": environment,
                            "config_sha256": digest, "frozen_config": config, "data_provenance": source_data,
                            "transform_metadata": metadata,
                            "schema_version": "1.0", "family": "mlp", "preprocessing": condition,
                        }
                        if path.exists():
                            if not args.resume:
                                raise ValueError(f"record already exists; explicit resume required: {path}")
                            existing = json.loads(path.read_text(encoding="utf-8"))
                            validate_record(existing, expected, _effective_training(config, weight_decay))
                            continue
                        row = run_validation_unit(
                            run_id=config["run_id"], dataset=name, model_id="MLP", seed=seed,
                            split_id=split_id, condition=condition, regularization=regularization,
                            weight_decay=float(weight_decay), x=transformed, y=y, edge_index=edge_index,
                            train_indices=train_indices, validation_indices=validation_indices,
                            training=_effective_training(config, weight_decay), source_commit=source,
                            environment=environment, data_provenance=source_data,
                            transform_metadata=metadata, device=device, h2_adjacencies=h2,
                            config_sha256_value=digest, frozen_config=config,
                        )
                        validate_record(row, expected, _effective_training(config, weight_decay))
                        write_json_exclusive(path, row)
                        print(json.dumps({"completed": str(path.relative_to(args.output_root)), "seconds": row["duration_seconds"]}), flush=True)
            del x, y, edge_index
        complete = {
            "status": "complete", "run_id": config["run_id"], "expected_records": config["expected_records"],
            "expected_trials": config["expected_trials"], "config_sha256": digest,
            "source_fingerprint": fingerprint, "test_evaluations_after_selection": 0,
        }
        if complete_path.exists():
            if json.loads(complete_path.read_text(encoding="utf-8")) != complete:
                raise ValueError("existing complete artifact does not match current run")
        else:
            write_json_exclusive(complete_path, complete)
    except Exception as exc:
        write_json_exclusive(args.output_root / f"failure_{time.time_ns()}.json", {"run_id": config["run_id"], "status": "error", "type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
