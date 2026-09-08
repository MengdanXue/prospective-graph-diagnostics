"""Strict validation helpers shared by validation-only diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from experiments.run_prospective_benchmark import select_trial


def _finite(value: Any, *, allow_none: bool = False) -> bool:
    if value is None:
        return allow_none
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_history(trial: dict[str, Any], training: dict[str, Any]) -> None:
    history = trial.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("missing trial history")
    if len(history) != trial.get("epochs_completed") or len(history) > int(training["max_epochs"]):
        raise ValueError("history length/max_epochs mismatch")
    for epoch, entry in enumerate(history):
        if not isinstance(entry, dict) or entry.get("epoch") != epoch or not _nonnegative_int(entry.get("epoch")):
            raise ValueError("history epochs are not contiguous")
        for key in ("train_loss", "validation_loss", "validation_accuracy"):
            if not _finite(entry.get(key)):
                raise ValueError("nonfinite history metric")
        if float(entry["train_loss"]) < 0 or float(entry["validation_loss"]) < 0 or not 0 <= float(entry["validation_accuracy"]) <= 1:
            raise ValueError("history metric out of range")
    for key in ("validation_accuracy", "validation_loss", "duration_seconds"):
        if not _finite(trial.get(key)):
            raise ValueError("nonfinite trial metric")
    if not _nonnegative_int(trial.get("best_epoch")) or not _nonnegative_int(trial.get("epochs_completed")):
        raise ValueError("invalid trial epoch audit")
    if trial["best_epoch"] >= len(history):
        raise ValueError("best epoch outside history")
    best = min(history, key=lambda entry: (-float(entry["validation_accuracy"]), float(entry["validation_loss"]), int(entry["epoch"])))
    if trial["best_epoch"] != best["epoch"]:
        raise ValueError("best checkpoint does not match history selection")
    if not math.isclose(float(trial["validation_accuracy"]), float(best["validation_accuracy"]), abs_tol=1e-12):
        raise ValueError("trial accuracy does not match history")
    if not math.isclose(float(trial["validation_loss"]), float(best["validation_loss"]), abs_tol=1e-12):
        raise ValueError("trial loss does not match history")
    if len(history) < int(training["max_epochs"]) and len(history) - 1 - trial["best_epoch"] < int(training["patience"]):
        raise ValueError("early stopping does not satisfy patience")


def _validate_partition(name: str, metrics: dict[str, Any], train_metrics: dict[str, Any] | None) -> None:
    for key in ("loss", "accuracy", "balanced_accuracy", "training_majority_accuracy", "dominant_prediction_fraction"):
        if not _finite(metrics.get(key)):
            raise ValueError(f"invalid {name} {key}")
    if float(metrics["loss"]) < 0 or not 0 <= float(metrics["accuracy"]) <= 1:
        raise ValueError(f"invalid {name} loss/accuracy")
    n = metrics.get("n")
    if not _nonnegative_int(n) or n <= 0:
        raise ValueError(f"invalid {name} size")
    labels = metrics.get("label_counts")
    predictions = metrics.get("prediction_counts")
    recalls = metrics.get("recall_by_class")
    if not isinstance(labels, dict) or not isinstance(predictions, dict) or set(labels) != set(predictions):
        raise ValueError(f"{name} count key mismatch")
    if not all(_nonnegative_int(value) for value in labels.values()) or not all(_nonnegative_int(value) for value in predictions.values()):
        raise ValueError(f"{name} counts must be nonnegative integers")
    if sum(labels.values()) != n or sum(predictions.values()) != n:
        raise ValueError(f"{name} count total mismatch")
    if not isinstance(recalls, dict) or set(recalls) != set(labels):
        raise ValueError(f"{name} recall key mismatch")
    supported: list[float] = []
    for class_id, support in labels.items():
        recall = recalls[class_id]
        if support == 0:
            if recall is not None:
                raise ValueError(f"{name} unsupported recall must be null")
        else:
            if not _finite(recall) or not 0 <= float(recall) <= 1:
                raise ValueError(f"{name} recall out of range")
            supported.append(float(recall))
    if not math.isclose(float(metrics["balanced_accuracy"]), float(np.mean(supported)), abs_tol=1e-7):
        raise ValueError(f"{name} balanced accuracy mismatch")
    weighted_recall = sum(labels[key] * float(recalls[key]) for key in labels if recalls[key] is not None) / n
    if not math.isclose(float(metrics["accuracy"]), weighted_recall, abs_tol=1e-7):
        raise ValueError(f"{name} accuracy/recall mismatch")
    majority = metrics.get("training_majority_class")
    if not _nonnegative_int(majority) or str(majority) not in labels:
        raise ValueError(f"{name} invalid majority class")
    if train_metrics is not None and majority != train_metrics.get("training_majority_class"):
        raise ValueError("partition majority mismatch")
    expected_majority = max((int(class_id) for class_id in labels), key=lambda class_id: (labels[str(class_id)], -class_id))
    if train_metrics is metrics and majority != expected_majority:
        raise ValueError("training majority class is not derived from train labels")
    if not math.isclose(float(metrics["training_majority_accuracy"]), labels[str(majority)] / n, abs_tol=1e-7):
        raise ValueError(f"{name} majority baseline mismatch")
    dominant = metrics.get("dominant_prediction_class")
    if not _nonnegative_int(dominant) or str(dominant) not in predictions or predictions[str(dominant)] != max(predictions.values()):
        raise ValueError(f"{name} invalid dominant class")
    if not math.isclose(float(metrics["dominant_prediction_fraction"]), predictions[str(dominant)] / n, abs_tol=1e-7):
        raise ValueError(f"{name} dominant fraction mismatch")


def validate_record(
    row: dict[str, Any],
    expected: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    *,
    model: str | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    """Validate a complete selected train/validation record and return its key."""
    required = ("schema_version", "run_id", "dataset", "model", "family", "seed", "split_id", "condition", "preprocessing", "regularization", "weight_decay", "source_commit", "source_fingerprint", "environment", "data_provenance", "config_sha256", "frozen_config", "transform_metadata")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"missing record fields: {missing}")
    expected_model = model or (expected or {}).get("model")
    expected_family = family or (expected or {}).get("family")
    if row.get("schema_version") != "1.0" or row.get("model") != expected_model or row.get("family") != expected_family:
        raise ValueError("record schema/model/family mismatch")
    if row.get("preprocessing") != row.get("condition") or row.get("condition") not in {"raw", "normalize_features", "normalize_centered_scaled"}:
        raise ValueError("record condition mismatch")
    if row.get("status") != "success":
        raise ValueError("record is not successful")
    if expected:
        mismatches = [key for key, value in expected.items() if row.get(key) != value]
        if mismatches:
            raise ValueError(f"record provenance mismatch: {mismatches}")
    if row.get("test_evaluations_after_selection") != 0:
        raise ValueError("test evaluations must be zero")
    forbidden = [key for key in row if key.startswith("test_") and key != "test_evaluations_after_selection"]
    if forbidden:
        raise ValueError(f"test metrics must not be recorded: {forbidden}")
    if not _finite(row.get("weight_decay")) or float(row["weight_decay"]) < 0 or not _finite(row.get("duration_seconds")):
        raise ValueError("invalid training scalar")
    if training is None:
        training = row.get("training_configuration")
    if not isinstance(training, dict) or row.get("training_configuration") != training:
        raise ValueError("training configuration mismatch")
    trials = row.get("trials")
    if not isinstance(trials, list) or len(trials) != len(training["trials"]):
        raise ValueError("trial count mismatch")
    for index, (trial, configuration) in enumerate(zip(trials, training["trials"])):
        if trial.get("trial_id") != f"trial_{index:03d}" or trial.get("configuration") != configuration:
            raise ValueError("trial grid mismatch")
        _validate_history(trial, training)
    selected = select_trial(trials)
    if row.get("selected_trial_id") != selected["trial_id"]:
        raise ValueError("selected trial mismatch")
    selection_accuracy = row.get("selection_validation_accuracy", selected["validation_accuracy"])
    selection_loss = row.get("selection_validation_loss", selected["validation_loss"])
    if (not _finite(selection_accuracy) or
            not math.isclose(float(selection_accuracy), float(selected["validation_accuracy"]), abs_tol=1e-12)):
        raise ValueError("selection validation accuracy mismatch")
    if (not _finite(selection_loss) or
            not math.isclose(float(selection_loss), float(selected["validation_loss"]), abs_tol=1e-12)):
        raise ValueError("selection validation loss mismatch")
    for key in ("validation_accuracy", "validation_loss"):
        if not _finite(row.get(key)):
            raise ValueError(f"reported {key} is not finite")
    partitions = row.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation"}:
        raise ValueError("partition scope mismatch")
    _validate_partition("train", partitions["train"], partitions["train"])
    _validate_partition("validation", partitions["validation"], partitions["train"])
    if not math.isclose(float(partitions["validation"]["accuracy"]), float(row["validation_accuracy"]), abs_tol=1e-7):
        raise ValueError("selected validation accuracy differs from partition")
    if abs(float(partitions["validation"]["loss"]) - float(row["validation_loss"])) > 1e-6:
        raise ValueError("selected validation loss differs from partition")
    metadata = row.get("transform_metadata")
    if not isinstance(metadata, dict) or metadata.get("fit_partition") != "train_features_only":
        raise ValueError("missing train-fitted transform metadata")
    if not isinstance(metadata.get("fitted_normalized_mean"), list) or not metadata["fitted_normalized_mean"] or not all(_finite(value) for value in metadata["fitted_normalized_mean"]):
        raise ValueError("invalid fitted mean")
    if not _finite(metadata.get("fitted_scale")):
        raise ValueError("invalid fitted scalar scale")
    return {"dataset": row["dataset"], "model": row["model"], "seed": row["seed"], "condition": row["condition"]}
