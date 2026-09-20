#!/usr/bin/env python3
"""Validate and summarize the validation-only MLP optimization diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_prospective_benchmark import config_sha256
from scripts import run_mlp_optimization_diagnostic as runner
from scripts.validate_preprocessing_records import validate_preprocessing_run


DEFAULT_CONFIG = ROOT / "configs/mlp_optimization_diagnostic_v1.json"
DEFAULT_AUDIT = runner.AUDIT_PATH
PREPROCESSING_CONFIG = ROOT / "configs/preprocessing_sensitivity_v1.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_value(value: Any) -> Any:
    """Convert NumPy scalar/array metadata to the JSON representation on disk."""
    return json.loads(json.dumps(value, allow_nan=False))


def _require(ok: bool, where: str | Path, message: str) -> None:
    if not ok:
        raise ValueError(f"{message}: {where}")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _close(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_close(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_close(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.0)
    return left == right


def _sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _validate_manifest(run_root: Path, config: dict[str, Any], config_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any], str, bool]:
    manifest_path = run_root / "run_manifest.json"
    complete_path = run_root / "complete.json"
    _require(manifest_path.is_file(), manifest_path, "missing run manifest")
    _require(complete_path.is_file(), complete_path, "missing complete marker")
    failures = sorted(run_root.rglob("failure_*.json"))
    _require(not failures, run_root, "failure artifact present: " + ", ".join(str(p.relative_to(run_root)) for p in failures))
    manifest = _load(manifest_path)
    complete = _load(complete_path)
    digest = config_sha256(config)
    derived_records = len(config["datasets"]) * len(config["conditions"]) * len(config["regularization"]) * len(config["seeds"])
    derived_trials = derived_records * len(config["training"]["trials"])
    _require(config.get("expected_records") == derived_records, "config", "expected_records does not match configured scope")
    _require(config.get("expected_trials") == derived_trials, "config", "expected_trials does not match configured scope")
    _require(manifest.get("config") == config, manifest_path, "manifest frozen config mismatch")
    _require(manifest.get("config_sha256") == digest, manifest_path, "manifest config digest mismatch")
    _require(manifest.get("run_id") == config["run_id"], manifest_path, "manifest run_id mismatch")
    _require(manifest.get("analysis_status") == config["analysis_status"], manifest_path, "manifest analysis_status mismatch")
    source = manifest.get("source_commit")
    _require(isinstance(source, str) and _COMMIT.fullmatch(source), manifest_path, "source_commit must be a full SHA")
    environment = manifest.get("environment")
    _require(isinstance(environment, Mapping) and environment, manifest_path, "missing environment")
    fingerprint = manifest.get("source_fingerprint")
    _require(isinstance(fingerprint, Mapping) and isinstance(fingerprint.get("files"), Mapping), manifest_path, "missing source fingerprint")
    _require(fingerprint.get("config_sha256") == digest, manifest_path, "source fingerprint config mismatch")
    _require(all(isinstance(v, str) and _SHA256.fullmatch(v) for v in fingerprint["files"].values()), manifest_path, "invalid source fingerprint")
    for key, expected in (("schema_version", "1.0"), ("run_id", config["run_id"]), ("expected_records", config["expected_records"]),
                          ("expected_trials", config["expected_trials"]), ("test_evaluations_after_selection", 0)):
        _require(manifest.get(key) == expected, manifest_path, f"manifest {key} mismatch")
    for key, expected in (("status", "complete"), ("run_id", config["run_id"]), ("expected_records", config["expected_records"]),
                          ("expected_trials", config["expected_trials"]), ("config_sha256", digest), ("test_evaluations_after_selection", 0)):
        _require(complete.get(key) == expected, complete_path, f"complete marker {key} mismatch")
    _require(complete.get("source_fingerprint") == manifest["source_fingerprint"], complete_path, "complete source fingerprint mismatch")
    source_content_verified = False
    if config_path is not None and config_path.is_file():
        actual_fingerprint = runner.source_fingerprint(config_path, digest)
        _require(actual_fingerprint == manifest["source_fingerprint"], manifest_path, "source fingerprint does not match current executable snapshot")
        source_content_verified = True
    return manifest, complete, digest, source_content_verified


def _expected_paths(run_root: Path, config: Mapping[str, Any]) -> dict[Path, tuple[str, str, str, int]]:
    expected: dict[Path, tuple[str, str, str, int]] = {}
    for condition in config["conditions"]:
        for dataset in config["datasets"]:
            for regularization in config["regularization"]:
                for seed in config["seeds"]:
                    relative = Path("records") / condition / dataset / regularization / f"seed_{int(seed):03d}.json"
                    expected[relative] = (condition, dataset, regularization, int(seed))
    return expected


def _reconstruct_inputs(config: Mapping[str, Any], data_root: Path, audit: Mapping[str, Any], run_root: Path, manifest: Mapping[str, Any]) -> tuple[dict[tuple[str, str, str, int], dict[str, Any]], dict[tuple[str, int], str], dict[str, Any]]:
    records_root = run_root / "records"
    expected_paths = _expected_paths(run_root, config)
    actual_paths = {p.relative_to(run_root) for p in records_root.rglob("*.json")}
    _require(actual_paths == set(expected_paths), records_root, "missing, extra, or duplicate semantic records")
    audit_split_ids: dict[tuple[str, int], str] = {}
    for unit in audit.get("units", []):
        key = (unit.get("dataset"), int(unit.get("seed", -1)))
        if key[0] in config["datasets"] and key[1] in config["seeds"]:
            _require(key not in audit_split_ids, "audit", f"duplicate target unit {key}")
            audit_split_ids[key] = unit["split_id"]
    expected_split_count = len(config["datasets"]) * len(config["seeds"])
    _require(len(audit_split_ids) == expected_split_count, "audit", "missing target split IDs")

    records: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    transform_metadata: dict[tuple[str, int, str], dict[str, Any]] = {}
    transform_hashes: dict[tuple[str, int, str], str] = {}
    for dataset, specification in config["datasets"].items():
        x, y, edge_index, splits = runner.load_inputs(data_root, dataset, specification, audit)
        for seed in config["seeds"]:
            split, split_id = splits[int(seed)]
            _require(split_id == audit_split_ids[(dataset, int(seed))], dataset, "runner split differs from frozen audit")
            train_indices = torch.as_tensor(split["train"], dtype=torch.long)
            for condition in config["conditions"]:
                transformed, expected_metadata = runner.fit_transform_features(x, train_indices, condition)
                expected_metadata = _json_value(expected_metadata)
                expected_hash = hashlib.sha256(transformed.numpy().tobytes()).hexdigest()
                for regularization, weight_decay in config["regularization"].items():
                    key = (condition, dataset, regularization, int(seed))
                    relative = next(path for path, semantic in expected_paths.items() if semantic == key)
                    path = run_root / relative
                    row = _load(path)
                    effective_training = runner._effective_training(config, float(weight_decay))
                    expected_provenance = {
                        "raw_filename": specification["filename"], "raw_sha256": specification["sha256"],
                        "split_id": split_id, "transformed_feature_sha256": expected_hash,
                        "source_fingerprint": manifest["source_fingerprint"],
                    }
                    expected = {
                        "run_id": config["run_id"], "dataset": dataset, "model": "MLP", "seed": int(seed),
                        "split_id": split_id, "condition": condition, "preprocessing": condition,
                        "regularization": regularization, "weight_decay": float(weight_decay),
                        "source_commit": manifest["source_commit"], "source_fingerprint": manifest["source_fingerprint"],
                        "environment": manifest["environment"], "config_sha256": config_sha256(config),
                        "frozen_config": config, "data_provenance": expected_provenance,
                    }
                    try:
                        runner.validate_record(row, expected, effective_training)
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(f"invalid diagnostic record {path}: {exc}") from exc
                    _require(row.get("transform_metadata") == expected_metadata, path, "transform metadata does not match train-fitted reconstruction")
                    _require(row.get("data_provenance", {}).get("transformed_feature_sha256") == expected_hash, path, "transformed feature hash mismatch")
                    _require(row.get("schema_version") == "1.0" and row.get("family") == "mlp", path, "record schema or family mismatch")
                    _require(row.get("model") == "MLP" and row.get("preprocessing") == condition, path, "record model/preprocessing mismatch")
                    n_classes = int(y.max().item()) + 1
                    for partition_name, indices in (("train", split["train"]), ("validation", split["validation"])):
                        partition = row.get("partitions", {}).get(partition_name)
                        labels = y[torch.as_tensor(indices, dtype=torch.long)]
                        counts = torch.bincount(labels, minlength=n_classes)
                        expected_counts = {str(index): int(counts[index].item()) for index in range(n_classes)}
                        _require(isinstance(partition, Mapping) and partition.get("n") == len(indices), path, f"{partition_name} count n mismatch")
                        _require(partition.get("label_counts") == expected_counts, path, f"{partition_name} label counts mismatch")
                    records[key] = row
                    transform_metadata[(dataset, int(seed), condition)] = expected_metadata
                    transform_hashes[(dataset, int(seed), condition)] = expected_hash
        del x, y, edge_index
    return records, audit_split_ids, {"transform_metadata": transform_metadata, "transform_hashes": transform_hashes}


def _metric(row: Mapping[str, Any], metric: str) -> float:
    if metric == "validation_accuracy":
        return float(row["validation_accuracy"])
    if metric == "validation_loss":
        return float(row["validation_loss"])
    metrics = row["partitions"]["validation"]
    return float(metrics[metric.removeprefix("validation_")])


def _aggregate(records: Mapping[tuple[str, str, str, int], dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in config["datasets"]:
        output[dataset] = {}
        for condition in config["conditions"]:
            output[dataset][condition] = {}
            for regularization in config["regularization"]:
                rows = [records[(condition, dataset, regularization, int(seed))] for seed in config["seeds"]]
                validation = [row["partitions"]["validation"] for row in rows]
                train = [row["partitions"]["train"] for row in rows]
                fractions = [float(metrics["dominant_prediction_fraction"]) for metrics in validation]
                majority_all = sum(int(metrics["prediction_counts"].get(str(metrics["training_majority_class"]), 0) == metrics["n"]) for metrics in validation)
                entry: dict[str, Any] = {"n": len(rows), "train": {}, "validation": {}, "validation_all_predictions_training_majority_count": majority_all,
                                         "validation_dominant_prediction_fraction_range": [min(fractions), max(fractions)]}
                for partition_name, metrics, prefix in (("train", train, "train"), ("validation", validation, "validation")):
                    for field in ("accuracy", "loss", "balanced_accuracy", "training_majority_accuracy"):
                        values = [float(item[field]) for item in metrics]
                        entry[partition_name][field] = {"mean": _mean(values), "sample_sd": _sample_sd(values)}
                output[dataset][condition][regularization] = entry
    return output


CONTRAST_METRICS = ("validation_accuracy", "validation_loss", "validation_balanced_accuracy", "validation_dominant_prediction_fraction")


def _contrast(records: Mapping[tuple[str, str, str, int], dict[str, Any]], dataset: str, left: tuple[str, str], right: tuple[str, str], metric: str, seeds: list[int], label: str) -> dict[str, Any]:
    diffs = [_metric(records[(left[0], dataset, left[1], seed)], metric) - _metric(records[(right[0], dataset, right[1], seed)], metric) for seed in seeds]
    return {"label": label, "left_minus_right": f"{left} - {right}", "metric": metric, "n": len(diffs), "differences": [{"seed": seed, "difference": diff} for seed, diff in zip(seeds, diffs)],
            "mean_difference": _mean(diffs), "range": [min(diffs), max(diffs)], "positive_count": sum(diff > 0 for diff in diffs),
            "zero_count": sum(diff == 0 for diff in diffs), "negative_count": sum(diff < 0 for diff in diffs)}


def _contrasts(records: Mapping[tuple[str, str, str, int], dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    seeds = [int(seed) for seed in config["seeds"]]
    wd_names = list(config["regularization"])
    baseline = next(name for name in wd_names if float(config["regularization"][name]) == 0.0005)
    zero = next(name for name in wd_names if float(config["regularization"][name]) == 0.0)
    out: dict[str, Any] = {}
    pairs = (("normalize_features", "raw", "normalized_minus_raw"), ("normalize_scaled", "normalize_features", "scaled_minus_normalized"),
             ("normalize_centered", "normalize_features", "centered_minus_normalized"), ("normalize_centered_scaled", "normalize_scaled", "centered_scaled_minus_scaled"),
             ("normalize_centered_scaled", "normalize_centered", "centered_scaled_minus_centered"))
    for dataset in config["datasets"]:
        out[dataset] = {"by_decay": {}, "weight_decay": {}, "scale_interactions": {}}
        for regularization in wd_names:
            out[dataset]["by_decay"][regularization] = {name: {metric: _contrast(records, dataset, (left, regularization), (right, regularization), metric, seeds, name) for metric in CONTRAST_METRICS} for left, right, name in pairs}
        for condition in config["conditions"]:
            out[dataset]["weight_decay"][condition] = {metric: _contrast(records, dataset, (condition, zero), (condition, baseline), metric, seeds, f"{zero}_minus_{baseline}") for metric in CONTRAST_METRICS}
        for condition, scaled, name in (("normalize_features", "normalize_scaled", "uncentered_scale_effect"), ("normalize_centered", "normalize_centered_scaled", "centered_scale_effect")):
            for metric in CONTRAST_METRICS:
                at_zero = [_metric(records[(scaled, dataset, zero, seed)], metric) - _metric(records[(condition, dataset, zero, seed)], metric) for seed in seeds]
                at_base = [_metric(records[(scaled, dataset, baseline, seed)], metric) - _metric(records[(condition, dataset, baseline, seed)], metric) for seed in seeds]
                diffs = [a - b for a, b in zip(at_zero, at_base)]
                out[dataset]["scale_interactions"][name + "_zero_minus_nonzero_" + metric] = {"metric": metric, "n": len(diffs), "differences": [{"seed": seed, "difference": diff} for seed, diff in zip(seeds, diffs)], "mean_difference": _mean(diffs), "range": [min(diffs), max(diffs)], "positive_count": sum(diff > 0 for diff in diffs), "zero_count": sum(diff == 0 for diff in diffs), "negative_count": sum(diff < 0 for diff in diffs)}
    return out


def _controls(records: Mapping[tuple[str, str, str, int], dict[str, Any]], preprocessing_root: Path, data_root: Path, *, manifest: Mapping[str, Any], transform_hashes: Mapping[tuple[str, int, str], str]) -> dict[str, Any]:
    old_config = _load(PREPROCESSING_CONFIG)
    old_audit = _load(DEFAULT_AUDIT)
    old_result = validate_preprocessing_run(preprocessing_root, config=old_config, audit=old_audit, data_root=data_root)
    old = old_result["records"]
    _require(old_result["metadata"]["environment"] == manifest["environment"], preprocessing_root, "original and diagnostic environments differ")
    results: dict[str, Any] = {}
    for dataset in old_config["datasets"]:
        results[dataset] = {}
        for condition in ("raw", "normalize_features"):
            rows = [((condition, dataset, "wd_5e-4", int(seed)), old[(condition, dataset, int(seed))]["MLP"]) for seed in old_config["seeds"]]
            accuracy_diffs = [float(records[key]["validation_accuracy"]) - float(row["validation_accuracy"]) for key, row in rows]
            loss_diffs = [float(records[key]["validation_loss"]) - float(row["validation_loss"]) for key, row in rows]
            trial_diffs = [{"seed": key[3], "new": records[key]["selected_trial_id"], "original": row["selected_trial_id"], "equal": records[key]["selected_trial_id"] == row["selected_trial_id"]} for key, row in rows]
            hash_equal = []
            for key, row in rows:
                old_hash = row.get("data_provenance", {}).get("transformed_feature_sha256")
                new_hash = transform_hashes[(dataset, key[3], condition)]
                hash_equal.append(old_hash == new_hash)
            _require(all(hash_equal), f"{dataset}/{condition}", "original and diagnostic transformed feature hashes differ")
            results[dataset][condition] = {"record_count": len(rows), "validation_accuracy_differences": [{"seed": key[3], "new": records[key]["validation_accuracy"], "original": row["validation_accuracy"], "difference": diff} for (key, row), diff in zip(rows, accuracy_diffs)], "validation_accuracy_exact_mismatch_count": sum(diff != 0 for diff in accuracy_diffs), "validation_accuracy_max_absolute_difference": max(map(abs, accuracy_diffs)), "validation_loss_differences": [{"seed": key[3], "new": records[key]["validation_loss"], "original": row["validation_loss"], "difference": diff} for (key, row), diff in zip(rows, loss_diffs)], "validation_loss_exact_mismatch_count": sum(diff != 0 for diff in loss_diffs), "validation_loss_max_absolute_difference": max(map(abs, loss_diffs)), "selected_trial_differences": trial_diffs, "selected_trial_id_mismatch_count": sum(not item["equal"] for item in trial_diffs), "transformed_feature_hash_mismatch_count": sum(not equal for equal in hash_equal)}
    return results


def _unit_rows(records: Mapping[tuple[str, str, str, int], dict[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    units = []
    for key in sorted(records, key=lambda item: (item[1], item[3], item[0], item[2])):
        condition, dataset, regularization, seed = key
        row = records[key]
        validation = row["partitions"]["validation"]
        units.append({"dataset": dataset, "seed": seed, "condition": condition, "regularization": regularization, "weight_decay": row["weight_decay"], "selected_trial_id": row["selected_trial_id"], "validation_accuracy": row["validation_accuracy"], "validation_loss": row["validation_loss"], "train_accuracy": row["partitions"]["train"]["accuracy"], "train_loss": row["partitions"]["train"]["loss"], "validation_balanced_accuracy": validation["balanced_accuracy"], "validation_dominant_prediction_fraction": validation["dominant_prediction_fraction"], "validation_training_majority_class": validation["training_majority_class"], "validation_training_majority_accuracy": validation["training_majority_accuracy"], "validation_all_predictions_training_majority": validation["prediction_counts"].get(str(validation["training_majority_class"]), 0) == validation["n"]})
    return units


def summarize(run_root: Path, data_root: Path, preprocessing_root: Path, config: dict[str, Any], audit: dict[str, Any] | None = None, *, config_path: Path | None = None, enforce_fixed_scope: bool = False) -> dict[str, Any]:
    audit = audit or _load(DEFAULT_AUDIT)
    if enforce_fixed_scope:
        _require(config.get("expected_records") == 200 and config.get("expected_trials") == 800, "config", "fixed diagnostic scope must be exactly 200 records and 800 trials")
        _require(config.get("seeds") == list(range(10)) and config.get("models") == ["MLP"], "config", "fixed diagnostic seed/model scope mismatch")
        _require(config.get("conditions") == ["raw", "normalize_features", "normalize_scaled", "normalize_centered", "normalize_centered_scaled"], "config", "fixed diagnostic condition scope mismatch")
        _require(list(config.get("regularization", {})) == ["wd_5e-4", "wd_0"], "config", "fixed diagnostic regularization scope mismatch")
    manifest, complete, digest, source_content_verified = _validate_manifest(run_root, config, config_path)
    records, split_ids, reconstruction = _reconstruct_inputs(config, data_root, audit, run_root, manifest)
    summary = {"schema_version": "1.0", "analysis_status": config["analysis_status"], "run_id": config["run_id"], "config_sha256": digest,
               "source_commit": manifest["source_commit"], "source_fingerprint": manifest["source_fingerprint"], "source_content_verified": source_content_verified, "environment": manifest["environment"], "record_count": len(records), "expected_records": config["expected_records"],
               "split_bindings": len(split_ids), "public_source_visibility_verified": False, "public_source_visibility_note": "public source visibility is outside this local audit",
               "transform_reconstruction": {"data_root": str(data_root), "fit_partition": "train_features_only", "all_records_reconstructed": True},
               "units": _unit_rows(records, config), "aggregates": _aggregate(records, config), "contrasts": _contrasts(records, config), "original_preprocessing_controls": _controls(records, preprocessing_root, data_root, manifest=manifest, transform_hashes=reconstruction["transform_hashes"])}
    scales: dict[str, Any] = {}
    for dataset in config["datasets"]:
        scales[dataset] = {}
        for condition in config["conditions"]:
            values = [float(reconstruction["transform_metadata"][(dataset, int(seed), condition)]["fitted_scale"]) for seed in config["seeds"]]
            scales[dataset][condition] = {"mean": _mean(values), "range": [min(values), max(values)]}
    summary["transform_reconstruction"]["fitted_scale_summary"] = scales
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# MLP optimization diagnostic", "", f"Validated {summary['record_count']} records for `{summary['run_id']}`.", "", "This is a validation-only post hoc diagnostic. It reports paired descriptive contrasts; it makes no population significance claim and does not use test predictions.", "", "## Selected validation summaries", "", "| Dataset | Condition | Decay | Val accuracy mean (sample SD) | Val loss mean (sample SD) | Balanced accuracy mean | Dominant prediction fraction range | All validation predictions training-majority |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for dataset, conditions in summary["aggregates"].items():
        for condition, decays in conditions.items():
            for decay, entry in decays.items():
                acc, loss, bal = entry["validation"]["accuracy"], entry["validation"]["loss"], entry["validation"]["balanced_accuracy"]
                lines.append(f"| {dataset} | {condition} | {decay} | {acc['mean']:.6g} ({acc['sample_sd']:.6g}) | {loss['mean']:.6g} ({loss['sample_sd']:.6g}) | {bal['mean']:.6g} | {entry['validation_dominant_prediction_fraction_range'][0]:.6g}–{entry['validation_dominant_prediction_fraction_range'][1]:.6g} | {entry['validation_all_predictions_training_majority_count']}/10 |")
    lines += ["", "## Validation contrasts", "", "| Dataset | Decay | Contrast | Accuracy mean difference | Positive / zero / negative |", "|---|---|---|---:|---:|"]
    for dataset, contrast_groups in summary["contrasts"].items():
        for decay, contrasts in contrast_groups["by_decay"].items():
            for name, metrics in contrasts.items():
                item = metrics["validation_accuracy"]
                lines.append(f"| {dataset} | {decay} | {name} | {item['mean_difference']:.6g} | {item['positive_count']} / {item['zero_count']} / {item['negative_count']} |")
        for condition, metrics in contrast_groups["weight_decay"].items():
            item = metrics["validation_accuracy"]
            lines.append(f"| {dataset} | weight decay | {condition} (zero minus 5e-4) | {item['mean_difference']:.6g} | {item['positive_count']} / {item['zero_count']} / {item['negative_count']} |")
        for name, item in contrast_groups["scale_interactions"].items():
            if name.endswith("validation_accuracy"):
                lines.append(f"| {dataset} | interaction | {name} | {item['mean_difference']:.6g} | {item['positive_count']} / {item['zero_count']} / {item['negative_count']} |")
    lines += ["", "## Original preprocessing controls", "", "| Dataset | Condition | Accuracy exact mismatches | Accuracy max absolute difference | Loss exact mismatches | Loss max absolute difference | Trial mismatches | Feature hash mismatches |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for dataset, conditions in summary["original_preprocessing_controls"].items():
        for condition, item in conditions.items():
            lines.append(f"| {dataset} | {condition} | {item['validation_accuracy_exact_mismatch_count']} | {item['validation_accuracy_max_absolute_difference']:.6g} | {item['validation_loss_exact_mismatch_count']} | {item['validation_loss_max_absolute_difference']:.6g} | {item['selected_trial_id_mismatch_count']} | {item['transformed_feature_hash_mismatch_count']} |")
    lines += ["", "## Integrity", "", f"- Config digest: `{summary['config_sha256']}`", f"- Source commit: `{summary['source_commit']}`", f"- Original split bindings: {summary['split_bindings']}", "- Raw NPZ checksums and all transformed feature hashes were reconstructed.", f"- Current executable source snapshot matched the manifest: {summary['source_content_verified']}.", "- Public source visibility was not verified.", "- Test evaluations: 0; test metrics are absent.", ""]
    return "\n".join(lines)


def _exclusive_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--preprocessing-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load(args.config)
    torch.set_num_threads(4)
    summary = summarize(args.run_root, args.data_root, args.preprocessing_root, config, _load(args.audit), config_path=args.config, enforce_fixed_scope=True)
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _exclusive_write(args.output_root / "mlp_optimization_diagnostic_summary.json", payload)
    _exclusive_write(args.output_root / "mlp_optimization_diagnostic_summary.md", render_markdown(summary))
    print(json.dumps({"status": "validated", "record_count": summary["record_count"], "output_root": str(args.output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
