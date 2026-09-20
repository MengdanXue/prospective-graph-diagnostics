#!/usr/bin/env python3
"""Strict validation and descriptive summary for the graph transfer diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_prospective_benchmark import config_sha256
from scripts import run_mlp_optimization_diagnostic as transform_runner
from scripts.validate_preprocessing_records import validate_preprocessing_run


DEFAULT_CONFIG = ROOT / "configs/graph_parameterization_diagnostic_v1.json"
DEFAULT_AUDIT = transform_runner.AUDIT_PATH
PREPROCESSING_CONFIG = ROOT / "configs/preprocessing_sensitivity_v1.json"
MLP_CONFIG = ROOT / "configs/mlp_optimization_diagnostic_v1.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _runner():
    """Import the graph runner lazily so lightweight unit tests can load helpers."""
    try:
        from scripts import run_graph_parameterization_diagnostic
    except ImportError as exc:  # pragma: no cover - present before execution agent lands
        raise RuntimeError("graph parameterization runner is not available") from exc
    return run_graph_parameterization_diagnostic


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(ok: bool, where: str | Path, message: str) -> None:
    if not ok:
        raise ValueError(f"{message}: {where}")


def _verify_historical_source(manifest: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    """Verify a prior control run against the source commit recorded in its manifest.

    The current MLP helper is intentionally allowed to evolve for this graph
    diagnostic.  Historical controls therefore use their own recorded commit
    and fingerprint rather than silently treating the current file as the old
    executable.
    """
    source = manifest.get("source_commit")
    _require(isinstance(source, str) and _COMMIT.fullmatch(source), "historical control manifest", "source_commit must be a full SHA")
    fingerprint = manifest.get("source_fingerprint")
    _require(isinstance(fingerprint, Mapping) and isinstance(fingerprint.get("files"), Mapping), "historical control manifest", "missing source fingerprint")
    verified: dict[str, dict[str, Any]] = {}
    for relative, expected in fingerprint["files"].items():
        if relative == "config":
            payload = config_path.read_bytes()
        else:
            completed = subprocess.run(["git", "show", f"{source}:{relative}"], cwd=ROOT, check=False, capture_output=True)
            _require(completed.returncode == 0, relative, "historical source path is unavailable at recorded commit")
            payload = completed.stdout
        actual = hashlib.sha256(payload).hexdigest()
        verified[relative] = {"expected_sha256": expected, "actual_sha256": actual, "verified": actual == expected}
        _require(actual == expected, relative, "historical source fingerprint mismatch")
    return {"source_commit": source, "files": verified, "verified": True}


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = _mean(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _metric(row: Mapping[str, Any], metric: str) -> float:
    if metric == "validation_accuracy":
        return float(row["validation_accuracy"])
    if metric == "validation_loss":
        return float(row["validation_loss"])
    return float(row["partitions"]["validation"][metric.removeprefix("validation_")])


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _validate_manifest(run_root: Path, config: Mapping[str, Any], config_path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    manifest_path = run_root / "run_manifest.json"
    complete_path = run_root / "complete.json"
    _require(manifest_path.is_file(), manifest_path, "missing run manifest")
    _require(complete_path.is_file(), complete_path, "missing complete marker")
    failures = sorted(run_root.rglob("failure_*.json"))
    _require(not failures, run_root, "failure artifact present: " + ", ".join(str(p.relative_to(run_root)) for p in failures))
    manifest, complete = _load(manifest_path), _load(complete_path)
    digest = config_sha256(config)
    derived_records = len(config["datasets"]) * len(config["seeds"]) * len(config["models"]) * len(config["conditions"])
    derived_trials = derived_records * len(config["training"]["trials"])
    _require(derived_records == 420 and config.get("expected_records") == derived_records, "config", "scope is not exactly 420 records")
    _require(config.get("expected_trials") == derived_trials == 1680, "config", "scope is not exactly 1,680 trials")
    _require(manifest.get("config") == config and manifest.get("config_sha256") == digest, manifest_path, "manifest config binding mismatch")
    _require(manifest.get("run_id") == config["run_id"] and manifest.get("analysis_status") == config["analysis_status"], manifest_path, "manifest identity mismatch")
    _require(manifest.get("expected_records") == 420 and manifest.get("expected_trials") == 1680 and manifest.get("test_evaluations_after_selection") == 0, manifest_path, "manifest scope/evaluation mismatch")
    source = manifest.get("source_commit")
    _require(isinstance(source, str) and _COMMIT.fullmatch(source), manifest_path, "source_commit must be a full SHA")
    environment = manifest.get("environment")
    _require(isinstance(environment, Mapping) and environment, manifest_path, "missing environment")
    fingerprint = manifest.get("source_fingerprint")
    _require(isinstance(fingerprint, Mapping) and isinstance(fingerprint.get("files"), Mapping), manifest_path, "missing source fingerprint")
    _require(fingerprint.get("config_sha256") == digest and all(isinstance(v, str) and _SHA256.fullmatch(v) for v in fingerprint["files"].values()), manifest_path, "invalid source fingerprint")
    actual = _runner().source_fingerprint(config_path, digest)
    _require(actual == fingerprint, manifest_path, "source fingerprint does not match current executable snapshot")
    for key, expected in (("status", "complete"), ("run_id", config["run_id"]), ("expected_records", 420), ("expected_trials", 1680), ("config_sha256", digest), ("test_evaluations_after_selection", 0), ("source_fingerprint", fingerprint)):
        _require(complete.get(key) == expected, complete_path, f"complete marker {key} mismatch")
    return manifest, complete, digest


def _expected_paths(run_root: Path, config: Mapping[str, Any]) -> dict[Path, tuple[str, str, str, int]]:
    return {Path("records") / condition / dataset / model / f"seed_{int(seed):03d}.json": (condition, dataset, model, int(seed)) for condition in config["conditions"] for dataset in config["datasets"] for model in config["models"] for seed in config["seeds"]}


def _partition_expected(y: torch.Tensor, indices: list[int]) -> tuple[int, dict[str, int]]:
    labels = y[torch.as_tensor(indices, dtype=torch.long)]
    n_classes = int(y.max().item()) + 1
    counts = torch.bincount(labels, minlength=n_classes)
    return len(indices), {str(index): int(counts[index].item()) for index in range(n_classes)}


def _reconstruct_records(run_root: Path, data_root: Path, config: Mapping[str, Any], audit: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[dict[tuple[str, str, str, int], dict[str, Any]], dict[tuple[str, int], str], dict[tuple[str, int, str], str]]:
    records_root = run_root / "records"
    expected_paths = _expected_paths(run_root, config)
    actual_paths = {path.relative_to(run_root) for path in records_root.rglob("*.json")}
    _require(actual_paths == set(expected_paths), records_root, "missing, extra, or duplicate semantic records")
    split_ids: dict[tuple[str, int], str] = {}
    for unit in audit.get("units", []):
        key = (unit.get("dataset"), int(unit.get("seed", -1)))
        if key[0] in config["datasets"] and key[1] in config["seeds"]:
            _require(key not in split_ids, "audit", f"duplicate target split {key}")
            split_ids[key] = unit["split_id"]
    _require(len(split_ids) == len(config["datasets"]) * len(config["seeds"]), "audit", "missing original split IDs")
    graph_runner = _runner()
    records: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    transformed_hashes: dict[tuple[str, int, str], str] = {}
    for dataset, specification in config["datasets"].items():
        x, y, edge_index, splits = transform_runner.load_inputs(data_root, dataset, specification, audit)
        for seed in config["seeds"]:
            split, split_id = splits[int(seed)]
            _require(split_id == split_ids[(dataset, int(seed))], dataset, "runner split differs from frozen audit")
            train_indices = torch.as_tensor(split["train"], dtype=torch.long)
            for condition in config["conditions"]:
                transformed, metadata = transform_runner.fit_transform_features(x, train_indices, condition)
                metadata = _json_value(metadata)
                transformed_hash = hashlib.sha256(transformed.numpy().tobytes()).hexdigest()
                for model in config["models"]:
                    key = (condition, dataset, model, int(seed))
                    path = run_root / next(relative for relative, semantic in expected_paths.items() if semantic == key)
                    row = _load(path)
                    family = "mlp" if model == "MLP" else "graph"
                    expected_provenance = {"raw_filename": specification["filename"], "raw_sha256": specification["sha256"], "split_id": split_id, "transformed_feature_sha256": transformed_hash, "source_fingerprint": manifest["source_fingerprint"]}
                    expected = {"run_id": config["run_id"], "dataset": dataset, "model": model, "family": family, "seed": int(seed), "split_id": split_id, "condition": condition, "preprocessing": condition, "regularization": "wd_5e-4", "weight_decay": float(config["training"]["weight_decay"]), "source_commit": manifest["source_commit"], "source_fingerprint": manifest["source_fingerprint"], "environment": manifest["environment"], "config_sha256": config_sha256(config), "frozen_config": config, "data_provenance": expected_provenance}
                    try:
                        graph_runner.validate_record(row, expected, config["training"], model=model, family=family)
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(f"invalid graph record {path}: {exc}") from exc
                    _require(row.get("transform_metadata") == metadata, path, "transform metadata mismatch")
                    _require(row.get("data_provenance", {}).get("transformed_feature_sha256") == transformed_hash, path, "transformed feature hash mismatch")
                    _require(row.get("schema_version") == "1.0" and row.get("model") == model and row.get("family") == family, path, "record schema/model family mismatch")
                    for partition, indices in (("train", split["train"]), ("validation", split["validation"])):
                        n, counts = _partition_expected(y, indices)
                        metrics = row.get("partitions", {}).get(partition)
                        _require(isinstance(metrics, Mapping) and metrics.get("n") == n and metrics.get("label_counts") == counts, path, f"{partition} class counts mismatch")
                    records[key] = row
                    transformed_hashes[(dataset, int(seed), condition)] = transformed_hash
        del x, y, edge_index
    return records, split_ids, transformed_hashes


CONTRAST_METRICS = ("validation_accuracy", "validation_loss", "validation_balanced_accuracy", "validation_dominant_prediction_fraction")


def _paired(records: Mapping[tuple[str, str, str, int], dict[str, Any]], dataset: str, model: str, left: str, right: str, metric: str, seeds: list[int], label: str) -> dict[str, Any]:
    differences = [_metric(records[(left, dataset, model, seed)], metric) - _metric(records[(right, dataset, model, seed)], metric) for seed in seeds]
    return {"label": label, "metric": metric, "n": len(differences), "differences": [{"seed": seed, "difference": value} for seed, value in zip(seeds, differences)], "mean_difference": _mean(differences), "sample_sd": _sample_sd(differences), "range": [min(differences), max(differences)], "positive_count": sum(value > 0 for value in differences), "zero_count": sum(value == 0 for value in differences), "negative_count": sum(value < 0 for value in differences)}


def _aggregates(records: Mapping[tuple[str, str, str, int], dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in config["datasets"]:
        output[dataset] = {}
        for model in config["models"]:
            output[dataset][model] = {}
            for condition in config["conditions"]:
                rows = [records[(condition, dataset, model, int(seed))] for seed in config["seeds"]]
                entry: dict[str, Any] = {"n": len(rows), "train": {}, "validation": {}}
                for partition in ("train", "validation"):
                    metrics = [row["partitions"][partition] for row in rows]
                    for field in ("loss", "accuracy", "balanced_accuracy", "training_majority_accuracy"):
                        values = [float(item[field]) for item in metrics]
                        entry[partition][field] = {"mean": _mean(values), "sample_sd": _sample_sd(values)}
                    fractions = [float(item["dominant_prediction_fraction"]) for item in metrics]
                    entry[partition]["dominant_prediction_fraction_range"] = [min(fractions), max(fractions)]
                validation = [row["partitions"]["validation"] for row in rows]
                entry["validation"]["all_predictions_training_majority_count"] = sum(int(item["prediction_counts"].get(str(item["training_majority_class"]), 0) == item["n"]) for item in validation)
                output[dataset][model][condition] = entry
    return output


def _contrasts(records: Mapping[tuple[str, str, str, int], dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    seeds = [int(seed) for seed in config["seeds"]]
    pairs = (("normalize_centered_scaled", "normalize_features", "centered_scaled_minus_normalized"), ("normalize_centered_scaled", "raw", "centered_scaled_minus_raw"), ("normalize_features", "raw", "normalized_minus_raw"))
    out: dict[str, Any] = {}
    for dataset in config["datasets"]:
        out[dataset] = {}
        for model in config["models"]:
            out[dataset][model] = {label: {metric: _paired(records, dataset, model, left, right, metric, seeds, label) for metric in CONTRAST_METRICS} for left, right, label in pairs}
    return out


def _unit_rows(records: Mapping[tuple[str, str, str, int], dict[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    units = []
    for condition, dataset, model, seed in sorted(records, key=lambda key: (key[1], key[3], key[2], key[0])):
        row = records[(condition, dataset, model, seed)]
        selection_accuracy = float(row.get("selection_validation_accuracy", row["validation_accuracy"]))
        selection_loss = float(row.get("selection_validation_loss", row["validation_loss"]))
        units.append({"dataset": dataset, "seed": seed, "model": model, "condition": condition, "selected_trial_id": row["selected_trial_id"], "selection_validation_accuracy": selection_accuracy, "selection_validation_loss": selection_loss, "validation_accuracy": row["validation_accuracy"], "validation_loss": row["validation_loss"], "selection_minus_reported_validation_accuracy": selection_accuracy - float(row["validation_accuracy"]), "selection_minus_reported_validation_loss": selection_loss - float(row["validation_loss"]), "train": row["partitions"]["train"], "validation": row["partitions"]["validation"]})
    return units


def _controls(records: Mapping[tuple[str, str, str, int], dict[str, Any]], transformed_hashes: Mapping[tuple[str, int, str], str], manifest: Mapping[str, Any], preprocessing_root: Path, mlp_root: Path, data_root: Path) -> dict[str, Any]:
    old_config = _load(PREPROCESSING_CONFIG)
    audit = _load(DEFAULT_AUDIT)
    old_result = validate_preprocessing_run(preprocessing_root, config=old_config, audit=audit, data_root=data_root)
    _require(old_result["metadata"]["environment"] == manifest["environment"], preprocessing_root, "original preprocessing environment differs")
    old = old_result["records"]
    controls: dict[str, Any] = {"preprocessing_280": {}, "mlp_60": {}}
    for dataset in old_config["datasets"]:
        controls["preprocessing_280"][dataset] = {}
        for condition in ("raw", "normalize_features"):
            entries = []
            for model in old_config["models"]:
                for seed in old_config["seeds"]:
                    key = (condition, dataset, model, int(seed))
                    old_row = old[(condition, dataset, int(seed))][model]
                    new_row = records[key]
                    entries.append({"model": model, "seed": int(seed), "new_validation_accuracy": new_row["validation_accuracy"], "original_validation_accuracy": old_row["validation_accuracy"], "validation_accuracy_difference": float(new_row["validation_accuracy"]) - float(old_row["validation_accuracy"]), "new_validation_loss": new_row["validation_loss"], "original_validation_loss": old_row["validation_loss"], "validation_loss_difference": float(new_row["validation_loss"]) - float(old_row["validation_loss"]), "new_selected_trial_id": new_row["selected_trial_id"], "original_selected_trial_id": old_row["selected_trial_id"], "feature_hash_equal": old_row["data_provenance"]["transformed_feature_sha256"] == transformed_hashes[(dataset, int(seed), condition)]})
            _require(all(item["feature_hash_equal"] for item in entries), f"preprocessing/{dataset}/{condition}", "original and graph transformed feature hashes differ")
            controls["preprocessing_280"][dataset][condition] = _control_stats(entries, feature_key="feature_hash_equal")
    mlp_config = _load(MLP_CONFIG)
    mlp_summary = __import__("scripts.summarize_mlp_optimization_diagnostic", fromlist=["_reconstruct_inputs", "_validate_manifest"])
    # The previous MLP run is a frozen control.  Its manifest points to the
    # pre-graph source commit, so verify that historical snapshot directly
    # instead of comparing it to the current graph runner file.
    mlp_manifest, _, _, _ = mlp_summary._validate_manifest(mlp_root, mlp_config, None)
    historical_mlp_source = _verify_historical_source(mlp_manifest, MLP_CONFIG)
    _require(mlp_manifest["environment"] == manifest["environment"], mlp_root, "prior MLP environment differs")
    mlp_records, _, _ = mlp_summary._reconstruct_inputs(mlp_config, data_root, audit, mlp_root, mlp_manifest)
    for dataset in mlp_config["datasets"]:
        controls["mlp_60"][dataset] = {}
        for condition in ("raw", "normalize_features", "normalize_centered_scaled"):
            entries = []
            for seed in mlp_config["seeds"]:
                key = (condition, dataset, "MLP", int(seed))
                old_row = mlp_records[(condition, dataset, "wd_5e-4", int(seed))]
                new_row = records[key]
                entries.append({"seed": int(seed), "new_validation_accuracy": new_row["validation_accuracy"], "original_validation_accuracy": old_row["validation_accuracy"], "validation_accuracy_difference": float(new_row["validation_accuracy"]) - float(old_row["validation_accuracy"]), "new_validation_loss": new_row["validation_loss"], "original_validation_loss": old_row["validation_loss"], "validation_loss_difference": float(new_row["validation_loss"]) - float(old_row["validation_loss"]), "new_selected_trial_id": new_row["selected_trial_id"], "original_selected_trial_id": old_row["selected_trial_id"], "feature_hash_equal": old_row["data_provenance"]["transformed_feature_sha256"] == transformed_hashes[(dataset, int(seed), condition)]})
            _require(all(item["feature_hash_equal"] for item in entries), f"mlp/{dataset}/{condition}", "prior MLP and graph transformed feature hashes differ")
            controls["mlp_60"][dataset][condition] = _control_stats(entries, feature_key="feature_hash_equal")
    controls["mlp_60"]["_source_verification"] = historical_mlp_source
    return controls


def _control_stats(entries: list[dict[str, Any]], *, feature_key: str) -> dict[str, Any]:
    accuracy = [float(item["validation_accuracy_difference"]) for item in entries]
    loss = [float(item["validation_loss_difference"]) for item in entries]
    return {"record_count": len(entries), "entries": entries, "validation_accuracy_exact_mismatch_count": sum(value != 0 for value in accuracy), "validation_accuracy_max_absolute_difference": max(map(abs, accuracy)), "validation_loss_exact_mismatch_count": sum(value != 0 for value in loss), "validation_loss_max_absolute_difference": max(map(abs, loss)), "selected_trial_id_mismatch_count": sum(item["new_selected_trial_id"] != item["original_selected_trial_id"] for item in entries), "feature_hash_mismatch_count": sum(not item[feature_key] for item in entries)}


def _selection(records: Mapping[tuple[str, str, str, int], dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    graph_models = [model for model in config["models"] if model != "MLP"]
    output: dict[str, Any] = {}
    for dataset in config["datasets"]:
        output[dataset] = {}
        for condition in config["conditions"]:
            counts = {model: 0 for model in graph_models}
            rows = []
            for seed in config["seeds"]:
                candidates = [records[(condition, dataset, model, int(seed))] for model in graph_models]
                selected = sorted(candidates, key=lambda row: (-float(row["validation_accuracy"]), float(row["validation_loss"]), str(row["model"])))[0]
                counts[selected["model"]] += 1
                mlp = records[(condition, dataset, "MLP", int(seed))]
                rows.append({"seed": int(seed), "selected_graph_model": selected["model"], "selected_graph_validation_accuracy": selected["validation_accuracy"], "mlp_validation_accuracy": mlp["validation_accuracy"], "graph_minus_mlp_validation_gap": float(selected["validation_accuracy"]) - float(mlp["validation_accuracy"])})
            output[dataset][condition] = {"selection_counts": counts, "selected_units": rows, "selection_rule": "validation_accuracy_desc, validation_loss_asc, model_id_asc", "selection_optimism_note": "same validation set selected and reported; graph portfolio has 24 trials versus four MLP trials; no test regret or action claim"}
    return output


def summarize(run_root: Path, data_root: Path, preprocessing_root: Path, mlp_root: Path, config: dict[str, Any], audit: dict[str, Any] | None = None, *, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    audit = audit or _load(DEFAULT_AUDIT)
    manifest, _, digest = _validate_manifest(run_root, config, config_path)
    records, split_ids, transformed_hashes = _reconstruct_records(run_root, data_root, config, audit, manifest)
    return {"schema_version": "1.0", "analysis_status": config["analysis_status"], "run_id": config["run_id"], "config_sha256": digest, "source_commit": manifest["source_commit"], "source_fingerprint": manifest["source_fingerprint"], "source_content_verified": True, "environment": manifest["environment"], "record_count": len(records), "expected_records": 420, "split_bindings": len(split_ids), "public_source_visibility_verified": False, "public_source_visibility_note": "public source visibility is outside this local audit", "units": _unit_rows(records, config), "aggregates": _aggregates(records, config), "contrasts": _contrasts(records, config), "controls": _controls(records, transformed_hashes, manifest, preprocessing_root, mlp_root, data_root), "secondary_selection": _selection(records, config)}


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Graph parameterization diagnostic", "", f"Validated {summary['record_count']} selected units for `{summary['run_id']}`.", "", "This is a validation-only transfer diagnostic. Contrasts are paired descriptive summaries across repeated seeds; the validation-selected graph comparison is selection-optimistic and does not make a test or population claim.", "", "## Validation accuracy by dataset and architecture", "", "| Dataset | Model | Raw mean (SD) | NormalizeFeatures mean (SD) | Centered-scaled mean (SD) |", "|---|---|---:|---:|---:|"]
    for dataset, models in summary["aggregates"].items():
        for model, conditions in models.items():
            cells = []
            for condition in ("raw", "normalize_features", "normalize_centered_scaled"):
                item = conditions[condition]["validation"]["accuracy"]
                cells.append(f"{item['mean']:.6g} ({item['sample_sd']:.6g})")
            lines.append(f"| {dataset} | {model} | {' | '.join(cells)} |")
    lines += ["", "## Main paired contrasts", "", "| Dataset | Model | Contrast | Mean validation-accuracy difference | SD | Range | Positive / zero / negative |", "|---|---|---|---:|---:|---:|---:|"]
    for dataset, models in summary["contrasts"].items():
        for model, contrasts in models.items():
            for label, metrics in contrasts.items():
                item = metrics["validation_accuracy"]
                lines.append(f"| {dataset} | {model} | {label} | {item['mean_difference']:.6g} | {item['sample_sd']:.6g} | {item['range'][0]:.6g}–{item['range'][1]:.6g} | {item['positive_count']} / {item['zero_count']} / {item['negative_count']} |")
    lines += ["", "## Controls", "", "Control mismatch counts and magnitudes are disclosed for review; small metric drift is not silently treated as agreement.", "", "| Control group | Dataset | Condition | Accuracy mismatches | Max | Loss mismatches | Max | Trial mismatches | Feature-hash mismatches |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for group, datasets in summary["controls"].items():
        for dataset, conditions in datasets.items():
            if str(dataset).startswith("_"):
                continue
            for condition, item in conditions.items():
                lines.append(f"| {group} | {dataset} | {condition} | {item['validation_accuracy_exact_mismatch_count']} | {item['validation_accuracy_max_absolute_difference']:.6g} | {item['validation_loss_exact_mismatch_count']} | {item['validation_loss_max_absolute_difference']:.6g} | {item['selected_trial_id_mismatch_count']} | {item['feature_hash_mismatch_count']} |")
    lines += ["", "## Integrity", "", f"- Config digest: `{summary['config_sha256']}`", f"- Source commit: `{summary['source_commit']}`", f"- Split bindings: {summary['split_bindings']}", "- Raw NPZ data, train-fitted transforms, transformed hashes, and train/validation class counts were reconstructed.", "- Public source visibility was not verified.", "- New test evaluations and test metrics are absent.", ""]
    lines.insert(-1, "- The prior MLP control was verified against its historical source commit and fingerprint.")
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
    parser.add_argument("--mlp-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(4)
    config = _load(args.config)
    summary = summarize(args.run_root, args.data_root, args.preprocessing_root, args.mlp_root, config, _load(args.audit), config_path=args.config)
    _exclusive_write(args.output_root / "graph_parameterization_diagnostic_summary.json", json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _exclusive_write(args.output_root / "graph_parameterization_diagnostic_summary.md", render_markdown(summary))
    print(json.dumps({"status": "validated", "record_count": summary["record_count"], "output_root": str(args.output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
