#!/usr/bin/env python3
"""Read-only structural and provenance validation for preprocessing records.

The validator deliberately does not retrain models or infer transformed feature
arrays.  It binds the immutable records to a trusted config and diagnostic
audit, checks the complete semantic scope, and verifies the trial audit used by
the frozen runner.  If ``data_root`` is supplied, the two raw NPZ checksums are
also checked against the config; otherwise raw provenance is checked only as a
record-to-config binding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "configs" / "preprocessing_sensitivity_v1.json"
DEFAULT_AUDIT = ROOT / "results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _load(value: Mapping[str, Any] | Path | str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return json.loads(Path(value).read_text(encoding="utf-8"))


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fail(path: Path | str, message: str) -> ValueError:
    return ValueError(f"{message}: {path}")


def _require(condition: bool, path: Path | str, message: str) -> None:
    if not condition:
        raise _fail(path, message)


def _check_finite_metrics(value: Any, path: str = "record") -> None:
    """Reject NaN/infinity and impossible accuracy/loss values recursively."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            _check_finite_metrics(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _check_finite_metrics(child, f"{path}[{index}]")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    if not math.isfinite(float(value)):
        raise ValueError(f"non-finite metric at {path}")
    key = path.rsplit(".", 1)[-1].split("[", 1)[0].lower()
    if "accuracy" in key:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"accuracy outside [0, 1] at {path}")
    elif key == "loss" or key.endswith("_loss"):
        if float(value) < 0.0:
            raise ValueError(f"negative loss at {path}")


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_trial_histories(record: Mapping[str, Any], training: Mapping[str, Any], path: Path) -> None:
    """Check that each reported checkpoint is supported by its epoch history."""
    max_epochs = training.get("max_epochs")
    patience = training.get("patience")
    _require(isinstance(max_epochs, int) and not isinstance(max_epochs, bool) and max_epochs > 0, path, "invalid max_epochs")
    _require(isinstance(patience, int) and not isinstance(patience, bool) and patience > 0, path, "invalid patience")
    for index, trial in enumerate(record.get("trials") or []):
        trial_path = f"{path}:trials[{index}]"
        history = trial.get("history")
        _require(isinstance(history, list) and history, trial_path, "missing trial history")
        completed = trial.get("epochs_completed")
        _require(isinstance(completed, int) and not isinstance(completed, bool), trial_path, "invalid epochs_completed")
        _require(completed == len(history) and 1 <= completed <= max_epochs, trial_path, "epochs_completed/history length mismatch")
        for epoch, entry in enumerate(history):
            _require(isinstance(entry, Mapping), trial_path, "invalid history entry")
            _require(entry.get("epoch") == epoch, trial_path, "history epochs are not contiguous")
            for metric in ("train_loss", "validation_loss", "validation_accuracy"):
                _require(metric in entry and isinstance(entry[metric], (int, float)) and not isinstance(entry[metric], bool), trial_path, f"missing history {metric}")
            _check_finite_metrics(entry, trial_path)
        best = min(history, key=lambda entry: (-float(entry["validation_accuracy"]), float(entry["validation_loss"]), int(entry["epoch"])))
        _require(trial.get("best_epoch") == best["epoch"], trial_path, "best_epoch does not match history")
        for key in ("validation_accuracy", "validation_loss"):
            _require(math.isclose(float(trial.get(key)), float(best[key]), rel_tol=0.0, abs_tol=1e-12), trial_path, f"trial {key} does not match history")
        duration = trial.get("duration_seconds")
        _require(isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(float(duration)) and float(duration) >= 0.0, trial_path, "invalid trial duration")
        if completed < max_epochs:
            _require(completed - 1 - int(best["epoch"]) >= patience, trial_path, "early-stop history does not show the patience cutoff")


def _audit_units(config: Mapping[str, Any], audit: Mapping[str, Any], audit_path: Path | str) -> dict[tuple[str, int], dict[str, Any]]:
    units = audit.get("units")
    _require(isinstance(units, list), audit_path, "audit units must be a list")
    wanted = {(dataset, int(seed)) for dataset in config["datasets"] for seed in config["seeds"]}
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for index, unit in enumerate(units):
        if not isinstance(unit, Mapping) or "dataset" not in unit or "seed" not in unit:
            continue
        key = (str(unit["dataset"]), int(unit["seed"]))
        if key not in wanted:
            continue
        if key in selected:
            raise _fail(audit_path, f"duplicate audit unit for {key}")
        _require(isinstance(unit.get("split_id"), str) and unit["split_id"], f"audit.units[{index}]", "audit unit lacks split_id")
        selected[key] = dict(unit)
    missing = sorted(wanted - set(selected))
    _require(not missing, audit_path, f"audit is missing target units {missing}")
    return selected


def validate_preprocessing_run(
    run_root: Path | str,
    *,
    config: Mapping[str, Any] | Path | str = DEFAULT_CONFIG,
    audit: Mapping[str, Any] | Path | str = DEFAULT_AUDIT,
    data_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate a complete run and return ``records`` plus compact metadata.

    ``records`` is keyed as ``(preprocessing, dataset, seed)`` and then model,
    matching the existing sensitivity summarizer.  ``metadata`` states exactly
    which provenance checks were possible without reconstructing NPZ features.
    """
    run_root = Path(run_root)
    config_path = Path(config) if not isinstance(config, Mapping) else None
    audit_path = Path(audit) if not isinstance(audit, Mapping) else None
    config_obj = _load(config)
    audit_obj = _load(audit)
    manifest_path = run_root / "run_manifest.json"
    complete_path = run_root / "complete.json"
    records_root = run_root / "records"
    _require(manifest_path.is_file(), manifest_path, "missing run manifest")
    _require(complete_path.is_file(), complete_path, "missing complete marker")
    _require(records_root.is_dir(), records_root, "missing records directory")
    manifest = _load(manifest_path)
    complete = _load(complete_path)
    failure_files = sorted(run_root.rglob("failure_*.json"))
    _require(not failure_files, run_root, "failure artifact present: " + ", ".join(str(path.relative_to(run_root)) for path in failure_files))

    digest = canonical_config_sha256(config_obj)
    _require(manifest.get("config_sha256") == digest, manifest_path, "manifest config digest mismatch")
    _require(manifest.get("config") == config_obj, manifest_path, "manifest config differs from trusted config")
    _require(manifest.get("run_id") == config_obj.get("run_id"), manifest_path, "manifest run_id mismatch")
    _require(manifest.get("analysis_status") == config_obj.get("analysis_status"), manifest_path, "manifest analysis_status mismatch")
    _require(manifest.get("preflight") is False, manifest_path, "preflight manifest cannot be audited as a completed run")
    source_commit = manifest.get("source_commit")
    environment = manifest.get("environment")
    _require(isinstance(source_commit, str) and _COMMIT.fullmatch(source_commit), manifest_path, "source_commit must be a full 40-hex SHA")
    _require(isinstance(environment, Mapping) and environment, manifest_path, "missing execution environment")
    _require(complete.get("status") == "complete", complete_path, "run is not complete")

    datasets = config_obj.get("datasets")
    conditions = config_obj.get("conditions")
    models = config_obj.get("models")
    seeds = config_obj.get("seeds")
    training = config_obj.get("training")
    _require(isinstance(datasets, Mapping) and datasets, "config", "datasets must be nonempty")
    _require(isinstance(conditions, list) and conditions, "config", "conditions must be nonempty")
    _require(isinstance(models, list) and models, "config", "models must be nonempty")
    _require(isinstance(seeds, list) and seeds, "config", "seeds must be nonempty")
    _require(isinstance(training, Mapping), "config", "training is missing")
    _require(len(set(conditions)) == len(conditions), "config", "duplicate condition")
    _require(len(set(models)) == len(models), "config", "duplicate model")
    _require(len(set(seeds)) == len(seeds), "config", "duplicate seed")
    expected_count = len(datasets) * len(conditions) * len(models) * len(seeds)
    _require(complete.get("expected_records") == expected_count, complete_path, "complete marker count mismatch")
    _require(complete.get("config_sha256") == digest, complete_path, "complete marker config digest mismatch")

    audit_units = _audit_units(config_obj, audit_obj, audit_path or "audit")
    expected_paths: dict[Path, tuple[str, str, str, int]] = {}
    for condition in conditions:
        for dataset in datasets:
            for model in models:
                for seed in seeds:
                    seed = int(seed)
                    relative = Path("records") / str(condition) / str(dataset) / str(model) / f"seed_{seed:03d}.json"
                    expected_paths[relative] = (str(condition), str(dataset), str(model), seed)
    actual_paths = {path.relative_to(run_root) for path in records_root.rglob("*.json")}
    _require(actual_paths == set(expected_paths), records_root, "missing, extra, or duplicate semantic JSON records")

    records: defaultdict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    transformed_digests: dict[tuple[str, str], str] = {}
    environments: set[str] = set()
    source_commits: set[str] = set()
    for relative, semantic in sorted(expected_paths.items(), key=lambda item: str(item[0])):
        condition, dataset, model, seed = semantic
        path = run_root / relative
        row = _load(path)
        _require(row.get("status") == "success", path, "record status is not success")
        for metric in ("validation_accuracy", "validation_loss", "test_accuracy"):
            _require(metric in row and isinstance(row[metric], (int, float)) and not isinstance(row[metric], bool), path, f"missing or invalid {metric}")
        _check_finite_metrics(row, str(path))
        for key, value in (("schema_version", config_obj.get("schema_version")), ("run_id", config_obj["run_id"]),
                           ("config_sha256", digest), ("source_commit", source_commit), ("environment", environment),
                           ("dataset", dataset), ("model", model), ("seed", seed), ("preprocessing", condition)):
            _require(row.get(key) == value, path, f"{key} mismatch")
        _require(row.get("frozen_config") == config_obj, path, "frozen_config mismatch")
        _require(row.get("training_configuration") == training, path, "training_configuration mismatch")
        _require(row.get("test_evaluations_after_selection") == 1, path, "test evaluation count must be exactly one")
        _validate_trial_histories(row, training, path)
        provenance = row.get("data_provenance")
        _require(isinstance(provenance, Mapping), path, "missing data_provenance")
        specification = datasets[dataset]
        for key, value in (("raw_filename", specification.get("filename")), ("raw_sha256", specification.get("sha256")),
                           ("preprocessing", condition)):
            _require(provenance.get(key) == value, path, f"data provenance {key} mismatch")
        _require(isinstance(provenance.get("raw_sha256"), str) and _SHA256.fullmatch(provenance["raw_sha256"]), path, "invalid raw_sha256")
        transformed = provenance.get("transformed_feature_sha256")
        _require(isinstance(transformed, str) and _SHA256.fullmatch(transformed), path, "invalid transformed feature digest")
        provenance_key = (dataset, condition)
        if provenance_key in transformed_digests:
            _require(transformed_digests[provenance_key] == transformed, path, "inconsistent transformed feature digest")
        else:
            transformed_digests[provenance_key] = transformed
        audit_unit = audit_units[(dataset, seed)]
        _require(row.get("split_id") == audit_unit["split_id"], path, "split_id does not match frozen audit")
        try:
            # This is the same selector/fidelity check used by the frozen runner.
            from experiments.run_prospective_benchmark import validate_trial_audit
            validate_trial_audit(row, training)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid selected trial audit at {path}: {exc}") from exc
        _require((condition, dataset, seed) not in records or model not in records[(condition, dataset, seed)], path, "duplicate semantic record")
        records[(condition, dataset, seed)][model] = row
        environments.add(json.dumps(row["environment"], sort_keys=True, separators=(",", ":")))
        source_commits.add(row["source_commit"])

    if data_root is not None:
        data_root = Path(data_root)
        for dataset, specification in datasets.items():
            raw_path = data_root / str(specification["filename"])
            _require(raw_path.is_file(), raw_path, "raw data file is missing")
            _require(_raw_sha256(raw_path) == specification["sha256"], raw_path, "raw data checksum mismatch")

    metadata = {
        "schema_version": "1.0",
        "status": "validated",
        "run_id": config_obj["run_id"],
        "config_sha256": digest,
        "source_commit": source_commit,
        "environment": dict(environment),
        "record_count": expected_count,
        "scope": {"datasets": list(datasets), "conditions": list(conditions), "models": list(models), "seeds": [int(s) for s in seeds]},
        "split_bindings": len(audit_units),
        "transformed_feature_digest_groups": len(transformed_digests),
        "uniform_source_commit": len(source_commits) == 1,
        "uniform_environment": len(environments) == 1,
        "raw_npz_hashes_checked": data_root is not None,
        "raw_provenance_check": "record raw_filename/raw_sha256 compared with trusted config",
        "transformed_feature_check": "digest consistency checked within each dataset/condition; NPZ feature hashes were not reconstructed",
        "public_source_visibility_verified": False,
        "public_source_visibility_note": "source existence and public archival visibility are outside this local record audit",
    }
    return {"records": {key: dict(value) for key, value in records.items()}, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--data-root", type=Path, help="Optional local NPZ root for raw checksum verification")
    parser.add_argument("--output", type=Path, help="Optional path for the validation report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_preprocessing_run(args.run_root, config=args.config, audit=args.audit, data_root=args.data_root)
    report = {"metadata": result["metadata"]}
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
