"""Strict validator for the input-robustness formal record tree."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from scripts.input_robustness_formal_records import (
    CONDITIONS, DATASETS, MODELS, FormalRecordError, canonical_json, digest,
    expected_keys, file_digest, record_key, validate_record,
)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormalRecordError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FormalRecordError(f"JSON artifact is not an object: {path}")
    return value


def _scope_from_manifest(manifest: Mapping[str, Any]) -> set[tuple[str, str, str, int]]:
    scope = manifest.get("scope", manifest)
    return expected_keys(
        datasets=scope.get("datasets", DATASETS),
        conditions=scope.get("conditions", CONDITIONS),
        models=scope.get("models", MODELS),
        seeds=scope.get("seeds", range(10)),
    )


def _record_digest(records: list[tuple[Path, Mapping[str, Any]]]) -> str:
    entries = [{"path": path.relative_to(path.parents[4]).as_posix() if len(path.parents) > 4 else path.name,
                "sha256": file_digest(path), "key": list(record_key(record))}
               for path, record in records]
    entries.sort(key=lambda item: (item["key"], item["path"]))
    return hashlib.sha256(canonical_json(entries)).hexdigest()


def _config_digest(path: Path) -> str:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormalRecordError(f"cannot read authoritative config: {path}: {exc}") from exc
    return digest(value)


def _authoritative_config(root: Path, config_path: Path, data_binding_path: Path,
                          manifest: Mapping[str, Any], scope: set[tuple[str, str, str, int]]) -> dict[str, Any]:
    config_path = Path(config_path)
    binding_path = Path(data_binding_path)
    if not config_path.is_file() or not binding_path.is_file():
        raise FormalRecordError("authoritative config and data-binding evidence are required")
    config = _read(config_path)
    binding = _read(binding_path)
    config_hash = _config_digest(config_path)
    if manifest.get("config_sha256") != config_hash:
        raise FormalRecordError("manifest config_sha256 does not match the authoritative config")
    if config.get("formal_training_enabled") is not False:
        raise FormalRecordError("authoritative config must keep formal training disabled for validation")
    config_scope = expected_keys(datasets=config.get("datasets", []), conditions=config.get("conditions", []),
                                 models=config.get("models", []), seeds=config.get("seeds", []))
    if config_scope != scope:
        raise FormalRecordError("record scope differs from the authoritative config scope")
    manifest_scope = manifest.get("scope", {})
    for field in ("datasets", "conditions", "models", "seeds"):
        if list(manifest_scope.get(field, [])) != list(config.get(field, [])):
            raise FormalRecordError(f"manifest {field} order/content differs from the authoritative config")
    execution = config.get("execution", {})
    if (execution.get("model_devices", {}).get("H2GCN") != "cpu"
            or execution.get("torch_num_threads") != 4
            or config.get("h2gcn_checkpoint_mode", execution.get("h2gcn_checkpoint_mode")) != "original_full_state_copy"):
        raise FormalRecordError("authoritative H2GCN CPU/checkpoint settings are not preserved")
    expected_records = config.get("expected_records")
    if expected_records != len(scope) or config.get("expected_trials") != len(scope) * 4:
        raise FormalRecordError("authoritative config record/trial counts are inconsistent")
    bound = config.get("bound_input_source") or {}
    configured_binding_path = bound.get("path")
    if isinstance(configured_binding_path, str):
        configured_path = (Path(__file__).resolve().parents[1] / configured_binding_path).resolve()
        if configured_path != binding_path.resolve():
            raise FormalRecordError("selected data-binding path differs from the config-bound source path")
    expected_binding = bound.get("sha256")
    actual_binding = file_digest(binding_path)
    if expected_binding != actual_binding:
        raise FormalRecordError("data-binding file differs from the config-bound source")
    if manifest.get("data_binding_sha256") != actual_binding:
        raise FormalRecordError("manifest data_binding_sha256 does not match the bound data source")
    if binding.get("bound_split_count") != len(config.get("datasets", [])) * len(config.get("seeds", [])):
        raise FormalRecordError("authoritative data binding has an unexpected split count")
    return {"config": config, "binding": binding, "config_sha256": config_hash,
            "data_binding_sha256": actual_binding, "config_path": str(config_path),
            "data_binding_path": str(binding_path)}


def validate_run(root: Path, *, expected_keys: set[tuple[str, str, str, int]] | None = None,
                 synthetic: bool = False, require_complete: bool = True,
                 config_path: Path | None = None, data_binding_path: Path | None = None) -> dict[str, Any]:
    """Validate every accepted record and return a summary suitable for analysis.

    Only JSON files below ``records/`` are accepted as model records.  Failure
    artifacts may remain below ``failures/`` and are deliberately excluded from
    the successful record set; they can never fill a missing unit.
    """
    root = Path(root)
    manifest = _read(root / "manifest.json")
    if manifest.get("schema_version") != "1.0":
        raise FormalRecordError("unsupported formal manifest schema")
    scope = set(expected_keys) if expected_keys is not None else _scope_from_manifest(manifest)
    if not scope:
        raise FormalRecordError("formal scope must not be empty")
    authority = None
    if not synthetic:
        if config_path is None or data_binding_path is None:
            raise FormalRecordError("formal validation requires authoritative config and data-binding evidence")
        authority = _authoritative_config(root, config_path, data_binding_path, manifest, scope)
    files = sorted((root / "records").rglob("*.json")) if (root / "records").exists() else []
    if not files:
        raise FormalRecordError("formal records directory is empty")
    records: list[tuple[Path, Mapping[str, Any]]] = []
    identities: dict[tuple[str, str, str, int], Path] = {}
    transform_bindings: dict[tuple[str, str, int], tuple[str, str, str]] = {}
    split_bindings: dict[tuple[str, int], str] = {}
    provenance: dict[str, Any] | None = None
    for path in files:
        record = _read(path)
        validate_record(record, expected=manifest, synthetic=synthetic)
        key = record_key(record)
        if key in identities:
            raise FormalRecordError(f"duplicate unit identity {key}: {identities[key]} and {path}")
        identities[key] = path
        if key not in scope:
            raise FormalRecordError(f"unexpected unit identity {key}")
        if provenance is None:
            provenance = {field: record[field] for field in ("run_id", "source_commit", "config_sha256", "data_binding_sha256")}
        elif any(record[field] != provenance[field] for field in provenance):
            raise FormalRecordError("mixed run or source provenance")
        binding = record["transform_binding"]
        binding_key = (key[0], key[1], key[3])
        binding_signature = (binding["split_id"], binding["transformed_feature_sha256"], binding["fit_statistics_sha256"])
        previous = transform_bindings.setdefault(binding_key, binding_signature)
        if previous != binding_signature:
            raise FormalRecordError(f"transform binding differs across models for {binding_key}")
        split_key = (key[0], key[3])
        prior_split = split_bindings.setdefault(split_key, str(record["split_id"]))
        if prior_split != record["split_id"]:
            raise FormalRecordError(f"split binding differs across conditions for {split_key}")
        if authority is not None:
            dataset_binding = authority["binding"].get("datasets", {}).get(key[0], {})
            split_rows = {int(row.get("seed")): row for row in dataset_binding.get("split_checks", [])}
            split_row = split_rows.get(key[3])
            if not isinstance(split_row, Mapping) or split_row.get("split_id") != record["split_id"]:
                raise FormalRecordError(f"record split is not bound to the authoritative data source for {key}")
            if record.get("frozen_config") != authority["config"]:
                raise FormalRecordError(f"record frozen_config differs from the authoritative config for {key}")
            if record.get("training_configuration") != authority["config"].get("training"):
                raise FormalRecordError(f"training configuration differs from the authoritative config for {key}")
            diagnostic_rows = {
                int(row.get("seed")): row for row in dataset_binding.get("full_diagnostic_checks", [])
                if isinstance(row, Mapping)
            }
            diagnostic_row = diagnostic_rows.get(key[3], {})
            authoritative_diag = diagnostic_row.get("diagnostics", {}) if isinstance(diagnostic_row, Mapping) else {}
            for field in ("homophily", "mean_degree", "delta_h"):
                expected_value = authoritative_diag.get(field)
                observed_value = record["diagnostics"].get(field)
                if expected_value is not None and (observed_value is None or not math.isclose(float(observed_value), float(expected_value), abs_tol=1e-12)):
                    raise FormalRecordError(f"diagnostic {field} is not bound to the authoritative source for {key}")
            candidates = [row for row in dataset_binding.get("candidate_checks", [])
                          if isinstance(row, Mapping) and int(row.get("seed", -1)) == key[3]
                          and str(row.get("fit_metadata", {}).get("condition")) == key[1]]
            if candidates:
                expected_transform = candidates[0].get("transformed_feature_sha256")
                if expected_transform and binding.get("transformed_feature_sha256") != expected_transform:
                    raise FormalRecordError(f"transformed feature binding differs from authoritative source for {key}")
            elif key[1] == "normalize_features":
                expected_transform = dataset_binding.get("normalize_features_sha256")
                if expected_transform and binding.get("transformed_feature_sha256") != expected_transform:
                    raise FormalRecordError(f"normalize feature binding differs from authoritative source for {key}")
        from scripts.input_robustness_checkpoint_store import verify_checkpoint
        for checkpoint in record["checkpoint_manifest"]:
            if not checkpoint.get("state_sha256"):
                raise FormalRecordError("checkpoint manifest must include state_sha256")
            try:
                verify_checkpoint(root, checkpoint)
            except Exception as exc:
                raise FormalRecordError(f"checkpoint evidence failed for {path}: {exc}") from exc
        records.append((path, record))
    actual = set(identities)
    missing = sorted(scope - actual)
    extra = sorted(actual - scope)
    if missing:
        raise FormalRecordError(f"missing formal records: {missing[:5]}" + (" ..." if len(missing) > 5 else ""))
    if extra:
        raise FormalRecordError(f"unexpected formal records: {extra[:5]}" + (" ..." if len(extra) > 5 else ""))
    if manifest.get("expected_records", len(scope)) != len(scope):
        raise FormalRecordError("manifest expected_records does not match its scope")
    trial_count = len(records) * 4
    if manifest.get("expected_trials", trial_count) != trial_count:
        raise FormalRecordError("manifest expected_trials does not match its scope")
    digest_value = _record_digest(records)
    if require_complete:
        complete = _read(root / "complete.json")
        if complete.get("status") != "complete" or complete.get("run_id") != manifest.get("run_id"):
            raise FormalRecordError("complete marker is missing or has the wrong identity")
        if complete.get("expected_records") != len(records) or complete.get("expected_trials") != trial_count:
            raise FormalRecordError("complete marker counts differ from records")
        if complete.get("record_digest") != digest_value:
            raise FormalRecordError("complete marker digest differs from records")
        if complete.get("synthetic") is not synthetic:
            raise FormalRecordError("complete marker execution mode differs")
    return {
        "status": "passed", "root": str(root), "run_id": manifest.get("run_id"),
        "record_count": len(records), "trial_count": trial_count, "expected_records": len(scope),
        "expected_trials": len(scope) * 4, "record_digest": digest_value,
        "identities": sorted([list(key) for key in actual]), "records": [record for _, record in records],
        "validation_evaluations": len(records) * 4, "test_evaluations": len(records),
        "formal_training_enabled": all(record.get("formal_training_enabled") is True for _, record in records),
        "synthetic": synthetic,
        "authoritative_evidence": authority,
    }


def validate_complete_run(root: Path, *, expected_keys: set[tuple[str, str, str, int]] | None = None,
                          synthetic: bool = False, config_path: Path | None = None,
                          data_binding_path: Path | None = None) -> dict[str, Any]:
    return validate_run(root, expected_keys=expected_keys, synthetic=synthetic, require_complete=True,
                        config_path=config_path, data_binding_path=data_binding_path)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-binding", type=Path)
    args = parser.parse_args()
    result = validate_complete_run(args.root, synthetic=args.synthetic, config_path=args.config,
                                   data_binding_path=args.data_binding)
    print(json.dumps({key: value for key, value in result.items() if key not in {"records", "identities"}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
