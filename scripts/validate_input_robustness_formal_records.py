"""Strict validator for the input-robustness formal record tree."""

from __future__ import annotations

import hashlib
import json
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


def validate_run(root: Path, *, expected_keys: set[tuple[str, str, str, int]] | None = None,
                 synthetic: bool = False, require_complete: bool = True) -> dict[str, Any]:
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
    }


def validate_complete_run(root: Path, *, expected_keys: set[tuple[str, str, str, int]] | None = None,
                          synthetic: bool = False) -> dict[str, Any]:
    return validate_run(root, expected_keys=expected_keys, synthetic=synthetic, require_complete=True)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    result = validate_complete_run(args.root, synthetic=args.synthetic)
    print(json.dumps({key: value for key, value in result.items() if key not in {"records", "identities"}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

