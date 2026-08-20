#!/usr/bin/env python3
"""Build a deterministic, path-redacted archive of the formal raw artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any


FORMAL_COUNTS = {
    "prospective_records": 770,
    "prospective_diagnostics": 110,
    "degree_records": 30,
}
GROUPS = {
    "prospective_records": (
        Path("route_a_prospective_v2/records"),
        Path("prospective/records"),
        "route_a_prospective_v2",
        1,
    ),
    "prospective_diagnostics": (
        Path("route_a_prospective_v2/diagnostics"),
        Path("prospective/diagnostics"),
        "route_a_prospective_v2",
        1,
    ),
    "degree_records": (
        Path("route_a_degree_matched_v1/records"),
        Path("degree_matched/records"),
        "route_a_degree_matched_v1",
        3,
    ),
}
PRIVATE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\"),
    re.compile(r"/(?:home|Users)/", re.IGNORECASE),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def redact_processed_roots(payload: Any, dataset: str) -> int:
    """Replace only roots nested under a ``processed_files`` collection."""
    redactions = 0
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "processed_files":
                if not isinstance(value, list):
                    raise ValueError("processed_files must be a list")
                for item in value:
                    if not isinstance(item, dict) or not isinstance(item.get("root"), str):
                        raise ValueError("every processed file must contain a string root")
                    item["root"] = f"data/{dataset}/processed"
                    redactions += 1
            else:
                redactions += redact_processed_roots(value, dataset)
    elif isinstance(payload, list):
        for item in payload:
            redactions += redact_processed_roots(item, dataset)
    return redactions


def iter_strings(payload: Any):
    if isinstance(payload, dict):
        for value in payload.values():
            yield from iter_strings(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from iter_strings(value)
    elif isinstance(payload, str):
        yield payload


def assert_no_private_paths(payload: Any, label: str) -> None:
    for value in iter_strings(payload):
        if any(pattern.search(value) for pattern in PRIVATE_PATH_PATTERNS):
            raise ValueError(f"private absolute path remains after redaction: {label}")


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def write_archive_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    archive.writestr(
        zip_info(name), data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
    )


def archive_readme(release_commit: str, total: int) -> bytes:
    text = f"""# Formal Raw Artifact Archive

This archive contains {total} JSON records from the frozen confirmatory runs:

- 770 prospective model records;
- 110 diagnostic records whose label-dependent graph statistics use training labels only;
- 30 degree-matched paired intervention records.

Privacy transformation: each machine-local value under a `processed_files[].root`
field was replaced with `data/<dataset>/processed`. No outcome, configuration,
data checksum, edge list, split identifier, or model-selection field was changed.
`MANIFEST.json` records the SHA-256 of each immutable source artifact and its
path-redacted public counterpart. The immutable source files remain unchanged.

Public release commit: `{release_commit}`.
"""
    return text.encode("utf-8")


def build_release(
    source_root: Path,
    output: Path,
    *,
    expected_counts: dict[str, int],
    release_commit: str,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if set(expected_counts) != set(GROUPS):
        raise ValueError("expected_counts must define every formal artifact group")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")

    manifest_files: list[dict[str, Any]] = []
    group_counts: dict[str, int] = {}
    total_redactions = 0
    source_commits: set[str] = set()
    config_sha256: set[str] = set()

    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for group, (relative_root, archive_root, run_id, expected_redactions) in GROUPS.items():
                folder = source_root / relative_root
                files = sorted(folder.rglob("*.json"))
                if len(files) != expected_counts[group]:
                    raise ValueError(
                        f"{group}: expected {expected_counts[group]} JSON files, found {len(files)}"
                    )
                group_counts[group] = len(files)
                for path in files:
                    source_bytes = path.read_bytes()
                    payload = json.loads(source_bytes.decode("utf-8"))
                    if payload.get("status") != "success":
                        raise ValueError(f"non-success artifact: {path}")
                    if payload.get("run_id") != run_id:
                        raise ValueError(f"unexpected run_id in {path}")
                    dataset = payload.get("dataset")
                    if not isinstance(dataset, str) or not dataset:
                        raise ValueError(f"missing dataset in {path}")
                    redactions = redact_processed_roots(payload, dataset)
                    if redactions != expected_redactions:
                        raise ValueError(
                            f"{path}: expected {expected_redactions} root redactions, found {redactions}"
                        )
                    assert_no_private_paths(payload, str(path.relative_to(source_root)))
                    public_bytes = canonical_json_bytes(payload)
                    archive_path = (archive_root / path.relative_to(folder)).as_posix()
                    write_archive_member(archive, archive_path, public_bytes)
                    total_redactions += redactions
                    manifest_files.append(
                        {
                            "group": group,
                            "path": archive_path,
                            "public_bytes": len(public_bytes),
                            "public_sha256": sha256_bytes(public_bytes),
                            "redactions": redactions,
                            "source_bytes": len(source_bytes),
                            "source_sha256": sha256_bytes(source_bytes),
                        }
                    )
                    if isinstance(payload.get("source_commit"), str):
                        source_commits.add(payload["source_commit"])
                    if isinstance(payload.get("config_sha256"), str):
                        config_sha256.add(payload["config_sha256"])

            manifest = {
                "archive_schema_version": "1.0",
                "config_sha256": sorted(config_sha256),
                "files": manifest_files,
                "group_counts": group_counts,
                "redaction_policy": {
                    "field": "processed_files[].root",
                    "replacement": "data/<dataset>/processed",
                    "scope": "path_only",
                },
                "release_commit": release_commit,
                "source_commits": sorted(source_commits),
                "total_json_files": sum(group_counts.values()),
                "total_redactions": total_redactions,
            }
            write_archive_member(
                archive,
                "README.md",
                archive_readme(release_commit, manifest["total_json_files"]),
            )
            write_archive_member(archive, "MANIFEST.json", canonical_json_bytes(manifest))
        temporary.rename(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_release(
        args.source_root,
        args.output,
        expected_counts=FORMAL_COUNTS,
        release_commit=args.release_commit,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "group_counts": manifest["group_counts"],
                "total_json_files": manifest["total_json_files"],
                "total_redactions": manifest["total_redactions"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
