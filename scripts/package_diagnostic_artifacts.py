#!/usr/bin/env python3
"""Package frozen post-hoc diagnostic records with reproducibility provenance.

The packages intentionally contain JSON records and source/config snapshots,
but no datasets, virtual environments, or model checkpoints; the separate
reliability-audit package retains its worker state dictionaries as raw output.
Source files are read from the exact Git commits recorded by each run
manifest; the current working-tree runner is never substituted for a
historical source file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_bytes(repo: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=repo,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"git source is unavailable at {commit}: {relative}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def add_entry(entries: dict[str, bytes], name: str, data: bytes) -> None:
    normalized = str(PurePosixPath(name))
    require(
        normalized == name and not normalized.startswith("../") and normalized != ".",
        f"unsafe archive path: {name}",
    )
    require(normalized not in entries, f"duplicate archive path: {normalized}")
    entries[normalized] = data


def add_file(entries: dict[str, bytes], name: str, path: Path) -> None:
    require(path.is_file(), f"missing file: {path}")
    add_entry(entries, name, path.read_bytes())


def add_tree(entries: dict[str, bytes], prefix: str, root: Path, expected_count: int) -> None:
    files = sorted(root.rglob("*.json"))
    require(len(files) == expected_count, f"{root}: expected {expected_count} JSON records, found {len(files)}")
    require(not list(root.rglob("failure_*.json")), f"failure artifact under {root}")
    for path in files:
        relative = path.relative_to(root).as_posix()
        add_file(entries, f"{prefix}/{relative}", path)


def validate_run(run_root: Path, expected_records: int, expected_test_evals: int) -> dict[str, Any]:
    manifest_path = run_root / "run_manifest.json"
    complete_path = run_root / "complete.json"
    manifest = load_json(manifest_path)
    complete = load_json(complete_path)
    # The older preprocessing manifest stores these two fields only inside its
    # frozen config; normalize them in memory without rewriting that artifact.
    manifest.setdefault("expected_records", manifest.get("config", {}).get("expected_records", expected_records))
    manifest.setdefault(
        "test_evaluations_after_selection",
        manifest.get("config", {}).get("training", {}).get("test_evaluations_after_selection", expected_test_evals),
    )
    require(manifest.get("expected_records") == expected_records, f"{manifest_path}: expected_records mismatch")
    require(manifest.get("test_evaluations_after_selection") == expected_test_evals, f"{manifest_path}: test-evaluation binding mismatch")
    require(complete.get("status") == "complete", f"{complete_path}: run is not complete")
    require(complete.get("expected_records") == expected_records, f"{complete_path}: expected_records mismatch")
    if "test_evaluations_after_selection" in complete:
        require(complete.get("test_evaluations_after_selection") == expected_test_evals, f"{complete_path}: test-evaluation binding mismatch")
    source = manifest.get("source_commit")
    require(isinstance(source, str) and COMMIT.fullmatch(source), f"{manifest_path}: source_commit is not a full SHA")
    require(isinstance(manifest.get("config_sha256"), str) and SHA256.fullmatch(manifest["config_sha256"]), f"{manifest_path}: invalid config digest")
    return manifest


def verify_record_bindings(run_root: Path, manifest: dict[str, Any], expected_records: int) -> None:
    records = sorted((run_root / "records").rglob("*.json"))
    require(len(records) == expected_records, f"{run_root}: record count mismatch")
    for path in records:
        row = load_json(path)
        require(row.get("run_id") == manifest.get("run_id"), f"{path}: run_id mismatch")
        require(row.get("source_commit") == manifest.get("source_commit"), f"{path}: source_commit mismatch")
        require(row.get("config_sha256") == manifest.get("config_sha256"), f"{path}: config_sha256 mismatch")
        require(row.get("test_evaluations_after_selection") == manifest.get("test_evaluations_after_selection"), f"{path}: test-evaluation mismatch")


def source_snapshot(
    entries: dict[str, bytes],
    repo: Path,
    manifest: dict[str, Any],
    config_path: Path,
    prefix: str,
    extra_sources: tuple[str, ...] = (),
) -> dict[str, Any]:
    commit = manifest["source_commit"]
    fingerprint = manifest.get("source_fingerprint", {})
    expected_files = fingerprint.get("files", {}) if isinstance(fingerprint, dict) else {}
    require(isinstance(expected_files, dict), f"{prefix}: malformed source fingerprint")
    config_relative = config_path.relative_to(repo).as_posix()
    relative_paths = list(expected_files)
    if "config" in relative_paths:
        relative_paths[relative_paths.index("config")] = config_relative
    for relative in extra_sources:
        if relative not in relative_paths:
            relative_paths.append(relative)
    actual: dict[str, dict[str, Any]] = {}
    for relative in sorted(set(relative_paths)):
        data = git_bytes(repo, commit, relative)
        expected = expected_files.get("config" if relative == config_relative else relative)
        digest = sha256(data)
        require(expected is None or digest == expected, f"{prefix}: historical source hash mismatch for {relative}")
        archive_name = f"{prefix}/{commit}/{relative}"
        add_entry(entries, archive_name, data)
        actual[relative] = {"sha256": digest, "size": len(data), "expected_sha256": expected}
    return {"commit": commit, "files": actual, "config_relative": config_relative}


def add_manifest_and_config(
    entries: dict[str, bytes],
    repo: Path,
    run_root: Path,
    manifest: dict[str, Any],
    config_path: Path,
    prefix: str,
    expected_records: int,
    expected_test_evals: int,
) -> dict[str, Any]:
    verify_record_bindings(run_root, manifest, expected_records)
    add_file(entries, f"{prefix}/run_manifest.json", run_root / "run_manifest.json")
    add_file(entries, f"{prefix}/complete.json", run_root / "complete.json")
    add_file(entries, f"{prefix}/config/{config_path.name}", config_path)
    add_tree(entries, f"{prefix}/records", run_root / "records", expected_records)
    binding = source_snapshot(entries, repo, manifest, config_path, f"{prefix}/source_snapshot")
    add_entry(entries, f"{prefix}/source_binding.json", (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return {
        "run_id": manifest["run_id"],
        "records": expected_records,
        "test_evaluations_after_selection": expected_test_evals,
        "source_commit": manifest["source_commit"],
        "config_sha256": manifest["config_sha256"],
    }


def add_partial_attempt(
    entries: dict[str, bytes], repo: Path, run_root: Path, config_path: Path, prefix: str
) -> dict[str, Any]:
    """Retain a failed earlier attempt without treating it as a completed run."""
    manifest = load_json(run_root / "run_manifest.json")
    records = sorted((run_root / "records").rglob("*.json"))
    failures = sorted(run_root.glob("failure_*.json"))
    require(len(records) == 198, f"{run_root}: expected the 198 partial records")
    require(len(failures) == 1, f"{run_root}: expected one failure artifact")
    require(not (run_root / "complete.json").exists(), f"{run_root}: partial attempt has a complete marker")
    source = manifest.get("source_commit")
    require(isinstance(source, str) and COMMIT.fullmatch(source), f"{run_root}: invalid partial source commit")
    config_digest = manifest.get("config_sha256")
    require(isinstance(config_digest, str) and SHA256.fullmatch(config_digest), f"{run_root}: invalid partial config digest")
    for path in records:
        row = load_json(path)
        require(row.get("run_id") == manifest.get("run_id"), f"{path}: partial run_id mismatch")
        require(row.get("source_commit") == source, f"{path}: partial source_commit mismatch")
        require(row.get("config_sha256") == config_digest, f"{path}: partial config_sha256 mismatch")
    add_file(entries, f"{prefix}/run_manifest.json", run_root / "run_manifest.json")
    for path in records:
        add_file(entries, f"{prefix}/records/{path.relative_to(run_root / 'records').as_posix()}", path)
    add_file(entries, f"{prefix}/{failures[0].name}", failures[0])
    binding = source_snapshot(entries, repo, manifest, config_path, f"{prefix}/source_snapshot")
    scope = {
        "status": "partial_failed_attempt",
        "records_present": len(records),
        "expected_records": manifest.get("expected_records"),
        "failure_artifacts": [{"name": failures[0].name, "sha256": sha256(failures[0].read_bytes())}],
        "source_commit": source,
        "config_sha256": config_digest,
        "source_binding": binding,
    }
    add_entry(entries, f"{prefix}/partial_scope.json", (json.dumps(scope, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return scope


def source_manifest_without_fingerprint(
    entries: dict[str, bytes], repo: Path, manifest: dict[str, Any], config_path: Path, prefix: str
) -> dict[str, Any]:
    """Add the older preprocessing source even though its manifest has no file hashes."""
    commit = manifest["source_commit"]
    require(COMMIT.fullmatch(commit), f"{prefix}: invalid source commit")
    files = (
        "experiments/prospective_data.py",
        "experiments/prospective_models.py",
        "experiments/run_prospective_benchmark.py",
        "scripts/run_preprocessing_sensitivity.py",
    )
    relative_paths = (config_path.relative_to(repo).as_posix(),) + files
    actual: dict[str, Any] = {}
    for relative in sorted(relative_paths):
        data = git_bytes(repo, commit, relative)
        add_entry(entries, f"{prefix}/{commit}/{relative}", data)
        actual[relative] = {"sha256": sha256(data), "size": len(data)}
    return {"commit": commit, "files": actual, "config_relative": config_path.relative_to(repo).as_posix()}


def add_exact_commit_file(entries: dict[str, bytes], repo: Path, commit: str, relative: str, prefix: str) -> dict[str, Any]:
    require(COMMIT.fullmatch(commit), f"invalid source commit: {commit}")
    data = git_bytes(repo, commit, relative)
    add_entry(entries, f"{prefix}/{commit}/{relative}", data)
    return {"commit": commit, "path": relative, "sha256": sha256(data), "size": len(data)}


def add_reliability_audit(entries: dict[str, bytes], repo: Path, run_root: Path) -> dict[str, Any]:
    """Package the completed 22-worker audit and its exact source bindings."""
    manifest = load_json(run_root / "run_manifest.json")
    complete = load_json(run_root / "complete.json")
    require(manifest.get("run_id") == "training_reproducibility_audit_v1", "unexpected reliability-audit run_id")
    require(manifest.get("test_evaluations") == 0, "reliability audit contains test evaluations")
    require(manifest.get("source_commit") and COMMIT.fullmatch(manifest["source_commit"]), "invalid reliability-audit source commit")
    workers = sorted((run_root / "workers").iterdir())
    require(len(workers) == 22, f"expected 22 reliability workers, found {len(workers)}")
    require(all((worker / "record.json").is_file() for worker in workers), "missing reliability worker record")
    require(not list(run_root.rglob("failure_*.json")), "reliability audit contains failure artifact")
    file_hashes = complete.get("files_sha256")
    require(isinstance(file_hashes, dict), "reliability complete marker has no file hash map")
    for relative, expected in file_hashes.items():
        path = run_root / Path(relative.replace("\\", "/"))
        require(path.is_file(), f"missing reliability audit file: {relative}")
        require(sha256(path.read_bytes()) == expected, f"reliability audit hash mismatch: {relative}")
    for name in ("run_manifest.json", "complete.json", "frozen_config.json", "audit_summary.json"):
        add_file(entries, f"audit/{name}", run_root / name)
    for path in sorted(run_root.glob("logs/*.log")):
        add_file(entries, f"audit/logs/{path.name}", path)
    for path in sorted((run_root / "workers").rglob("*")):
        if path.is_file():
            add_file(entries, f"audit/workers/{path.relative_to(run_root / 'workers').as_posix()}", path)
    sources: dict[str, Any] = {}
    for relative, expected in sorted(manifest["source_files_sha256"].items()):
        result = subprocess.run(["git", "show", f"{manifest['source_commit']}:{relative}"], cwd=repo, check=False, capture_output=True)
        mode = "git_commit"
        if result.returncode == 0:
            data = result.stdout
        else:
            path = repo / relative
            require(path.is_file(), f"reliability source unavailable: {relative}")
            data = path.read_bytes()
            mode = "working_tree_exact"
        digest = sha256(data)
        require(digest == expected, f"reliability source hash mismatch: {relative}")
        add_entry(entries, f"audit/source_snapshot/{mode}/{relative}", data)
        sources[relative] = {"sha256": digest, "mode": mode, "size": len(data)}
    add_entry(entries, "audit/source_binding.json", (json.dumps({"source_commit": manifest["source_commit"], "files": sources}, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    add_entry(entries, "audit/README.md", (
        "This package contains the completed 22-worker reliability audit: 22/22 workers succeeded, "
        "zero failure artifacts were recorded, and test_evaluations is zero. Worker .pt files are retained "
        "as raw audit output. The audit script was untracked at the recorded commit, so its exact working-tree "
        "bytes are stored under source_snapshot/working_tree_exact/ and checked against the run manifest hash.\n"
    ).encode("utf-8"))
    return {
        "run_id": manifest["run_id"],
        "workers": len(workers),
        "test_evaluations": manifest["test_evaluations"],
        "source_commit": manifest["source_commit"],
        "source_files": sources,
        "summary_sha256": sha256((run_root / "audit_summary.json").read_bytes()),
    }


def hash_manifest(entries: dict[str, bytes], metadata: dict[str, Any]) -> bytes:
    payload = dict(metadata)
    payload["files"] = {
        name: {"sha256": sha256(data), "size": len(data)}
        for name, data in sorted(entries.items())
    }
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def readme(package_name: str, package_kind: str) -> str:
    controls = package_kind == "controls"
    return f"""# {package_name}\n\nThis archive contains frozen JSON records and exact source/configuration provenance for the {package_kind} diagnostic. It deliberately excludes NPZ datasets, virtual environments, and generated caches; the reliability-audit package intentionally retains its worker state/log artifacts as raw output.\n\n## Verify the archive\n\n`hash_manifest.json` lists every archive entry except itself. After extraction, recompute each listed SHA-256 before using a record. The external `.sha256` file authenticates the ZIP byte stream.\n\n## Rebuild or inspect\n\nThe `run_manifest.json` and `complete.json` files are the authoritative scope and completion bindings. The records contain only the partitions declared by the manifest. The graph and MLP diagnostics bind `test_evaluations_after_selection` to zero; no test predictions are present in those packages.\n\nThe `source_snapshot/` tree is read directly from the commit recorded by the run manifest. Use a Git worktree at that commit, or copy the snapshot into a clean checkout preserving its relative paths, to inspect or run the frozen runner. Do not edit the current runner to chase a new hash. The archived source is the executable provenance for this record set.\n\nThe package does not include the two NPZ inputs. Obtain files named in the frozen config separately and verify their SHA-256 before a full reconstruction. A full summary rerun also needs the repository's frozen audit referenced by the summarizer and the analysis dependencies.\n\n{('The preprocessing control records are retained here as supplementary evidence. They intentionally report one post-selection test evaluation per record; this is disclosed in their manifest and is not part of the validation-only graph/MLP claim.' if controls else 'The graph package contains 420 records (2 datasets × 10 seeds × 7 models × 3 conditions), its run manifest, completion marker, frozen config, generated summary, and historical source snapshot.') }\n"""


def write_zip(path: Path, entries: dict[str, bytes], metadata: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = hash_manifest(entries, metadata)
    all_entries = dict(entries)
    all_entries["hash_manifest.json"] = manifest
    readme_data = readme(metadata["package_name"], metadata["package_kind"]).encode("utf-8")
    all_entries["README_rebuild.md"] = readme_data
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(all_entries):
            info = zipfile.ZipInfo(name, ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, all_entries[name])
    digest = sha256(path.read_bytes())
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return {"path": str(path), "sha256": digest, "entries": len(all_entries), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--graph-run-root", type=Path, required=True)
    parser.add_argument("--graph-summary-root", type=Path, required=True)
    parser.add_argument("--initial-graph-run-root", type=Path)
    parser.add_argument("--mlp-run-root", type=Path, required=True)
    parser.add_argument("--preprocessing-run-root", type=Path, required=True)
    parser.add_argument("--audit-run-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    graph_config = repo / "configs/graph_parameterization_diagnostic_v1.json"
    mlp_config = repo / "configs/mlp_optimization_diagnostic_v1.json"
    preprocessing_config = repo / "configs/preprocessing_sensitivity_v1.json"

    graph_manifest = validate_run(args.graph_run_root, 420, 0)
    mlp_manifest = validate_run(args.mlp_run_root, 200, 0)
    preprocessing_manifest = validate_run(args.preprocessing_run_root, 280, 1)
    graph_entries: dict[str, bytes] = {}
    graph_meta = add_manifest_and_config(graph_entries, repo, args.graph_run_root, graph_manifest, graph_config, "graph", 420, 0)
    initial_root = args.initial_graph_run_root or args.graph_run_root.parent / "graph-parameterization-v1"
    initial_scope = add_partial_attempt(graph_entries, repo, initial_root, graph_config, "initial_attempt_198")
    add_entry(
        graph_entries,
        "initial_attempt_198/README.md",
        (
            "This is the first failed graph-parameterization attempt. It contains 198 partial records, "
            "the original run manifest, one failure_*.json artifact, and no complete marker. The records "
            "are retained for audit history and are excluded from the completed 420-record summary.\n"
        ).encode("utf-8"),
    )
    summary_json = args.graph_summary_root / "graph_parameterization_diagnostic_summary.json"
    summary_md = args.graph_summary_root / "graph_parameterization_diagnostic_summary.md"
    add_file(graph_entries, "graph/summary/graph_parameterization_diagnostic_summary.json", summary_json)
    add_file(graph_entries, "graph/summary/graph_parameterization_diagnostic_summary.md", summary_md)
    summary = load_json(summary_json)
    require(summary.get("record_count") == 420, "graph summary record_count is not 420")
    require(summary.get("source_commit") == graph_manifest["source_commit"], "graph summary source commit mismatch")
    summarizer_source = add_exact_commit_file(
        graph_entries,
        repo,
        "a22606a6a50c118b29b6f4777ce3163e7cce7675",
        "scripts/summarize_graph_parameterization_diagnostic.py",
        "summarizer_source_snapshot",
    )
    # The graph package includes both the exact graph source and the MLP source
    # commit used to validate the historical control in the new summarizer.
    mlp_history = source_snapshot(graph_entries, repo, mlp_manifest, mlp_config, "control_source_snapshot")
    graph_metadata = {
        "schema_version": "1.0",
        "package_name": "posthoc_graph_parameterization_v1",
        "package_kind": "graph parameterization",
        "run": graph_meta,
        "historical_control_source": mlp_history,
        "summarizer_source": summarizer_source,
        "initial_attempt": initial_scope,
        "summary_sha256": sha256(summary_json.read_bytes()),
    }
    graph_result = write_zip(args.output_dir / "posthoc_graph_parameterization_v1.zip", graph_entries, graph_metadata)

    control_entries: dict[str, bytes] = {}
    mlp_meta = add_manifest_and_config(control_entries, repo, args.mlp_run_root, mlp_manifest, mlp_config, "mlp", 200, 0)
    pre_meta = add_manifest_and_config(control_entries, repo, args.preprocessing_run_root, preprocessing_manifest, preprocessing_config, "preprocessing", 280, 1)
    # The old preprocessing manifest did not carry a per-file source fingerprint;
    # retain an exact commit snapshot and record its computed hashes explicitly.
    pre_source = source_manifest_without_fingerprint(control_entries, repo, preprocessing_manifest, preprocessing_config, "preprocessing_source_snapshot")
    controls_metadata = {
        "schema_version": "1.0",
        "package_name": "posthoc_diagnostic_controls_v1",
        "package_kind": "supplementary controls",
        "mlp": mlp_meta,
        "preprocessing": pre_meta,
        "preprocessing_source": pre_source,
    }
    controls_result = write_zip(args.output_dir / "posthoc_diagnostic_controls_v1.zip", control_entries, controls_metadata)
    output: dict[str, Any] = {"graph": graph_result, "controls": controls_result}
    if args.audit_run_root is not None:
        audit_entries: dict[str, bytes] = {}
        audit_metadata = {
            "schema_version": "1.0",
            "package_name": "training_reproducibility_audit_v1",
            "package_kind": "reliability audit",
            "audit": add_reliability_audit(audit_entries, repo, args.audit_run_root),
        }
        output["audit"] = write_zip(args.output_dir / "training_reproducibility_audit_v1.zip", audit_entries, audit_metadata)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
