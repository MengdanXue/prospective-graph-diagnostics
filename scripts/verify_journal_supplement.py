"""Verify journal evidence, including explicit metadata-only archive derivation.

This entry does not load datasets, evaluate models or train. Historical ZIPs
keep their identities; the journal repeatability copy has a separate identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_training_reproducibility import summarize_records
from scripts.verify_diagnostic_artifacts import verify_archive

ORIGINAL_AUDIT = "39d6c1c126be7d3db0b24153ef0a0c43d6a95052d98ff3a6c57bde6e7ea8f1c6"
ORIGINAL_ARCHIVES = {
    "prospective-graph-formal-artifacts-v0.1.0.zip": "209d386f078e0c4b2b15eb221b52c52bb002fc136a473494684e01be95b70389",
    "posthoc_diagnostic_controls_v1.zip": "4ab10775a8bbcce924d206d74a101c5a8f7e751a0d50a68ba3d5a0f0a6972c56",
    "posthoc_graph_parameterization_v1.zip": "7ee948ec89856c4223ab7f9043d3858f3f337491c2eaa7167196f14a64f89a06",
}
DERIVED = "training_repeatability_audit_journal_v1.zip"
PATH_KEYS = {"python_executable", "torch_file", "torch_geometric_file"}
REDACTED = "<local-path-redacted>"
LOCAL_PATH = re.compile(r"(?:[A-Za-z]:[\\/]+|/home/|/Users/)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def safe_names(names):
    require(len(names) == len(set(names)), "duplicate ZIP member")
    for name in names:
        path = PurePosixPath(name)
        require(not path.is_absolute() and path.as_posix() == name and "\\" not in name
                and ":" not in name and all(p not in {".", "..", ""} for p in name.split("/")),
                "unsafe ZIP member")


def metadata_free(value):
    if isinstance(value, dict):
        return {k: metadata_free(v) for k, v in value.items() if k not in PATH_KEYS}
    if isinstance(value, list):
        return [metadata_free(v) for v in value]
    return value


def redact_paths(value, pointer=""):
    """Return a copy and exact allowed JSON pointers; reject broader redaction."""
    if isinstance(value, dict):
        out, pointers = {}, []
        for key, item in value.items():
            child = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
            if isinstance(item, str) and LOCAL_PATH.search(item):
                require(key in PATH_KEYS, "local path outside allowed metadata field")
                out[key] = REDACTED
                pointers.append(child)
            else:
                out[key], changes = redact_paths(item, child)
                pointers.extend(changes)
        return out, pointers
    if isinstance(value, list):
        out, pointers = [], []
        for index, item in enumerate(value):
            item, changes = redact_paths(item, pointer + "/" + str(index))
            out.append(item)
            pointers.extend(changes)
        return out, pointers
    require(not isinstance(value, str) or not LOCAL_PATH.search(value), "unclassified local path")
    return value, []


def pointer_value(value, pointer):
    for part in pointer.split("/")[1:]:
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def verify_formal(path):
    require(sha(path.read_bytes()) == ORIGINAL_ARCHIVES[path.name], "formal archive identity mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        safe_names(names)
        manifest = json.loads(archive.read("MANIFEST.json"))
        require(manifest["group_counts"] == {"prospective_records": 770, "prospective_diagnostics": 110,
                                             "degree_records": 30}, "formal scope mismatch")
        require({row["path"] for row in manifest["files"]} == set(names) - {"README.md", "MANIFEST.json"},
                "formal manifest coverage mismatch")
        for row in manifest["files"]:
            data = archive.read(row["path"])
            require(sha(data) == row["public_sha256"] and len(data) == row["public_bytes"], "formal member mismatch")
    return {"records_hashed": 910, "model_records_reaggregated": False}


def verify_derived(path, compact_path, original_path=None):
    outer = verify_archive(path)
    with zipfile.ZipFile(path) as archive:
        mapping = json.loads(archive.read("JOURNAL_DERIVATION.json"))
        require(mapping["original_archive_sha256"] == ORIGINAL_AUDIT, "original identity mismatch")
        require(mapping["allowed_metadata_fields"] == sorted(PATH_KEYS), "redaction field policy mismatch")
        names = set(archive.namelist())
        require(set(mapping["files"]) == names - {"hash_manifest.json", "JOURNAL_DERIVATION.json"},
                "derivation coverage mismatch")
        originals = zipfile.ZipFile(original_path) if original_path else None
        if originals:
            require(sha(original_path.read_bytes()) == ORIGINAL_AUDIT, "supplied original identity mismatch")
        try:
            for name, entry in mapping["files"].items():
                data = archive.read(name)
                require(sha(data) == entry["derived_sha256"] and len(data) == entry["derived_bytes"],
                        "derived byte mapping mismatch")
                if entry["operation"] == "unchanged":
                    require(entry["original_sha256"] == entry["derived_sha256"]
                            and entry["original_bytes"] == entry["derived_bytes"], "unchanged member mismatch")
                elif entry["operation"] == "metadata_paths_only":
                    obj = json.loads(data)
                    require(sha(canonical(metadata_free(obj))) == entry["scientific_sha256"], "scientific value changed")
                    require(bool(entry["redacted_json_pointers"]), "missing redaction pointers")
                    for pointer in entry["redacted_json_pointers"]:
                        require(pointer.split("/")[-1] in PATH_KEYS, "illegal redaction field")
                        require(pointer_value(obj, pointer) == REDACTED, "redaction value mismatch")
                    require(not redact_paths(obj)[1], "remaining local paths")
                    if originals:
                        original = json.loads(originals.read(name))
                        transformed, pointers = redact_paths(original)
                        require(transformed == obj and pointers == entry["redacted_json_pointers"],
                                "derivation differs beyond approved metadata")
                        require(metadata_free(original) == metadata_free(obj), "original scientific fields changed")
                else:
                    require(name == "audit/complete.json" and entry["operation"] == "completion_hash_rebinding",
                            "unsupported derivation operation")
                    obj = json.loads(data)
                    fields = {k: v for k, v in obj.items() if k != "files_sha256"}
                    require(sha(canonical(fields)) == entry["scientific_sha256"], "completion science changed")
                    if originals:
                        original = json.loads(originals.read(name))
                        require(fields == {k: v for k, v in original.items() if k != "files_sha256"},
                                "completion fields changed")
                        require(set(obj["files_sha256"]) == set(original["files_sha256"]), "completion inventory changed")
                if originals:
                    original = originals.read(name)
                    require(sha(original) == entry["original_sha256"] and len(original) == entry["original_bytes"],
                            "original byte mapping mismatch")
        finally:
            if originals:
                originals.close()
        manifest = json.loads(archive.read("audit/run_manifest.json"))
        summary = json.loads(archive.read("audit/audit_summary.json"))
        complete = json.loads(archive.read("audit/complete.json"))
        require(complete["status"] == "complete" and complete["worker_count"] == complete["success_count"] == 22
                and complete["test_evaluations"] == 0, "completion scope mismatch")
        require(len(complete["files_sha256"]) == 113, "completion coverage mismatch")
        for name, expected in complete["files_sha256"].items():
            require(sha(archive.read("audit/" + name.replace("\\", "/"))) == expected, "internal completion hash mismatch")
        records = []
        for worker in manifest["workers"]:
            prefix = "audit/workers/" + worker["worker_id"] + "/"
            record = json.loads(archive.read(prefix + "record.json"))
            for name, expected in record["artifacts_sha256"].items():
                require(sha(archive.read(prefix + name)) == expected, "worker model artifact hash mismatch")
            records.append(record)
        require(summary["manifest"] == manifest and summary["records"] == records, "embedded records mismatch")
        rebuilt = summarize_records(manifest["config"], records)
        for key, value in rebuilt.items():
            require(summary[key] == value, "worker aggregate mismatch: " + key)
        package = json.loads(archive.read("hash_manifest.json"))
        for name, expected in package["audit"]["source_files"].items():
            source = "audit/source_snapshot/" + expected["mode"] + "/" + name
            require(sha(archive.read(source)) == expected["sha256"], "source snapshot binding mismatch")
            require(manifest["source_files_sha256"][name] == expected["sha256"], "run/source binding mismatch")
        compact = json.loads(compact_path.read_bytes())
        require(compact["full_summary_sha256"] == mapping["files"]["audit/audit_summary.json"]["original_sha256"],
                "original compact summary binding mismatch")
        require(compact["full_summary_bytes"] == mapping["files"]["audit/audit_summary.json"]["original_bytes"],
                "original compact summary size mismatch")
        require(compact["groups"] == summary["groups"], "compact scientific groups mismatch")
    return {**outer, "workers_reaggregated": 22, "completion_files_hashed": 113,
            "original_bytes_compared": original_path is not None, "original_model_records_reaggregated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--original-repeatability", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    reports = []
    for name, expected in ORIGINAL_ARCHIVES.items():
        path = args.evidence_dir / name
        require(sha(path.read_bytes()) == expected, "original archive changed: " + name)
        reports.append(verify_formal(path) if name.startswith("prospective-") else verify_archive(path))
    compact = ROOT / "results/diagnostic/training_reproducibility_audit_v1/analysis/audit_summary.json"
    reports.append(verify_derived(args.evidence_dir / DERIVED, compact, args.original_repeatability))
    result = {"status": "verified", "archives": reports, "training": False, "dataset_loaders": False,
              "new_model_evaluations": False, "complete_770_record_reaggregation": False}
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        with args.report.open("x", encoding="utf-8") as handle:
            handle.write(output)
    print(output)


if __name__ == "__main__":
    main()
