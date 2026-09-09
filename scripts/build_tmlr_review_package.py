#!/usr/bin/env python3
"""Build an anonymous review supplement from retained evidence, without training.

Original files are read-only. Review copies redact identifying strings and
PDF metadata; numeric JSON leaves and all other JSON leaves are checked.
The complete author archive remains the source of checkpoint tensors and Git
history. This script neither uploads files nor submits a manuscript.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import zipfile

from pypdf import PdfReader, PdfWriter

try:
    from scripts.verify_diagnostic_artifacts import verify_archive
    from scripts.check_tmlr_freeze import check
except ModuleNotFoundError:
    from verify_diagnostic_artifacts import verify_archive
    from check_tmlr_freeze import check


ORIGINAL_SHA256 = "209d386f078e0c4b2b15eb221b52c52bb002fc136a473494684e01be95b70389"
IDENTITY = re.compile(r"MengdanXue|Mengdan\s+Xue|\bMengdan\b|https-github-com-mengdanxue-prospective-graph", re.I)
LOCAL_PATH = re.compile(r"(?:[A-Za-z]:[\\/](?:Users|home)[\\/]|/(?:Users|home)/)[^\r\n\"<>]*", re.I)
OWN_LINK = re.compile(r"https?://[^\s\"{}<>]*(?:MengdanXue|prospective-graph-diagnostics)[^\s\"{}<>]*", re.I)
ARCHIVES = {
    "controls": "posthoc_diagnostic_controls_v1.zip",
    "graph": "posthoc_graph_parameterization_v1.zip",
    "reliability": "training_reproducibility_audit_v1.zip",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def redact_text(text: str) -> str:
    text = OWN_LINK.sub("[anonymous artifact supplied with submission]", text)
    text = LOCAL_PATH.sub("<redacted-local-path>", text)
    return IDENTITY.sub("Anonymous Authors", text)


def redact_json(value):
    if isinstance(value, dict):
        # Identifying paths in this evidence occur in values, not schema keys.
        if any(redact_text(key) != key for key in value):
            raise ValueError("identifying JSON key requires an explicit schema review")
        return {key: redact_json(child) for key, child in value.items()}
    if isinstance(value, list):
        return [redact_json(child) for child in value]
    return redact_text(value) if isinstance(value, str) else value


def check_json_projection(original, anonymous) -> int:
    """Allow only the declared string replacement; bind all numbers and schema."""
    if isinstance(original, dict):
        assert isinstance(anonymous, dict) and original.keys() == anonymous.keys()
        return sum(check_json_projection(v, anonymous[k]) for k, v in original.items())
    if isinstance(original, list):
        assert isinstance(anonymous, list) and len(original) == len(anonymous)
        return sum(check_json_projection(a, b) for a, b in zip(original, anonymous))
    if isinstance(original, str):
        assert anonymous == redact_text(original)
        return int(original != anonymous)
    assert type(original) is type(anonymous) and original == anonymous
    return 0


def safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if (path.is_absolute() or "\\" in name or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"unsafe archive path: {name}")


def anonymous_bytes(name: str, data: bytes) -> tuple[bytes, int]:
    suffix = PurePosixPath(name).suffix.lower()
    if suffix == ".json":
        original = json.loads(data)
        anonymous = redact_json(original)
        redactions = check_json_projection(original, anonymous)
        if redactions:
            return (json.dumps(anonymous, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(), redactions
        return data, 0
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        metadata = str(reader.metadata or "")
        if IDENTITY.search(metadata) or LOCAL_PATH.search(metadata) or OWN_LINK.search(metadata):
            writer = PdfWriter()
            writer.clone_document_from_reader(reader)
            writer.metadata = None
            if "/Metadata" in writer.root_object:
                del writer.root_object["/Metadata"]
            out = io.BytesIO()
            writer.write(out)
            return out.getvalue(), 1
        return data, 0
    if suffix in {".py", ".md", ".tex", ".bib", ".sty", ".bst", ".txt", ".log", ".svg", ".cff"} or name.endswith("LICENSE"):
        original = data.decode("utf-8")
        anonymous = redact_text(original)
        return anonymous.encode("utf-8"), int(original != anonymous)
    return data, 0


README = """# Anonymous supplement

This supplement contains manuscript source, evaluation code, compact results,
and retained records for the frozen graph-versus-MLP study. The main PDF
contains the reproducibility and training-diagnostic appendices after the
references. No model training or new test-set evaluation was performed when
preparing this supplement.

## Verify the delivered files and final numerical claims

From this directory, using Python 3.12 or later (standard library only):

```text
python scripts/check_tmlr_freeze.py
```

The checker validates every delivered file against REVIEW_MANIFEST.json and
checks the final LODO quantities against the retained result JSON. A digest
check proves consistency of the delivered files, not correctness of learning
outcomes or identity with a new training run.

For existing scientific checks, install the pinned CPU analysis dependencies
from requirements-ci.lock.txt and run:

```text
python scripts/audit_route_a_claims.py --root . --main main_tmlr.tex
```

Use Tectonic 0.17.0 to rebuild the manuscript (create a separate output folder):

```text
tectonic main_tmlr.tex --outdir ../pdf-check --keep-logs --untrusted
python scripts/check_latex_log.py ../pdf-check/main_tmlr.log
```

## Evidence layout and scope

- results/: compact reported summaries and figures.
- records/frozen/: 770 model records, 110 diagnostic records, and 30 paired
  edge-intervention records from the original release.
- records/controls/: 280 preprocessing and 200 MLP diagnostic records.
- records/graph/: the completed 420-record graph diagnosis and the separately
  labelled partial failed attempt retained for provenance.
- records/reliability/: the 22 worker records, training histories, retained
  logs, summary, and source snapshots for the bounded repeatability audit.
- experiments/, scripts/, configs/: implementation and frozen recipes.
- templates/tmlr/: unchanged official style and its third-party license.

Large checkpoint tensors (66 .pt files) and the Git repository history are
omitted to meet the supplement size limit and protect review anonymity. Raw
dataset inputs are obtained from the dataset sources recorded in the configs.
The omitted tensors limit checkpoint-forward and state-file verification from
this supplement. Historical Git-dependent reconstruction commands require the
complete author archive, not just this anonymous source copy.

## Anonymization and provenance

REVIEW_MANIFEST.json covers every other file in this supplement and records
each file's delivered digest and original digest when applicable. Identifying
local-path strings, author identifiers, and PDF metadata are removed only in
review copies. All JSON schemas, numbers, booleans, null values, and strings
unrelated to identification are preserved and checked against the originals.

Manifests named original_hash_manifest.json, completion markers, and source
fingerprints inside the record groups refer to the original execution/archive
bytes. They are retained provenance, not replacement checksums for redacted
copies. Use REVIEW_MANIFEST.json for the delivered copies. The complete
unmodified evidence archive is retained by the authors. Its availability and
all final author declarations must be confirmed before formal submission.

The original frozen comparison and the post-hoc controls remain separate;
their uncertainty, information boundaries, and training limitations are
described in the manuscript. This archive does not certify repeatability of
the complete original benchmark under stronger baselines.
"""


def build(root: Path, original_archive: Path, assets: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    stage = output / "anonymous_supplement"
    if os.name == "nt":
        stage = Path("\\\\?\\" + str(stage.resolve()))
    stage.mkdir()
    manifest = {"schema_version": 1, "files": {}, "new_training": False,
                "new_test_evaluations": False, "omitted_checkpoint_files": 0,
                "anonymization": "identifying strings and PDF metadata in copies only"}

    def add(name: str, data: bytes, origin: str) -> None:
        safe_name(name)
        if name in manifest["files"]:
            raise ValueError(f"duplicate output: {name}")
        modified, count = anonymous_bytes(name, data)
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(modified)
        manifest["files"][name] = {"bytes": len(modified), "sha256": digest(modified),
                                   "original_sha256": digest(data), "redacted_fields": count,
                                   "origin": origin}

    source_files = []
    for folder in ("experiments", "scripts", "configs", "results", "sections_tmlr", "templates/tmlr"):
        source_files.extend(p for p in (root / folder).rglob("*") if p.is_file()
                            and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".log"})
    source_files += [root / n for n in ("main_tmlr.tex", "references.bib", "LICENSE",
                                       "sections/06_fixed_degree_analysis.tex")]
    source_files += list(root.glob("requirements*.txt"))
    excluded = {"scripts/build_tmlr_review_package.py", "templates/tmlr/official-example.tex"}
    for path in sorted(set(source_files)):
        relative = path.relative_to(root).as_posix()
        if relative not in excluded:
            add(relative, path.read_bytes(), "current source")

    with original_archive.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == ORIGINAL_SHA256
    with zipfile.ZipFile(original_archive) as archive:
        original_manifest = json.loads(archive.read("MANIFEST.json"))
        for entry in original_manifest["files"]:
            data = archive.read(entry["path"])
            assert digest(data) == entry["public_sha256"]
            assert len(data) == entry["public_bytes"]
        for name in archive.namelist():
            if name != "README.md":
                add("records/frozen/" + name, archive.read(name), "original frozen archive")
    for group, filename in ARCHIVES.items():
        verify_archive(assets / filename)
        with zipfile.ZipFile(assets / filename) as archive:
            for name in archive.namelist():
                if name.endswith(".pt"):
                    manifest["omitted_checkpoint_files"] += 1
                    continue
                if name == "README_rebuild.md":
                    continue
                target = "original_hash_manifest.json" if name == "hash_manifest.json" else name
                add(f"records/{group}/{target}", archive.read(name), f"{group} original archive")
    assert manifest["omitted_checkpoint_files"] == 66
    add("README.md", README.encode(), "review documentation")
    manifest["record_counts"] = {
        "frozen_models": len(list((stage / "records/frozen/prospective/records").rglob("*.json"))),
        "frozen_diagnostics": len(list((stage / "records/frozen/prospective/diagnostics").rglob("*.json"))),
        "degree_pairs": len(list((stage / "records/frozen/degree_matched/records").rglob("*.json"))),
        "mlp": len(list((stage / "records/controls/mlp/records").rglob("*.json"))),
        "preprocessing": len(list((stage / "records/controls/preprocessing/records").rglob("*.json"))),
        "graph": len(list((stage / "records/graph/graph/records").rglob("*.json"))),
        "reliability_workers": len(list((stage / "records/reliability/audit/workers").glob("*/record.json"))),
    }
    assert sorted(manifest["record_counts"].values()) == [22, 30, 110, 200, 280, 420, 770], manifest["record_counts"]
    (stage / "REVIEW_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = check(stage)
    # Scan actual text and PDF metadata after copying, not just the source list.
    for path in stage.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".pdf":
            reader = PdfReader(path)
            text = str(reader.metadata or "") + "\n".join(p.extract_text() or "" for p in reader.pages)
        elif path.suffix in {".png"}:
            continue
        else:
            text = path.read_text(encoding="utf-8")
        assert not IDENTITY.search(text), str(path.relative_to(stage))
        assert not LOCAL_PATH.search(text), str(path.relative_to(stage))
        assert not OWN_LINK.search(text), str(path.relative_to(stage))
    zip_path = output / "tmlr_anonymous_supplement.zip"
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(path.relative_to(stage).as_posix(), date_time=(2026, 9, 9, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes(), compresslevel=9)
    assert zip_path.stat().st_size < 100_000_000, zip_path.stat().st_size
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(manifest["files"]) + 1
        for name, record in manifest["files"].items():
            assert digest(archive.read(name)) == record["sha256"]
    sha = digest(zip_path.read_bytes())
    zip_path.with_suffix(".zip.sha256").write_text(f"{sha}  {zip_path.name}\n", encoding="ascii")
    report.update({"supplement_bytes": zip_path.stat().st_size, "supplement_sha256": sha,
                   "record_counts": manifest["record_counts"], "anonymous_scan": "passed",
                   "omitted_checkpoint_files": manifest["omitted_checkpoint_files"],
                   "modified_review_copies": sum(bool(r["redacted_fields"]) for r in manifest["files"].values())})
    (output / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--original-archive", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.original_archive, args.assets_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
