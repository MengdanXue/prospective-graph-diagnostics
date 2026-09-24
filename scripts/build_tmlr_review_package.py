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
import subprocess
import sys
import zipfile

from pypdf import PdfReader, PdfWriter

try:
    from scripts.verify_diagnostic_artifacts import verify_archive
except ModuleNotFoundError:
    from verify_diagnostic_artifacts import verify_archive


ORIGINAL_SHA256 = "209d386f078e0c4b2b15eb221b52c52bb002fc136a473494684e01be95b70389"
IDENTITY = re.compile(r"MengdanXue|Mengdan\s+Xue|\bMengdan\b|https-github-com-mengdanxue-prospective-graph", re.I)
LOCAL_PATH = re.compile(r"(?:[A-Za-z]:[\\/](?:Users|home)[\\/]|/(?:Users|home)/)[^\r\n\"<>]*", re.I)
OWN_LINK = re.compile(r"https?://[^\s\"{}<>]*(?:MengdanXue|prospective-graph-diagnostics)[^\s\"{}<>]*", re.I)
# These aliases affect review copies only. Original identifiers and byte hashes
# remain in the author archive; do not rewrite the execution records in place.
BENCHMARK_ALIAS = "benchmark_spec_v1"
LINKABLE_BENCHMARK = "route_a_diagnostic_v1"
SHORT_SOURCE_REVISION = re.compile(r"\bdca835a\b")
HEX40 = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{40}(?![0-9a-fA-F])")
SNAPSHOT_COMMIT = re.compile(r"(?:[a-z_]*source_snapshot)/([0-9a-f]{40})(?:/|$)")
GIT_FIELD = re.compile(r"(?:^|_)(?:commit|commits|git_revision|source_revision)$")
TEXT_SUFFIXES = {".py", ".md", ".tex", ".bib", ".sty", ".bst", ".txt", ".log", ".svg", ".cff"}
PROTOCOL_FILES = (
    "docs/preregistration_diagnostic_benchmark.md",
    "docs/protocol_amendment_prospective_v2.md",
    "docs/protocol_amendment_prospective_v2_1.md",
    "docs/review_protocol_index.md",
)
ARCHIVES = {
    "controls": "posthoc_diagnostic_controls_v1.zip",
    "graph": "posthoc_graph_parameterization_v1.zip",
    "reliability": "training_reproducibility_audit_v1.zip",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ReviewAliases:
    """One archive-wide map, with explicit discovery and no guessed SHA meaning.

    Full Git revisions are discovered from revision-valued JSON fields and
    source-snapshot paths. Other 40-hex strings fail closed, except upstream
    blob checksums explicitly attributed in the official template provenance.
    The private map is never added to the review stage.
    """

    def __init__(self, git_revisions=(), upstream_blobs=(), original_digests=()):
        self.git = {sha: f"source-revision-{index:03d}"
                    for index, sha in enumerate(sorted(set(git_revisions)), 1)}
        self.upstream_blobs = set(upstream_blobs)
        self.original_digests = {sha: f"archived-byte-digest-{index:05d}"
                                 for index, sha in enumerate(sorted(set(original_digests)), 1)}
        self.derived_digests = {}
        self.protected_original_digests = set(original_digests)

    @classmethod
    def discover(cls, inputs):
        revisions, upstream, all_hex = set(), set(), set()

        def visit(value, key=""):
            if isinstance(value, dict):
                for child_key, child in value.items():
                    revisions.update(SNAPSHOT_COMMIT.findall(child_key))
                    visit(child, child_key)
            elif isinstance(value, list):
                for child in value:
                    visit(child, key)
            elif isinstance(value, str):
                revisions.update(SNAPSHOT_COMMIT.findall(value))
                if GIT_FIELD.search(key) and HEX40.fullmatch(value):
                    revisions.add(value)

        for name, data, _origin in inputs:
            revisions.update(SNAPSHOT_COMMIT.findall(name))
            all_hex.update(HEX40.findall(name))
            if PurePosixPath(name).suffix == ".pdf":
                reader = PdfReader(io.BytesIO(data))
                value = str(reader.metadata or "") + "\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                try:
                    value = data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            all_hex.update(HEX40.findall(value))
            if PurePosixPath(name).suffix == ".json":
                visit(json.loads(value))
            if name == "templates/tmlr/PROVENANCE.md":
                assert "Verified upstream Git blob SHA-1" in value
                upstream.update(re.findall(r"\| (?:LICENSE|fancyhdr\.sty|official-example\.tex|tmlr\.bst|tmlr\.sty) \| ([0-9a-f]{40}) \|", value))
        unknown = all_hex - revisions - upstream
        if unknown:
            raise ValueError(f"unclassified 40-hex identifiers require provenance review: {sorted(unknown)}")
        return cls(revisions, upstream)

    def protect_original_digests(self, inputs):
        """Protect older pre-redaction bytes absent from the supplied archives."""
        protected = set()
        for name, data, _origin in inputs:
            if name == "records/frozen/MANIFEST.json":
                for entry in json.loads(data)["files"]:
                    if entry.get("redactions", 0):
                        protected.add(entry["source_sha256"])
        self.original_digests = {sha: f"archived-byte-digest-{index:05d}"
                                 for index, sha in enumerate(sorted(protected), 1)}
        self.protected_original_digests.update(protected)

    def private_record(self):
        return {"git_revision_aliases": self.git,
                "protected_original_byte_digests": self.original_digests,
                "review_derived_digest_bindings": self.derived_digests,
                "preserved_third_party_blob_sha1": sorted(self.upstream_blobs)}


def redact_text(text: str, aliases: ReviewAliases | None = None) -> str:
    text = OWN_LINK.sub("[anonymous artifact supplied with submission]", text)
    text = LOCAL_PATH.sub("<redacted-local-path>", text)
    text = IDENTITY.sub("Anonymous Authors", text)
    text = text.replace(LINKABLE_BENCHMARK, BENCHMARK_ALIAS)
    text = text.replace(LINKABLE_BENCHMARK.replace("_", r"\_"), BENCHMARK_ALIAS.replace("_", r"\_"))
    text = text.replace("route_a", "benchmark").replace(r"route\_a", "benchmark")
    text = SHORT_SOURCE_REVISION.sub("[pre-freeze source revision]", text)
    if aliases is not None:
        text = HEX40.sub(lambda match: aliases.git.get(match.group(), match.group()), text)
        text = re.sub(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])",
                      lambda match: aliases.derived_digests.get(match.group(), aliases.original_digests.get(match.group(), match.group())), text)
    return text


def redact_json(value, aliases: ReviewAliases | None = None):
    if isinstance(value, dict):
        output = {}
        for key, child in value.items():
            anonymous_key = redact_text(key, aliases)
            if anonymous_key in output:
                raise ValueError(f"anonymous JSON key collision: {anonymous_key}")
            output[anonymous_key] = redact_json(child, aliases)
        return output
    if isinstance(value, list):
        return [redact_json(child, aliases) for child in value]
    return redact_text(value, aliases) if isinstance(value, str) else value


def check_json_projection(original, anonymous, aliases: ReviewAliases | None = None) -> int:
    """Allow only the declared string replacement; bind all numbers and schema."""
    if isinstance(original, dict):
        renamed = [redact_text(key, aliases) for key in original]
        assert len(set(renamed)) == len(renamed), "anonymous JSON key collision"
        assert isinstance(anonymous, dict) and set(renamed) == set(anonymous)
        return sum(int(k != ak) + check_json_projection(original[k], anonymous[ak], aliases)
                   for k, ak in zip(original, renamed))
    if isinstance(original, list):
        assert isinstance(anonymous, list) and len(original) == len(anonymous)
        return sum(check_json_projection(a, b, aliases) for a, b in zip(original, anonymous))
    if isinstance(original, str):
        assert anonymous == redact_text(original, aliases)
        return int(original != anonymous)
    assert type(original) is type(anonymous) and original == anonymous
    return 0


def safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if (path.is_absolute() or "\\" in name or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"unsafe archive path: {name}")


def anonymous_bytes(name: str, data: bytes, aliases: ReviewAliases | None = None) -> tuple[bytes, int]:
    suffix = PurePosixPath(name).suffix.lower()
    if suffix == ".json":
        original = json.loads(data)
        anonymous = redact_json(original, aliases)
        redactions = check_json_projection(original, anonymous, aliases)
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
    if suffix in TEXT_SUFFIXES or name.endswith("LICENSE"):
        original = data.decode("utf-8")
        anonymous = redact_text(original, aliases)
        if aliases is not None and name == "scripts/validate_preprocessing_records.py":
            # An explicit review-only representation adapter, not a permissive
            # replacement of the validator: require one of this archive's exact
            # aliases and retain every subsequent provenance/metric assertion.
            pattern = '_COMMIT = re.compile(r"[0-9a-f]{40}\\Z")'
            assert original.count(pattern) == 1
            whitelist = "(?:" + "|".join(re.escape(value) for value in aliases.git.values()) + r")\Z"
            anonymous = anonymous.replace(pattern, f'_COMMIT = re.compile(r"{whitelist}")')
            anonymous = anonymous.replace("source_commit must be a full 40-hex SHA",
                                          "source_commit must be a registered review revision alias")
        return anonymous.encode("utf-8"), int(original != anonymous)
    return data, 0


README = """# Anonymous supplement

This supplement contains manuscript source, evaluation code, compact results,
and retained records for the frozen graph-versus-MLP study. The main PDF
contains the record-reconstruction and training-diagnostic appendices after the
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

To reconstruct the main decision analysis and its sensitivities, create an
empty sibling output directory and run the following commands. They read saved
records; they do not train or evaluate models on test data. The portfolio
command first verifies the delivered record manifest, reconstructs the full
770-model/110-diagnostic audit, and then computes all 63 subsets and seven
leave-one-dataset-out calibration rows. The degree intervention explicitly
uses the v2 configuration retained in its records.

```text
mkdir ../reconstruction
python scripts/summarize_portfolio_robustness.py --records-root records/frozen/prospective/records --output ../reconstruction/portfolio.json
python scripts/summarize_preprocessing_sensitivity.py --root records/controls/preprocessing --output ../reconstruction/preprocessing.json
python scripts/summarize_fallback_sensitivity.py --output ../reconstruction/fallback.json
python scripts/summarize_threshold_headroom.py --records-root records/frozen/prospective/records --output ../reconstruction/threshold_headroom.json
python scripts/plot_threshold_headroom.py --input ../reconstruction/threshold_headroom.json --output ../reconstruction/threshold_headroom.pdf
python scripts/summarize_degree_matched_benchmark.py --config configs/prospective_benchmark_v2.json --input-root records/frozen/degree_matched --output ../reconstruction/degree.json
```

The threshold command enumerates exact decision intervals using the retained
test outcomes; its minima are optimistic post-hoc bounds, distinct from the
unchanged held-out calibration. It also reconstructs the practical-margin
confusion counts, raw regret contributions, and additional fallback checks.

Outputs must not already exist. These commands reconstruct retained outcomes;
they do not authenticate the hidden Git identities or rehash absent raw data.

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
- docs/review_protocol_index.md: index and interpretation of the three archived
  protocol documents included in docs/. Their current archived versions retain
  later provenance and label-scope clarifications; they are not presented as
  unmodified registration-time snapshots. The comparison-family and paper-level
  decision-rule passages are retained apart from declared identifier aliases.

Large checkpoint tensors (66 .pt files) and the Git repository history are
omitted to meet the supplement size limit and protect review anonymity. Raw
dataset inputs are obtained from the dataset sources recorded in the configs.
The omitted tensors limit checkpoint-forward and state-file verification from
this supplement. Historical Git-dependent reconstruction commands require the
complete author archive, not just this anonymous source copy.

## Anonymization and provenance

REVIEW_MANIFEST.json covers every other file with its delivered SHA-256 and
byte count. It does not expose pre-redaction file digests. Identifying paths,
author identifiers, and PDF metadata are removed only in review copies. Run
and benchmark names are consistently aliased in paths, code, JSON keys, and
records. Git revisions and matching snapshot-directory names become
source-revision-NNN aliases: these are review identifiers, not Git references.
The author-only map and original evidence are kept outside this supplement.
Canonical configuration digests and available file digests/byte counts are
recomputed for the review copies, including nested archive manifests. These
derived metadata bindings do not claim identity with original execution bytes.
Older pre-redaction digests whose bytes are absent become
archived-byte-digest-NNNNN metadata aliases, which are not checksums. All
scientific numbers, booleans, nulls, collection membership, and other strings
are preserved and checked; only declared file-size metadata is recalculated.
Genuine model/data fingerprints and explicitly attributed upstream template
blob checksums remain intact. The review copy of the preprocessing validator
replaces its Git-SHA format check with this archive's exact revision-alias
whitelist; all consistency, scope, history, selection and metric checks remain.

Historical manifests, completion markers, and source bindings carry the
declared review-copy metadata substitutions. Original-source bindings can be
verified only in the complete author archive. Use
REVIEW_MANIFEST.json to check delivered-file integrity. The protocol documents
make the archived rule inspectable, but this ZIP cannot independently establish
pre-outcome timestamps or the complete Git history. Removing direct identifiers
does not guarantee unlinkability to public research artifacts.

The original frozen comparison and the post-hoc controls remain separate;
their uncertainty, information boundaries, and training limitations are
described in the manuscript. This archive does not certify repeatability of
the complete original benchmark under stronger baselines.
"""


def canonical_digest(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def rebind_size_metadata(original, anonymous, available, aliases, path=""):
    """Update size/bytes only beside an exact SHA-256 of an available file.

    available maps original file digest to (original length, delivered bytes).
    An unbound checksum (for example an omitted tensor) cannot authorize any
    numeric change. A false original byte count fails rather than being fixed.
    """
    changes = []
    if isinstance(original, dict):
        source = available.get(original.get("sha256"))
        if source is not None:
            original_size, delivered = source
            for key in ("size", "bytes"):
                if key in original:
                    if type(original[key]) is not int or original[key] != original_size:
                        raise ValueError(f"original byte count does not match its bound file: {path}/{key}")
                    if anonymous[key] != len(delivered):
                        changes.append({"path": f"{path}/{key}", "before": anonymous[key], "after": len(delivered)})
                        anonymous[key] = len(delivered)
        for key, child in original.items():
            changes.extend(rebind_size_metadata(child, anonymous[redact_text(key, aliases)],
                                                available, aliases, f"{path}/{redact_text(key, aliases)}"))
    elif isinstance(original, list):
        for index, child in enumerate(original):
            changes.extend(rebind_size_metadata(child, anonymous[index], available, aliases, f"{path}/{index}"))
    return changes


def prepare_review_inputs(inputs, aliases):
    """Rebind anonymous config/file metadata, leaving scientific leaves intact.

    The original inputs and their hashes remain author-only. Config digests are
    recomputed from complete, aliased configurations. Archive entry checksums
    and byte lengths refer to their delivered copies. No assertion in the
    downstream assembler/evaluator is removed or made less strict.
    """
    aliases.protect_original_digests(inputs)
    configurations = {}

    def find_configurations(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"config", "frozen_config"} and isinstance(child, dict):
                    configurations[canonical_digest(child)] = child
                find_configurations(child)
        elif isinstance(value, list):
            for child in value:
                find_configurations(child)

    for name, data, _origin in inputs:
        if PurePosixPath(name).suffix == ".json":
            parsed = json.loads(data)
            if "configs" in PurePosixPath(name).parts:
                configurations[canonical_digest(parsed)] = parsed
            find_configurations(parsed)
    old_file_digests = {name: digest(data) for name, data, _origin in inputs}
    old_file_sizes = {name: len(data) for name, data, _origin in inputs}
    size_metadata = {name: json.loads(data) for name, data, _origin in inputs
                     if name.endswith(".json") and b'"sha256"' in data
                     and (b'"size"' in data or b'"bytes"' in data)}
    reverse_names = {redact_text(name, aliases): name for name, _data, _origin in inputs}
    if len(reverse_names) != len(inputs):
        raise ValueError("duplicate/colliding anonymous file paths")
    previous = None
    metadata_changes = {}
    # The retained manifests form an acyclic provenance tree. Refuse cycles or
    # an unexpected new dependency depth rather than emitting stale bindings.
    for _iteration in range(20):
        config_bindings = {original: canonical_digest(redact_json(config, aliases))
                           for original, config in configurations.items()}
        config_bindings = {old: new for old, new in config_bindings.items() if old != new}
        aliases.derived_digests.update(config_bindings)
        rendered = {name: anonymous_bytes(name, data, aliases)
                    for name, data, _origin in inputs}
        # Only these documented archive metadata fields may change numerically.
        metadata_changes = {}
        available = {}
        for name, (data, _count) in rendered.items():
            old_sha = old_file_digests[name]
            bound = (old_file_sizes[name], data)
            if old_sha in available and available[old_sha] != bound:
                raise ValueError("one original file has inconsistent review projections")
            available[old_sha] = bound
        for name, original_metadata in size_metadata.items():
            value = json.loads(rendered[name][0])
            changes = rebind_size_metadata(original_metadata, value, available, aliases)
            if changes:
                rendered[name] = ((json.dumps(value, indent=2, sort_keys=True) + "\n").encode(), rendered[name][1] + len(changes))
                metadata_changes[name] = changes
        for name, _data, _origin in inputs:
            if name == "records/frozen/MANIFEST.json":
                value = json.loads(rendered[name][0])
                changes = []
                for index, entry in enumerate(value["files"]):
                    source_name = "records/frozen/" + entry["path"]
                    # Manifest paths have been renamed; resolve through the
                    # same bijection used when files are copied.
                    source_name = reverse_names.get(source_name)
                    if source_name is None:
                        raise ValueError("missing frozen manifest target")
                    delivered = rendered[source_name][0]
                    for field, new in (("public_sha256", digest(delivered)), ("public_bytes", len(delivered))):
                        if entry[field] != new:
                            changes.append({"path": f"files/{index}/{field}", "before": entry[field], "after": new})
                            entry[field] = new
                rendered[name] = ((json.dumps(value, indent=2, sort_keys=True) + "\n").encode(), rendered[name][1] + len(changes))
                metadata_changes.setdefault(name, []).extend(changes)
            elif name.endswith("/original_hash_manifest.json"):
                value = json.loads(rendered[name][0])
                prefix = name.rsplit("/", 1)[0] + "/"
                changes = []
                for relative, entry in value["files"].items():
                    source_name = reverse_names.get(prefix + relative)
                    if source_name is None:
                        # Omitted checkpoint tensors keep their genuine hashes.
                        if not relative.endswith(".pt") and relative != "README_rebuild.md":
                            raise ValueError(f"unresolved retained archive entry: {relative}")
                        continue
                    delivered = rendered[source_name][0]
                    for field, new in (("sha256", digest(delivered)), ("size", len(delivered))):
                        if entry[field] != new:
                            changes.append({"path": f"files/{relative}/{field}", "before": entry[field], "after": new})
                            entry[field] = new
                rendered[name] = ((json.dumps(value, indent=2, sort_keys=True) + "\n").encode(), rendered[name][1] + len(changes))
                metadata_changes.setdefault(name, []).extend(changes)
        file_bindings = {old_file_digests[name]: digest(data)
                         for name, (data, _count) in rendered.items()
                         if old_file_digests[name] != digest(data)}
        merged = {**file_bindings, **config_bindings}
        if previous == {name: digest(value[0]) for name, value in rendered.items()} and aliases.derived_digests == merged:
            aliases.protected_original_digests.update(merged)
            return rendered, metadata_changes
        previous = {name: digest(value[0]) for name, value in rendered.items()}
        aliases.derived_digests = merged
    raise ValueError("anonymous provenance rebinding did not reach a stable state")


def add_review_file(stage, manifest, private_files, aliases, name, data, origin, rendered=None):
    """Copy one file with a single mapping for content and relative paths."""
    safe_name(name)
    anonymous_name = redact_text(name, aliases)
    safe_name(anonymous_name)
    if anonymous_name in manifest["files"]:
        raise ValueError(f"duplicate/colliding review output: {anonymous_name}")
    modified, count = rendered if rendered is not None else anonymous_bytes(name, data, aliases)
    path = stage / anonymous_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(modified)
    manifest["files"][anonymous_name] = {
        "bytes": len(modified), "sha256": digest(modified),
        "redacted_fields": count, "renamed_path": anonymous_name != name, "origin": origin,
    }
    private_files[anonymous_name] = {"original_path": name, "original_sha256": digest(data),
                                     "delivered_sha256": digest(modified)}


def write_review_zip(stage, zip_path, manifest):
    """Only stage files are eligible; the author-only map lives outside stage."""
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                relative = path.relative_to(stage).as_posix()
                assert relative in manifest["files"] or relative == "REVIEW_MANIFEST.json", relative
                assert "author-only" not in relative, relative
                info = zipfile.ZipInfo(relative, date_time=(2026, 9, 9, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes(), compresslevel=9)
    assert zip_path.stat().st_size < 100_000_000, zip_path.stat().st_size
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(manifest["files"]) + 1
        for name, record in manifest["files"].items():
            assert digest(archive.read(name)) == record["sha256"]


def assert_anonymous_text(name, text, aliases):
    for value in (name, text):
        assert not IDENTITY.search(value), name
        assert not LOCAL_PATH.search(value), name
        assert not OWN_LINK.search(value), name
        assert "route_a" not in value and r"route\_a" not in value, name
        assert not SHORT_SOURCE_REVISION.search(value), name
        assert not (set(HEX40.findall(value)) - (aliases.upstream_blobs if name == "templates/tmlr/PROVENANCE.md" else set())), name
        assert not (set(re.findall(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", value)) & aliases.protected_original_digests), name


def build(root: Path, original_archive: Path, assets: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    stage = output / "anonymous_supplement"
    if os.name == "nt":
        stage = Path("\\\\?\\" + str(stage.resolve()))
    stage.mkdir()
    manifest = {"schema_version": 2, "files": {}, "new_training": False,
                "new_test_evaluations": False, "omitted_checkpoint_files": 0,
                "anonymization": "review-only aliases for paths, identifiers, Git revisions and protected original byte digests; author map excluded"}
    inputs = []

    source_files = []
    for folder in ("experiments", "scripts", "configs", "results", "sections_tmlr", "templates/tmlr"):
        source_files.extend(p for p in (root / folder).rglob("*") if p.is_file()
                            and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".log"})
    source_files += [root / n for n in ("main_tmlr.tex", "references.bib", "LICENSE",
                                       "sections/06_fixed_degree_analysis.tex")]
    source_files += [root / name for name in PROTOCOL_FILES]
    source_files += list(root.glob("requirements*.txt"))
    excluded = {"scripts/build_tmlr_review_package.py", "templates/tmlr/official-example.tex"}
    for path in sorted(set(source_files)):
        relative = path.relative_to(root).as_posix()
        if relative not in excluded:
            inputs.append((relative, path.read_bytes(), "current source"))

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
                inputs.append(("records/frozen/" + name, archive.read(name), "original frozen archive"))
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
                inputs.append((f"records/{group}/{target}", archive.read(name), f"{group} original archive"))
    assert manifest["omitted_checkpoint_files"] == 66
    inputs.append(("README.md", README.encode(), "review documentation"))
    aliases = ReviewAliases.discover(inputs)
    rendered, metadata_changes = prepare_review_inputs(inputs, aliases)
    private_files = {}
    for name, data, origin in inputs:
        add_review_file(stage, manifest, private_files, aliases, name, data, origin, rendered[name])
    private = aliases.private_record()
    private.update({"scope": "author-only; do not submit", "files": private_files,
                    "explicit_archive_metadata_changes": metadata_changes,
                    "original_frozen_archive_sha256": ORIGINAL_SHA256})
    private_path = output / "author-only-provenance.json"
    assert private_path.parent.resolve() != stage.resolve()
    private_path.write_text(json.dumps(private, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del inputs, rendered
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
    # Execute the delivered checker: its imported names and paths carry the same
    # aliases as the delivered result tree. Do not bypass its strict assertions.
    checked = subprocess.run([sys.executable, "-B", str(stage / "scripts/check_tmlr_freeze.py"),
                              "--root", str(stage)], cwd=stage, capture_output=True, text=True)
    if checked.returncode:
        raise RuntimeError(f"delivered freeze checker failed:\n{checked.stdout}\n{checked.stderr}")
    report = json.loads(checked.stdout)
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
        assert_anonymous_text(path.relative_to(stage).as_posix(), text, aliases)
    zip_path = output / "tmlr_anonymous_supplement.zip"
    write_review_zip(stage, zip_path, manifest)
    sha = digest(zip_path.read_bytes())
    zip_path.with_suffix(".zip.sha256").write_text(f"{sha}  {zip_path.name}\n", encoding="ascii")
    report.update({"supplement_bytes": zip_path.stat().st_size, "supplement_sha256": sha,
                   "record_counts": manifest["record_counts"], "anonymous_scan": "passed",
                   "git_revision_aliases": len(aliases.git),
                   "protected_original_byte_digests": len(aliases.protected_original_digests),
                   "archived_digest_aliases": len(aliases.original_digests),
                   "review_derived_digest_bindings": len(aliases.derived_digests),
                   "preserved_third_party_blob_sha1": len(aliases.upstream_blobs),
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
