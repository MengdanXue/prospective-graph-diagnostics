#!/usr/bin/env python3
"""Verify complete entry coverage and byte integrity before archive extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


ARCHIVES = (
    "posthoc_graph_parameterization_v1.zip",
    "posthoc_diagnostic_controls_v1.zip",
    "training_reproducibility_audit_v1.zip",
)


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_archive(path: Path) -> dict:
    digest = file_sha256(path)
    checksum = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip()
    if checksum != f"{digest}  {path.name}":
        raise ValueError(f"ZIP checksum mismatch: {path.name}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate ZIP entries: {path.name}")
        for name in names:
            p = PurePosixPath(name)
            if (p.is_absolute() or p.as_posix() != name or "\\" in name
                    or ":" in name or any(part in (".", "..") for part in name.split("/"))):
                raise ValueError(f"unsafe ZIP entry: {name}")
        manifest = json.loads(archive.read("hash_manifest.json"))
        files = manifest["files"]
        if set(files) != set(names) - {"hash_manifest.json"}:
            raise ValueError(f"incomplete manifest coverage: {path.name}")
        for name, expected in files.items():
            with archive.open(name) as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != expected["sha256"] or archive.getinfo(name).file_size != expected["size"]:
                raise ValueError(f"entry integrity mismatch: {name}")
    return {"archive": path.name, "sha256": digest, "entries": len(names),
            "verified_entries": len(files), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    results = {"status": "verified", "archives": [verify_archive(args.assets_dir / n) for n in ARCHIVES]}
    text = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.report:
        with args.report.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
