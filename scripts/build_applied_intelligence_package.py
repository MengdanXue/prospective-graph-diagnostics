"""Flatten the Applied Intelligence manuscript into a journal source directory.

This packages existing manuscript assets only. It never runs an experiment or
replaces an existing output directory. Compile manuscript.tex before submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def build(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    seen: set[Path] = set()
    source_hashes: dict[str, str] = {}

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def expand(path: Path) -> str:
        path = path.resolve()
        path.relative_to(root)
        if path in seen:
            raise ValueError(f"Repeated/cyclic input: {path}")
        seen.add(path)
        source_hashes[path.relative_to(root).as_posix()] = sha(path)
        text = path.read_text(encoding="utf-8")
        return re.sub(
            r"\\input\{([^}]+)\}",
            lambda m: expand(root / (m[1] if Path(m[1]).suffix else m[1] + ".tex")),
            text,
        )

    text = expand(root / "main_applied_intelligence.tex")
    assets: dict[str, str] = {}

    def figure(match: re.Match) -> str:
        source = (root / match[2]).resolve()
        source.relative_to(root)
        if not source.suffix:
            candidates = [source.with_suffix(ext) for ext in (".pdf", ".png", ".jpg")]
            source = next((p for p in candidates if p.is_file()), source)
        digest = sha(source)
        if source.name in assets and assets[source.name] != digest:
            raise ValueError(f"Figure basename collision: {source.name}")
        assets[source.name] = digest
        shutil.copy2(source, output / source.name)
        source_hashes[source.relative_to(root).as_posix()] = digest
        return match[1] + "{" + source.name + "}"

    text = re.sub(r"(\\includegraphics(?:\[[^]]*\])?)\{([^}]+)\}", figure, text)
    text = re.sub(r"\\graphicspath\{(?:\{[^}]*\})+\}\s*", "", text)
    (output / "manuscript.tex").write_text(text, encoding="utf-8")
    for name in ("references.bib", "sn-jnl.cls", "sn-basic.bst"):
        shutil.copy2(root / name, output / name)
        source_hashes[name] = sha(root / name)
    manifest = {
        "purpose": "manuscript source package, not the research-record supplement",
        "source_sha256": source_hashes,
        "flat_source_sha256": {p.name: sha(p) for p in sorted(output.iterdir())},
    }
    # Keep administrative metadata outside files uploaded to the LaTeX compiler.
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {output}; compile manuscript.tex to verify the flat package.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[1], args.output.resolve())
