#!/usr/bin/env python3
"""Audit active LaTeX sources for claims forbidden by the Route A design."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    rule: str


LINE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "structure_information_bound",
        re.compile(r"Structure\s+Information\s+Bound", re.IGNORECASE),
    ),
    (
        "bounded_efficiency_range",
        re.compile(
            r"Efficiency[^\n]{0,100}\\in\s*\[\s*-?1\s*,\s*1\s*\]",
            re.IGNORECASE,
        ),
    ),
    ("historical_selector_score_32_36", re.compile(r"(?<!\d)32\s*/\s*36(?!\d)")),
    ("historical_selector_score_7_9", re.compile(r"(?<!\d)7\s*/\s*9(?!\d)")),
    ("historical_selector_score_12_12", re.compile(r"(?<!\d)12\s*/\s*12(?!\d)")),
    (
        "architecture_independent_claim",
        re.compile(r"regardless\s+of\s+architecture", re.IGNORECASE),
    ),
    (
        "false_conditional_mi_error_bound",
        re.compile(
            r"I\s*\(\s*Y\s*;\s*G\s*(?:\||\\mid)\s*X\s*\)[^\n]{0,80}\\log\s*C",
            re.IGNORECASE,
        ),
    ),
    (
        "headroom_absolute_gap_claim",
        re.compile(
            r"\\mathcal\s*\{B\}[^\n]{0,80}\\Rightarrow[^\n]{0,80}\|\s*\\Delta\s*\|",
            re.IGNORECASE,
        ),
    ),
)

INPUT_PATTERN = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
DEGREE_PRESERVING_PATTERN = re.compile(
    r"degree[-\s]preserving[^\n]{0,80}(?:edge\s+shuffle|randomi[sz])"
    r"|(?:edge\s+shuffle|randomi[sz])[^\n]{0,80}degree[-\s]preserving",
    re.IGNORECASE,
)
DEGREE_CHECKSUM_PATTERN = re.compile(
    r"degree(?:-sequence|\s+sequence|\s+sequence's|\s+sequence\s+checksum|\s+checksum)"
    r"[^\n]{0,120}(?:identical|unchanged|checksum)",
    re.IGNORECASE,
)


def strip_latex_comment(line: str) -> str:
    """Remove content after the first unescaped percent sign."""

    for index, char in enumerate(line):
        if char != "%":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and line[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return line[:index]
    return line


def resolve_active_sources(root: Path, main: Path) -> list[Path]:
    """Return the main source and recursively included LaTeX files."""

    ordered: list[Path] = []
    visited: set[Path] = set()

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in visited:
            return
        if not resolved.is_file():
            raise FileNotFoundError(f"Active LaTeX source does not exist: {resolved}")
        visited.add(resolved)
        ordered.append(resolved)

        text = resolved.read_text(encoding="utf-8")
        active_text = "\n".join(strip_latex_comment(line) for line in text.splitlines())
        for match in INPUT_PATTERN.finditer(active_text):
            relative = Path(match.group(1).strip())
            if not relative.suffix:
                relative = relative.with_suffix(".tex")
            visit(root / relative)

    visit(main)
    return ordered


def scan_sources(root: Path, sources: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for path in sources:
        active_lines = [strip_latex_comment(line) for line in path.read_text(encoding="utf-8").splitlines()]
        active_text = "\n".join(active_lines)
        has_degree_checksum = bool(DEGREE_CHECKSUM_PATTERN.search(active_text))
        has_verified_degree_summary = verified_degree_summary_exists(root)

        for line_number, line in enumerate(active_lines, start=1):
            for rule, pattern in LINE_RULES:
                if pattern.search(line):
                    findings.append(Finding(path.relative_to(root), line_number, rule))
            if (
                DEGREE_PRESERVING_PATTERN.search(line)
                and not has_degree_checksum
                and not has_verified_degree_summary
            ):
                findings.append(
                    Finding(path.relative_to(root), line_number, "unverified_degree_preserving_claim")
                )
    return findings


def verified_degree_summary_exists(root: Path) -> bool:
    """Return true only for the completed frozen 3-dataset/30-pair summary."""

    path = (
        root
        / "results"
        / "diagnostic"
        / "route_a_degree_matched_v1"
        / "summary"
        / "summary.json"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return (
        payload.get("status") == "success"
        and payload.get("run_id") == "route_a_degree_matched_v1"
        and payload.get("record_count") == 30
        and set(payload.get("datasets", {})) == {"Cora", "CiteSeer", "PubMed"}
    )


def audit(root: Path, main_name: str) -> list[Finding]:
    root = root.resolve()
    main = (root / main_name).resolve()
    return scan_sources(root, resolve_active_sources(root, main))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--main", default="main_neurocomputing.tex")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        findings = audit(args.root, args.main)
    except (FileNotFoundError, UnicodeDecodeError, ValueError) as exc:
        print(f"Route A claim audit error: {exc}")
        return 2

    if findings:
        for finding in findings:
            print(f"{finding.path.as_posix()}:{finding.line}:{finding.rule}")
        print(f"Route A claim audit failed with {len(findings)} finding(s).")
        return 1

    print("Route A claim audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
