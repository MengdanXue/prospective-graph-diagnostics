#!/usr/bin/env python3
"""Verify final manuscript quantities and, when present, review-package digests.

This reads existing results only; it does not train or evaluate any model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def check(root: Path) -> dict:
    result_dir = root / "results/diagnostic/route_a_prospective_v2/analysis"
    robustness = json.loads((result_dir / "portfolio_robustness.json").read_text(encoding="utf-8"))
    full = robustness["calibrated_threshold"]["full"]
    folds = full["folds"]
    assert len(folds) == 11
    constant = [name for name, fold in folds.items() if fold["threshold"] == -1]
    assert len(constant) == 10
    assert all(folds[name]["graph_actions"] == folds[name]["units"] for name in constant)
    roman = folds["Roman-empire"]
    assert roman["threshold"] == 0.11 and roman["graph_actions"] == 0
    total = sum(fold["regret_pp"] for fold in folds.values())
    assert abs(total / 11 - full["mean_regret_pp"]) < 1e-12
    share = 100 * roman["regret_pp"] / total
    assert round(share) == 89
    assert round(full["mean_regret_pp"], 2) == 2.41
    section = (root / "sections_tmlr/04a_equal_budget_sensitivity.tex").read_text(encoding="utf-8")
    assert "10 of the 11 folds" in section
    assert r"approximately 89\%" in section
    assert "GCN and GAT" in section and "GPR-GNN" in section
    main = (root / "main_tmlr.tex").read_text(encoding="utf-8")
    assert r"\usepackage{tmlr}" in main
    assert main.index(r"\bibliography{references}") < main.index(r"\appendix")
    audit = json.loads((result_dir / "diagnostic_audit.json").read_text(encoding="utf-8"))
    expected = {"historical_combined": 7.46, "always_graph": 0.26, "validation_selection": 0.22}
    for policy, value in expected.items():
        observed = audit["methods"][policy]["full_set_mean_regret"] * 100
        assert round(observed, 2) == value
    report = {
        "status": "verified", "new_training": False, "new_test_evaluations": False,
        "calibrated_full_regret_pp": full["mean_regret_pp"],
        "constant_graph_folds": len(constant), "folds": len(folds),
        "roman_share_of_calibrated_regret_pct": share,
        "frozen_regrets_pp": expected,
    }
    manifest_path = root / "REVIEW_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts}
        assert actual == set(manifest["files"]) | {"REVIEW_MANIFEST.json"}
        for relative, expected_file in manifest["files"].items():
            data = (root / relative).read_bytes()
            assert hashlib.sha256(data).hexdigest() == expected_file["sha256"], relative
            assert len(data) == expected_file["bytes"], relative
        report["review_files_verified"] = len(manifest["files"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(check(args.root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
