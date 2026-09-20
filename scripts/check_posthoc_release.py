"""Rebuild all post-hoc tables from checksum-verified public CPU records."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_equal_budget_sensitivity import (
    GRAPH_ARCHITECTURES, evaluate_portfolio, load_units, require_matching_audit,
)
from scripts.summarize_portfolio_robustness import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT / "results/diagnostic/route_a_prospective_v2/analysis"
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    audit = read(directory / "diagnostic_audit.json")
    units = load_units(args.records_root, audit=audit,
                       config=read(ROOT / "configs/prospective_benchmark_v2.json"))
    equal = read(directory / "equal_budget_sensitivity.json")
    require_matching_audit(evaluate_portfolio(audit, units, GRAPH_ARCHITECTURES), equal["full_portfolio"])
    for name in GRAPH_ARCHITECTURES:
        require_matching_audit(evaluate_portfolio(audit, units, [name]), equal["equal_budget_single_architecture"][name])
    actual = json.loads(json.dumps(summarize(audit, units), allow_nan=False))
    require_matching_audit(actual, read(directory / "portfolio_robustness.json"))
    print("Verified release manifest, complete frozen audit, six single architectures, all 63 subsets, and seven held-out threshold analyses.")


if __name__ == "__main__":
    main()
