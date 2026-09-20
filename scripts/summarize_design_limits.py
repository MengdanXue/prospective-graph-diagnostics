"""Document threshold support, actual split sizes, and executed inference modes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def summarize(audit, diagnostics_root):
    h = [u["homophily"] for u in audit["units"]]
    datasets = sorted({u["dataset"] for u in audit["units"]})
    splits = {}
    for name in datasets:
        rows = [json.loads(p.read_text()) for p in (diagnostics_root / name).glob("*.json")]
        counts = [r["provenance"]["data"]["eligibility"]["class_counts"] for r in rows]
        if len(rows) != 10 or any(c != counts[0] for c in counts):
            raise ValueError("incomplete or inconsistent class-count provenance")
        n = sum(counts[0].values())
        train = sum(max(1, int(c * .6)) for c in counts[0].values())
        validation = sum(max(1, int(c * .2)) for c in counts[0].values())
        splits[name] = {"nodes": n, "train": train, "validation": validation, "test": n-train-validation,
                        "class_counts": counts[0]}
    return {"schema_version": "1.0", "source_benchmark_id": audit["benchmark_id"],
            "threshold_empty_interval": {"lower": max(v for v in h if v < .5), "upper": min(v for v in h if v >= .5),
                                         "interpretation": "all thresholds strictly inside this interval give identical full-set graph/MLP actions on these 110 units"},
            "splits": splits,
            "inference": {"datasets": len(datasets), "family_size": 8,
                          "identical_full_set_policy": "homophily_only",
                          "homophily_only_mode": "all_zero_short_circuit_p_1", "homophily_only_enumerated_patterns": 0,
                          "other_seven_comparisons_mode": "exhaustive", "enumerated_patterns_per_other_comparison": 2**len(datasets),
                          "random_50_50": "analytic expected regret, not a sampled action sequence",
                          "configured_monte_carlo_samples": 10000, "configured_seed": 20260809,
                          "monte_carlo_parameters_used_for_these_comparisons": False,
                          "family_preserved": "no post-outcome deletion of the identical-policy comparison"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((ROOT / "results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json").read_text())
    result = summarize(audit, args.diagnostics_root)
    with args.output.open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
