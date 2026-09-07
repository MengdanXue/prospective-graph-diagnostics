"""Post-hoc subset and leave-one-dataset-out threshold analyses; no retraining."""
import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from scripts.summarize_equal_budget_sensitivity import (
    GRAPH_ARCHITECTURES, evaluate_portfolio, load_units, select_by_validation,
)

# Fixed grid, including constant policies. Ties choose the smallest threshold.
THRESHOLDS = (-1.0,) + tuple(i / 100 for i in range(101)) + (2.0,)
EXCLUSIONS = {"without_webkb_pair": ("Cornell", "Wisconsin"),
              "without_wikipedia_pair": ("Chameleon", "Squirrel")}


def mean(values):
    return math.fsum(values) / len(values)


def threshold_loss(rows, threshold):
    return mean([max(r["graph_test"], r["mlp_test"]) -
                 (r["graph_test"] if r["homophily"] >= threshold else r["mlp_test"])
                 for r in rows])


def fit_threshold(training_rows):
    """Only training-dataset outcomes enter threshold choice; clusters are equal-weighted."""
    groups = defaultdict(list)
    for row in training_rows:
        groups[row["dataset"]].append(row)
    if not groups:
        raise ValueError("threshold fitting requires training datasets")
    losses = [mean([threshold_loss(rows, t) for rows in groups.values()]) for t in THRESHOLDS]
    minimum = min(losses)
    return next(t for t, loss in zip(THRESHOLDS, losses) if loss <= minimum + 1e-12)


def leave_one_dataset_out(rows):
    datasets = sorted({r["dataset"] for r in rows})
    if len(datasets) < 2:
        raise ValueError("leave-one-dataset-out requires at least two datasets")
    folds = {}
    for dataset in datasets:
        training = [r for r in rows if r["dataset"] != dataset]
        heldout = [r for r in rows if r["dataset"] == dataset]
        threshold = fit_threshold(training)
        folds[dataset] = {"threshold": threshold, "regret_pp": 100 * threshold_loss(heldout, threshold),
                          "graph_actions": sum(r["homophily"] >= threshold for r in heldout),
                          "units": len(heldout), "training_datasets": len(datasets) - 1}
    return {"mean_regret_pp": mean([f["regret_pp"] for f in folds.values()]), "folds": folds}


def summarize(audit, units):
    portfolios = []
    calibrated = {}
    for size in range(1, len(GRAPH_ARCHITECTURES) + 1):
        for names in itertools.combinations(GRAPH_ARCHITECTURES, size):
            entry = evaluate_portfolio(audit, units, names)
            entry["delta_pp"] = entry["mean_regret_pp"]["combined"] - entry["mean_regret_pp"]["always_graph"]
            differences = {k: v["combined"] - v["always_graph"] for k, v in entry["datasets"].items()}
            entry["leave_one_out_delta_range_pp"] = [
                min(mean([v for k, v in differences.items() if k != omit]) for omit in differences),
                max(mean([v for k, v in differences.items() if k != omit]) for omit in differences)]
            entry["exclusion_delta_pp"] = {key: mean([v for k, v in differences.items() if k not in excluded])
                                            for key, excluded in EXCLUSIONS.items()}
            rows = []
            for unit in audit["units"]:
                records = units[(unit["dataset"], int(unit["seed"]))]
                graph = select_by_validation(records[name] for name in names)
                rows.append({"dataset": unit["dataset"], "homophily": unit["homophily"],
                             "graph_test": float(graph["test_accuracy"]), "mlp_test": float(records["MLP"]["test_accuracy"]),
                             "combined_graph": unit["decisions"]["historical_combined"]["action"] == "graph"})
            direct = 100 * mean([0 if r["combined_graph"] else r["graph_test"] - r["mlp_test"] for r in rows])
            if not math.isclose(direct, entry["delta_pp"], abs_tol=1e-10):
                raise ValueError("oracle-cancellation identity failed")
            if size in (1, len(GRAPH_ARCHITECTURES)):
                calibrated[names[0] if size == 1 else "full"] = leave_one_dataset_out(rows)
            portfolios.append(entry)
    by_size = []
    for size in range(1, 7):
        group = [r for r in portfolios if len(r["portfolio"]) == size]
        by_size.append({"size": size, "count": len(group),
                        "combined_better": sum(r["delta_pp"] < -1e-9 for r in group),
                        "min_delta_pp": min(r["delta_pp"] for r in group),
                        "max_delta_pp": max(r["delta_pp"] for r in group)})
    return {"schema_version": "1.0", "analysis_status": "post_hoc_descriptive",
            "source_benchmark_id": audit["benchmark_id"], "subsets": portfolios, "by_size": by_size,
            "exclusions": EXCLUSIONS, "threshold_grid": THRESHOLDS,
            "threshold_tie_rule": "smallest threshold within 1e-12 of minimum training loss",
            "threshold_training": "equal-weight mean raw regret over the other ten dataset clusters; their selected-model test outcomes are meta-training targets",
            "threshold_evaluation": "held-out dataset outcomes never select its threshold; same original datasets, not independent prospective validation",
            "calibrated_threshold": calibrated}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT / "results/diagnostic/route_a_prospective_v2/analysis"
    audit = json.loads((directory / "diagnostic_audit.json").read_text())
    config = json.loads((ROOT / "configs/prospective_benchmark_v2.json").read_text())
    units = load_units(args.records_root, audit=audit, config=config)
    result = summarize(audit, units)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("Verified all 63 subsets; leave-one-dataset-out threshold regrets (pp):")
    print({k: round(v["mean_regret_pp"], 4) for k, v in result["calibrated_threshold"].items()})


if __name__ == "__main__":
    main()
