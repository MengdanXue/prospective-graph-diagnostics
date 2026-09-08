"""Summarize the paired raw-vs-NormalizeFeatures post-hoc run."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_equal_budget_sensitivity import select_by_validation

MODELS = ("MLP", "GCN", "GAT", "GraphSAGE", "H2GCN", "LINKX", "GPR-GNN")
GRAPH = MODELS[1:]
POLICIES = ("always_graph", "always_mlp", "historical_combined", "validation_selection")


def mean(values):
    return math.fsum(values) / len(values)


def summarize(root, audit):
    records = defaultdict(dict)
    for path in root.glob("records/*/*/*/seed_*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "success":
            raise ValueError(f"non-success record: {path}")
        records[(row["preprocessing"], row["dataset"], int(row["seed"]))][row["model"]] = row
    expected = {("normalize_features" if c == "normalize_features" else "raw", u["dataset"], int(u["seed"]))
                for c in ("normalize_features", "raw") for u in audit["units"]
                if u["dataset"] in ("Roman-empire", "Amazon-ratings")}
    if set(records) != expected or any(set(v) != set(MODELS) for v in records.values()):
        raise ValueError("record scope does not match the configured 280 model records")
    audit_units = {(u["dataset"], int(u["seed"])): u for u in audit["units"]}
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    out = {"schema_version": "1.0", "analysis_status": "post_hoc_paired_sensitivity",
           "source_run": {"run_id": manifest["run_id"], "source_commit": manifest["source_commit"],
                          "config_sha256": manifest["config_sha256"], "environment": manifest["environment"],
                          "raw_record_count": complete["expected_records"], "scope_verified": True},
           "datasets": {}, "conditions": {}, "paired_differences": {}}
    for condition in ("normalize_features", "raw"):
        all_units = []
        for dataset in ("Roman-empire", "Amazon-ratings"):
            rows = []
            for seed in range(10):
                unit = audit_units[(dataset, seed)]
                models = records[(condition, dataset, seed)]
                graph = select_by_validation(models[n] for n in GRAPH)
                mlp = models["MLP"]
                graph_acc, mlp_acc = float(graph["test_accuracy"]), float(mlp["test_accuracy"])
                oracle = max(graph_acc, mlp_acc)
                target = "graph" if graph_acc - mlp_acc > .01 else "mlp"
                actions = {"always_graph": "graph", "always_mlp": "mlp",
                           "historical_combined": "graph" if unit["homophily"] >= .55 else "mlp",
                           "validation_selection": "graph" if float(graph["validation_accuracy"]) - float(mlp["validation_accuracy"]) > .01 else "mlp"}
                row = {"dataset": dataset, "seed": seed, "selected_graph": graph["model"],
                       "graph_test": graph_acc, "mlp_test": mlp_acc, "graph_validation": float(graph["validation_accuracy"]),
                       "mlp_validation": float(mlp["validation_accuracy"]), "target": target,
                       "test_gap": graph_acc - mlp_acc,
                       "actions": actions,
                       "regret": {k: oracle - (graph_acc if a == "graph" else mlp_acc) for k, a in actions.items()}}
                rows.append(row); all_units.append(row)
            ds = out["datasets"].setdefault(dataset, {})
            ds[condition] = {"units": rows, "target_graph": sum(r["target"] == "graph" for r in rows),
                             "mean_graph_test": mean([r["graph_test"] for r in rows]),
                             "mean_mlp_test": mean([r["mlp_test"] for r in rows]),
                             "mean_test_gap": mean([r["test_gap"] for r in rows]),
                             "selected_graph_counts": {m: sum(r["selected_graph"] == m for r in rows) for m in GRAPH},
                             "mean_regret_pp": {p: 100 * mean([r["regret"][p] for r in rows]) for p in POLICIES}}
        out["conditions"][condition] = {"units": len(all_units), "target_graph": sum(r["target"] == "graph" for r in all_units),
                                         "mean_graph_test": mean([r["graph_test"] for r in all_units]),
                                         "mean_mlp_test": mean([r["mlp_test"] for r in all_units]),
                                         "mean_test_gap": mean([r["test_gap"] for r in all_units]),
                                         "mean_regret_pp": {p: 100 * mean([r["regret"][p] for r in all_units]) for p in POLICIES}}
    for dataset in ("Roman-empire", "Amazon-ratings"):
        for key in ("mlp_test", "graph_test", "test_gap"):
            a = out["datasets"][dataset]["normalize_features"]["units"]
            b = out["datasets"][dataset]["raw"]["units"]
            out["paired_differences"].setdefault(dataset, {})[key] = {"raw_minus_normalized_by_seed": [r[key] - a[i][key] for i, r in enumerate(b)],
                                                                        "mean_raw_minus_normalized": mean([r[key] - a[i][key] for i, r in enumerate(b)])}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((ROOT / "results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json").read_text())
    result = summarize(args.root, audit)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for condition, row in result["conditions"].items():
        print(condition, {k: round(v, 3) for k, v in row["mean_regret_pp"].items()}, "target_graph", row["target_graph"])


if __name__ == "__main__":
    main()
