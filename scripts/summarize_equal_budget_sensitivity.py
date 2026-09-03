#!/usr/bin/env python3
"""Equal-total-compute sensitivity analysis for the graph-vs-MLP decision.

The frozen benchmark compares asymmetric portfolios: the graph action selects
among six architectures (24 trials) while the feature-only action contains one
MLP (four trials). This script asks what changes when the graph action is
restricted to a single architecture, which makes the comparison four trials
against four trials.

No model is retrained. Every quantity is recomputed from the immutable
per-unit records in the v0.1.0 release archive, using the same portfolio
selection rule as experiments/evaluate_diagnostics.py: validation accuracy
descending, then model identifier ascending.

The diagnostic policies themselves are portfolio-independent. Edge homophily
and MLP validation accuracy do not change when the graph portfolio is
restricted, so each rule's action is reused unchanged from the frozen audit;
only the realized outcomes, the decision target, and validation selection are
recomputed.

Usage:

    python scripts/summarize_equal_budget_sensitivity.py \\
      --records-root tmp/artifacts/formal-v0.1.0/prospective/records \\
      --audit results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json \\
      --output tmp/equal_budget_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

GRAPH_ARCHITECTURES = ("GAT", "GCN", "GPR-GNN", "GraphSAGE", "H2GCN", "LINKX")
MLP = "MLP"
PRACTICAL_MARGIN = 0.01


def load_units(records_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Index every per-unit model record by (dataset, seed) and model."""

    units: dict[tuple[str, int], dict[str, Any]] = defaultdict(dict)
    for path in sorted(records_root.glob("*/*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "success":
            raise ValueError(f"record is not successful: {path}")
        units[(record["dataset"], int(record["seed"]))][record["model"]] = record
    if not units:
        raise ValueError(f"no records found under {records_root}")
    return units


def select_by_validation(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Portfolio selection rule used by the frozen evaluator."""

    return sorted(
        records, key=lambda row: (-float(row["validation_accuracy"]), str(row["model"]))
    )[0]


def evaluate_portfolio(
    audit: dict[str, Any],
    units: dict[tuple[str, int], dict[str, Any]],
    portfolio: Sequence[str],
) -> dict[str, Any]:
    """Recompute targets and policy outcomes for one graph portfolio."""

    per_dataset: dict[str, dict[str, float]] = defaultdict(
        lambda: {"always_graph": 0.0, "combined": 0.0, "validation": 0.0, "units": 0.0}
    )
    graph_targets = 0
    accuracy = {"always_graph": 0, "combined": 0, "validation": 0}
    total = {"always_graph": 0.0, "combined": 0.0, "validation": 0.0}

    for unit in audit["units"]:
        models = units[(unit["dataset"], int(unit["seed"]))]
        graph = select_by_validation(models[name] for name in portfolio)
        mlp = models[MLP]
        graph_test = float(graph["test_accuracy"])
        mlp_test = float(mlp["test_accuracy"])
        oracle = max(graph_test, mlp_test)
        target = "graph" if graph_test - mlp_test > PRACTICAL_MARGIN else "mlp"
        graph_targets += target == "graph"

        combined = unit["decisions"]["historical_combined"]["action"]
        combined = "mlp" if combined == "abstain" else combined
        validation = (
            "graph"
            if float(graph["validation_accuracy"]) - float(mlp["validation_accuracy"])
            > PRACTICAL_MARGIN
            else "mlp"
        )

        actions = {"always_graph": "graph", "combined": combined, "validation": validation}
        bucket = per_dataset[unit["dataset"]]
        bucket["units"] += 1
        for policy, action in actions.items():
            regret = oracle - (graph_test if action == "graph" else mlp_test)
            total[policy] += regret
            bucket[policy] += regret
            accuracy[policy] += action == target

    count = len(audit["units"])
    datasets = {
        name: {
            policy: 100 * bucket[policy] / bucket["units"]
            for policy in ("always_graph", "combined", "validation")
        }
        for name, bucket in per_dataset.items()
    }
    wins = sum(1 for d in datasets.values() if d["combined"] < d["always_graph"] - 1e-9)
    losses = sum(1 for d in datasets.values() if d["combined"] > d["always_graph"] + 1e-9)
    nonzero = wins + losses

    return {
        "portfolio": list(portfolio),
        "total_trials": 4 * len(portfolio),
        "graph_targets": graph_targets,
        "units": count,
        "mean_regret_pp": {
            policy: 100 * value / count for policy, value in sorted(total.items())
        },
        "selection_accuracy": {
            policy: value / count for policy, value in sorted(accuracy.items())
        },
        "combined_vs_always_graph": {
            "dataset_wins": wins,
            "dataset_losses": losses,
            "dataset_ties": len(datasets) - nonzero,
            "nonzero_datasets": nonzero,
            "sign_flip_p_floor": 2 / 2**nonzero if nonzero else None,
        },
        "datasets": dict(sorted(datasets.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    units = load_units(args.records_root)

    full = evaluate_portfolio(audit, units, GRAPH_ARCHITECTURES)
    published = {
        "graph_targets": 89,
        "always_graph": 0.26,
        "combined": 7.46,
        "validation": 0.22,
    }
    checks = {
        "graph_targets": full["graph_targets"] == published["graph_targets"],
        **{
            policy: abs(full["mean_regret_pp"][policy] - published[policy]) < 0.005
            for policy in ("always_graph", "combined", "validation")
        },
    }
    if not all(checks.values()):
        failed = sorted(name for name, ok in checks.items() if not ok)
        raise SystemExit(
            "full-portfolio reproduction does not match the published values: "
            + ", ".join(failed)
        )

    summary = {
        "schema_version": "1.0",
        "source_benchmark_id": audit["benchmark_id"],
        "practical_margin": PRACTICAL_MARGIN,
        "reproduction_check": "full portfolio reproduces the published values",
        "full_portfolio": full,
        "equal_budget_single_architecture": {
            name: evaluate_portfolio(audit, units, [name]) for name in GRAPH_ARCHITECTURES
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"full portfolio (24 trials): graph targets {full['graph_targets']}/110, "
          f"always-graph {full['mean_regret_pp']['always_graph']:.2f} pp, "
          f"combined {full['mean_regret_pp']['combined']:.2f} pp  [reproduces published values]")
    for name in GRAPH_ARCHITECTURES:
        entry = summary["equal_budget_single_architecture"][name]
        cmp_ = entry["combined_vs_always_graph"]
        print(
            f"  {name:<10} (4 trials): graph targets {entry['graph_targets']:>3}/110, "
            f"always-graph {entry['mean_regret_pp']['always_graph']:>5.2f} pp, "
            f"combined {entry['mean_regret_pp']['combined']:>5.2f} pp, "
            f"dataset W/L/T {cmp_['dataset_wins']}/{cmp_['dataset_losses']}/{cmp_['dataset_ties']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
