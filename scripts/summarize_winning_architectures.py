#!/usr/bin/env python3
"""Summarize which graph architecture wins each unit, and where regret concentrates.

This script performs a deterministic re-aggregation of the frozen diagnostic
audit. It trains nothing, changes no action, and introduces no new experimental
setting. Every quantity it emits is a function of fields already present in
``diagnostic_audit.json``.

The output supports the structural-mismatch analysis: an edge-homophily
threshold selects the graph action only on high-homophily datasets, whereas the
graph portfolio's realized advantage is largely supplied by heterophily-aware
architectures on low-homophily datasets.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Architectures in the frozen graph portfolio that are designed to remain
# effective when one-hop homophily is low. Fixed by the benchmark configuration,
# not selected after observing outcomes.
HETEROPHILY_AWARE = ("GPR-GNN", "H2GCN", "LINKX")

COMBINED = "historical_combined"


def unit_regret(unit: dict[str, Any], action: str) -> float:
    """Oracle-portfolio regret for one unit under a realized action."""

    graph = unit["selected_graph_test"]
    mlp = unit["selected_mlp_test"]
    chosen = graph if action == "graph" else mlp
    return max(graph, mlp) - chosen


def resolved_action(unit: dict[str, Any], policy: str, fallback: str = "mlp") -> str:
    """Action after the frozen abstention fallback."""

    action = unit["decisions"][policy]["action"]
    return fallback if action == "abstain" else action


def summarize(audit: dict[str, Any]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in audit["units"]:
        by_dataset[unit["dataset"]].append(unit)

    datasets: dict[str, Any] = {}
    for name, units in by_dataset.items():
        counts = Counter(unit["selected_graph"] for unit in units)
        heterophily_aware = sum(
            count for model, count in counts.items() if model in HETEROPHILY_AWARE
        )
        combined_regret = sum(
            unit_regret(unit, resolved_action(unit, COMBINED)) for unit in units
        ) / len(units)
        always_graph_regret = sum(
            unit_regret(unit, "graph") for unit in units
        ) / len(units)
        datasets[name] = {
            "units": len(units),
            "mean_train_label_homophily": sum(u["homophily"] for u in units) / len(units),
            "winning_graph_architectures": dict(sorted(counts.items())),
            "heterophily_aware_wins": heterophily_aware,
            "combined_selects_graph": sum(
                1 for u in units if resolved_action(u, COMBINED) == "graph"
            ),
            "mean_combined_regret": combined_regret,
            "mean_always_graph_regret": always_graph_regret,
        }

    total_units = len(audit["units"])
    total_heterophily_aware = sum(
        1 for u in audit["units"] if u["selected_graph"] in HETEROPHILY_AWARE
    )
    regret_total = sum(d["mean_combined_regret"] for d in datasets.values())
    for name, entry in datasets.items():
        entry["share_of_combined_regret"] = (
            entry["mean_combined_regret"] / regret_total if regret_total else 0.0
        )

    ranked = sorted(
        datasets.items(), key=lambda kv: kv[1]["mean_combined_regret"], reverse=True
    )
    top_four = [name for name, _ in ranked[:4]]

    return {
        "schema_version": "1.0",
        "source_benchmark_id": audit["benchmark_id"],
        "heterophily_aware_architectures": list(HETEROPHILY_AWARE),
        "totals": {
            "units": total_units,
            "heterophily_aware_wins": total_heterophily_aware,
            "heterophily_aware_win_share": total_heterophily_aware / total_units,
        },
        "regret_concentration": {
            "top_four_datasets": top_four,
            "top_four_share_of_combined_regret": sum(
                datasets[name]["share_of_combined_regret"] for name in top_four
            ),
            "top_four_units": sum(datasets[name]["units"] for name in top_four),
            "top_four_heterophily_aware_wins": sum(
                datasets[name]["heterophily_aware_wins"] for name in top_four
            ),
            "top_four_combined_graph_actions": sum(
                datasets[name]["combined_selects_graph"] for name in top_four
            ),
            "top_four_max_homophily": max(
                datasets[name]["mean_train_label_homophily"] for name in top_four
            ),
        },
        "datasets": dict(sorted(datasets.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    summary = summarize(audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    totals = summary["totals"]
    concentration = summary["regret_concentration"]
    print(
        f"heterophily-aware wins: {totals['heterophily_aware_wins']}/{totals['units']} "
        f"({100 * totals['heterophily_aware_win_share']:.0f}%)"
    )
    print(
        f"top-four regret share: {100 * concentration['top_four_share_of_combined_regret']:.1f}% "
        f"over {concentration['top_four_units']} units; "
        f"{concentration['top_four_heterophily_aware_wins']} heterophily-aware wins; "
        f"{concentration['top_four_combined_graph_actions']} combined graph actions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
