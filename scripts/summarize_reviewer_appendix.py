#!/usr/bin/env python3
"""Derive reviewer-facing descriptive summaries from the frozen audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REQUIRED_POLICIES = (
    "historical_combined",
    "always_graph",
    "homophily_only",
    "validation_selection",
)


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _primary_action(unit: dict[str, Any], policy: str, *, abstention_fallback: str = "mlp") -> str:
    action = str(unit["decisions"][policy]["action"])
    if action == "abstain":
        return abstention_fallback
    if action not in {"graph", "mlp"}:
        raise ValueError(f"unsupported action for {policy}: {action!r}")
    return action


def _regret(unit: dict[str, Any], action: str) -> float:
    graph = float(unit["selected_graph_test"])
    mlp = float(unit["selected_mlp_test"])
    return max(graph, mlp) - (graph if action == "graph" else mlp)


def _validate_group(dataset: str, units: list[dict[str, Any]], expected_seeds: int) -> None:
    if len(units) != expected_seeds:
        raise ValueError(f"{dataset}: expected {expected_seeds} units, observed {len(units)}")
    seeds = [int(unit["seed"]) for unit in units]
    if len(set(seeds)) != expected_seeds:
        raise ValueError(f"{dataset}: duplicate or missing seeds")
    for unit in units:
        missing = [policy for policy in REQUIRED_POLICIES if policy not in unit.get("decisions", {})]
        if missing:
            raise ValueError(f"{dataset}/seed={unit['seed']}: missing policies {missing}")
        for key in (
            "homophily",
            "mean_degree",
            "delta_h",
            "selected_graph_test",
            "selected_mlp_test",
            "target_action",
            "test_gap",
        ):
            if key not in unit:
                raise ValueError(f"{dataset}/seed={unit['seed']}: missing {key}")


def summarize_audit(
    audit: dict[str, Any],
    dataset_metadata: dict[str, dict[str, int]],
    *,
    expected_seeds: int = 10,
) -> dict[str, Any]:
    if audit.get("status") != "scored":
        raise ValueError("audit status must be 'scored'")
    units = list(audit.get("units") or [])
    if not units:
        raise ValueError("audit contains no units")
    expected_units = int(audit.get("record_counts", {}).get("evaluation_units", -1))
    if len(units) != expected_units:
        raise ValueError(f"audit unit count mismatch: expected {expected_units}, observed {len(units)}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        grouped[str(unit["dataset"])].append(unit)
    if set(grouped) != set(dataset_metadata):
        raise ValueError("dataset metadata does not match audit datasets")

    dataset_rows: dict[str, dict[str, Any]] = {}
    combined_counts: Counter[str] = Counter()
    combined_primary_actions: list[str] = []
    homophily_primary_actions: list[str] = []
    primary_regrets: list[float] = []
    graph_fallback_regrets: list[float] = []
    primary_correct: list[bool] = []
    graph_fallback_correct: list[bool] = []
    abstention_units: list[dict[str, Any]] = []

    for dataset in sorted(grouped):
        rows = sorted(grouped[dataset], key=lambda row: int(row["seed"]))
        _validate_group(dataset, rows, expected_seeds)
        metadata = dataset_metadata[dataset]
        for key in ("node_count", "edge_count", "class_count", "feature_count"):
            if key not in metadata:
                raise ValueError(f"{dataset}: missing metadata field {key}")

        local_counts: Counter[str] = Counter()
        local_combined_regrets: list[float] = []
        local_graph_regrets: list[float] = []
        local_validation_regrets: list[float] = []
        for unit in rows:
            raw_combined = str(unit["decisions"]["historical_combined"]["action"])
            if raw_combined not in {"graph", "mlp", "abstain"}:
                raise ValueError(f"unsupported combined action: {raw_combined!r}")
            local_counts[raw_combined] += 1
            combined_counts[raw_combined] += 1
            if raw_combined == "abstain":
                abstention_units.append(unit)

            primary = _primary_action(unit, "historical_combined", abstention_fallback="mlp")
            graph_fallback = _primary_action(unit, "historical_combined", abstention_fallback="graph")
            homophily = _primary_action(unit, "homophily_only")
            graph_action = _primary_action(unit, "always_graph")
            validation = _primary_action(unit, "validation_selection")
            combined_primary_actions.append(primary)
            homophily_primary_actions.append(homophily)
            primary_regret = _regret(unit, primary)
            graph_fallback_regret = _regret(unit, graph_fallback)
            primary_regrets.append(primary_regret)
            graph_fallback_regrets.append(graph_fallback_regret)
            local_combined_regrets.append(primary_regret)
            local_graph_regrets.append(_regret(unit, graph_action))
            local_validation_regrets.append(_regret(unit, validation))
            primary_correct.append(primary == unit["target_action"])
            graph_fallback_correct.append(graph_fallback == unit["target_action"])

        dataset_rows[dataset] = {
            **{key: int(metadata[key]) for key in ("node_count", "edge_count", "class_count", "feature_count")},
            "mean_degree": _mean([float(unit["mean_degree"]) for unit in rows]),
            "mean_train_edge_homophily": _mean([float(unit["homophily"]) for unit in rows]),
            "mean_two_hop_difference": _mean([float(unit["delta_h"]) for unit in rows]),
            "graph_target_count": sum(unit["target_action"] == "graph" for unit in rows),
            "mean_graph_minus_mlp_test_gap_pp": 100.0 * _mean([float(unit["test_gap"]) for unit in rows]),
            "combined_mean_regret_pp": 100.0 * _mean(local_combined_regrets),
            "always_graph_mean_regret_pp": 100.0 * _mean(local_graph_regrets),
            "validation_selection_mean_regret_pp": 100.0 * _mean(local_validation_regrets),
            "combined_action_counts": {
                action: int(local_counts[action]) for action in ("graph", "mlp", "abstain")
            },
        }

    graph_targets_among_abstentions = sum(
        unit["target_action"] == "graph" for unit in abstention_units
    )
    graph_raw_wins_among_abstentions = sum(
        float(unit["selected_graph_test"]) > float(unit["selected_mlp_test"])
        for unit in abstention_units
    )
    return {
        "schema_version": "1.0",
        "status": "derived_from_frozen_audit",
        "record_counts": {
            "datasets": len(dataset_rows),
            "evaluation_units": len(units),
            "seeds_per_dataset": expected_seeds,
        },
        "overall": {
            "combined_action_counts": {
                action: int(combined_counts[action]) for action in ("graph", "mlp", "abstain")
            },
            "combined_mlp_fallback_selection_accuracy_pct": 100.0 * _mean(primary_correct),
            "combined_mlp_fallback_mean_regret_pp": 100.0 * _mean(primary_regrets),
            "combined_vs_homophily_primary_action_differences": sum(
                left != right for left, right in zip(combined_primary_actions, homophily_primary_actions)
            ),
        },
        "fallback_sensitivity": {
            "analysis_scope": "post_hoc_descriptive_only",
            "primary_fallback": "mlp",
            "alternative_fallback": "graph",
            "abstention_count": len(abstention_units),
            "graph_targets_among_abstentions": int(graph_targets_among_abstentions),
            "graph_raw_wins_among_abstentions": int(graph_raw_wins_among_abstentions),
            "graph_fallback_selection_accuracy_pct": 100.0 * _mean(graph_fallback_correct),
            "graph_fallback_mean_regret_pp": 100.0 * _mean(graph_fallback_regrets),
        },
        "datasets": dataset_rows,
    }


def load_dataset_metadata(data_root: Path, dataset_names: list[str]) -> dict[str, dict[str, int]]:
    import numpy as np

    from experiments.prospective_data import canonical_undirected_edges
    from experiments.run_prospective_benchmark import _load_dataset

    metadata: dict[str, dict[str, int]] = {}
    for name in sorted(dataset_names):
        dataset = _load_dataset(name, data_root)
        data = dataset[0]
        edges = canonical_undirected_edges(data.edge_index.cpu().numpy(), node_count=data.num_nodes)
        labels = data.y.view(-1).cpu().numpy()
        metadata[name] = {
            "node_count": int(data.num_nodes),
            "edge_count": int(len(edges)),
            "class_count": int(len(np.unique(labels))),
            "feature_count": int(data.x.size(1)),
        }
    return metadata


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=10)
    args = parser.parse_args()
    audit_bytes = args.audit.read_bytes()
    audit = json.loads(audit_bytes)
    dataset_names = sorted({str(unit["dataset"]) for unit in audit.get("units", [])})
    metadata = load_dataset_metadata(args.data_root, dataset_names)
    summary = summarize_audit(audit, metadata, expected_seeds=args.expected_seeds)
    summary["provenance"] = {
        "source_audit": args.audit.name,
        "source_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "dataset_metadata_source": "read_only_pyg_cache",
        "primary_evaluator_unchanged": True,
    }
    _write_exclusive(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
