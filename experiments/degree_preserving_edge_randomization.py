#!/usr/bin/env python3
"""Auditable degree-preserving randomization for simple undirected graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path
from typing import Any, Iterable


Edge = tuple[int, int]


def canonical_edge(left: int, right: int) -> Edge:
    return (left, right) if left < right else (right, left)


def normalize_edges(node_count: int, edges: Iterable[Iterable[int]]) -> set[Edge]:
    if not isinstance(node_count, int) or isinstance(node_count, bool) or node_count < 1:
        raise ValueError("node_count must be a positive integer")
    normalized: set[Edge] = set()
    for raw_edge in edges:
        values = list(raw_edge)
        if len(values) != 2:
            raise ValueError("each edge must contain exactly two node identifiers")
        left, right = values
        if (
            not isinstance(left, int)
            or isinstance(left, bool)
            or not isinstance(right, int)
            or isinstance(right, bool)
        ):
            raise ValueError("node identifiers must be integers")
        if not 0 <= left < node_count or not 0 <= right < node_count:
            raise ValueError("edge endpoint lies outside [0, node_count)")
        if left == right:
            raise ValueError("self-loops are not allowed in the simple-graph intervention")
        edge = canonical_edge(left, right)
        if edge in normalized:
            raise ValueError("duplicate undirected edges are not allowed")
        normalized.add(edge)
    if len(normalized) < 2:
        raise ValueError("at least two edges are required")
    return normalized


def degree_sequence(node_count: int, edges: set[Edge]) -> list[int]:
    degrees = [0] * node_count
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    return degrees


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def double_edge_swap(
    edges: set[Edge],
    *,
    n_swaps: int,
    seed: int,
    max_attempts: int | None = None,
) -> tuple[set[Edge], dict[str, int]]:
    if not isinstance(n_swaps, int) or isinstance(n_swaps, bool) or n_swaps < 0:
        raise ValueError("n_swaps must be a non-negative integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if max_attempts is None:
        max_attempts = max(100, n_swaps * 1000)
    if max_attempts < n_swaps:
        raise ValueError("max_attempts must be at least n_swaps")

    randomized = set(edges)
    edge_list = sorted(randomized)
    rng = random.Random(seed)
    successful = 0
    attempts = 0
    while successful < n_swaps and attempts < max_attempts:
        attempts += 1
        first_index, second_index = rng.sample(range(len(edge_list)), 2)
        first = edge_list[first_index]
        second = edge_list[second_index]
        left_a, right_a = first
        left_b, right_b = second
        if len({left_a, right_a, left_b, right_b}) != 4:
            continue

        if rng.randrange(2) == 0:
            candidate_one = canonical_edge(left_a, left_b)
            candidate_two = canonical_edge(right_a, right_b)
        else:
            candidate_one = canonical_edge(left_a, right_b)
            candidate_two = canonical_edge(right_a, left_b)
        if candidate_one == candidate_two:
            continue
        if candidate_one in randomized or candidate_two in randomized:
            continue

        randomized.remove(first)
        randomized.remove(second)
        randomized.add(candidate_one)
        randomized.add(candidate_two)
        edge_list[first_index] = candidate_one
        edge_list[second_index] = candidate_two
        successful += 1

    if successful != n_swaps:
        raise RuntimeError(
            f"completed {successful} of {n_swaps} requested swaps after {attempts} attempts"
        )
    return randomized, {
        "requested_swaps": n_swaps,
        "successful_swaps": successful,
        "attempts": attempts,
        "max_attempts": max_attempts,
    }


def adjacency(node_count: int, edges: set[Edge]) -> list[set[int]]:
    neighbors = [set() for _ in range(node_count)]
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    return neighbors


def component_statistics(neighbors: list[set[int]]) -> tuple[int, float]:
    unvisited = set(range(len(neighbors)))
    sizes = []
    while unvisited:
        start = min(unvisited)
        stack = [start]
        unvisited.remove(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in neighbors[node]:
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return len(sizes), max(sizes) / len(neighbors)


def degree_assortativity(edges: set[Edge], degrees: list[int]) -> float | None:
    left_values = []
    right_values = []
    for left, right in sorted(edges):
        left_values.extend([degrees[left], degrees[right]])
        right_values.extend([degrees[right], degrees[left]])
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    numerator = sum(
        (left - left_mean) * (right - right_mean)
        for left, right in zip(left_values, right_values)
    )
    left_scale = sum((value - left_mean) ** 2 for value in left_values)
    right_scale = sum((value - right_mean) ** 2 for value in right_values)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator > 0.0 else None


def average_clustering(neighbors: list[set[int]]) -> float:
    coefficients = []
    for node_neighbors in neighbors:
        degree = len(node_neighbors)
        if degree < 2:
            coefficients.append(0.0)
            continue
        ordered = sorted(node_neighbors)
        links = sum(
            ordered[right_index] in neighbors[ordered[left_index]]
            for left_index in range(len(ordered))
            for right_index in range(left_index + 1, len(ordered))
        )
        coefficients.append(links / (degree * (degree - 1) / 2))
    return sum(coefficients) / len(coefficients)


def graph_statistics(
    node_count: int, edges: set[Edge], labels: list[Any] | None
) -> dict[str, Any]:
    neighbors = adjacency(node_count, edges)
    degrees = degree_sequence(node_count, edges)
    components, largest_fraction = component_statistics(neighbors)
    if labels is None:
        homophily = None
    else:
        if len(labels) != node_count:
            raise ValueError("labels must have node_count entries")
        homophily = sum(labels[left] == labels[right] for left, right in edges) / len(edges)
    return {
        "homophily": homophily,
        "component_count": components,
        "largest_component_fraction": largest_fraction,
        "degree_assortativity": degree_assortativity(edges, degrees),
        "average_clustering": average_clustering(neighbors),
    }


def structural_changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    changes = {}
    for key in before:
        if before[key] is None or after[key] is None:
            changes[key] = None
        else:
            changes[key] = after[key] - before[key]
    return changes


def invariant_audit(node_count: int, before: set[Edge], after: set[Edge]) -> dict[str, Any]:
    degrees_before = degree_sequence(node_count, before)
    degrees_after = degree_sequence(node_count, after)
    sorted_before = sorted(degrees_before)
    sorted_after = sorted(degrees_after)
    return {
        "node_count_before": node_count,
        "node_count_after": node_count,
        "node_count_identical": True,
        "edge_count_before": len(before),
        "edge_count_after": len(after),
        "edge_count_identical": len(before) == len(after),
        "degree_sequence_by_node_identical": degrees_before == degrees_after,
        "sorted_degree_sequence_identical": sorted_before == sorted_after,
        "degree_sequence_by_node_sha256_before": sha256_json(degrees_before),
        "degree_sequence_by_node_sha256_after": sha256_json(degrees_after),
        "degree_sequence_sha256_before": sha256_json(sorted_before),
        "degree_sequence_sha256_after": sha256_json(sorted_after),
        "edge_set_sha256_before": sha256_json(sorted(before)),
        "edge_set_sha256_after": sha256_json(sorted(after)),
        "simple_graph_before": len(before) == len(set(before)),
        "simple_graph_after": len(after) == len(set(after)),
    }


def run_intervention(
    *,
    node_count: int,
    edges: Iterable[Iterable[int]],
    labels: list[Any] | None,
    n_swaps: int,
    seed: int,
) -> dict[str, Any]:
    original = normalize_edges(node_count, edges)
    randomized, swap_status = double_edge_swap(original, n_swaps=n_swaps, seed=seed)
    invariants = invariant_audit(node_count, original, randomized)
    if not all(
        invariants[key]
        for key in (
            "node_count_identical",
            "edge_count_identical",
            "degree_sequence_by_node_identical",
            "sorted_degree_sequence_identical",
            "simple_graph_before",
            "simple_graph_after",
        )
    ):
        raise RuntimeError("graph invariant audit failed")
    before = graph_statistics(node_count, original, labels)
    after = graph_statistics(node_count, randomized, labels)
    return {
        "schema_version": "1.0",
        "algorithm": "simple_undirected_double_edge_swap",
        "status": "success",
        "node_count": node_count,
        "seed": seed,
        "n_swaps": n_swaps,
        "swap_status": swap_status,
        "original_edges": [list(edge) for edge in sorted(original)],
        "randomized_edges": [list(edge) for edge in sorted(randomized)],
        "audit": {
            "invariants": invariants,
            "before": before,
            "after": after,
            "changes": structural_changes(before, after),
        },
        "environment": {"python": platform.python_version()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        print(f"Refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0":
            raise ValueError("schema_version must equal 1.0")
        result = run_intervention(
            node_count=payload["node_count"],
            edges=payload["edges"],
            labels=payload.get("labels"),
            n_swaps=payload["n_swaps"],
            seed=payload["seed"],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result["swap_status"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
