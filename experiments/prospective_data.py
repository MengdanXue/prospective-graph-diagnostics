#!/usr/bin/env python3
"""Deterministic data and train-only diagnostic utilities for Route A."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


_DATASET_SPECIFICATIONS = {
    "Cora": {"class": "Planetoid", "name": "Cora"},
    "CiteSeer": {"class": "Planetoid", "name": "CiteSeer"},
    "PubMed": {"class": "Planetoid", "name": "PubMed"},
    "Texas": {"class": "WebKB", "name": "Texas"},
    "Wisconsin": {"class": "WebKB", "name": "Wisconsin"},
    "Cornell": {"class": "WebKB", "name": "Cornell"},
    "Chameleon": {"class": "WikipediaNetwork", "name": "chameleon"},
    "Squirrel": {"class": "WikipediaNetwork", "name": "squirrel"},
    "Actor": {"class": "Actor"},
    "Roman-empire": {"class": "HeterophilousGraphDataset", "name": "Roman-empire"},
    "Amazon-ratings": {"class": "HeterophilousGraphDataset", "name": "Amazon-ratings"},
    "Coauthor-CS": {"class": "Coauthor", "name": "CS"},
}


def dataset_specifications() -> dict[str, dict[str, str]]:
    return {name: dict(specification) for name, specification in _DATASET_SPECIFICATIONS.items()}


def validate_dataset_eligibility(
    labels: np.ndarray,
    *,
    minimum_class_count: int,
) -> dict[str, int]:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if minimum_class_count < 1:
        raise ValueError("minimum_class_count must be positive")
    values, counts = np.unique(labels, return_counts=True)
    class_counts = {
        str(value.item() if hasattr(value, "item") else value): int(count)
        for value, count in zip(values, counts)
    }
    if any(count < minimum_class_count for count in class_counts.values()):
        raise ValueError(
            f"dataset is ineligible: minimum_class_count={minimum_class_count}; "
            f"class_counts={class_counts}"
        )
    return class_counts


def make_stratified_split(
    labels: np.ndarray,
    *,
    seed: int,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("split ratios must be positive and sum to less than one")
    rng = np.random.default_rng(seed)
    parts: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 3:
            raise ValueError(f"class {label!r} has fewer than three examples")
        indices = rng.permutation(indices)
        train_count = max(1, int(np.floor(len(indices) * train_ratio)))
        validation_count = max(1, int(np.floor(len(indices) * validation_ratio)))
        if train_count + validation_count >= len(indices):
            train_count = len(indices) - 2
            validation_count = 1
        parts["train"].extend(indices[:train_count].tolist())
        parts["validation"].extend(
            indices[train_count : train_count + validation_count].tolist()
        )
        parts["test"].extend(indices[train_count + validation_count :].tolist())
    return {
        name: np.asarray(sorted(indices), dtype=np.int64)
        for name, indices in parts.items()
    }


def split_identifier(split: dict[str, np.ndarray]) -> str:
    payload = {
        name: np.asarray(split[name], dtype=np.int64).tolist()
        for name in ("train", "validation", "test")
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_undirected_edges(edge_index: Any, *, node_count: int) -> np.ndarray:
    values = np.asarray(edge_index, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if node_count < 1:
        raise ValueError("node_count must be positive")
    edges: set[tuple[int, int]] = set()
    for left, right in values.T.tolist():
        if not 0 <= left < node_count or not 0 <= right < node_count:
            raise ValueError("edge endpoint outside node range")
        if left == right:
            continue
        edges.add((left, right) if left < right else (right, left))
    if not edges:
        raise ValueError("graph has no non-loop edges")
    return np.asarray(sorted(edges), dtype=np.int64)


def train_only_diagnostics(
    edges: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    edges = np.asarray(edges, dtype=np.int64)
    labels = np.asarray(labels)
    train_mask = np.asarray(train_mask, dtype=bool)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape [num_edges, 2]")
    if labels.ndim != 1 or train_mask.shape != labels.shape:
        raise ValueError("labels and train_mask must be aligned vectors")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")

    node_count = len(labels)
    mean_degree = 2.0 * len(edges) / node_count
    train_edges = edges[train_mask[edges[:, 0]] & train_mask[edges[:, 1]]]
    if len(train_edges):
        one_hop = float(np.mean(labels[train_edges[:, 0]] == labels[train_edges[:, 1]]))
    else:
        one_hop = None

    neighbors: list[list[int]] = [[] for _ in range(node_count)]
    for left, right in train_edges.tolist():
        neighbors[left].append(right)
        neighbors[right].append(left)
    eligible = np.asarray(
        [node for node, values in enumerate(neighbors) if len(values) >= 2],
        dtype=np.int64,
    )
    accepted = 0
    two_hop = None
    if len(eligible):
        weights = np.asarray(
            [len(neighbors[node]) * (len(neighbors[node]) - 1) for node in eligible],
            dtype=np.float64,
        )
        weights /= weights.sum()
        rng = np.random.default_rng(seed)
        agreements = np.empty(sample_count, dtype=np.float64)
        centers = rng.choice(eligible, size=sample_count, replace=True, p=weights)
        for index, center in enumerate(centers.tolist()):
            endpoint_a, endpoint_b = rng.choice(neighbors[center], size=2, replace=False)
            agreements[index] = labels[endpoint_a] == labels[endpoint_b]
        accepted = sample_count
        two_hop = float(agreements.mean())

    return {
        "homophily": one_hop,
        "mean_degree": mean_degree,
        "h_2": two_hop,
        "delta_h": None if one_hop is None or two_hop is None else two_hop - one_hop,
        "train_edge_count": int(len(train_edges)),
        "two_hop_requested_walks": int(sample_count),
        "two_hop_attempted_walks": int(sample_count if len(eligible) else 0),
        "two_hop_accepted_walks": int(accepted),
        "two_hop_sampling_seed": int(seed),
        "two_hop_estimand": "path_weighted_length_two_endpoint_agreement",
        "homophily_label_scope": "train_only",
        "two_hop_label_scope": "train_only",
        "uses_test_labels": False,
    }


def checksum_files(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return records
