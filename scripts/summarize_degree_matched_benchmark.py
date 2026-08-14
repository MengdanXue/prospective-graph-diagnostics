#!/usr/bin/env python3
"""Validate and summarize immutable degree-matched paired records."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.run_prospective_benchmark import (
    config_sha256,
    validate_trial_audit,
    write_json_exclusive,
)
from experiments.degree_preserving_edge_randomization import invariant_audit, normalize_edges


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "prospective_benchmark_v1.json"


def _bootstrap_interval(
    differences: np.ndarray, samples: int, rng: np.random.Generator
) -> list[float]:
    indices = rng.integers(0, differences.size, size=(samples, differences.size))
    means = differences[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _sign_flip_p_value(
    differences: np.ndarray, samples: int, rng: np.random.Generator
) -> float:
    observed = abs(float(differences.mean()))
    if differences.size <= 18:
        signs = np.asarray(
            list(itertools.product([-1.0, 1.0], repeat=differences.size))
        )
        null_statistics = np.abs((signs * differences).mean(axis=1))
        return float(np.count_nonzero(null_statistics >= observed - 1e-15) / len(signs))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(samples, differences.size))
    null_statistics = np.abs((signs * differences).mean(axis=1))
    return float((np.count_nonzero(null_statistics >= observed) + 1) / (samples + 1))


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, (count - index) * value)
        adjusted[name] = min(1.0, running)
    return adjusted


def _validate_record(
    record: dict[str, Any],
    dataset: str,
    seed: int,
    expected_config: dict[str, Any],
) -> None:
    if record.get("status") != "success":
        raise ValueError(f"failed record for {dataset}/{seed}")
    if record.get("dataset") != dataset or record.get("seed") != seed:
        raise ValueError(f"record key mismatch for {dataset}/{seed}")
    if record.get("model") != "GCN" or not record.get("split_id"):
        raise ValueError(f"invalid model or split for {dataset}/{seed}")
    expected_run_id = expected_config["edge_intervention"]["run_id"]
    if record.get("run_id") != expected_run_id:
        raise ValueError(f"run id mismatch for {dataset}/{seed}")
    if record.get("difference_definition") != "randomized_test_accuracy_minus_original_test_accuracy":
        raise ValueError(f"difference definition mismatch for {dataset}/{seed}")
    expected_config_sha = config_sha256(expected_config)
    if not record.get("source_commit"):
        raise ValueError(f"missing provenance for {dataset}/{seed}")
    if record.get("config_sha256") != expected_config_sha or record.get("frozen_config") != expected_config:
        raise ValueError(f"frozen configuration mismatch for {dataset}/{seed}")
    if record.get("randomization_seed") != 100000 + seed:
        raise ValueError(f"randomization seed mismatch for {dataset}/{seed}")
    conditions = record.get("conditions", {})
    if set(conditions) != {"original", "randomized"}:
        raise ValueError(f"missing paired condition for {dataset}/{seed}")
    original = float(conditions["original"]["test_accuracy"])
    randomized = float(conditions["randomized"]["test_accuracy"])
    difference = float(record["paired_test_difference"])
    if not math.isclose(difference, randomized - original, abs_tol=1e-12):
        raise ValueError(f"paired difference mismatch for {dataset}/{seed}")
    for condition_name, condition in conditions.items():
        expected_nested = {
            "run_id": expected_run_id,
            "dataset": dataset,
            "model": "GCN",
            "seed": seed,
            "split_id": record["split_id"],
            "source_commit": record["source_commit"],
            "config_sha256": expected_config_sha,
            "frozen_config": expected_config,
            "data_provenance": record["data_provenance"],
        }
        if any(condition.get(key) != value for key, value in expected_nested.items()):
            raise ValueError(f"condition provenance mismatch for {dataset}/{seed}/{condition_name}")
        try:
            validate_trial_audit(condition, expected_config["training"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid condition trial audit for {dataset}/{seed}/{condition_name}: {exc}") from exc
    invariants = record.get("randomization_audit", {}).get("invariants", {})
    node_count = int(record["node_count"])
    original_edges = normalize_edges(node_count, record["original_edges"])
    randomized_edges = normalize_edges(node_count, record["randomized_edges"])
    recomputed = invariant_audit(node_count, original_edges, randomized_edges)
    if invariants != recomputed:
        raise ValueError(f"randomization audit mismatch for {dataset}/{seed}")
    metadata = record.get("randomization_metadata", {})
    swap_status = metadata.get("swap_status", {})
    if metadata.get("status") != "success" or swap_status.get(
        "successful_swaps"
    ) != record.get("requested_swaps"):
        raise ValueError(f"incomplete randomization execution for {dataset}/{seed}")
    for key in (
        "node_count_identical",
        "edge_count_identical",
        "degree_sequence_by_node_identical",
        "simple_graph_before",
        "simple_graph_after",
    ):
        if not invariants.get(key, False):
            raise ValueError(f"failed {key} for {dataset}/{seed}")


def summarize_records(
    records: list[dict[str, Any]],
    *,
    expected_datasets: list[str],
    expected_seeds: list[int],
    expected_config: dict[str, Any],
    bootstrap_samples: int,
    bootstrap_seed: int,
    permutation_samples: int,
    permutation_seed: int,
) -> dict[str, Any]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (record.get("dataset"), record.get("seed"))
        if key in indexed:
            raise ValueError(f"duplicate record: {key}")
        indexed[key] = record
    expected = {(dataset, seed) for dataset in expected_datasets for seed in expected_seeds}
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(f"record scope mismatch; missing={missing}, extra={extra}")
    if len({record.get("source_commit") for record in records}) != 1:
        raise ValueError("mixed source commits")
    if len({record.get("config_sha256") for record in records}) != 1:
        raise ValueError("mixed configuration digests")
    for dataset in expected_datasets:
        provenance = {
            json.dumps(indexed[(dataset, seed)].get("data_provenance"), sort_keys=True)
            for seed in expected_seeds
        }
        if len(provenance) != 1:
            raise ValueError(f"mixed data provenance for {dataset}")

    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    permutation_rng = np.random.default_rng(permutation_seed)
    dataset_rows: dict[str, dict[str, Any]] = {}
    raw_p_values = {}
    for dataset in expected_datasets:
        paired_rows = []
        for seed in expected_seeds:
            record = indexed[(dataset, seed)]
            _validate_record(record, dataset, seed, expected_config)
            original = float(record["conditions"]["original"]["test_accuracy"])
            randomized = float(record["conditions"]["randomized"]["test_accuracy"])
            paired_rows.append(
                {
                    "seed": seed,
                    "split_id": record["split_id"],
                    "randomization_seed": record["randomization_seed"],
                    "requested_swaps": record["requested_swaps"],
                    "original_test_accuracy": original,
                    "randomized_test_accuracy": randomized,
                    "paired_difference": randomized - original,
                }
            )
        differences = np.asarray(
            [row["paired_difference"] for row in paired_rows], dtype=float
        )
        p_value = _sign_flip_p_value(
            differences, permutation_samples, permutation_rng
        )
        raw_p_values[dataset] = p_value
        dataset_rows[dataset] = {
            "pair_count": len(paired_rows),
            "mean_original_test_accuracy": float(
                np.mean([row["original_test_accuracy"] for row in paired_rows])
            ),
            "mean_randomized_test_accuracy": float(
                np.mean([row["randomized_test_accuracy"] for row in paired_rows])
            ),
            "mean_paired_difference": float(differences.mean()),
            "median_paired_difference": float(np.median(differences)),
            "sample_standard_deviation": float(differences.std(ddof=1)),
            "bootstrap_95_ci": _bootstrap_interval(
                differences, bootstrap_samples, bootstrap_rng
            ),
            "sign_flip_p_value": p_value,
            "paired_seed_records": paired_rows,
        }
    adjusted = _holm_adjust(raw_p_values)
    for dataset, value in adjusted.items():
        dataset_rows[dataset]["holm_adjusted_p_value"] = value
    dataset_means = np.asarray(
        [dataset_rows[dataset]["mean_paired_difference"] for dataset in expected_datasets],
        dtype=float,
    )
    return {
        "schema_version": "1.0",
        "run_id": expected_config["edge_intervention"]["run_id"],
        "status": "success",
        "difference_definition": "randomized_test_accuracy_minus_original_test_accuracy",
        "record_count": len(records),
        "datasets": dataset_rows,
        "dataset_clustered_overall": {
            "dataset_count": int(dataset_means.size),
            "estimand": "equal_weight_mean_of_dataset_level_mean_paired_differences",
            "mean_paired_difference": float(dataset_means.mean()),
            "median_dataset_mean_difference": float(np.median(dataset_means)),
            "bootstrap_95_ci": _bootstrap_interval(
                dataset_means, bootstrap_samples, bootstrap_rng
            ),
            "sign_flip_p_value": _sign_flip_p_value(
                dataset_means, permutation_samples, permutation_rng
            ),
        },
        "inference": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "permutation_samples": permutation_samples,
            "permutation_seed": permutation_seed,
            "multiple_testing": "Holm across dataset-level sign-flip tests",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    intervention = config["edge_intervention"]
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.input_root / "records").glob("*/*.json"))
    ]
    result = summarize_records(
        records,
        expected_datasets=intervention["datasets"],
        expected_seeds=intervention["seeds"],
        expected_config=config,
        bootstrap_samples=int(intervention["bootstrap"]["samples"]),
        bootstrap_seed=int(intervention["bootstrap"]["seed"]),
        permutation_samples=int(intervention["permutation"]["samples"]),
        permutation_seed=int(intervention["permutation"]["seed"]),
    )
    write_json_exclusive(args.output, result)
    print(json.dumps({"status": "success", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
