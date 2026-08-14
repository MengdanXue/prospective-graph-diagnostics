#!/usr/bin/env python3
"""Run immutable paired GCN training on original and degree-matched graphs."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from experiments.degree_preserving_edge_randomization import (
    normalize_edges,
    run_intervention,
)
from experiments.prospective_data import (
    canonical_undirected_edges,
    make_stratified_split,
    split_identifier,
)
from experiments.run_prospective_benchmark import (
    DEFAULT_CONFIG,
    _bidirectional_edge_index,
    _load_dataset,
    _processed_provenance,
    _selected,
    _source_commit,
    config_sha256,
    environment_snapshot,
    run_model_unit,
    validate_resume_record,
    validate_trial_audit,
    write_json_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]


def randomization_seed(intervention: dict[str, Any], seed: int) -> int:
    return int(intervention["randomization_seed_offset"]) + int(seed)


def requested_swaps(intervention: dict[str, Any], edge_count: int) -> int:
    return int(intervention["successful_swaps_per_edge"]) * int(edge_count)


def randomize_graph(
    *,
    node_count: int,
    edges: Iterable[Iterable[int]],
    labels: list[Any],
    seed: int,
    intervention: dict[str, Any],
) -> dict[str, Any]:
    normalized = normalize_edges(
        node_count, ((int(left), int(right)) for left, right in edges)
    )
    return run_intervention(
        node_count=node_count,
        edges=normalized,
        labels=labels,
        n_swaps=requested_swaps(intervention, len(normalized)),
        seed=randomization_seed(intervention, seed),
    )


def run_paired_unit(
    *,
    run_id: str,
    dataset: str,
    seed: int,
    split_id: str,
    x: torch.Tensor,
    y: torch.Tensor,
    original_edge_index: torch.Tensor,
    randomized_edge_index: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    test_indices: torch.Tensor,
    training: dict[str, Any],
    randomization_seed: int,
    randomization_audit: dict[str, Any],
    randomization_metadata: dict[str, Any],
    original_edges: list[list[int]],
    randomized_edges: list[list[int]],
    requested_swaps: int,
    source_commit: str,
    environment: dict[str, Any],
    data_provenance: dict[str, Any],
    device: torch.device,
    config_sha256: str = "test-only",
    frozen_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    common = {
        "run_id": run_id,
        "dataset": dataset,
        "model_id": "GCN",
        "seed": seed,
        "split_id": split_id,
        "x": x,
        "y": y,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "training": training,
        "source_commit": source_commit,
        "environment": environment,
        "data_provenance": data_provenance,
        "device": device,
        "config_sha256": config_sha256,
        "frozen_config": frozen_config,
    }
    original = run_model_unit(edge_index=original_edge_index, **common)
    randomized = run_model_unit(edge_index=randomized_edge_index, **common)
    difference = float(randomized["test_accuracy"]) - float(original["test_accuracy"])
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "success",
        "dataset": dataset,
        "model": "GCN",
        "seed": int(seed),
        "split_id": split_id,
        "randomization_seed": int(randomization_seed),
        "requested_swaps": int(requested_swaps),
        "conditions": {"original": original, "randomized": randomized},
        "paired_test_difference": difference,
        "difference_definition": "randomized_test_accuracy_minus_original_test_accuracy",
        "randomization_audit": randomization_audit,
        "randomization_metadata": randomization_metadata,
        "node_count": int(x.size(0)),
        "original_edges": original_edges,
        "randomized_edges": randomized_edges,
        "source_commit": source_commit,
        "environment": environment,
        "data_provenance": data_provenance,
        "config_sha256": config_sha256,
        "frozen_config": frozen_config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> int:
    run_started = time.perf_counter()
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config_digest = config_sha256(config)
    intervention = config["edge_intervention"]
    datasets = _selected(args.datasets, intervention["datasets"], "datasets")
    seeds = _selected(
        [str(seed) for seed in args.seeds] if args.seeds else None,
        [str(seed) for seed in intervention["seeds"]],
        "seeds",
    )
    seeds = [int(seed) for seed in seeds]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    environment = environment_snapshot(device)
    commit = _source_commit()

    for dataset_name in datasets:
        failure_root = args.output_root / "failures" / dataset_name
        if args.resume and failure_root.exists() and any(failure_root.rglob("*.json")):
            raise RuntimeError(f"existing failure blocks resume: {failure_root}")
        try:
            dataset = _load_dataset(dataset_name, args.data_root)
            data = dataset[0]
            x = data.x.float().cpu()
            y = data.y.long().view(-1).cpu()
            canonical = canonical_undirected_edges(
                data.edge_index.cpu().numpy(), node_count=data.num_nodes
            )
            original_edge_index = _bidirectional_edge_index(canonical)
            provenance = _processed_provenance(dataset)
        except Exception as exc:
            write_json_exclusive(
                failure_root / "dataset_setup.json",
                {
                    "schema_version": "1.0",
                    "run_id": intervention["run_id"],
                    "status": "error",
                    "dataset": dataset_name,
                    "stage": "dataset_setup",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                    "source_commit": commit,
                    "config_sha256": config_digest,
                    "frozen_config": config,
                    "environment": environment,
                    "command": sys.argv,
                    "elapsed_seconds": time.perf_counter() - run_started,
                },
            )
            return 1
        for seed in seeds:
            split = make_stratified_split(y.numpy(), seed=seed)
            current_split_id = split_identifier(split)
            failure_path = (
                args.output_root
                / "failures"
                / dataset_name
                / f"seed_{seed:03d}.json"
            )
            if failure_path.exists():
                raise RuntimeError(f"existing failure blocks resume: {failure_path}")
            output = (
                args.output_root
                / "records"
                / dataset_name
                / f"seed_{seed:03d}.json"
            )
            if output.exists():
                if args.resume:
                    validate_resume_record(
                        output,
                        {
                            "run_id": intervention["run_id"],
                            "dataset": dataset_name,
                            "model": "GCN",
                            "seed": seed,
                            "split_id": current_split_id,
                            "source_commit": commit,
                            "config_sha256": config_digest,
                            "data_provenance": provenance,
                        },
                        None,
                    )
                    existing = json.loads(output.read_text(encoding="utf-8"))
                    for condition in existing.get("conditions", {}).values():
                        validate_trial_audit(condition, config["training"])
                    continue
                raise FileExistsError(output)
            try:
                randomized = randomize_graph(
                    node_count=data.num_nodes,
                    edges=canonical,
                    labels=y.tolist(),
                    seed=seed,
                    intervention=intervention,
                )
                randomized_edges = np.asarray(
                    randomized["randomized_edges"], dtype=np.int64
                )
                record = run_paired_unit(
                    run_id=intervention["run_id"],
                    dataset=dataset_name,
                    seed=seed,
                    split_id=current_split_id,
                    x=x,
                    y=y,
                    original_edge_index=original_edge_index,
                    randomized_edge_index=_bidirectional_edge_index(randomized_edges),
                    train_indices=torch.from_numpy(split["train"]),
                    validation_indices=torch.from_numpy(split["validation"]),
                    test_indices=torch.from_numpy(split["test"]),
                    training=config["training"],
                    randomization_seed=randomized["seed"],
                    randomization_audit=randomized["audit"],
                    randomization_metadata={
                        "algorithm": randomized["algorithm"],
                        "status": randomized["status"],
                        "swap_status": randomized["swap_status"],
                    },
                    original_edges=randomized["original_edges"],
                    randomized_edges=randomized["randomized_edges"],
                    requested_swaps=randomized["n_swaps"],
                    source_commit=commit,
                    environment=environment,
                    data_provenance=provenance,
                    device=device,
                    config_sha256=config_digest,
                    frozen_config=config,
                )
                write_json_exclusive(output, record)
                print(
                    json.dumps(
                        {
                            "dataset": dataset_name,
                            "seed": seed,
                            "status": "success",
                            "output": str(output),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as exc:
                write_json_exclusive(
                    failure_path,
                    {
                        "schema_version": "1.0",
                        "run_id": intervention["run_id"],
                        "status": "error",
                        "dataset": dataset_name,
                        "model": "GCN",
                        "seed": seed,
                        "split_id": current_split_id,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "traceback": traceback.format_exc(),
                        "source_commit": commit,
                        "environment": environment,
                        "data_provenance": provenance,
                        "config_sha256": config_digest,
                        "frozen_config": config,
                        "command": sys.argv,
                        "elapsed_seconds": time.perf_counter() - run_started,
                    },
                )
                print(f"{type(exc).__name__}: {exc}", flush=True)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
