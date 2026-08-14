#!/usr/bin/env python3
"""Assemble immutable prospective records into the frozen evaluator schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evaluate_diagnostics import audit_payload
from experiments.run_prospective_benchmark import validate_trial_audit, write_json_exclusive


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "prospective_benchmark_v1.json"
FROZEN_MODELS = ["MLP", "GCN", "GAT", "GraphSAGE", "H2GCN", "LINKX", "GPR-GNN"]


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def assemble_payload(records_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    records_root = Path(records_root)
    failures_root = records_root / "failures"
    _require(
        not failures_root.exists() or not any(failures_root.rglob("*.json")),
        "failure artifacts are present",
    )
    _require(config["models"] == FROZEN_MODELS, "configuration model set is not frozen")
    model_records = []
    diagnostic_records = []
    expected_paths: set[Path] = set()
    expected_config_sha = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    common_commit: str | None = None
    provenance_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            unit_splits = set()
            for model in config["models"]:
                path = records_root / "records" / dataset / model / f"seed_{seed:03d}.json"
                expected_paths.add(path.resolve())
                record = _read(path)
                _require(record.get("status") == "success", f"record is not successful: {path}")
                _require(record.get("run_id") == config["run_id"], f"run id mismatch: {path}")
                _require(record.get("dataset") == dataset, f"dataset mismatch: {path}")
                _require(record.get("model") == model, f"model mismatch: {path}")
                _require(int(record.get("seed")) == seed, f"seed mismatch: {path}")
                commit = str(record.get("source_commit") or "")
                _require(bool(commit), f"missing source commit: {path}")
                if common_commit is None:
                    common_commit = commit
                _require(commit == common_commit, f"mixed source commits: {path}")
                _require(record.get("environment"), f"missing environment: {path}")
                _require(record.get("data_provenance") is not None, f"missing data provenance: {path}")
                if dataset not in provenance_by_dataset:
                    provenance_by_dataset[dataset] = record["data_provenance"]
                _require(
                    record["data_provenance"] == provenance_by_dataset[dataset],
                    f"mixed data provenance: {path}",
                )
                _require(record.get("config_sha256") == expected_config_sha, f"config digest mismatch: {path}")
                _require(record.get("frozen_config") == config, f"frozen config mismatch: {path}")
                try:
                    validate_trial_audit(record, config["training"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid trial audit: {path}: {exc}") from exc
                expected_family = "mlp" if model == "MLP" else "graph"
                _require(record.get("family") == expected_family, f"family mismatch: {path}")
                unit_splits.add(str(record["split_id"]))
                model_records.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "split_id": str(record["split_id"]),
                        "model": model,
                        "family": expected_family,
                        "validation_accuracy": float(record["validation_accuracy"]),
                        "test_accuracy": float(record["test_accuracy"]),
                    }
                )
            diagnostic_path = (
                records_root / "diagnostics" / dataset / f"seed_{seed:03d}.json"
            )
            expected_paths.add(diagnostic_path.resolve())
            diagnostic = _read(diagnostic_path)
            _require(
                diagnostic.get("status") == "success",
                f"diagnostic is not successful: {diagnostic_path}",
            )
            _require(
                diagnostic.get("run_id") == config["run_id"],
                f"run id mismatch: {diagnostic_path}",
            )
            _require(diagnostic.get("dataset") == dataset, f"dataset mismatch: {diagnostic_path}")
            _require(int(diagnostic.get("seed")) == seed, f"seed mismatch: {diagnostic_path}")
            provenance = diagnostic.get("provenance") or {}
            _require(
                provenance.get("uses_test_labels") is False,
                f"test-label leakage marker: {diagnostic_path}",
            )
            _require(
                provenance.get("homophily_label_scope")
                == ("train_only" if diagnostic.get("homophily") is not None else "unavailable"),
                f"homophily scope mismatch: {diagnostic_path}",
            )
            _require(
                provenance.get("two_hop_label_scope")
                == ("train_only" if diagnostic.get("delta_h") is not None else "unavailable"),
                f"two-hop scope mismatch: {diagnostic_path}",
            )
            _require(provenance.get("source_commit") == common_commit, f"commit mismatch: {diagnostic_path}")
            _require(provenance.get("environment"), f"missing environment: {diagnostic_path}")
            _require(provenance.get("data") == provenance_by_dataset[dataset], f"data provenance mismatch: {diagnostic_path}")
            _require(provenance.get("config_sha256") == expected_config_sha, f"config digest mismatch: {diagnostic_path}")
            _require(provenance.get("frozen_config") == config, f"frozen config mismatch: {diagnostic_path}")
            unit_splits.add(str(diagnostic["split_id"]))
            _require(len(unit_splits) == 1, f"inconsistent split identifiers for {(dataset, seed)}")
            diagnostic_records.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "split_id": str(diagnostic["split_id"]),
                    "homophily": diagnostic.get("homophily"),
                    "mean_degree": diagnostic.get("mean_degree"),
                    "delta_h": diagnostic.get("delta_h"),
                    "provenance": {
                        "uses_test_labels": False,
                        "homophily_label_scope": provenance["homophily_label_scope"],
                        "two_hop_label_scope": provenance["two_hop_label_scope"],
                    },
                }
            )

    actual_paths = {
        path.resolve()
        for directory in (records_root / "records", records_root / "diagnostics")
        if directory.exists()
        for path in directory.rglob("*.json")
    }
    _require(actual_paths == expected_paths, "unexpected or duplicate source records are present")
    evaluator = config["evaluator"]
    payload = {
        "schema_version": "1.0",
        "benchmark_id": evaluator["specification_version"],
        "outcome": "accuracy",
        "practical_margin": evaluator["practical_margin"],
        "mlp_models": ["MLP"],
        "graph_models": FROZEN_MODELS[1:],
        "model_records": model_records,
        "diagnostic_records": diagnostic_records,
        "bootstrap": evaluator["bootstrap"],
        "permutation": evaluator["permutation"],
    }
    audit_payload(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    payload = assemble_payload(args.records_root, config)
    write_json_exclusive(args.output, payload)
    print(
        json.dumps(
            {
                "model_records": len(payload["model_records"]),
                "diagnostic_records": len(payload["diagnostic_records"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
