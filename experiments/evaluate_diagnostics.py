#!/usr/bin/env python3
"""Score a prospective diagnostic benchmark from paired per-seed records."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


BENCHMARK_ID = "route_a_diagnostic_v1"
PRACTICAL_MARGIN = 0.01
MLP_MODELS = ["MLP"]
GRAPH_MODELS = ["GCN", "GAT", "GraphSAGE", "H2GCN", "LINKX", "GPR-GNN"]
METHODS = [
    "always_mlp",
    "always_graph",
    "random_50_50",
    "homophily_only",
    "degree_only",
    "homophily_plus_degree",
    "validation_selection",
    "two_hop_only",
    "historical_combined",
]
REFERENCE_METHOD = "historical_combined"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_specification(payload: dict[str, Any]) -> None:
    require(payload.get("schema_version") == "1.0", "schema_version must equal 1.0")
    require(payload.get("benchmark_id") == BENCHMARK_ID, f"benchmark_id must equal {BENCHMARK_ID}")
    require(payload.get("outcome") == "accuracy", "outcome must equal accuracy")
    require(payload.get("practical_margin") == PRACTICAL_MARGIN, "practical_margin must equal 0.01")
    require(payload.get("mlp_models") == MLP_MODELS, "mlp_models do not match the frozen set")
    require(payload.get("graph_models") == GRAPH_MODELS, "graph_models do not match the frozen set")
    require(payload.get("model_records"), "model_records must not be empty")
    require(payload.get("diagnostic_records"), "diagnostic_records must not be empty")
    for key in ("bootstrap", "permutation"):
        require(int(payload[key]["samples"]) > 0, f"{key}.samples must be positive")
        require(isinstance(payload[key]["seed"], int), f"{key}.seed must be an integer")


def unit_key(record: dict[str, Any]) -> tuple[str, int, str]:
    return str(record["dataset"]), int(record["seed"]), str(record["split_id"])


def validate_accuracy(value: Any, label: str) -> float:
    number = float(value)
    require(math.isfinite(number) and 0.0 <= number <= 1.0, f"{label} must lie in [0, 1]")
    return number


def select_by_validation(records: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(records, key=lambda row: (-float(row["validation_accuracy"]), str(row["model"])))[0]


def confidence(value: float, scale: float = 1.0) -> float:
    return float(min(1.0, abs(value) / scale))


def decision(action: str, score: float | None) -> dict[str, Any]:
    return {"action": action, "confidence": None if score is None else float(score)}


def fixed_decisions(unit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    h = unit["homophily"]
    degree = unit["mean_degree"]
    delta_h = unit["delta_h"]
    mlp_validation = unit["selected_mlp_validation"]
    graph_validation = unit["selected_graph_validation"]
    choices: dict[str, dict[str, Any]] = {
        "always_mlp": decision("mlp", 1.0),
        "always_graph": decision("graph", 1.0),
        "random_50_50": decision("expected_random", None),
    }

    if h is None:
        choices["homophily_only"] = decision("abstain", None)
    else:
        h_score = float(h) - 0.5
        choices["homophily_only"] = decision(
            "graph" if h_score >= 0.0 else "mlp", confidence(h_score, 0.5)
        )

    if degree is None:
        choices["degree_only"] = decision("abstain", None)
    else:
        degree_score = math.tanh(math.log(max(float(degree), 1e-12) / 10.0))
        choices["degree_only"] = decision(
            "graph" if degree_score >= 0.0 else "mlp", confidence(degree_score)
        )

    if h is None or degree is None:
        choices["homophily_plus_degree"] = decision("abstain", None)
    else:
        degree_score = math.tanh(math.log(max(float(degree), 1e-12) / 10.0))
        combined_score = 0.5 * (2.0 * float(h) - 1.0) + 0.5 * degree_score
        choices["homophily_plus_degree"] = decision(
            "graph" if combined_score >= 0.0 else "mlp", confidence(combined_score)
        )

    validation_score = graph_validation - mlp_validation - PRACTICAL_MARGIN
    choices["validation_selection"] = decision(
        "graph" if validation_score > 0.0 else "mlp",
        confidence(validation_score, 0.25),
    )

    if delta_h is None:
        choices["two_hop_only"] = decision("abstain", None)
    else:
        two_hop_score = float(delta_h) - 0.05
        choices["two_hop_only"] = decision(
            "graph" if two_hop_score > 0.0 else "mlp", confidence(two_hop_score, 0.25)
        )

    if h is None:
        choices["historical_combined"] = decision("abstain", None)
    elif float(h) >= 0.55:
        choices["historical_combined"] = decision(
            "graph", confidence(float(h) - 0.55, 0.45)
        )
    elif mlp_validation >= 0.40:
        historical_confidence = max(0.55 - float(h), mlp_validation - 0.40)
        choices["historical_combined"] = decision(
            "mlp", confidence(historical_confidence, 0.60)
        )
    else:
        choices["historical_combined"] = decision("abstain", None)
    return choices


def build_units(payload: dict[str, Any]) -> list[dict[str, Any]]:
    models_by_unit: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in payload["model_records"]:
        models_by_unit[unit_key(record)].append(record)

    diagnostics_by_unit: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in payload["diagnostic_records"]:
        key = unit_key(record)
        require(key not in diagnostics_by_unit, f"duplicate diagnostic record for {key}")
        provenance = record.get("provenance") or {}
        require(
            provenance.get("uses_test_labels") is False,
            f"diagnostic for {key} must explicitly state that it does not use test labels",
        )
        expected_homophily_scope = "train_only" if record.get("homophily") is not None else "unavailable"
        expected_two_hop_scope = "train_only" if record.get("delta_h") is not None else "unavailable"
        require(
            provenance.get("homophily_label_scope") == expected_homophily_scope,
            f"homophily provenance for {key} must equal {expected_homophily_scope}",
        )
        require(
            provenance.get("two_hop_label_scope") == expected_two_hop_scope,
            f"two-hop provenance for {key} must equal {expected_two_hop_scope}",
        )
        diagnostics_by_unit[key] = record

    require(set(models_by_unit) == set(diagnostics_by_unit), "model and diagnostic unit keys must match")
    units: list[dict[str, Any]] = []
    expected_models = set(MLP_MODELS + GRAPH_MODELS)
    for key in sorted(models_by_unit):
        records = models_by_unit[key]
        model_names = [str(record["model"]) for record in records]
        require(len(model_names) == len(set(model_names)), f"duplicate model record for {key}")
        require(set(model_names) == expected_models, f"model set is incomplete for {key}")
        for record in records:
            validate_accuracy(record["validation_accuracy"], "validation_accuracy")
            validate_accuracy(record["test_accuracy"], "test_accuracy")
            expected_family = "mlp" if record["model"] in MLP_MODELS else "graph"
            require(record.get("family") == expected_family, f"incorrect model family for {record['model']}")

        mlp = select_by_validation([row for row in records if row["model"] in MLP_MODELS])
        graph = select_by_validation([row for row in records if row["model"] in GRAPH_MODELS])
        diagnostics = diagnostics_by_unit[key]
        h = diagnostics.get("homophily")
        degree = diagnostics.get("mean_degree")
        delta_h = diagnostics.get("delta_h")
        if h is not None:
            h = float(h)
            require(0.0 <= h <= 1.0, f"homophily must lie in [0, 1] for {key}")
        if degree is not None:
            degree = float(degree)
            require(math.isfinite(degree) and degree > 0.0, f"mean_degree must be positive for {key}")
        if delta_h is not None:
            delta_h = float(delta_h)
            require(math.isfinite(delta_h), f"delta_h must be finite for {key}")

        mlp_test = float(mlp["test_accuracy"])
        graph_test = float(graph["test_accuracy"])
        test_gap = graph_test - mlp_test
        unit = {
            "dataset": key[0],
            "seed": key[1],
            "split_id": key[2],
            "selected_mlp": mlp["model"],
            "selected_graph": graph["model"],
            "selected_mlp_validation": float(mlp["validation_accuracy"]),
            "selected_graph_validation": float(graph["validation_accuracy"]),
            "selected_mlp_test": mlp_test,
            "selected_graph_test": graph_test,
            "test_gap": test_gap,
            "target_action": "graph" if test_gap > PRACTICAL_MARGIN else "mlp",
            "homophily": h,
            "mean_degree": degree,
            "delta_h": delta_h,
        }
        unit["decisions"] = fixed_decisions(unit)
        units.append(unit)
    return units


def action_regret(unit: dict[str, Any], action: str) -> float:
    oracle = max(unit["selected_mlp_test"], unit["selected_graph_test"])
    selected = unit["selected_graph_test"] if action == "graph" else unit["selected_mlp_test"]
    return float(oracle - selected)


def risk_coverage_curve(units: list[dict[str, Any]], method: str) -> list[dict[str, float]]:
    covered = []
    for unit in units:
        choice = unit["decisions"][method]
        if choice["action"] not in {"mlp", "graph"}:
            continue
        covered.append((float(choice["confidence"]), unit, choice["action"]))
    covered.sort(key=lambda item: (-item[0], item[1]["dataset"], item[1]["seed"]))
    curve = []
    for size in range(1, len(covered) + 1):
        subset = covered[:size]
        errors = sum(action != unit["target_action"] for _, unit, action in subset)
        regrets = [action_regret(unit, action) for _, unit, action in subset]
        curve.append(
            {
                "coverage": size / len(units),
                "selective_risk": errors / size,
                "covered_mean_regret": float(np.mean(regrets)),
                "minimum_confidence": subset[-1][0],
            }
        )
    return curve


def unit_regrets(units: list[dict[str, Any]], method: str) -> list[float]:
    regrets = []
    for unit in units:
        if method == "random_50_50":
            regrets.append(0.5 * (action_regret(unit, "mlp") + action_regret(unit, "graph")))
            continue
        action = unit["decisions"][method]["action"]
        fallback_action = "mlp" if action == "abstain" else action
        regrets.append(action_regret(unit, fallback_action))
    return regrets


def score_method(units: list[dict[str, Any]], method: str) -> dict[str, Any]:
    regrets = unit_regrets(units, method)
    if method == "random_50_50":
        return {
            "covered": len(units),
            "abstained": 0,
            "coverage": 1.0,
            "selection_accuracy": 0.5,
            "selective_accuracy": 0.5,
            "selective_risk": 0.5,
            "full_set_mean_regret": float(np.mean(regrets)),
            "covered_mean_regret": float(np.mean(regrets)),
            "risk_coverage_curve": [],
            "random_metrics_are_analytic_expectations": True,
        }

    covered_units = []
    full_correct = 0
    for unit in units:
        action = unit["decisions"][method]["action"]
        fallback_action = "mlp" if action == "abstain" else action
        full_correct += fallback_action == unit["target_action"]
        if action in {"mlp", "graph"}:
            covered_units.append((unit, action))
    covered = len(covered_units)
    covered_correct = sum(action == unit["target_action"] for unit, action in covered_units)
    covered_regrets = [action_regret(unit, action) for unit, action in covered_units]
    return {
        "covered": covered,
        "abstained": len(units) - covered,
        "coverage": covered / len(units),
        "selection_accuracy": full_correct / len(units),
        "selective_accuracy": covered_correct / covered if covered else None,
        "selective_risk": 1.0 - covered_correct / covered if covered else None,
        "full_set_mean_regret": float(np.mean(regrets)),
        "covered_mean_regret": float(np.mean(covered_regrets)) if covered_regrets else None,
        "risk_coverage_curve": risk_coverage_curve(units, method),
        "abstention_fallback": "mlp",
    }


def dataset_mean_regret(units: list[dict[str, Any]], method: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for unit, regret in zip(units, unit_regrets(units, method)):
        grouped[unit["dataset"]].append(regret)
    return {dataset: float(np.mean(values)) for dataset, values in grouped.items()}


def stable_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (base + int.from_bytes(digest[:4], "big")) % (2**32)


def bootstrap_mean_ci(values: np.ndarray, *, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def sign_flip_p(values: np.ndarray, *, samples: int, seed: int) -> float:
    observed = abs(float(values.mean()))
    if np.allclose(values, 0.0):
        return 1.0
    if values.size <= 18:
        signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=values.size)))
        permuted = np.abs((signs * values).mean(axis=1))
        return float(np.count_nonzero(permuted >= observed - 1e-15) / permuted.size)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(samples, values.size))
    permuted = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed - 1e-15) + 1) / (permuted.size + 1))


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["raw_p"])
    running = 0.0
    total = len(rows)
    for rank, (original_index, row) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(row["raw_p"]))
        running = max(running, adjusted)
        rows[original_index]["holm_adjusted_p"] = running


def paired_comparisons(units: list[dict[str, Any]], payload: dict[str, Any]) -> list[dict[str, Any]]:
    reference = dataset_mean_regret(units, REFERENCE_METHOD)
    datasets = sorted(reference)
    rows = []
    for method in METHODS:
        if method == REFERENCE_METHOD:
            continue
        candidate = dataset_mean_regret(units, method)
        differences = np.asarray([candidate[name] - reference[name] for name in datasets])
        sd = float(differences.std(ddof=1)) if differences.size > 1 else 0.0
        rows.append(
            {
                "method": method,
                "reference": REFERENCE_METHOD,
                "estimand": "dataset_mean_regret_method_minus_reference",
                "mean_difference": float(differences.mean()),
                "bootstrap_95_ci": bootstrap_mean_ci(
                    differences,
                    samples=int(payload["bootstrap"]["samples"]),
                    seed=stable_seed(int(payload["bootstrap"]["seed"]), method),
                ),
                "paired_standardized_mean_difference": (
                    float(differences.mean() / sd) if sd > 0.0 else None
                ),
                "win_rate": float(np.mean(differences < 0.0)),
                "raw_p": sign_flip_p(
                    differences,
                    samples=int(payload["permutation"]["samples"]),
                    seed=stable_seed(int(payload["permutation"]["seed"]), method),
                ),
            }
        )
    holm_adjust(rows)
    return rows


def audit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    validate_specification(payload)
    units = build_units(payload)
    return {
        "schema_version": "1.0",
        "benchmark_id": BENCHMARK_ID,
        "status": "scored",
        "target": {
            "outcome": "accuracy",
            "practical_margin": PRACTICAL_MARGIN,
            "positive_action": "graph only when selected-graph test accuracy exceeds selected-MLP test accuracy by more than the margin",
            "tie_policy": "all gaps less than or equal to the practical margin map to mlp",
        },
        "record_counts": {
            "model_records": len(payload["model_records"]),
            "diagnostic_records": len(payload["diagnostic_records"]),
            "evaluation_units": len(units),
            "datasets": len({unit["dataset"] for unit in units}),
        },
        "units": units,
        "methods": {method: score_method(units, method) for method in METHODS},
        "paired_comparisons": paired_comparisons(units, payload),
        "inference": {
            "resampling_unit": "dataset",
            "seed_aggregation": "mean regret within dataset before resampling",
            "bootstrap": payload["bootstrap"],
            "paired_test": "two-sided paired sign-flip permutation test on dataset mean regret",
            "permutation": payload["permutation"],
            "multiple_comparison_correction": "Holm family-wise correction across eight predeclared comparisons",
            "non_significance_policy": "non-significance is not interpreted as equivalence",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
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
        audit = audit_payload(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(audit["record_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
