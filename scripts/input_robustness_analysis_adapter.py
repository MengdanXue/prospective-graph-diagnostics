"""Pure two-condition analysis for validated input-robustness records.

The adapter never trains a model and never reads the test set to choose a
trial, action, or calibration threshold.  It selects a graph architecture from
each candidate portfolio by the frozen validation rule, then evaluates all
nine predeclared strategies on the selected test result.  Leave-one-dataset-out
thresholds are fit only on the other dataset clusters within the same
condition and portfolio.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.input_robustness_formal_records import (
    CONDITIONS, GRAPH_MODELS, STRATEGIES, THRESHOLDS, FormalRecordError,
    record_key, select_trial,
)
from experiments.evaluate_diagnostics import (
    fixed_decisions as frozen_fixed_decisions,
    score_method as frozen_score_method,
    PRACTICAL_MARGIN as FROZEN_PRACTICAL_MARGIN,
)


PRACTICAL_MARGIN = 0.01


def enumerate_graph_portfolios(models: Sequence[str] = GRAPH_MODELS) -> list[tuple[str, ...]]:
    names = tuple(models)
    if set(names) != set(GRAPH_MODELS) or len(names) != len(GRAPH_MODELS):
        raise FormalRecordError("portfolio enumeration requires the six approved graph models")
    return [combo for size in range(1, len(names) + 1) for combo in itertools.combinations(names, size)]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise FormalRecordError("cannot average an empty analysis stratum")
    return math.fsum(values) / len(values)


def _selected_model(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    return sorted(rows, key=lambda row: (-float(row["validation_accuracy"]), str(row["model"])))[0]


def _units(records: Sequence[Mapping[str, Any]], condition: str) -> dict[tuple[str, int], dict[str, Mapping[str, Any]]]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        if record.get("condition") != condition:
            continue
        key = (str(record["dataset"]), int(record["seed"]))
        model = str(record["model"])
        if model in grouped[key]:
            raise FormalRecordError(f"duplicate model in analysis unit {key}: {model}")
        grouped[key][model] = record
    if not grouped:
        raise FormalRecordError(f"no records for condition {condition}")
    expected = set(GRAPH_MODELS) | {"MLP"}
    for key, models in grouped.items():
        if set(models) != expected:
            raise FormalRecordError(f"unit {key} does not contain all seven models")
    return grouped


def _portfolio_rows(units: Mapping[tuple[str, int], Mapping[str, Mapping[str, Any]]],
                    portfolio: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for (dataset, seed), models in sorted(units.items()):
        graph = _selected_model(models[name] for name in portfolio)
        mlp = models["MLP"]
        diagnostics = graph.get("diagnostics", {})
        # Diagnostics are train-only and copied across model records.  These
        # names and semantics are the frozen evaluator's input contract.
        for field in ("homophily", "mean_degree", "delta_h"):
            value = diagnostics.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))):
                raise FormalRecordError(f"missing/non-finite train-only diagnostic {field} in {dataset}/{seed}")
        rows.append({
            "dataset": dataset, "seed": seed, "split_id": str(graph["split_id"]),
            "portfolio": tuple(portfolio),
            "graph_model": graph["model"], "graph_validation": float(graph["validation_accuracy"]),
            "mlp_validation": float(mlp["validation_accuracy"]),
            "graph_test": float(graph["test_accuracy"]), "mlp_test": float(mlp["test_accuracy"]),
            "homophily": None if diagnostics["homophily"] is None else float(diagnostics["homophily"]),
            "mean_degree": None if diagnostics["mean_degree"] is None else float(diagnostics["mean_degree"]),
            "delta_h": None if diagnostics["delta_h"] is None else float(diagnostics["delta_h"]),
        })
    return rows


def _frozen_units(rows: Sequence[Mapping[str, Any]], *, practical_margin: float) -> list[dict[str, Any]]:
    """Convert adapter rows to the exact frozen evaluator unit contract."""
    units = []
    for row in rows:
        graph_test = float(row["graph_test"])
        mlp_test = float(row["mlp_test"])
        unit = {
            "dataset": str(row["dataset"]),
            "seed": int(row["seed"]),
            "split_id": str(row.get("split_id", f"{row['dataset']}-seed-{row['seed']}")),
            "selected_mlp": "MLP",
            "selected_graph": str(row["graph_model"]),
            "selected_mlp_validation": float(row["mlp_validation"]),
            "selected_graph_validation": float(row["graph_validation"]),
            "selected_mlp_test": mlp_test,
            "selected_graph_test": graph_test,
            "test_gap": graph_test - mlp_test,
            "target_action": "graph" if graph_test - mlp_test > practical_margin else "mlp",
            "homophily": row["homophily"],
            "mean_degree": row["mean_degree"],
            "delta_h": row["delta_h"],
        }
        unit["decisions"] = frozen_fixed_decisions(unit)
        units.append(unit)
    if not units:
        raise FormalRecordError("cannot evaluate an empty strategy stratum")
    return units


def evaluate_strategy(rows: Sequence[Mapping[str, Any]], strategy: str, *,
                      degree_threshold: float | None = None,
                      homophily_threshold: float | None = None,
                      two_hop_threshold: float | None = None,
                      practical_margin: float = PRACTICAL_MARGIN) -> dict[str, Any]:
    """Score using the frozen evaluator; threshold arguments are compatibility-only."""
    if strategy not in STRATEGIES:
        raise FormalRecordError(f"strategy is not predeclared: {strategy}")
    if practical_margin != FROZEN_PRACTICAL_MARGIN:
        raise FormalRecordError("the adapter must use the frozen practical margin of 0.01")
    units = _frozen_units(rows, practical_margin=practical_margin)
    scored = dict(frozen_score_method(units, strategy))
    outcomes = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        decision = unit["decisions"][strategy]
        action = decision["action"]
        effective = "mlp" if action == "abstain" else action
        if action == "expected_random":
            regret = 0.5 * (
                max(unit["selected_mlp_test"], unit["selected_graph_test"]) - unit["selected_mlp_test"]
                + max(unit["selected_mlp_test"], unit["selected_graph_test"]) - unit["selected_graph_test"]
            )
            accuracy = 0.5
        else:
            oracle = max(unit["selected_mlp_test"], unit["selected_graph_test"])
            realized = unit["selected_graph_test"] if effective == "graph" else unit["selected_mlp_test"]
            regret = float(oracle - realized)
            accuracy = float(effective == unit["target_action"])
        item = {"dataset": unit["dataset"], "seed": unit["seed"], "action": action,
                "effective_action": effective, "target": unit["target_action"],
                "confidence": decision["confidence"], "regret": regret, "accuracy": accuracy}
        outcomes.append(item)
        by_dataset[item["dataset"]].append(item)
    scored.update({
        "strategy": strategy,
        "units": len(units),
        "mean_regret_pp": 100.0 * float(scored["full_set_mean_regret"]),
        "actions": {action: sum(item["action"] == action for item in outcomes)
                    for action in ("graph", "mlp", "abstain", "expected_random")},
        "datasets": {
            dataset: {
                "units": len(items),
                "mean_regret_pp": 100.0 * _mean(item["regret"] for item in items),
                "selection_accuracy": _mean(item["accuracy"] for item in items),
            }
            for dataset, items in sorted(by_dataset.items())
        },
        "outcomes": outcomes,
    })
    return scored


def _threshold_loss(rows: Sequence[Mapping[str, Any]], threshold: float, feature: str) -> float:
    return _mean(max(float(row["graph_test"]), float(row["mlp_test"])) -
                 (float(row["graph_test"]) if float(row[feature]) >= threshold else float(row["mlp_test"]))
                 for row in rows)


def fit_threshold(rows: Sequence[Mapping[str, Any]], *, feature: str = "homophily") -> float:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["dataset"])].append(row)
    if not groups:
        raise FormalRecordError("threshold fitting requires training datasets")
    losses = [_mean(_threshold_loss(group, threshold, feature) for group in groups.values()) for threshold in THRESHOLDS]
    minimum = min(losses)
    return next(threshold for threshold, loss in zip(THRESHOLDS, losses) if loss <= minimum + 1e-12)


def leave_one_dataset_out(rows: Sequence[Mapping[str, Any]], *, feature: str = "homophily") -> dict[str, Any]:
    datasets = sorted({str(row["dataset"]) for row in rows})
    if len(datasets) < 2:
        raise FormalRecordError("leave-one-dataset-out requires at least two datasets")
    folds = {}
    for heldout in datasets:
        training = [row for row in rows if row["dataset"] != heldout]
        test = [row for row in rows if row["dataset"] == heldout]
        threshold = fit_threshold(training, feature=feature)
        folds[heldout] = {
            "threshold": threshold, "training_datasets": len(datasets) - 1,
            "units": len(test), "regret_pp": 100.0 * _threshold_loss(test, threshold, feature),
            "graph_actions": sum(float(row[feature]) >= threshold for row in test),
        }
    return {"feature": feature, "mean_regret_pp": _mean(fold["regret_pp"] for fold in folds.values()), "folds": folds}


def adapt_records(records: Sequence[Mapping[str, Any]], *, conditions: Sequence[str] = CONDITIONS,
                  practical_margin: float = PRACTICAL_MARGIN) -> dict[str, Any]:
    if tuple(conditions) != CONDITIONS:
        raise FormalRecordError("the adapter requires the two approved conditions in config order")
    if practical_margin != FROZEN_PRACTICAL_MARGIN:
        raise FormalRecordError("the adapter must use the frozen practical margin of 0.01")
    results: dict[str, Any] = {}
    for condition in conditions:
        units = _units(records, condition)
        portfolios = []
        calibrations = {}
        for portfolio in enumerate_graph_portfolios():
            rows = _portfolio_rows(units, portfolio)
            strategies = {name: evaluate_strategy(rows, name, practical_margin=practical_margin) for name in STRATEGIES}
            label = "+".join(portfolio)
            portfolios.append({"portfolio": list(portfolio), "label": label, "strategies": strategies})
            calibrations[label] = leave_one_dataset_out(rows)
        results[condition] = {
            "units": len(units), "dataset_count": len({dataset for dataset, _ in units}),
            "portfolio_count": len(portfolios), "strategy_count": len(STRATEGIES),
            "portfolios": portfolios, "leave_one_dataset_out": calibrations,
        }
    first, second = conditions
    by_label = {entry["label"]: entry for entry in results[first]["portfolios"]}
    cross = []
    for entry in results[second]["portfolios"]:
        prior = by_label[entry["label"]]
        cross.append({"portfolio": entry["portfolio"], "label": entry["label"], "strategies": {
            strategy: {"mean_regret_pp_delta": entry["strategies"][strategy]["mean_regret_pp"] - prior["strategies"][strategy]["mean_regret_pp"],
                       "selection_accuracy_delta": entry["strategies"][strategy]["selection_accuracy"] - prior["strategies"][strategy]["selection_accuracy"]}
            for strategy in STRATEGIES
        }})
    return {
        "schema_version": "1.0", "analysis_status": "post_hoc_two_condition_adapter",
        "conditions": list(conditions), "strategies": list(STRATEGIES),
        "graph_models": list(GRAPH_MODELS), "portfolio_count": 63,
        "practical_margin": practical_margin, "portfolios_by_condition": results,
        "cross_condition": cross,
        "calibration_contract": "each threshold is fit on the other dataset clusters within the same condition and portfolio; held-out test outcomes never select it",
    }


def adapt_validated_root(root: Path, *, synthetic: bool = False) -> dict[str, Any]:
    from scripts.validate_input_robustness_formal_records import validate_complete_run
    config_path = None
    binding_path = None
    if not synthetic:
        repo_root = Path(__file__).resolve().parents[1]
        config_path = repo_root / "configs" / "input_robustness_11_v2.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        binding_path = repo_root / str(config["bound_input_source"]["path"])
    validated = validate_complete_run(root, synthetic=synthetic, config_path=config_path,
                                      data_binding_path=binding_path)
    return adapt_records(validated["records"])
