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
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.input_robustness_formal_records import (
    CONDITIONS, GRAPH_MODELS, STRATEGIES, THRESHOLDS, FormalRecordError,
    record_key, select_trial,
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
        # Diagnostics are train-only and copied across model records.  Require
        # the values needed by a strategy rather than silently inventing them.
        for field in ("homophily", "mean_degree", "two_hop_agreement"):
            value = diagnostics.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise FormalRecordError(f"missing/non-finite train-only diagnostic {field} in {dataset}/{seed}")
        historical = diagnostics.get("historical_action", "mlp")
        if historical not in ("graph", "mlp", "abstain"):
            raise FormalRecordError("historical_action must be graph, mlp, or abstain")
        rows.append({
            "dataset": dataset, "seed": seed, "portfolio": tuple(portfolio),
            "graph_model": graph["model"], "graph_validation": float(graph["validation_accuracy"]),
            "mlp_validation": float(mlp["validation_accuracy"]),
            "graph_test": float(graph["test_accuracy"]), "mlp_test": float(mlp["test_accuracy"]),
            "homophily": float(diagnostics["homophily"]), "mean_degree": float(diagnostics["mean_degree"]),
            "two_hop_agreement": float(diagnostics["two_hop_agreement"]),
            "historical_action": historical,
        })
    return rows


def _action(strategy: str, row: Mapping[str, Any], *, degree_threshold: float = 2.0,
            homophily_threshold: float = 0.5, two_hop_threshold: float = 0.5,
            practical_margin: float = PRACTICAL_MARGIN) -> str:
    if strategy == "always_graph":
        return "graph"
    if strategy == "always_mlp":
        return "mlp"
    if strategy == "degree_only":
        return "graph" if row["mean_degree"] >= degree_threshold else "mlp"
    if strategy == "historical_combined":
        return "mlp" if row["historical_action"] == "abstain" else row["historical_action"]
    if strategy == "homophily_only":
        return "graph" if row["homophily"] >= homophily_threshold else "mlp"
    if strategy == "homophily_plus_degree":
        return "graph" if row["homophily"] >= homophily_threshold and row["mean_degree"] >= degree_threshold else "mlp"
    if strategy == "two_hop_only":
        return "graph" if row["two_hop_agreement"] >= two_hop_threshold else "mlp"
    if strategy == "validation_selection":
        return "graph" if row["graph_validation"] - row["mlp_validation"] > practical_margin else "mlp"
    if strategy == "random_50_50":
        return "random"
    raise FormalRecordError(f"unknown analysis strategy: {strategy}")


def evaluate_strategy(rows: Sequence[Mapping[str, Any]], strategy: str, *, degree_threshold: float = 2.0,
                      homophily_threshold: float = 0.5, two_hop_threshold: float = 0.5,
                      practical_margin: float = PRACTICAL_MARGIN) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise FormalRecordError(f"strategy is not predeclared: {strategy}")
    outcomes = []
    for row in rows:
        graph_test, mlp_test = float(row["graph_test"]), float(row["mlp_test"])
        oracle = max(graph_test, mlp_test)
        target = "graph" if graph_test - mlp_test > practical_margin else "mlp"
        action = _action(strategy, row, degree_threshold=degree_threshold,
                         homophily_threshold=homophily_threshold,
                         two_hop_threshold=two_hop_threshold,
                         practical_margin=practical_margin)
        if action == "random":
            regret = oracle - 0.5 * (graph_test + mlp_test)
            accuracy = 0.5 if graph_test != mlp_test else 1.0
        else:
            realized = graph_test if action == "graph" else mlp_test
            regret = oracle - realized
            accuracy = float(action == target)
        outcomes.append({"dataset": row["dataset"], "seed": row["seed"], "action": action,
                         "target": target, "regret": regret, "accuracy": accuracy})
    if not outcomes:
        raise FormalRecordError("cannot evaluate an empty strategy stratum")
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in outcomes:
        by_dataset[item["dataset"]].append(item)
    return {
        "strategy": strategy, "units": len(outcomes), "coverage": 1.0,
        "mean_regret_pp": 100.0 * _mean(item["regret"] for item in outcomes),
        "selection_accuracy": _mean(item["accuracy"] for item in outcomes),
        "actions": {action: sum(item["action"] == action for item in outcomes)
                    for action in ("graph", "mlp", "random")},
        "datasets": {dataset: {"units": len(items), "mean_regret_pp": 100.0 * _mean(i["regret"] for i in items),
                               "selection_accuracy": _mean(i["accuracy"] for i in items)}
                     for dataset, items in sorted(by_dataset.items())},
        "outcomes": outcomes,
    }


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
    validated = validate_complete_run(root, synthetic=synthetic)
    return adapt_records(validated["records"])

