"""Train-label-only graph statistics and a post-hoc decision-evaluation adapter.

The statistics implement Eq. (2) and the uniform-edge version of Eq. (3) in
Platonov et al., NeurIPS 2023. The train-induced graph and calibrated decision
rules are this study's adaptation, not an original classifier from that paper.
No feature values, validation labels or test labels enter the metric API.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.evaluate_diagnostics import action_regret, score_method, unit_regrets
from scripts.input_robustness_analysis_adapter import (
    _frozen_units, _portfolio_rows, _units, enumerate_graph_portfolios,
)
from scripts.input_robustness_formal_records import (
    CONDITIONS, FormalRecordError, canonical_json, digest,
)

PAPER_URL = "https://papers.neurips.cc/paper_files/paper/2023/file/01b681025fdbda8e935a66cc5bb6e9de-Paper-Conference.pdf"
METRICS = ("adjusted_homophily", "edge_label_informativeness")
METHOD = "published_metric_calibrated"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalRecordError(message)


def _integer_array(value: Any, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    _require(array.ndim == ndim and array.dtype.kind in "iu", f"{name} must be a {ndim}-D integer array")
    _require(not array.size or (int(array.min()) >= 0 and int(array.max()) <= np.iinfo(np.int64).max),
             f"{name} contains an invalid integer")
    return array.astype(np.int64, copy=False)


def compute_train_metrics(edge_index: Any, train_nodes: Any, train_labels: Any,
                          *, num_nodes: int) -> dict[str, Any]:
    """Accept labels for train_nodes *only*, in the same order.

    Work and storage depend on supplied edges/training nodes/observed class
    pairs, never num_nodes squared or the largest node/class identifier.
    Edges are restricted to train nodes, made undirected, deduplicated and
    stripped of self-loops. Each remaining edge gives two ordered label pairs.
    The graph passed to model training is never changed.
    """
    _require(isinstance(num_nodes, int) and not isinstance(num_nodes, bool) and num_nodes > 0,
             "num_nodes must be a positive integer")
    edges = _integer_array(edge_index, "edge_index", 2)
    _require(edges.shape[0] == 2, "edge_index must have shape (2, E)")
    nodes = _integer_array(train_nodes, "train_nodes", 1)
    labels = _integer_array(train_labels, "train_labels", 1)
    _require(nodes.size == labels.size, "pass only one label per training node")
    _require(not edges.size or int(edges.max()) < num_nodes, "edge endpoint is outside num_nodes")
    _require(not nodes.size or int(nodes.max()) < num_nodes, "training node is outside num_nodes")
    order = np.argsort(nodes)
    nodes, labels = nodes[order], labels[order]
    _require(not nodes.size or np.unique(nodes).size == nodes.size, "duplicate training node")
    if nodes.size:
        positions = np.searchsorted(nodes, edges)
        inside = positions < nodes.size
        bounded = np.minimum(positions, nodes.size - 1)
        inside &= nodes[bounded] == edges
        keep = inside.all(axis=0) & (edges[0] != edges[1])
        canonical = np.unique(np.sort(edges[:, keep], axis=0).T, axis=0)
    else:
        canonical = np.empty((0, 2), dtype=np.int64)
    edge_count = int(canonical.shape[0])
    # Explicit little-endian serialization makes provenance platform independent.
    hashes = {"training_nodes_sha256": hashlib.sha256(nodes.astype("<i8").tobytes()).hexdigest(),
              "training_labels_sha256": hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest(),
              "training_simple_edges_sha256": hashlib.sha256(canonical.astype("<i8").tobytes()).hexdigest()}
    result: dict[str, Any] = {
        "schema_version": "1.0", "source": PAPER_URL,
        "graph_scope": "training_induced_undirected_simple_no_self_loops",
        "label_scope": "train_only", "uses_validation_labels": False, "uses_test_labels": False,
        "training_node_count": int(nodes.size), "training_edge_count": edge_count,
        "metrics": {}, **hashes,
    }
    if not edge_count:
        result["endpoint_class_count"] = 0
        result["metrics"] = {metric: {"value": None, "unavailable_reason": "no_training_edges"} for metric in METRICS}
        return result
    edge_labels = labels[np.searchsorted(nodes, canonical)]
    endpoints = Counter(int(label) for label in edge_labels.ravel())
    pairs: Counter = Counter()
    for first, second in edge_labels:
        pairs[(int(first), int(second))] += 1
        pairs[(int(second), int(first))] += 1
    result["endpoint_class_count"] = len(endpoints)
    if len(endpoints) == 1:
        result["metrics"] = {metric: {"value": None, "unavailable_reason": "single_endpoint_class"} for metric in METRICS}
        return result
    total = 2 * edge_count
    squared_degree_mass = sum(count * count for count in endpoints.values())
    same_count = sum(count for (first, second), count in pairs.items() if first == second)
    # Integer arithmetic avoids cancellation near maximal class imbalance.
    adjusted = (same_count * total - squared_degree_mass) / (total * total - squared_degree_mass)
    entropy = -math.fsum((count / total) * math.log(count / total) for count in endpoints.values())
    information = math.fsum((count / total) * math.log((count * total) / (endpoints[first] * endpoints[second]))
                            for (first, second), count in pairs.items())
    li = information / entropy
    # The exact functional lies in [0, 1]; allow only floating-point roundoff.
    _require(-32 * np.finfo(float).eps <= li <= 1 + 32 * np.finfo(float).eps,
             "edge label informativeness escaped its mathematical range")
    result["metrics"] = {
        "adjusted_homophily": {"value": float(adjusted), "unavailable_reason": None},
        "edge_label_informativeness": {"value": min(1.0, max(0.0, li)), "unavailable_reason": None},
    }
    return result


def make_metric_record(dataset: str, seed: int, split_id: str, statistics: Mapping[str, Any]) -> dict[str, Any]:
    result = {"dataset": str(dataset), "seed": int(seed), "split_id": str(split_id), **dict(statistics)}
    result["record_sha256"] = digest(result)
    return result


def _score(row: Mapping[str, Any], metric: str) -> float | None:
    value = row[metric]
    _require(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)
                              and math.isfinite(value)), f"invalid metric value: {metric}")
    return None if value is None else float(value)


def candidate_thresholds(rows: Sequence[Mapping[str, Any]], metric: str) -> list[dict[str, Any]]:
    _require(metric in METRICS, "unknown published metric")
    values = sorted({_score(row, metric) for row in rows if _score(row, metric) is not None})
    # Extremes are symbolic infinities: JSON stays finite and future held-out
    # values outside the training-fold range retain the intended action.
    result = [{"kind": "all_graph", "value": None}]
    for low, high in zip(values, values[1:]):
        midpoint = low + (high - low) / 2
        if midpoint <= low:  # Adjacent machine floats: high is the same partition.
            midpoint = high
        result.append({"kind": "cutoff", "value": midpoint})
    result.append({"kind": "all_mlp", "value": None})
    return result


def threshold_action(value: float | None, threshold: Mapping[str, Any]) -> str:
    if value is None:
        return "abstain"
    kind = threshold["kind"]
    if kind == "all_graph":
        return "graph"
    if kind == "all_mlp":
        return "mlp"
    _require(kind == "cutoff" and math.isfinite(float(threshold["value"])), "invalid threshold")
    return "graph" if value >= float(threshold["value"]) else "mlp"


def _mean(values: Sequence[float]) -> float:
    _require(bool(values), "empty averaging group")
    return math.fsum(values) / len(values)


def _calibration_loss(rows: Sequence[Mapping[str, Any]], metric: str, threshold: Mapping[str, Any]) -> float:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        action = threshold_action(_score(row, metric), threshold)
        # Use the frozen loss, including its declared MLP abstention fallback.
        unit = {"selected_mlp_test": float(row["mlp_test"]), "selected_graph_test": float(row["graph_test"])}
        groups[str(row["dataset"])].append(action_regret(unit, "mlp" if action == "abstain" else action))
    return _mean([_mean(values) for values in groups.values()])


def fit_published_threshold(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    _require(bool(rows), "calibration requires training datasets")
    candidates = candidate_thresholds(rows, metric)
    losses = [_calibration_loss(rows, metric, threshold) for threshold in candidates]
    # Retain the existing calibration's declared numerical tie convention.
    # Candidates are ordered from -infinity to +infinity.
    minimum = min(losses)
    index = next(i for i, loss in enumerate(losses) if loss <= minimum + 1e-12)
    return {"threshold": candidates[index], "training_mean_regret_pp": 100 * losses[index],
            "candidate_count": len(candidates), "training_datasets": sorted({str(row["dataset"]) for row in rows})}


def evaluate_thresholds(rows: Sequence[Mapping[str, Any]], metric: str,
                        thresholds: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    units = _frozen_units(rows, practical_margin=0.01)
    for row, unit in zip(rows, units):
        action = threshold_action(_score(row, metric), thresholds[str(row["dataset"])])
        # Neither paper supplies decision confidence; use a constant for all
        # covered units and do not interpret a confidence-risk curve.
        unit["decisions"][METHOD] = {"action": action, "confidence": None if action == "abstain" else 1.0}
    scored = dict(score_method(units, METHOD))
    scored.pop("risk_coverage_curve")
    regrets = unit_regrets(units, METHOD)
    outcomes = [{"dataset": unit["dataset"], "seed": unit["seed"], "split_id": unit["split_id"],
                 "score": _score(row, metric), "threshold": dict(thresholds[str(unit["dataset"])]),
                 "action": unit["decisions"][METHOD]["action"],
                 "effective_action": "mlp" if unit["decisions"][METHOD]["action"] == "abstain" else unit["decisions"][METHOD]["action"],
                 "target": unit["target_action"], "selected_graph": unit["selected_graph"],
                 "regret": regret} for row, unit, regret in zip(rows, units, regrets)]
    groups = defaultdict(list)
    for outcome in outcomes:
        groups[outcome["dataset"]].append(outcome)
    datasets = {name: {"units": len(items), "mean_regret_pp": 100 * _mean([item["regret"] for item in items]),
                       "coverage": _mean([float(item["action"] != "abstain") for item in items])}
                for name, items in sorted(groups.items())}
    scored.update({"outcomes": outcomes, "datasets": datasets,
                   "mean_regret_pp": _mean([item["mean_regret_pp"] for item in datasets.values()]),
                   "aggregation": "equal_dataset_weight", "confidence": "not_defined_by_metric"})
    return scored


def leave_one_dataset_out_published(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    datasets = sorted({str(row["dataset"]) for row in rows})
    _require(len(datasets) >= 2, "LODO requires at least two datasets")
    folds = {}
    for heldout in datasets:
        training = [row for row in rows if row["dataset"] != heldout]
        folds[heldout] = fit_published_threshold(training, metric)
    result = evaluate_thresholds(rows, metric, {key: value["threshold"] for key, value in folds.items()})
    result.update({"folds": folds, "analysis_status": "post_hoc_extension_lodo",
                   "calibration_outcomes": "test_outcomes_of_other_datasets_only"})
    return result


def adapt_published_records(records: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]],
                             *, include_optimistic: bool = False) -> dict[str, Any]:
    """Pure adapter. Production callers must first validate the complete run."""
    metric_map = {}
    for record in metrics:
        key = (str(record["dataset"]), int(record["seed"]), str(record["split_id"]))
        _require(key not in metric_map, f"duplicate metric record: {key}")
        _require(record.get("record_sha256") == digest({k: v for k, v in record.items() if k != "record_sha256"}),
                 "metric record digest mismatch")
        _require(record.get("label_scope") == "train_only" and record.get("uses_test_labels") is False
                 and record.get("uses_validation_labels") is False, "metric label-scope violation")
        _require(set(record["metrics"]) == set(METRICS), "missing or unexpected metric")
        for metric in METRICS:
            value = record["metrics"][metric]
            _require((value["value"] is None) == bool(value.get("unavailable_reason")),
                     "undefined metrics require an explicit reason")
        metric_map[key] = record
    _require({row.get("condition") for row in records} == set(CONDITIONS), "unexpected input condition")
    splits = defaultdict(set)
    for record in records:
        splits[(record["dataset"], record["seed"])].add(record["split_id"])
    _require(all(len(values) == 1 for values in splits.values()), "models/conditions do not share a split")
    expected = {(str(record["dataset"]), int(record["seed"]), str(record["split_id"])) for record in records}
    _require(set(metric_map) == expected, "metric and validated model unit scope differ")
    results = {}
    for condition in CONDITIONS:
        grouped = _units(records, condition)
        _require({(dataset, seed) for dataset, seed in grouped} == {(d, s) for d, s, _ in expected},
                 "two-condition dataset/seed scopes differ")
        portfolios = []
        for portfolio in enumerate_graph_portfolios():
            rows = _portfolio_rows(grouped, portfolio)
            for row in rows:
                source = metric_map[(row["dataset"], row["seed"], row["split_id"])]
                row.update({metric: source["metrics"][metric]["value"] for metric in METRICS})
            methods = {metric: leave_one_dataset_out_published(rows, metric) for metric in METRICS}
            if include_optimistic:
                for metric in METRICS:
                    fit = fit_published_threshold(rows, metric)
                    bound = evaluate_thresholds(rows, metric, {row["dataset"]: fit["threshold"] for row in rows})
                    methods[metric]["optimistic_test_selected_floor"] = {
                        **bound, **fit, "analysis_status": "post_hoc_test_selected_optimistic_bound_not_deployment_estimate"}
            portfolios.append({"label": "+".join(portfolio), "portfolio": list(portfolio), "metrics": methods})
        results[condition] = portfolios
    return {"schema_version": "1.0", "analysis_status": "post_hoc_published_metric_extension",
            "source": PAPER_URL, "metric_count": 2, "portfolio_count": 63,
            "conditions": list(CONDITIONS), "portfolios_by_condition": results,
            "metric_records_sha256": hashlib.sha256(canonical_json(list(metrics))).hexdigest(),
            "decision_rule": "graph iff metric >= threshold; unavailable metric abstains with declared MLP fallback",
            "calibration_contract": "LODO candidates and loss use other datasets only; equal dataset weights; smallest threshold on ties",
            "interpretation": "published statistics with study-specific calibration; not original published selectors or confirmatory evidence"}
