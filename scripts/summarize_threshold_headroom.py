"""Exact post-hoc h1-cutoff/headroom reanalysis of verified frozen records.

No model training or new test evaluation occurs. The test-oracle cutoff uses
retained test outcomes and is an optimistic bound within one one-sided cutoff
family, not a fitted deployable selector or a new confirmatory comparison.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from experiments.evaluate_diagnostics import PRACTICAL_MARGIN, action_regret, fixed_decisions
from scripts.summarize_equal_budget_sensitivity import (
    DEFAULT_CONFIG, GRAPH_ARCHITECTURES, load_units, select_by_validation,
)
from scripts.summarize_fallback_sensitivity import (
    ANALYSIS, COMBINED, frozen_units, load_json, require, same_number,
)


def dataset_mean(values):
    """Equal dataset weights, with equal seed weights within each dataset."""
    grouped = defaultdict(list)
    for dataset, value in values:
        grouped[dataset].append(value)
    require(bool(grouped), "cannot average an empty dataset collection")
    return math.fsum(math.fsum(group) / len(group) for group in grouped.values()) / len(grouped)


def exact_threshold_curve(units):
    """Enumerate every distinct graph iff h >= t decision for t in [0, 1].

    For sorted observed h values, the constant-action intervals are [0,h_min],
    (h_min,h_next], ..., (h_max,1]. A value at zero creates the singleton [0,0];
    a value at one remains a graph action even at t=1. Evaluating each closed
    upper endpoint therefore covers the entire continuum without a grid or
    an epsilon perturbation at a decision boundary.
    """
    require(bool(units), "threshold analysis requires units")
    for unit in units:
        h = unit["homophily"]
        require(isinstance(h, (int, float)) and not isinstance(h, bool)
                and math.isfinite(h) and 0 <= h <= 1, "invalid homophily")
    endpoints = sorted({unit["homophily"] for unit in units} | {1.0})
    curve = []
    lower = 0.0
    for index, upper in enumerate(endpoints):
        per_dataset = defaultdict(list)
        graph_actions = 0
        for unit in units:
            action = "graph" if unit["homophily"] >= upper else "mlp"
            graph_actions += action == "graph"
            per_dataset[unit["dataset"]].append(action_regret(unit, action))
        curve.append({
            "lower": lower, "upper": upper,
            "lower_inclusive": index == 0, "upper_inclusive": True,
            "mean_regret_pp": 100 * dataset_mean(
                (dataset, loss) for dataset, losses in per_dataset.items() for loss in losses),
            "graph_actions": graph_actions, "coverage": 1.0,
            "datasets": {dataset: 100 * math.fsum(losses) / len(losses)
                         for dataset, losses in sorted(per_dataset.items())},
        })
        lower = upper
    return curve


def portfolio_units(frozen, records, portfolio):
    require(bool(portfolio) and len(set(portfolio)) == len(portfolio)
            and set(portfolio) <= set(GRAPH_ARCHITECTURES), "invalid portfolio")
    require(set(records) == set(frozen), "model records and frozen audit scopes differ")
    result = []
    for key, original in sorted(frozen.items()):
        models = records[key]
        require(set(models) == {"MLP", *GRAPH_ARCHITECTURES}, f"{key}: incomplete model set")
        graph = select_by_validation(models[name] for name in portfolio)
        mlp = models["MLP"]
        unit = dict(original)
        unit.update(selected_graph=graph["model"], selected_graph_test=graph["test_accuracy"],
                    selected_graph_validation=graph["validation_accuracy"],
                    selected_mlp_test=mlp["test_accuracy"],
                    selected_mlp_validation=mlp["validation_accuracy"])
        unit["test_gap"] = unit["selected_graph_test"] - unit["selected_mlp_test"]
        unit["target_action"] = "graph" if unit["test_gap"] > PRACTICAL_MARGIN else "mlp"
        unit["decisions"] = fixed_decisions(unit)
        require(unit["decisions"][COMBINED] == original["decisions"][COMBINED],
                f"{key}: portfolio restriction changed frozen Combined action")
        result.append(unit)
    return result


def summarize_portfolio(units, portfolio):
    policy_regret = {}
    for label, method in (("always_graph", "always_graph"), ("combined", COMBINED),
                          ("validation_selection", "validation_selection")):
        values = []
        for unit in units:
            action = unit["decisions"][method]["action"]
            values.append((unit["dataset"], action_regret(unit, "mlp" if action == "abstain" else action)))
        policy_regret[label] = 100 * dataset_mean(values)
    curve = exact_threshold_curve(units)
    minimum = min(row["mean_regret_pp"] for row in curve)
    policy_regret["test_oracle_h1_cutoff"] = minimum
    headroom = policy_regret["always_graph"]
    return {
        "portfolio": list(portfolio), "units": len(units),
        "datasets": len({unit["dataset"] for unit in units}),
        "headroom_pp": headroom, "policy_regret_pp": policy_regret,
        "headroom_closed_fraction": {
            label: (headroom - regret) / headroom if headroom else None
            for label, regret in policy_regret.items()},
        "exact_threshold_curve": curve,
        "minimum_threshold_intervals": [
            {key: row[key] for key in ("lower", "upper", "lower_inclusive", "upper_inclusive")}
            for row in curve if row["mean_regret_pp"] == minimum],
    }


def confusion_summary(units):
    groups = {}
    for action in ("graph", "mlp", "abstain"):
        selected = [unit for unit in units if unit["decisions"][COMBINED]["action"] == action]
        effective = "mlp" if action == "abstain" else action
        losses = [action_regret(unit, effective) for unit in selected]
        groups[action] = {
            "units": len(selected), "effective_action_with_declared_fallback": effective,
            "practical_margin_target_graph": sum(unit["target_action"] == "graph" for unit in selected),
            "practical_margin_target_mlp": sum(unit["target_action"] == "mlp" for unit in selected),
            "raw_graph_test_greater_than_mlp": sum(unit["test_gap"] > 0 for unit in selected),
            "positive_raw_regret_units": sum(loss > 0 for loss in losses),
            "sum_regret_pp_units": 100 * math.fsum(losses),
            "full_set_regret_contribution_pp": 100 * dataset_mean(
                (unit["dataset"], action_regret(unit, effective)
                 if unit["decisions"][COMBINED]["action"] == action else 0.0) for unit in units),
        }
    return {
        "target_definition": "graph iff graph_test - mlp_test > 0.01; otherwise mlp",
        "regret_definition": "max(graph_test, mlp_test) minus effective selected test accuracy; no margin",
        "abstention_fallback": "mlp", "groups": groups,
        "interpretation": "Observed action/outcome counts in the full frozen portfolio only; "
                          "a margin-based correct target can still incur a positive raw regret.",
    }


def validation_fallback_summary(units):
    """Change only Combined's abstentions to the frozen validation action."""
    losses = []
    changed_abstentions = 0
    for unit in units:
        action = unit["decisions"][COMBINED]["action"]
        if action == "abstain":
            action = unit["decisions"]["validation_selection"]["action"]
            changed_abstentions += 1
        losses.append((unit["dataset"], action_regret(unit, action)))
    return {"mean_regret_pp": 100 * dataset_mean(losses), "units": len(units),
            "abstentions_assigned_by_validation": changed_abstentions,
            "coverage_before_fallback": (len(units) - changed_abstentions) / len(units)}


def existing_sensitivities(units):
    fallback = {}
    for name, excluded in (("all_11", ()),
                           ("without_chameleon_squirrel", ("Chameleon", "Squirrel"))):
        fallback[name] = validation_fallback_summary([unit for unit in units if unit["dataset"] not in excluded])
        fallback[name]["excluded_datasets"] = list(excluded)
    omitted = {}
    for dataset in sorted({unit["dataset"] for unit in units}):
        losses = []
        for unit in units:
            if unit["dataset"] == dataset:
                continue
            action = unit["decisions"][COMBINED]["action"]
            losses.append((unit["dataset"], action_regret(unit, "mlp" if action == "abstain" else action)
                           - action_regret(unit, "graph")))
        omitted[dataset] = 100 * dataset_mean(losses)
    return {
        "scope": "Full frozen portfolio only; descriptive alternative fallback and dataset omissions. "
                 "No preprocessing results or independently retrained records are inserted.",
        "validation_fallback": fallback,
        "delete_one_dataset_combined_minus_always_graph_pp": omitted,
        "delete_one_dataset_delta_range_pp": [min(omitted.values()), max(omitted.values())],
    }


def summarize(audit, records):
    frozen = frozen_units(audit)
    portfolios = {}
    for name, models in (("full", GRAPH_ARCHITECTURES),
                         *((model, (model,)) for model in GRAPH_ARCHITECTURES)):
        units = portfolio_units(frozen, records, models)
        portfolios[name] = summarize_portfolio(units, models)
    for label, method in (("always_graph", "always_graph"), ("combined", COMBINED),
                          ("validation_selection", "validation_selection")):
        same_number(portfolios["full"]["policy_regret_pp"][label] / 100,
                    audit["methods"][method]["full_set_mean_regret"], f"full.{label}")
    return {
        "schema_version": "1.0", "analysis_status": "post_hoc_descriptive_retained_record_reanalysis",
        "source_benchmark_id": audit["benchmark_id"], "practical_margin": PRACTICAL_MARGIN,
        "scope": "Full six-architecture portfolio and every single-architecture restriction; "
                 "no new training, model selection trials, or test evaluations.",
        "weighting": "Equal dataset weights and equal seed weights within each dataset; frozen scope is 11 x 10.",
        "oracle_definition": "Per unit max(MLP test accuracy, validation-selected graph test accuracy) "
                             "within the evaluated portfolio; recomputed separately for each portfolio.",
        "headroom_definition": "Always-graph raw regret; maximum achievable reduction to the same unit oracle.",
        "headroom_closure_definition": "(always_graph_regret - policy_regret) / always_graph_regret; "
                                       "negative means worse; null when headroom is zero.",
        "threshold_rule": "graph iff h1 >= t, otherwise mlp; t in [0,1], full coverage and no abstention",
        "threshold_search": "Exact constant-action intervals using every observed h1 boundary; "
                            "evaluate closed upper endpoints, including zero/one boundaries; no grid.",
        "test_oracle_caveat": "Minimum regret is selected using all retained test outcomes. It is an "
                              "optimistic in-sample bound only for this one-sided cutoff family, "
                              "not a held-out or prospective estimate, not a new method, and not LODO calibration.",
        "coverage_caveat": "The cutoff has coverage one. Frozen Combined retains its original abstentions "
                           "and MLP fallback; full-set regret agreement does not imply equal coverage.",
        "validation_selection": "Reuse the frozen evaluator: choose graph only if validation gap exceeds 0.01.",
        "portfolios": portfolios,
        "full_portfolio_confusion": confusion_summary(list(frozen.values())),
        "existing_sensitivities": existing_sensitivities(list(frozen.values())),
    }


def read_verified_inputs(records_root, audit_path, config_path, manifest_path=None):
    """Strict release checks plus duplicate-JSON-key refusal before analysis."""
    records_root = Path(records_root)
    manifest_path = Path(manifest_path) if manifest_path else records_root.parent.parent / "MANIFEST.json"
    audit, config = load_json(audit_path), load_json(config_path)
    manifest = load_json(manifest_path)
    for directory in ("records", "diagnostics"):
        for path in (records_root.parent / directory).rglob("*.json"):
            load_json(path)
    records = load_units(records_root, audit=audit, config=config, manifest_path=manifest_path)
    files = [entry for entry in manifest["files"]
             if entry.get("group") in ("prospective_records", "prospective_diagnostics")]
    provenance = {
        "verification": "Release manifest file hashes and complete record scope; frozen assembly/trial/config/"
                        "split/provenance checks; reconstructed complete audit matched; duplicate JSON keys rejected.",
        "manifest_trust": "The manifest is a trust input, not a signature; obtain it from the verified release or review package.",
        "input_sources": {label: {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                  "bytes": Path(path).stat().st_size}
                          for label, path in (("audit", audit_path), ("config", config_path), ("manifest", manifest_path))},
        "verified_model_records": sum(entry["group"] == "prospective_records" for entry in files),
        "verified_diagnostic_records": sum(entry["group"] == "prospective_diagnostics" for entry in files),
    }
    return audit, records, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=ANALYSIS / "diagnostic_audit.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, help="Create a new JSON file; default prints without writing")
    args = parser.parse_args(argv)
    audit, records, provenance = read_verified_inputs(args.records_root, args.audit, args.config, args.manifest)
    result = summarize(audit, records)
    result["source_verification"] = provenance
    source_bytes = Path(__file__).read_bytes()
    result["analysis_source"] = {"path": "scripts/summarize_threshold_headroom.py",
                                 "sha256": hashlib.sha256(source_bytes).hexdigest(), "bytes": len(source_bytes)}
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")
    return result


if __name__ == "__main__":
    main()
