"""Descriptive fallback reanalysis of retained summaries; no training or mixed runs."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from experiments.evaluate_diagnostics import (
    BENCHMARK_ID, GRAPH_MODELS, PRACTICAL_MARGIN, action_regret, fixed_decisions,
    score_method, validate_accuracy,
)

ANALYSIS = ROOT / "results/diagnostic/route_a_prospective_v2/analysis"
DATASETS = (
    "Actor", "Amazon-ratings", "Chameleon", "CiteSeer", "Coauthor-CS", "Cora",
    "Cornell", "PubMed", "Roman-empire", "Squirrel", "Wisconsin",
)
PREPROCESSING_DATASETS = ("Amazon-ratings", "Roman-empire")
CONDITIONS = ("normalize_features", "raw")
COMBINED = "historical_combined"
FALLBACKS = ("mlp", "graph")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def required(mapping, key, context):
    require(isinstance(mapping, dict) and key in mapping, f"{context}: missing {key}")
    return mapping[key]


def same_number(actual, expected, context):
    require(isinstance(actual, (float, int)) and not isinstance(actual, bool)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
            f"{context}: saved value does not match reconstruction")


def same_decisions(actual, expected, context):
    """Require identical actions and confidences equal up to reconstruction tolerance.

    The frozen evaluator's degree scores use math.log and math.tanh, whose last
    bit depends on the platform's libm, so a saved confidence can differ from a
    reconstruction by one ulp without any change in the selected action.
    """
    require(isinstance(expected, dict) and set(expected) == set(actual),
            f"{context}: saved decisions differ from the frozen evaluator")
    for method, computed in actual.items():
        saved = expected[method]
        require(isinstance(saved, dict) and set(saved) == set(computed)
                and saved["action"] == computed["action"],
                f"{context}.{method}: saved decisions differ from the frozen evaluator")
        if computed["confidence"] is None:
            require(saved["confidence"] is None,
                    f"{context}.{method}: saved decisions differ from the frozen evaluator")
        else:
            same_number(saved["confidence"], computed["confidence"],
                        f"{context}.{method}.confidence")


def unique_json_object(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, f"duplicate JSON key: {key}")
        out[key] = value
    return out


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_json_object)


def record_key(row, context):
    dataset = required(row, "dataset", context)
    seed = required(row, "seed", context)
    require(isinstance(dataset, str) and type(seed) is int, f"{context}: invalid dataset/seed")
    return dataset, seed


def check_scope(index, datasets, context):
    expected = {(dataset, seed) for dataset in datasets for seed in range(10)}
    require(set(index) == expected, f"{context}: missing or unexpected dataset/seed keys")


def validate_unit_values(unit, context):
    for key in ("selected_mlp_test", "selected_graph_test", "selected_mlp_validation",
                "selected_graph_validation"):
        validate_accuracy(required(unit, key, context), f"{context}.{key}")
    h = required(unit, "homophily", context)
    degree = required(unit, "mean_degree", context)
    delta = required(unit, "delta_h", context)
    require(h is not None and math.isfinite(h) and 0 <= h <= 1,
            f"{context}: invalid homophily")
    require(degree is not None and math.isfinite(degree) and degree > 0,
            f"{context}: invalid mean_degree")
    require(delta is not None and math.isfinite(delta), f"{context}: invalid delta_h")
    gap = unit["selected_graph_test"] - unit["selected_mlp_test"]
    same_number(required(unit, "test_gap", context), gap, f"{context}.test_gap")
    target = "graph" if gap > PRACTICAL_MARGIN else "mlp"
    require(required(unit, "target_action", context) == target, f"{context}: wrong target")


def frozen_units(audit):
    require(required(audit, "benchmark_id", "audit") == BENCHMARK_ID, "unexpected benchmark")
    index = {}
    for row in required(audit, "units", "audit"):
        key = record_key(row, "audit unit")
        require(key not in index, f"audit: duplicate dataset/seed key {key}")
        split = required(row, "split_id", str(key))
        require(isinstance(split, str) and bool(split), f"{key}: empty split_id")
        require(required(row, "selected_graph", str(key)) in GRAPH_MODELS,
                f"{key}: unknown selected graph")
        require(required(row, "selected_mlp", str(key)) == "MLP", f"{key}: unexpected MLP")
        validate_unit_values(row, str(key))
        same_decisions(fixed_decisions(row), required(row, "decisions", str(key)), str(key))
        index[key] = row
    check_scope(index, DATASETS, "audit")
    units = [index[key] for key in sorted(index)]
    methods = required(audit, "methods", "audit")
    for method in (COMBINED, "always_graph"):
        saved = required(methods, method, "audit.methods")
        computed = score_method(units, method)
        for field in ("covered", "abstained", "coverage", "selection_accuracy",
                      "selective_accuracy", "full_set_mean_regret", "covered_mean_regret"):
            same_number(required(saved, field, method), computed[field], f"{method}.{field}")
    return index


def summarize_policy(units, fallback):
    require(fallback in FALLBACKS and bool(units), "invalid fallback or empty units")
    actions = [u["decisions"][COMBINED]["action"] for u in units]
    selected = [fallback if action == "abstain" else action for action in actions]
    regrets = [action_regret(u, action) for u, action in zip(units, selected)]
    total = math.fsum(regrets)
    reference = math.fsum(action_regret(u, "graph") for u in units) / len(units)
    groups = {}
    for name, abstain in (("covered", False), ("abstained", True)):
        losses = [loss for action, loss in zip(actions, regrets)
                  if (action == "abstain") == abstain]
        subtotal = math.fsum(losses)
        groups[name] = {
            "units": len(losses), "sum_regret_pp_units": 100 * subtotal,
            "subset_mean_regret_pp": 100 * subtotal / len(losses) if losses else None,
            "full_set_contribution_pp": 100 * subtotal / len(units),
            "share_of_total_regret": subtotal / total if total else None,
        }
    return {
        "units": len(units), "abstention_fallback": fallback,
        "pre_fallback_actions": dict(sorted(Counter(actions).items())),
        "coverage": groups["covered"]["units"] / len(units),
        "selection_accuracy": sum(a == u["target_action"] for a, u in zip(selected, units)) / len(units),
        "mean_regret_pp": 100 * total / len(units),
        "always_graph_mean_regret_pp": 100 * reference,
        "excess_over_always_graph_pp": 100 * (total / len(units) - reference),
        "decomposition": groups,
    }


def preprocessing_units(preprocessing, frozen):
    source = required(preprocessing, "source_run", "preprocessing")
    require(required(source, "run_id", "source_run") == "posthoc_preprocessing_v1",
            "unexpected preprocessing run")
    for field in ("source_commit", "config_sha256"):
        require(bool(required(source, field, "source_run")), f"source_run: empty {field}")
    datasets = required(preprocessing, "datasets", "preprocessing")
    require(set(datasets) == set(PREPROCESSING_DATASETS), "unexpected preprocessing datasets")
    index = {condition: {} for condition in CONDITIONS}
    split_verified = split_unavailable = 0
    for dataset in PREPROCESSING_DATASETS:
        require(set(datasets[dataset]) == set(CONDITIONS), f"{dataset}: missing/unexpected conditions")
        for condition in CONDITIONS:
            for row in required(datasets[dataset][condition], "units", f"{dataset}/{condition}"):
                key = record_key(row, f"{dataset}/{condition}")
                require(key[0] == dataset and key in frozen, f"{key}: missing diagnostic binding")
                require(key not in index[condition], f"{condition}: duplicate dataset/seed key {key}")
                diagnostic = frozen[key]
                if "split_id" in row:
                    require(row["split_id"] == diagnostic["split_id"], f"{key}: split_id mismatch")
                    split_verified += 1
                else:
                    split_unavailable += 1
                unit = {field: diagnostic[field] for field in ("homophily", "mean_degree", "delta_h")}
                unit.update({"dataset": dataset, "seed": key[1],
                             "test_gap": required(row, "test_gap", str(key)),
                             "target_action": required(row, "target", str(key))})
                for field in ("mlp_test", "graph_test", "mlp_validation", "graph_validation"):
                    unit[f"selected_{field}"] = required(row, field, str(key))
                require(required(row, "selected_graph", str(key)) in GRAPH_MODELS,
                        f"{key}: unknown selected graph")
                validate_unit_values(unit, f"{condition}/{key}")
                unit["decisions"] = fixed_decisions(unit)
                saved_actions = required(row, "actions", str(key))
                saved_regret = required(row, "regret", str(key))
                expected_policies = {"always_graph", "always_mlp", COMBINED, "validation_selection"}
                require(set(saved_actions) == expected_policies and set(saved_regret) == expected_policies,
                        f"{key}: missing/unexpected saved policies")
                for policy in expected_policies:
                    action = unit["decisions"][policy]["action"]
                    action = "mlp" if action == "abstain" else action
                    require(saved_actions[policy] == action, f"{key}: saved {policy} action mismatch")
                    same_number(saved_regret[policy], action_regret(unit, action), f"{key}/{policy}.regret")
                index[condition][key] = unit
    for condition in CONDITIONS:
        check_scope(index[condition], PREPROCESSING_DATASETS, condition)
        groups = [(f"{condition}/{dataset}", datasets[dataset][condition],
                   [unit for key, unit in index[condition].items() if key[0] == dataset])
                  for dataset in PREPROCESSING_DATASETS]
        condition_summary = required(required(preprocessing, "conditions", "preprocessing"),
                                     condition, "preprocessing.conditions")
        groups.append((condition, condition_summary, list(index[condition].values())))
        for context, saved, units in groups:
            same_number(required(saved, "target_graph", context),
                        sum(u["target_action"] == "graph" for u in units), f"{context}.target_graph")
            for summary_field, unit_field in (("mean_graph_test", "selected_graph_test"),
                                              ("mean_mlp_test", "selected_mlp_test"),
                                              ("mean_test_gap", "test_gap")):
                same_number(required(saved, summary_field, context),
                            math.fsum(u[unit_field] for u in units) / len(units),
                            f"{context}.{summary_field}")
            saved_regret = required(saved, "mean_regret_pp", context)
            for policy in ("always_graph", "always_mlp", COMBINED, "validation_selection"):
                expected = score_method(units, policy)["full_set_mean_regret"] * 100
                same_number(required(saved_regret, policy, context), expected,
                            f"{context}.{policy}.mean_regret_pp")
    binding = {
        "diagnostic_lookup": "exact dataset/seed key in the frozen audit",
        "split_ids_checked": split_verified,
        "split_ids_absent_from_preprocessing_summary": split_unavailable,
        "limitation": "Absent split IDs cannot be independently checked from these two summaries; "
                      "a dataset/seed lookup is not proof of split identity or raw-record provenance.",
        "performance_source": "All validation and test values are from the same preprocessing run; "
                              "no frozen-run model outcomes are inserted into either condition.",
    }
    return index, binding


def summarize(audit, preprocessing):
    frozen = frozen_units(audit)
    independent, binding = preprocessing_units(preprocessing, frozen)
    frozen_grid = {}
    for name, excluded in (("all_11", ()), ("without_chameleon_squirrel", ("Chameleon", "Squirrel"))):
        units = [row for key, row in sorted(frozen.items()) if key[0] not in excluded]
        frozen_grid[name] = {
            "excluded_datasets": list(excluded),
            "fallbacks": {fallback: summarize_policy(units, fallback) for fallback in FALLBACKS},
        }
    preprocessing_grid = {}
    for condition in CONDITIONS:
        units = [independent[condition][key] for key in sorted(independent[condition])]
        preprocessing_grid[condition] = {
            "fallbacks": {fallback: summarize_policy(units, fallback) for fallback in FALLBACKS},
            "datasets": {dataset: {
                fallback: summarize_policy([u for u in units if u["dataset"] == dataset], fallback)
                for fallback in FALLBACKS} for dataset in PREPROCESSING_DATASETS},
        }
    return {
        "schema_version": "1.0", "analysis_status": "post_hoc_descriptive_reanalysis",
        "scope": "Fixed six-architecture portfolio. Separate frozen 110-unit and paired two-dataset "
                 "preprocessing analyses; no mixed-run aggregation, new training, new test evaluations, "
                 "or confirmatory significance claims. Graph fallback is a post-hoc policy change. "
                 "Removing datasets does not test duplicate-filtered versions or identify duplicate-node causation.",
        "frozen_four_way_scope": "MLP/graph fallback crossed with all eleven datasets or exclusion "
                                 "of Chameleon and Squirrel only; this is not a range over all published exclusions.",
        "frozen_four_way": frozen_grid,
        "preprocessing_source": {key: preprocessing["source_run"][key]
                                 for key in ("run_id", "source_commit", "config_sha256")},
        "preprocessing_binding": binding,
        "preprocessing_two_by_two": preprocessing_grid,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ANALYSIS / "diagnostic_audit.json")
    parser.add_argument("--preprocessing", type=Path, default=ANALYSIS / "preprocessing_sensitivity.json")
    parser.add_argument("--output", type=Path, help="Create a new JSON file; default prints only and never overwrites")
    args = parser.parse_args(argv)
    result = summarize(load_json(args.audit), load_json(args.preprocessing))
    result["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in (("audit", args.audit), ("preprocessing", args.preprocessing))}
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
