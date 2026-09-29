"""Post-hoc exclusions and Combined fallback sensitivity on completed records.

Run this finite CLI under the existing budgeted supervisor. It neither trains
models nor creates a ledger, and emits progress for the enclosing run log.
The formal CLI has no synthetic/skip-validation switch. Pure functions below
also support isolated, in-memory test fixtures.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from experiments.evaluate_diagnostics import (
    METHODS, REFERENCE_METHOD, bootstrap_mean_ci, dataset_mean_regret,
    paired_comparisons, stable_seed, unit_regrets,
)
from scripts.input_robustness_analysis_adapter import (
    PRACTICAL_MARGIN, _frozen_units, _portfolio_rows, _units, evaluate_strategy,
)
from scripts.input_robustness_formal_records import (
    CONDITIONS, DATASETS, GRAPH_MODELS, FormalRecordError, file_digest,
)
from scripts.validate_input_robustness_formal_records import validate_complete_run


PORTFOLIOS = (GRAPH_MODELS, ("GCN",), ("GAT",), ("GCN", "GAT"))
FALLBACKS = ("mlp", "graph", "validation_selection")
SAMPLES = 10000


def settings(samples: int = SAMPLES) -> dict[str, Any]:
    return {"bootstrap": {"samples": samples, "seed": 20260808},
            "permutation": {"samples": samples, "seed": 20260809}}


def exclusions(datasets: Sequence[str]) -> dict[str, tuple[str, ...]]:
    names = set(datasets)
    if names != set(DATASETS):
        raise FormalRecordError("sensitivity requires the original eleven datasets")
    return {"all": (), "exclude_chameleon_squirrel": ("Chameleon", "Squirrel"),
            "exclude_cornell_wisconsin": ("Cornell", "Wisconsin"),
            **{f"leave_out:{name}": (name,) for name in sorted(names)}}


def resolve_combined(units: Sequence[dict[str, Any]], fallback: str) -> list[dict[str, Any]]:
    """Change only abstentions of Combined; retain all frozen explicit actions."""
    if fallback not in FALLBACKS:
        raise FormalRecordError(f"unsupported fallback: {fallback}")
    resolved = copy.deepcopy(units)
    for unit in resolved:
        choice = unit["decisions"][REFERENCE_METHOD]
        if choice["action"] == "abstain" and fallback != "mlp":
            choice["action"] = (unit["decisions"]["validation_selection"]["action"]
                                if fallback == "validation_selection" else "graph")
    return resolved


def _means(units: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for unit in units:
        grouped[unit["dataset"]].append(float(unit[key]))
    return {name: float(np.mean(values)) for name, values in sorted(grouped.items())}


def cross_condition(control: list[dict[str, Any]], candidate: list[dict[str, Any]],
                    label: str, *, samples: int = SAMPLES) -> dict[str, Any]:
    """Use the existing inference script's paired-dataset estimands and seeds."""
    identity = lambda unit: (unit["dataset"], unit["seed"], unit["split_id"])
    if {identity(unit) for unit in control} != {identity(unit) for unit in candidate}:
        raise FormalRecordError("cross-condition units or splits are not paired")
    ca = dataset_mean_regret(control, REFERENCE_METHOD)
    cb = dataset_mean_regret(candidate, REFERENCE_METHOD)
    ga = dataset_mean_regret(control, "always_graph")
    gb = dataset_mean_regret(candidate, "always_graph")
    series = {
        "combined_regret": {d: cb[d] - ca[d] for d in ca},
        "combined_minus_always_graph": {d: (cb[d] - gb[d]) - (ca[d] - ga[d]) for d in ca},
    }
    for name, key in (("mlp_test", "selected_mlp_test"),
                      ("selected_graph_test", "selected_graph_test")):
        a, b = _means(control, key), _means(candidate, key)
        series[name] = {d: b[d] - a[d] for d in a}
    output = {}
    for name, values in series.items():
        vector = np.asarray([values[d] for d in sorted(values)])
        output[name] = {
            "estimand": f"{CONDITIONS[1]} minus {CONDITIONS[0]}, equal-weight dataset mean",
            "mean": float(vector.mean()),
            "bootstrap_95_ci": bootstrap_mean_ci(
                vector, samples=samples, seed=stable_seed(20260913, f"{label}:{name}")),
            "datasets_improved": int((vector < 0).sum() if "regret" in name or "minus" in name
                                     else (vector > 0).sum()),
            "per_dataset": {d: values[d] for d in sorted(values)},
        }
    return output


def _require_same(expected: Any, observed: Any, location: str) -> None:
    """Reject mismatched summaries; numeric tolerance only covers JSON arithmetic."""
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(expected) != set(observed):
            raise FormalRecordError(f"evidence fields differ at {location}")
        for key, value in expected.items():
            _require_same(value, observed[key], f"{location}/{key}")
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            raise FormalRecordError(f"evidence rows differ at {location}")
        for index, (a, b) in enumerate(zip(expected, observed)):
            _require_same(a, b, f"{location}/{index}")
    elif isinstance(expected, float):
        if (isinstance(observed, bool) or not isinstance(observed, (float, int))
                or not math.isfinite(observed)
                or not math.isclose(expected, observed, abs_tol=1e-12, rel_tol=0.0)):
            raise FormalRecordError(f"evidence value differs at {location}")
    elif type(expected) is not type(observed) or expected != observed:
        raise FormalRecordError(f"evidence value differs at {location}")


def verify_prior_outputs(records: Sequence[Mapping[str, Any]], adapter: dict[str, Any],
                         inference: dict[str, Any], *, samples: int = SAMPLES) -> dict[str, int]:
    """Reconstruct the four relevant adapter strata and all existing inference."""
    if adapter.get("conditions") != list(CONDITIONS) or adapter.get("portfolio_count") != 63:
        raise FormalRecordError("adapter is not the approved two-condition/63-portfolio result")
    inference_portfolios = (GRAPH_MODELS,) + tuple((model,) for model in GRAPH_MODELS)
    labels = {"+".join(p) for p in inference_portfolios}
    if set(inference.get("paired_vs_combined", {})) != set(CONDITIONS):
        raise FormalRecordError("inference conditions are missing or unexpected")
    if set(inference.get("cross_condition", {})) != labels:
        raise FormalRecordError("inference cross-condition portfolio scope differs")
    units = {}
    try:
        for condition in CONDITIONS:
            grouped = _units(records, condition)
            entries = adapter["portfolios_by_condition"][condition]["portfolios"]
            by_label = {row["label"]: row for row in entries}
            if len(entries) != 63 or len(by_label) != 63:
                raise FormalRecordError("duplicate or missing adapter portfolios")
            for portfolio in PORTFOLIOS:
                label = "+".join(portfolio)
                rows = _portfolio_rows(grouped, portfolio)
                expected = {method: evaluate_strategy(rows, method) for method in METHODS}
                _require_same(expected, by_label[label]["strategies"], f"adapter/{condition}/{label}")
            if set(inference["paired_vs_combined"][condition]) != labels:
                raise FormalRecordError("inference portfolio scope differs")
            for portfolio in inference_portfolios:
                label = "+".join(portfolio)
                u = _frozen_units(_portfolio_rows(grouped, portfolio), practical_margin=PRACTICAL_MARGIN)
                units[condition, label] = u
                _require_same(paired_comparisons(u, settings(samples)),
                              inference["paired_vs_combined"][condition][label],
                              f"inference/{condition}/{label}")
        for label in labels:
            _require_same(cross_condition(units[CONDITIONS[0], label], units[CONDITIONS[1], label],
                                          label, samples=samples),
                          inference["cross_condition"][label], f"cross_condition/{label}")
    except (KeyError, TypeError) as exc:
        raise FormalRecordError(f"prior analysis structure is incomplete: {exc}") from exc
    return {"adapter_strategy_strata_reconstructed": 72,
            "within_condition_comparisons_reconstructed": 112,
            "cross_condition_series_reconstructed": 28}


def summarize_case(original: list[dict[str, Any]], resolved: list[dict[str, Any]],
                   *, samples: int = SAMPLES) -> dict[str, Any]:
    if not original or len(original) != len(resolved):
        raise FormalRecordError("empty or unpaired sensitivity case")
    per_dataset = dataset_mean_regret(resolved, REFERENCE_METHOD)
    actions = [unit["decisions"][REFERENCE_METHOD]["action"] for unit in original]
    effective = [unit["decisions"][REFERENCE_METHOD]["action"] for unit in resolved]
    correct = [float(("mlp" if action == "abstain" else action) == unit["target_action"])
               for action, unit in zip(effective, resolved)]
    correct_by_dataset: dict[str, list[float]] = defaultdict(list)
    for unit, value in zip(resolved, correct):
        correct_by_dataset[unit["dataset"]].append(value)
    resolution = {}
    for method in METHODS:
        if method == REFERENCE_METHOD:
            continue
        other = dataset_mean_regret(resolved, method)
        delta = np.asarray([other[d] - per_dataset[d] for d in sorted(per_dataset)])
        nonzero = int(np.count_nonzero(delta))
        resolution[method] = {"nonzero_datasets": nonzero,
                              "negative_datasets": int((delta < 0).sum()),
                              "positive_datasets": int((delta > 0).sum()),
                              "sign_flip_raw_p_floor": 2.0 ** (1 - nonzero) if nonzero else 1.0}
    return {
        "units": len(original), "dataset_count": len(per_dataset),
        "included_datasets": sorted(per_dataset),
        "combined_dataset_mean_regret_pp": 100 * float(np.mean(list(per_dataset.values()))),
        "combined_per_dataset_regret_pp": {d: 100 * v for d, v in sorted(per_dataset.items())},
        "always_graph_dataset_mean_regret_pp": 100 * float(np.mean(list(
            dataset_mean_regret(resolved, "always_graph").values()))),
        "validation_selection_dataset_mean_regret_pp": 100 * float(np.mean(list(
            dataset_mean_regret(resolved, "validation_selection").values()))),
        "original_diagnostic_actions": {action: actions.count(action) for action in ("graph", "mlp", "abstain")},
        "original_diagnostic_coverage": 1 - actions.count("abstain") / len(actions),
        "resolved_policy_unit_selection_accuracy": float(np.mean(correct)),
        "resolved_policy_dataset_selection_accuracy": float(np.mean(
            [np.mean(values) for values in correct_by_dataset.values()])),
        "paired_comparisons": paired_comparisons(resolved, settings(samples)),
        "comparison_resolution": resolution,
    }


def analyze_records(records: Sequence[Mapping[str, Any]], *, samples: int = SAMPLES,
                    progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    cases, cross, outcomes, cache = {}, {}, {}, {}
    omitted = exclusions(sorted({str(row["dataset"]) for row in records}))
    for condition in CONDITIONS:
        cases[condition], outcomes[condition] = {}, {}
        grouped = _units(records, condition)
        for portfolio in PORTFOLIOS:
            label = "+".join(portfolio)
            original = _frozen_units(_portfolio_rows(grouped, portfolio), practical_margin=PRACTICAL_MARGIN)
            cases[condition][label], outcomes[condition][label] = {}, {}
            for fallback in FALLBACKS:
                resolved = resolve_combined(original, fallback)
                regrets = unit_regrets(resolved, REFERENCE_METHOD)
                outcomes[condition][label][fallback] = [
                    {"dataset": a["dataset"], "seed": a["seed"], "split_id": a["split_id"],
                     "selected_graph": a["selected_graph"], "target": a["target_action"],
                     "action": a["decisions"][REFERENCE_METHOD]["action"],
                     "effective_action": ("mlp" if b["decisions"][REFERENCE_METHOD]["action"] == "abstain"
                                          else b["decisions"][REFERENCE_METHOD]["action"]),
                     "regret": regret}
                    for a, b, regret in zip(original, resolved, regrets)]
                for scenario, excluded in omitted.items():
                    a = [u for u in original if u["dataset"] not in excluded]
                    b = [u for u in resolved if u["dataset"] not in excluded]
                    cases[condition][label].setdefault(scenario, {})[fallback] = summarize_case(a, b, samples=samples)
                    cache[condition, label, scenario, fallback] = b
            if progress:
                progress(f"scored {condition} {label}")
    for portfolio in PORTFOLIOS:
        label = "+".join(portfolio)
        cross[label] = {}
        for scenario in omitted:
            cross[label][scenario] = {}
            for fallback in FALLBACKS:
                seed_label = label if (scenario, fallback) == ("all", "mlp") else f"{label}:{scenario}:{fallback}"
                cross[label][scenario][fallback] = cross_condition(
                    cache[CONDITIONS[0], label, scenario, fallback],
                    cache[CONDITIONS[1], label, scenario, fallback], seed_label, samples=samples)
    return {
        "schema_version": "1.0", "analysis_status": "post_hoc_existing_record_sensitivity",
        "conditions": list(CONDITIONS), "portfolios": [list(p) for p in PORTFOLIOS],
        "excluded_datasets_by_scenario": {k: list(v) for k, v in omitted.items()},
        "fallbacks": list(FALLBACKS), "primary_fallback": "mlp",
        "fallback_scope": "Only Combined abstentions change; all explicit and other-strategy decisions remain frozen.",
        "within_condition_inference": {**settings(samples), "unit": "equal-weight dataset mean",
            "holm_family": "eight frozen comparators against Combined within each condition/portfolio/scenario/fallback",
            "global_multiplicity_adjustment": False},
        "cross_condition_inference": {"bootstrap_samples": samples, "base_seed": 20260913,
            "seed_label": "<portfolio>:<metric> for all/mlp; otherwise <portfolio>:<scenario>:<fallback>:<metric>",
            "sign_flip_test": False, "multiplicity_adjustment": False},
        "notes": ["Original model records and primary MLP fallback are retained.",
                  "No retraining, threshold refitting, or new diagnostic method.",
                  "All sensitivity intervals are unadjusted, descriptive dataset-bootstrap intervals.",
                  "Per-case Holm does not establish global significance across sensitivity scenarios.",
                  "Point-estimate direction, interval evidence, and sign-flip evidence are reported separately."],
        "cases": cases, "cross_condition": cross, "unit_outcomes": outcomes,
    }


def run(root: Path, config: Path, data_binding: Path, adapter_path: Path,
        inference_path: Path, output: Path) -> dict[str, Any]:
    """Strict research entry, intended to be a child of the existing supervisor."""
    if output.exists():
        raise FormalRecordError(f"refusing to replace an existing sensitivity artifact: {output}")
    paths = {"manifest": root / "manifest.json", "complete": root / "complete.json",
             "config": config, "data_binding": data_binding,
             "adapter_result": adapter_path, "inference_result": inference_path}
    before = {name: file_digest(path) for name, path in paths.items()}
    repo = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__), repo / "experiments/evaluate_diagnostics.py",
                    repo / "scripts/input_robustness_analysis_adapter.py",
                    repo / "scripts/input_robustness_formal_records.py",
                    repo / "scripts/input_robustness_checkpoint_store.py",
                    repo / "scripts/validate_input_robustness_formal_records.py"]
    source_before = {p.relative_to(repo).as_posix(): file_digest(p) for p in source_paths}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    source_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *[str(p.relative_to(repo)) for p in source_paths]],
        cwd=repo, text=True)
    # The enclosing supervisor can stop us at any stage. Retain the accepted
    # record snapshot rather than silently blessing concurrently modified files.
    record_before = {p: (p.stat().st_size, p.stat().st_mtime_ns)
                     for p in (root / "records").rglob("*.json")}
    progress = lambda stage: print(json.dumps({"stage": stage}, ensure_ascii=False), flush=True)
    progress("strict validation of completed formal records and checkpoints")
    validated = validate_complete_run(root, config_path=config, data_binding_path=data_binding)
    if validated["record_count"] != 1540 or validated["trial_count"] != 6160:
        raise FormalRecordError("sensitivity requires the complete approved 1540-record/6160-trial run")
    records = validated["records"]
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    progress("reconstructing previous adapter and inference outputs")
    prior_checks = verify_prior_outputs(records, adapter, inference)
    result = analyze_records(records, progress=progress)
    if any(file_digest(path) != before[name] for name, path in paths.items()):
        raise FormalRecordError("an input evidence file changed during analysis")
    if source_before != {p.relative_to(repo).as_posix(): file_digest(p) for p in source_paths}:
        raise FormalRecordError("analysis source changed during execution")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != head:
        raise FormalRecordError("analysis HEAD changed during execution")
    record_after = {p: (p.stat().st_size, p.stat().st_mtime_ns)
                    for p in (root / "records").rglob("*.json")}
    if record_after != record_before:
        raise FormalRecordError("formal record files changed during analysis")
    result["provenance"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": validated["run_id"], "training_source_commit": records[0]["source_commit"],
        "config_sha256": records[0]["config_sha256"], "data_binding_sha256": records[0]["data_binding_sha256"],
        "record_digest": validated["record_digest"], "record_count": validated["record_count"],
        "trial_count": validated["trial_count"], "analysis_head_commit": head,
        "analysis_source_status": source_status,
        "analysis_source_sha256": source_before,
        "input_files": {name: {"path": str(path.resolve()), "sha256": before[name]}
                        for name, path in paths.items()},
        "prior_output_reconstruction": prior_checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves all previous evidence, including interrupted outputs.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    progress(f"completed {output}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "config", "data-binding", "adapter", "inference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.config, args.data_binding, args.adapter, args.inference, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
