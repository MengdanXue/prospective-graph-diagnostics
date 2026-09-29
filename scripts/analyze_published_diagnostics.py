"""Reconstruct both MLP budgets and published-metric decisions after completion.

This CLI has no synthetic bypass. Complete formal runs, authoritative configs,
data binding and real checkpoints must validate before any research metric is
computed. Execute it inside the existing charged/supervised analysis phase.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from scripts.input_robustness_analysis_adapter import adapt_records
from scripts.input_robustness_formal_records import (
    CONDITIONS, FormalRecordError, digest, file_digest, record_key,
)
from scripts.published_graph_diagnostics import (
    adapt_published_records, compute_train_metrics, make_metric_record,
)


def assemble_budget_analyses(base_records: Sequence[Mapping[str, Any]],
                             extended_records: Sequence[Mapping[str, Any]],
                             metric_records: Sequence[Mapping[str, Any]], *,
                             include_optimistic: bool = False) -> dict[str, Any]:
    """Pure fixture/integration entry; production caller validates file evidence."""
    base = {record_key(row): row for row in base_records}
    if len(base) != len(base_records):
        raise FormalRecordError("duplicate base record")
    extension = {record_key(row): row for row in extended_records}
    if len(extension) != len(extended_records):
        raise FormalRecordError("duplicate MLP extension record")
    expected = {key for key in base if key[2] == "MLP"}
    if not expected or set(extension) != expected:
        raise FormalRecordError("24-trial MLP records do not match the full base scope")
    for key, row in extension.items():
        old = base[key]
        for field in ("split_id", "diagnostics", "transform_binding"):
            if row.get(field) != old.get(field):
                raise FormalRecordError(f"24-trial MLP {field} differs from its paired base record")
    combined = [extension.get(key, row) for key, row in base.items()]
    output = {}
    for budget, records in ((4, list(base_records)), (24, combined)):
        model_groups = defaultdict(list)
        for record in records:
            model_groups[(record["condition"], record["dataset"], record["model"])].append(record)
        output[str(budget)] = {
            "mlp_trials": budget,
            "model_accuracy": [{"condition": key[0], "dataset": key[1], "model": key[2],
                                "units": len(group),
                                "mean_validation_accuracy": math.fsum(float(row["validation_accuracy"]) for row in group) / len(group),
                                "mean_test_accuracy": math.fsum(float(row["test_accuracy"]) for row in group) / len(group)}
                               for key, group in sorted(model_groups.items())],
            "original_strategies_and_h1_lodo": adapt_records(records),
            "published_metrics": adapt_published_records(records, metric_records, include_optimistic=include_optimistic),
        }
    differences = {}
    for condition in CONDITIONS:
        old = output["4"]["original_strategies_and_h1_lodo"]["portfolios_by_condition"][condition]
        new = output["24"]["original_strategies_and_h1_lodo"]["portfolios_by_condition"][condition]
        paired = []
        old_published = output["4"]["published_metrics"]["portfolios_by_condition"][condition]
        new_published = output["24"]["published_metrics"]["portfolios_by_condition"][condition]
        for first, second, first_metric, second_metric in zip(old["portfolios"], new["portfolios"], old_published, new_published):
            policies = {}
            for method, prior in first["strategies"].items():
                current = second["strategies"][method]
                policies[method] = {"mean_regret_pp_delta": current["mean_regret_pp"] - prior["mean_regret_pp"],
                                    "coverage_delta": current["coverage"] - prior["coverage"],
                                    "dataset_regret_pp_deltas": {dataset: current["datasets"][dataset]["mean_regret_pp"] - entry["mean_regret_pp"]
                                                                for dataset, entry in prior["datasets"].items()}}
            for metric, prior in first_metric["metrics"].items():
                current = second_metric["metrics"][metric]
                policies[metric + "_lodo"] = {"mean_regret_pp_delta": current["mean_regret_pp"] - prior["mean_regret_pp"],
                                               "coverage_delta": current["coverage"] - prior["coverage"],
                                               "dataset_regret_pp_deltas": {dataset: current["datasets"][dataset]["mean_regret_pp"] - entry["mean_regret_pp"]
                                                                           for dataset, entry in prior["datasets"].items()}}
            label = first["label"]
            old_h1, new_h1 = old["leave_one_dataset_out"][label], new["leave_one_dataset_out"][label]
            policies["original_h1_lodo"] = {"mean_regret_pp_delta": new_h1["mean_regret_pp"] - old_h1["mean_regret_pp"],
                                             "dataset_regret_pp_deltas": {dataset: new_h1["folds"][dataset]["regret_pp"] - entry["regret_pp"]
                                                                         for dataset, entry in old_h1["folds"].items()}}
            paired.append({"portfolio": first["portfolio"], "label": label, "policies": policies})
        differences[condition] = paired
    return {"schema_version": "1.0", "analysis_status": "post_hoc_two_inputs_two_mlp_budgets",
            "conditions": list(CONDITIONS), "mlp_budgets": [4, 24], "portfolio_count": 63,
            "fixed_strategy_count": 9, "published_metric_count": 2,
            "by_mlp_budget": output, "descriptive_budget_24_minus_4": differences,
            "scientific_scope": "published metrics with study-specific LODO calibration; no new graph training"}


def compute_bound_metric_records(base_records: Sequence[Mapping[str, Any]], *,
                                  data_root: Path, binding_path: Path) -> list[dict[str, Any]]:
    from scripts.input_robustness_data import load_bound_dataset
    binding = json.loads(Path(binding_path).read_text(encoding="utf-8"))
    expected = {(str(row["dataset"]), int(row["seed"]), str(row["split_id"])) for row in base_records}
    result = []
    for dataset in sorted({key[0] for key in expected}):
        features, labels, edges, splits, _ = load_bound_dataset(dataset, data_root, binding)
        num_nodes = int(features.size(0))
        del features
        for _, seed, split_id in sorted(key for key in expected if key[0] == dataset):
            partition, actual_split_id = splits[seed]
            if split_id != actual_split_id:
                raise FormalRecordError("metric source split differs from model records")
            train = partition["train"]
            # The metric function receives no non-training labels. Loading and
            # hashing bound tensors is provenance validation, not model selection.
            statistics = compute_train_metrics(edges.numpy(), train, labels[train].numpy(), num_nodes=num_nodes)
            result.append(make_metric_record(dataset, seed, split_id, statistics))
        del labels, edges, splits
    return result


def analyze_complete_extension(*, base_root: Path, extension_root: Path,
                                base_config_path: Path, extension_config_path: Path,
                                binding_path: Path, data_root: Path,
                                include_optimistic: bool = False) -> dict[str, Any]:
    from scripts.validate_input_robustness_formal_records import validate_complete_run
    from scripts.mlp_budget_extension import validate_extension_run
    base = validate_complete_run(base_root, config_path=base_config_path, data_binding_path=binding_path)
    extension = validate_extension_run(extension_root, config_path=extension_config_path, base_root=base_root,
                                        base_config_path=base_config_path, binding_path=binding_path,
                                        require_complete=True, allow_partial=False)
    metric_records = compute_bound_metric_records(base["records"], data_root=data_root, binding_path=binding_path)
    analyses = assemble_budget_analyses(base["records"], extension["records"], metric_records,
                                        include_optimistic=include_optimistic)
    analyses["metric_records"] = metric_records
    source = Path(__file__).resolve().parents[1]
    analyses["provenance"] = {
        "analysis_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "base_run_id": base["run_id"], "base_record_digest": base["record_digest"],
        "extension_record_digest": extension.get("record_digest"),
        "extension_validation_summary_sha256": digest({k: v for k, v in extension.items() if k != "records"}),
        "base_complete_sha256": file_digest(base_root / "complete.json"),
        "extension_complete_sha256": file_digest(extension_root / "complete.json"),
        "base_config_file_sha256": file_digest(base_config_path),
        "extension_config_file_sha256": file_digest(extension_config_path),
        "data_binding_sha256": file_digest(binding_path),
        "analysis_script_sha256": file_digest(Path(__file__)),
        "metric_script_sha256": file_digest(source / "scripts/published_graph_diagnostics.py"),
    }
    return analyses


def analyze_complete_base(*, base_root: Path, base_config_path: Path,
                          binding_path: Path, data_root: Path) -> dict[str, Any]:
    """Evaluate the two published statistics before additional MLP training.

    This is the same calibration adapter used after the extension. It cannot
    accept a success summary in place of the complete checkpoint-bound run.
    """
    from scripts.validate_input_robustness_formal_records import validate_complete_run
    source = Path(__file__).resolve().parents[1]
    paths = {"base_complete": base_root / "complete.json", "base_manifest": base_root / "manifest.json",
             "base_config": base_config_path, "data_binding": binding_path,
             "analysis_script": Path(__file__), "metric_script": source / "scripts/published_graph_diagnostics.py"}
    before = {name: file_digest(path) for name, path in paths.items()}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    print("validating the complete checkpoint-bound base for published metrics", flush=True)
    base = validate_complete_run(base_root, config_path=base_config_path, data_binding_path=binding_path)
    if base["record_count"] != 1540 or base["trial_count"] != 6160:
        raise FormalRecordError("published base analysis requires all 1540 records and 6160 trials")
    metric_records = compute_bound_metric_records(base["records"], data_root=data_root, binding_path=binding_path)
    print("calibrating both training-label metrics in all 63 portfolios", flush=True)
    results = adapt_published_records(base["records"], metric_records, include_optimistic=False)
    if before != {name: file_digest(path) for name, path in paths.items()} or commit != subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip():
        raise FormalRecordError("published analysis evidence/source changed during execution")
    results["mlp_trials"] = 4
    results["metric_records"] = metric_records
    results["provenance"] = {
        "analysis_source_commit": commit, "base_run_id": base["run_id"],
        "base_record_digest": base["record_digest"], "record_count": base["record_count"],
        "trial_count": base["trial_count"],
        "input_files": {name: {"path": str(path.resolve()), "sha256": before[name]} for name, path in paths.items()},
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--extension-config", type=Path, required=True)
    parser.add_argument("--data-binding", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-optimistic", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("analysis output already exists; preserve prior evidence")
    result = analyze_complete_extension(base_root=args.base_root, extension_root=args.extension_root,
                                        base_config_path=args.base_config, extension_config_path=args.extension_config,
                                        binding_path=args.data_binding, data_root=args.data_root,
                                        include_optimistic=args.include_optimistic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "complete", "output": str(args.output), "sha256": file_digest(args.output),
                      "conditions": result["conditions"], "mlp_budgets": result["mlp_budgets"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
