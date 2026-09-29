"""One owned MLP-extension worker, always launched by the existing guardian."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from experiments.run_prospective_benchmark import environment_snapshot
from scripts.input_robustness_data import fit_transform_features_bounded, load_bound_dataset
from scripts.input_robustness_formal_records import FormalRecordError, _exclusive_json, digest
from scripts.mlp_budget_extension import ExtensionRecordWriter, base_record, load_base_evidence, load_fixture_tensors, read_json, run_mlp24_unit, validate_extension_run
from scripts.input_robustness_formal_records import file_digest
from scripts.run_input_robustness_formal import _configure_runtime


def analyze_fixture_extension(*, evidence, output, config_path):
    from scripts.analyze_published_diagnostics import assemble_budget_analyses
    from scripts.published_graph_diagnostics import compute_train_metrics, make_metric_record
    from scripts.validate_input_robustness_formal_records import validate_complete_run
    config = read_json(config_path)
    if config.get("execution_mode") != "isolated_fixture":
        raise FormalRecordError("fixture analysis requires an isolated authoritative configuration")
    base = validate_complete_run(Path(evidence["root"]), config_path=Path(evidence["config_path"]),
                                 data_binding_path=Path(evidence["binding_path"]))
    extended = validate_extension_run(output, config_path=config_path,
        base_root=Path(evidence["root"]), base_config_path=Path(evidence["config_path"]),
        binding_path=Path(evidence["binding_path"]))
    metrics = []
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            tensors = load_fixture_tensors(evidence, dataset, config["conditions"][0], seed)
            train = tensors["train_indices"]
            statistics = compute_train_metrics(tensors["edge_index"].numpy(), train.numpy(),
                tensors["y"][train].numpy(), num_nodes=len(tensors["y"]))
            metrics.append(make_metric_record(dataset, seed,
                base_record(evidence, dataset, config["conditions"][0], seed)["split_id"], statistics))
    return assemble_budget_analyses(base["records"], extended["records"], metrics)


def execute_request(request):
    config = read_json(Path(request["config_path"]))
    _configure_runtime(config)
    if request["operation"] == "existing_analysis":
        from scripts.mlp_budget_extension_entry import validate_acceptance
        from scripts.mlp_budget_extension import validate_extension_config
        from scripts.input_robustness_sensitivity import run as run_sensitivity
        from scripts.analyze_published_diagnostics import analyze_complete_base
        validate_extension_config(config, read_json(Path(request["base_config_path"])))
        validate_acceptance(Path(request["acceptance_path"]), config_path=Path(request["config_path"]),
                            commit=request["source_commit"])
        output = Path(request["output_root"])
        sensitivity = output / "sensitivity.json"
        published = output / "published_metrics.json"
        if any(path.exists() for path in (sensitivity, published, output / "run_complete.json")):
            raise FormalRecordError("existing-record analysis output must be fresh; preserve prior attempts")
        run_sensitivity(Path(request["base_root"]), Path(request["base_config_path"]),
                        Path(request["binding_path"]), Path(request["adapter_path"]),
                        Path(request["inference_path"]), sensitivity)
        result = analyze_complete_base(base_root=Path(request["base_root"]),
            base_config_path=Path(request["base_config_path"]), binding_path=Path(request["binding_path"]),
            data_root=Path(request["data_root"]))
        _exclusive_json(published, result)
        summary = {"status": "complete", "source_commit": request["source_commit"],
                   "config_sha256": digest(config), "base_record_digest": result["provenance"]["base_record_digest"],
                   "analysis_status": "post_hoc_existing_records", "research_training_started": False,
                   "sensitivity": {"path": str(sensitivity.resolve()), "sha256": file_digest(sensitivity)},
                   "published_metrics": {"path": str(published.resolve()), "sha256": file_digest(published)}}
        _exclusive_json(output / "run_complete.json", summary)
        return summary
    if request["operation"] == "prepare":
        from scripts.mlp_budget_extension_entry import validate_acceptance
        validate_acceptance(Path(request["acceptance_path"]), config_path=Path(request["config_path"]),
                            commit=request["source_commit"])
        evidence = load_base_evidence(Path(request["base_root"]), config_path=Path(request["base_config_path"]),
                                      binding_path=Path(request["binding_path"]), extension_config=config)
        keys = []
        if request.get("resume"):
            previous = validate_extension_run(Path(request["output_root"]), config_path=Path(request["config_path"]),
                base_root=Path(request["base_root"]), base_config_path=Path(request["base_config_path"]),
                binding_path=Path(request["binding_path"]), require_complete=False, allow_partial=True)
            keys = [[row["dataset"], row["condition"], row["seed"]] for row in previous["records"]]
        return {"base_evidence": evidence, "completed_keys": keys}
    output = Path(request["output_root"])
    evidence = request["base_evidence"]
    if request["operation"] == "finalize":
        writer = ExtensionRecordWriter(output, config_path=Path(request["config_path"]),
            base_evidence=evidence, source_commit=request["source_commit"], resume=True)
        complete = writer.finalize()
        if config.get("execution_mode") == "isolated_fixture":
            analysis = analyze_fixture_extension(evidence=evidence, output=output, config_path=Path(request["config_path"]))
        else:
            from scripts.analyze_published_diagnostics import analyze_complete_extension
            if not request.get("analysis_data_root"):
                raise FormalRecordError("formal finalization requires the bound analysis data root")
            analysis = analyze_complete_extension(base_root=Path(evidence["root"]), extension_root=output,
                base_config_path=Path(evidence["config_path"]), extension_config_path=Path(request["config_path"]),
                binding_path=Path(evidence["binding_path"]), data_root=Path(request["analysis_data_root"]))
        if (output / "analysis.json").exists():
            if digest(read_json(output / "analysis.json")) != digest(analysis):
                raise FormalRecordError("saved analysis differs from strict reconstruction")
        else:
            _exclusive_json(output / "analysis.json", analysis)
        result = {"status": "complete", "records_complete_sha256": file_digest(output / "complete.json"),
                  "analysis_sha256": file_digest(output / "analysis.json"),
                  "record_digest": complete["record_digest"], "config_sha256": digest(config),
                  "source_commit": request["source_commit"], "conditions": config["conditions"],
                  "mlp_budgets": [4, 24], "portfolio_count": 63, "fixed_strategy_count": 9}
        _exclusive_json(output / "run_complete.json", result)
        return result
    if request["operation"] != "unit":
        raise FormalRecordError("unknown extension worker operation")
    fixture = config.get("execution_mode") == "isolated_fixture"
    if request.get("worker_sleep_seconds", 0):
        if not fixture or not 0 <= float(request["worker_sleep_seconds"]) <= 60:
            raise FormalRecordError("workload delay is restricted to isolated fixtures")
        time.sleep(float(request["worker_sleep_seconds"]))
    dataset, condition, seed = request["dataset"], request["condition"], int(request["seed"])
    inherited = base_record(evidence, dataset, condition, seed)
    if fixture:
        tensors = load_fixture_tensors(evidence, dataset, condition, seed)
        x, y, edges = tensors["x"], tensors["y"], tensors["edge_index"]
        train, validation, test = [tensors[f"{name}_indices"] for name in ("train", "validation", "test")]
        split_id, transform = inherited["split_id"], inherited["transform_binding"]
    else:
        binding = read_json(Path(evidence["binding_path"]))
        raw, y, edges, splits, entry = load_bound_dataset(dataset, Path(request["data_root"]), binding)
        split, split_id = splits[seed]
        train, validation, test = [torch.from_numpy(split[name]) for name in ("train", "validation", "test")]
        x, metadata = fit_transform_features_bounded(raw, train, condition)
        transform = {**inherited["transform_binding"], "fit_statistics_sha256": digest(metadata)}
    device = torch.device(config["execution"]["model_devices"]["MLP"])
    record = run_mlp24_unit(
        base_evidence=evidence, config=config, source_commit=request["source_commit"],
        dataset=dataset, condition=condition, seed=seed, output_root=output,
        checkpoint_dir=output / "checkpoints" / condition / dataset / f"seed_{seed:03d}" / request["attempt_id"],
        x=x, y=y, edge_index=edges, train_indices=train, validation_indices=validation,
        test_indices=test, split_id=split_id, transform_binding=transform,
        environment=environment_snapshot(device), device=device)
    writer = ExtensionRecordWriter(output, config_path=Path(request["config_path"]),
        base_evidence=evidence, source_commit=request["source_commit"], resume=True)
    path = writer.write_record(record)
    return {"record_path": str(path.resolve()), "record_sha256": file_digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    request = torch.load(args.request, map_location="cpu", weights_only=False)
    _exclusive_json(args.result, execute_request(request))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
