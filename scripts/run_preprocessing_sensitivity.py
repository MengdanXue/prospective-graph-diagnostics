"""Paired post-hoc preprocessing experiment, separate from frozen execution files."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from experiments.prospective_data import canonical_undirected_edges, make_stratified_split, split_identifier
from experiments.prospective_models import build_model, prepare_h2_adjacencies
from experiments.run_prospective_benchmark import (
    _bidirectional_edge_index, _source_commit, config_sha256, environment_snapshot,
    run_model_unit, seed_everything, validate_trial_audit, write_json_exclusive,
)


def transform_features(x, condition):
    if condition == "raw":
        return x.clone()
    if condition == "normalize_features":
        return NormalizeFeatures()(Data(x=x.clone())).x
    raise ValueError(f"unknown preprocessing: {condition}")


def load_inputs(directory, name, specification, audit):
    path = directory / specification["filename"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != specification["sha256"]:
        raise ValueError(f"raw data checksum mismatch: {name}")
    with np.load(path, allow_pickle=False) as data:
        x = torch.from_numpy(data["node_features"].copy()).float()
        y = torch.from_numpy(data["node_labels"].copy()).long().view(-1)
        edges = canonical_undirected_edges(data["edges"].T, node_count=len(y))
    if x.shape != tuple(specification["shape"]) or not torch.isfinite(x).all():
        raise ValueError("feature shape or finiteness mismatch")
    splits = {}
    for unit in audit["units"]:
        if unit["dataset"] != name:
            continue
        split = make_stratified_split(y.numpy(), seed=int(unit["seed"]))
        if split_identifier(split) != unit["split_id"]:
            raise ValueError("split differs from original benchmark")
        mask = np.zeros(len(y), dtype=bool)
        mask[split["train"]] = True
        eligible = edges[mask[edges[:, 0]] & mask[edges[:, 1]]]
        h = float(np.mean(y.numpy()[eligible[:, 0]] == y.numpy()[eligible[:, 1]]))
        if abs(h - unit["homophily"]) > 1e-12 or abs(2 * len(edges) / len(y) - unit["mean_degree"]) > 1e-12:
            raise ValueError("graph statistics differ from original benchmark")
        splits[int(unit["seed"])] = (split, unit["split_id"])
    if set(splits) != set(range(10)):
        raise ValueError("missing original split scope")
    return x, y, _bidirectional_edge_index(edges), splits


def check_existing(record, expected, training):
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"resume mismatch: {key}")
    if record.get("status") != "success" or record.get("test_evaluations_after_selection") != 1:
        raise ValueError("invalid completed record")
    validate_trial_audit(record, training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/preprocessing_sensitivity_v1.json")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--preflight", action="store_true", help="Only time three training steps; no validation/test evaluation")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    source = _source_commit()
    environment = environment_snapshot(device)
    digest = config_sha256(config)
    audit = json.loads((ROOT / "results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json").read_text())
    manifest = {"run_id": config["run_id"], "config_sha256": digest, "source_commit": source,
                "environment": environment, "analysis_status": config["analysis_status"],
                "preflight": args.preflight, "config": config}
    manifest_path = args.output_root / "run_manifest.json"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("output root belongs to another run or resume was not requested")
    else:
        if args.output_root.exists() and any(args.output_root.iterdir()):
            raise ValueError("nonempty output root has no matching manifest")
        write_json_exclusive(manifest_path, manifest)
    try:
        timings = []
        for name, specification in config["datasets"].items():
            x, y, edge_index, splits = load_inputs(args.data_root, name, specification, audit)
            h2 = prepare_h2_adjacencies(edge_index, num_nodes=len(y))
            if args.preflight:
                for model_id in config["models"]:
                    seed_everything(0)
                    model = build_model(model_id, num_nodes=len(y), in_channels=x.shape[1],
                                        hidden_channels=64, out_channels=int(y.max())+1, dropout=.5,
                                        edge_index=edge_index, h2_adjacencies=h2).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
                    local_x, local_y, local_e = x.to(device), y.to(device), edge_index.to(device)
                    indices = torch.as_tensor(splits[0][0]["train"], device=device)
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(3):
                        optimizer.zero_grad(set_to_none=True)
                        F.cross_entropy(model(local_x, local_e)[indices], local_y[indices]).backward()
                        optimizer.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    entry = {"dataset": name, "model": model_id, "training_steps": 3,
                             "seconds": time.perf_counter()-start,
                             "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
                    timings.append(entry)
                    print(json.dumps(entry), flush=True)
                    del optimizer, model, local_x, local_y, local_e
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                continue
            for seed in config["seeds"]:
                split, split_id = splits[seed]
                for condition in config["conditions"]:
                    transformed = transform_features(x, condition)
                    for model_id in config["models"]:
                        path = args.output_root / "records" / condition / name / model_id / f"seed_{seed:03d}.json"
                        provenance = {"raw_filename": specification["filename"], "raw_sha256": specification["sha256"],
                                      "preprocessing": condition, "transformed_feature_sha256": hashlib.sha256(transformed.numpy().tobytes()).hexdigest()}
                        expected = {"run_id": config["run_id"], "dataset": name, "model": model_id, "seed": seed,
                                    "split_id": split_id, "source_commit": source, "environment": environment,
                                    "config_sha256": digest, "frozen_config": config, "data_provenance": provenance,
                                    "preprocessing": condition}
                        if path.exists():
                            if not args.resume:
                                raise ValueError("record already exists; explicit resume required")
                            check_existing(json.loads(path.read_text()), expected, config["training"])
                            continue
                        row = run_model_unit(run_id=config["run_id"], dataset=name, model_id=model_id, seed=seed,
                                             split_id=split_id, x=transformed, y=y, edge_index=edge_index,
                                             train_indices=torch.as_tensor(split["train"]),
                                             validation_indices=torch.as_tensor(split["validation"]),
                                             test_indices=torch.as_tensor(split["test"]), training=config["training"],
                                             source_commit=source, environment=environment, data_provenance=provenance,
                                             device=device, h2_adjacencies=h2, config_sha256=digest, frozen_config=config)
                        row["preprocessing"] = condition
                        write_json_exclusive(path, row)
                        print(json.dumps({"completed": str(path.relative_to(args.output_root)), "seconds": row["duration_seconds"]}), flush=True)
                        gc.collect()
            del x, y, edge_index, h2
        if args.preflight:
            write_json_exclusive(args.output_root / "resource_profile.json", {"status": "preflight_only", "validation_or_test_evaluations": 0, "timings": timings})
        else:
            write_json_exclusive(args.output_root / "complete.json", {"status": "complete", "expected_records": len(config["datasets"])*len(config["seeds"])*len(config["models"])*len(config["conditions"]), "config_sha256": digest})
    except Exception as exc:
        write_json_exclusive(args.output_root / f"failure_{time.time_ns()}.json", {"status": "error", "type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()

