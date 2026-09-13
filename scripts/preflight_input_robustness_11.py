"""Outcome-blind resource and repeatability preflight; never a formal trainer.

The coordinator imports no Torch. Each dataset/condition repeat gets a fresh
process with backend environment set before Torch is imported. All outputs are
exclusive and live outside the frozen evidence. Short probes cannot be resumed
as selected-model records and this program has no full-training option.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/input_robustness_11_v1.json"
DEFAULT_BINDING = ROOT / "results/diagnostic/posthoc_input_robustness_11_v1/preflight/data_binding.json"
SOURCE_FILES = (
    "scripts/preflight_input_robustness_11.py",
    "scripts/input_robustness_data.py",
    "scripts/run_mlp_optimization_diagnostic.py",
    "scripts/run_preprocessing_sensitivity.py",
    "experiments/prospective_models.py",
    "experiments/prospective_data.py",
    "experiments/run_prospective_benchmark.py",
)
GIB = 1024 ** 3


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path, value):
    """Atomic exclusive JSON publication, including on Windows."""
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_config(config):
    original = json.loads((ROOT / "configs/prospective_benchmark_v2.json").read_text())
    for name in ("datasets", "seeds", "models", "training", "split", "model_parameters", "diagnostics"):
        if config[name] != original[name]:
            raise ValueError(f"frozen scope/recipe changed: {name}")
    if config["conditions"] != ["normalize_features", "normalize_centered_scaled"]:
        raise ValueError("exactly the two planned input conditions are required")
    if config.get("analysis_status") != "post_hoc_training_recipe_robustness":
        raise ValueError("post-hoc status is required")
    if config.get("stage") != "design_resource_preflight_only" or config.get("formal_training_enabled") is not False:
        raise ValueError("this program only accepts a preflight-stage configuration")
    preflight = config["preflight"]
    if (preflight["seed"], preflight["repeats"], preflight["training_steps"], preflight["expected_workers"]) != (0, 2, 3, 44):
        raise ValueError("preflight scope is fixed before execution")
    if any(preflight[key] != 0 for key in ("validation_evaluations", "test_evaluations", "formal_records")):
        raise ValueError("preflight must not evaluate held-out metrics")
    if config["expected_records"] != 1540 or config["expected_trials"] != 6160:
        raise ValueError("formal scope count mismatch")
    if config["execution"]["model_devices"] != {name: "cpu" if name == "H2GCN" else "cuda" for name in config["models"]}:
        raise ValueError("fixed per-architecture devices changed")
    strict = {"workers": 1, "torch_num_threads": 4, "torch_num_interop_threads": 1,
              "deterministic_algorithms": True, "deterministic_warn_only": False,
              "cublas_workspace_config": ":4096:8", "allow_tf32": False,
              "cudnn_benchmark": False, "cudnn_deterministic": True}
    if any(config["execution"].get(key) != value for key, value in strict.items()):
        raise ValueError("strict execution contract changed")
    if config["preflight"].get("retain_all_four_trial_states") is not True:
        raise ValueError("preflight must model four retained checkpoints")


def resolved_budget(config):
    budget = config["resource_budget"]
    return {"preflight_total_wall_seconds": budget["preflight_wall_seconds"],
            "preflight_worker_wall_seconds": budget["preflight_worker_wall_seconds"],
            "formal_model_unit_wall_seconds": budget["formal_model_unit_wall_seconds"],
            "formal_total_wall_seconds": budget["formal_total_worker_wall_seconds"],
            **{key + "_bytes": int(budget[key + "_gib"] * GIB)
               for key in ("max_worker_rss", "min_system_available", "max_cuda_allocated", "max_cuda_reserved", "min_disk_free")}}


def workers(config):
    return [{"dataset": dataset, "condition": condition, "repeat": repeat,
             "worker_id": f"{dataset}__{condition}__repeat{repeat:02d}"}
            for dataset in config["datasets"] for condition in config["conditions"]
            for repeat in range(config["preflight"]["repeats"])]


def backend_environment(base=None):
    environment = dict(os.environ if base is None else base)
    environment.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0",
                        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
    return environment


def configure_backend():
    if "torch" in sys.modules:
        raise RuntimeError("configure backend before importing Torch")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("worker startup environment is not deterministic")
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch


def tensor_hash(tensor):
    value = tensor.detach().cpu()
    if value.is_sparse:
        value = value.coalesce()
        return json_hash({"layout": str(value.layout), "shape": list(value.shape),
                          "indices": tensor_hash(value.indices()), "values": tensor_hash(value.values())})
    value = value.contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_hash(model):
    return json_hash({name: tensor_hash(value) for name, value in sorted(model.state_dict().items())})


def tensor_bytes(value):
    if value.is_sparse:
        value = value.coalesce()
        return tensor_bytes(value.indices()) + tensor_bytes(value.values())
    return value.numel() * value.element_size()


def train_probe(*, model, x, edge_index, train_indices, train_labels, trial, training_steps,
                weight_decay, torch, checkpoint_sink=None):
    """No validation/test indices or labels are accepted. No model is selected.

    An unscored full-node inference pass mirrors the epoch's evaluation workload.
    Fingerprints cover states/buffers, gradients and outputs in both modes.
    """
    import torch.nn.functional as functional
    optimizer = torch.optim.Adam(model.parameters(), lr=trial["learning_rate"], weight_decay=weight_decay)
    fingerprints, seconds = [], []
    initial = state_hash(model)
    for step in range(training_steps):
        if x.is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, edge_index)
        loss = functional.cross_entropy(logits[train_indices], train_labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss in resource probe")
        loss.backward()
        # Check gradients without observing any held-out outcome.
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            raise FloatingPointError("nonfinite training gradient in resource probe")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            inferred = model(x, edge_index)
        if not torch.isfinite(inferred).all():
            raise FloatingPointError("nonfinite unscored inference in resource probe")
        if checkpoint_sink is not None:
            # Copy on every step without evaluating a selection score. This
            # covers the worst case of replacing an improving CPU checkpoint
            # while Adam and prior trial states remain resident.
            checkpoint_sink(model)
        if x.is_cuda:
            torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)
        fingerprints.append({"step": step, "train_logits": tensor_hash(logits),
                             "train_loss": tensor_hash(loss), "state": state_hash(model),
                             "gradients": json_hash({name: tensor_hash(parameter.grad)
                                                     for name, parameter in model.named_parameters()
                                                     if parameter.grad is not None}),
                             "unscored_inference": tensor_hash(inferred)})
        del logits, inferred, loss
    return {"initial_state": initial, "trajectory": fingerprints, "epoch_seconds": seconds,
            "training_steps": training_steps, "validation_evaluations": 0, "test_evaluations": 0}


def runtime_snapshot(torch):
    from experiments.run_prospective_benchmark import environment_snapshot
    import psutil
    import importlib.util
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    return {**environment_snapshot(torch.device("cuda")), "psutil": psutil.__version__, "driver_version": driver,
            "optional_extensions_present": {name: importlib.util.find_spec(name) is not None
                                            for name in ("torch_scatter", "torch_sparse", "pyg_lib")},
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory}


def verify_source_binding(manifest, config, binding_path):
    if manifest.get("phase") != "preflight_only" or manifest.get("config_sha256") != json_hash(config):
        raise ValueError("worker/config differs from preflight manifest")
    if file_hash(binding_path) != manifest["binding_sha256"]:
        raise ValueError("data-binding manifest changed during preflight")
    if set(manifest["source_files"]) != set(SOURCE_FILES):
        raise ValueError("incomplete executable-source binding")
    for relative, expected in manifest["source_files"].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f"executable source changed during preflight: {relative}")


def run_worker(config, worker, data_root, binding, destination):
    torch = configure_backend()
    sys.path.insert(0, str(ROOT))
    import psutil
    from experiments.prospective_models import build_model, prepare_h2_adjacencies
    from experiments.run_prospective_benchmark import seed_everything
    from scripts.input_robustness_data import load_bound_dataset, fit_transform_features_bounded, array_sha256
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no unplanned CPU fallback")
    budget = resolved_budget(config)
    torch.cuda.set_per_process_memory_fraction(budget["max_cuda_reserved_bytes"] /
                                               torch.cuda.get_device_properties(0).total_memory)
    started = time.perf_counter()
    x, y, edges, splits, metadata = load_bound_dataset(worker["dataset"], data_root, binding)
    split, split_id = splits[config["preflight"]["seed"]]
    train = torch.as_tensor(split["train"], dtype=torch.long)
    transformed, transform_metadata = fit_transform_features_bounded(x, train, worker["condition"])
    if not torch.isfinite(transformed).all():
        raise FloatingPointError("nonfinite transformed input")
    transform_digest = tensor_hash(transformed)
    candidate_binding = next(row for row in metadata["candidate_checks"] if row["seed"] == config["preflight"]["seed"])
    expected_transform = (metadata["normalize_features_sha256"] if worker["condition"] == "normalize_features"
                          else candidate_binding["transformed_feature_sha256"])
    if array_sha256(transformed) != expected_transform:
        raise ValueError("transformed features differ from sealed data binding")
    if transform_metadata != {**candidate_binding["fit_metadata"], "condition": worker["condition"]}:
        raise ValueError("fitted statistics differ from sealed data binding")
    input_digest = tensor_hash(x)
    del x
    gc.collect()
    transform_seconds = time.perf_counter() - started
    environment = runtime_snapshot(torch)
    header = {**worker, "phase": "preflight_only", "analysis_status": config["analysis_status"],
              "run_id": config["run_id"], "config_sha256": json_hash(config),
              "split_id": split_id, "data_provenance": metadata,
              "input_sha256": input_digest, "transformed_sha256": transform_digest,
              "transform_metadata": transform_metadata, "environment": environment,
              "transform_seconds": transform_seconds,
              "validation_evaluations": 0, "test_evaluations": 0}
    write_exclusive(destination / "worker.json", header)
    # All classes occur in train by the original stratified eligibility rule.
    train_labels = y[train]
    num_classes = int(y[train].max()) + 1
    del y
    h2, h2_error, h2_seconds = None, None, 0.0
    failures = 0
    for model_id in config["models"]:
        device_name = config["execution"]["model_devices"][model_id]
        device = torch.device(device_name)
        local_x, local_edges = transformed.to(device), edges.to(device)
        local_train, local_labels = train.to(device), train_labels.to(device)
        retained_checkpoints = []
        if model_id == "H2GCN":
            prepare_started = time.perf_counter()
            try:
                h2 = prepare_h2_adjacencies(edges, num_nodes=local_x.size(0))
            except Exception as exc:
                h2_error = {"exception_type": type(exc).__name__, "exception": str(exc)}
            h2_seconds = time.perf_counter() - prepare_started
        for trial_index, trial in enumerate(config["training"]["trials"]):
            cell = {"model": model_id, "trial_id": f"trial_{trial_index:03d}", "configuration": trial,
                    "seed": config["preflight"]["seed"], "phase": "preflight_only",
                    "validation_evaluations": 0, "test_evaluations": 0,
                    "run_id": config["run_id"], "config_sha256": json_hash(config),
                    "worker_id": worker["worker_id"], "dataset": worker["dataset"],
                    "condition": worker["condition"], "repeat": worker["repeat"], "device": device_name}
            model = None
            checkpoint_holder = {}
            try:
                if h2_error and model_id == "H2GCN":
                    raise RuntimeError(f"H2 preprocessing failed: {h2_error}")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                seed_everything(config["preflight"]["seed"])
                model = build_model(model_id, num_nodes=local_x.size(0), in_channels=local_x.size(1),
                                    hidden_channels=config["training"]["hidden_channels"],
                                    out_channels=num_classes, dropout=trial["dropout"],
                                    edge_index=edges, h2_adjacencies=h2).to(device)
                def keep_probe_checkpoint(current_model):
                    checkpoint_holder["state"] = {name: value.detach().cpu().clone()
                                                  for name, value in current_model.state_dict().items()}
                result = train_probe(model=model, x=local_x, edge_index=local_edges,
                                     train_indices=local_train, train_labels=local_labels, trial=trial,
                                     training_steps=config["preflight"]["training_steps"],
                                     weight_decay=config["training"]["weight_decay"], torch=torch,
                                     checkpoint_sink=keep_probe_checkpoint)
                # The production unit retains all four trial checkpoints on CPU,
                # including H2GCN's sparse adjacency buffers. Keep them here too.
                state = checkpoint_holder["state"]
                if any(not torch.isfinite(value.coalesce().values() if value.is_sparse else value).all()
                       for value in state.values()):
                    raise FloatingPointError("nonfinite model state in resource probe")
                retained_checkpoints.append(state)
                checkpoint_bytes = sum(tensor_bytes(value) for value in state.values())
                peak_reserved = torch.cuda.max_memory_reserved()
                if peak_reserved > budget["max_cuda_reserved_bytes"] or torch.cuda.max_memory_allocated() > budget["max_cuda_allocated_bytes"]:
                    raise MemoryError("CUDA reserved-memory limit exceeded")
                cell.update(status="success", **result, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_cuda_reserved_bytes=peak_reserved, rss_bytes=psutil.Process().memory_info().rss,
                            four_checkpoints_bytes=4 * checkpoint_bytes,
                            retained_checkpoint_count=len(retained_checkpoints),
                            h2_preparation_seconds=h2_seconds if model_id == "H2GCN" else 0.0)
            except Exception as exc:
                failures += 1
                cell.update(status="failed", exception_type=type(exc).__name__, exception=str(exc),
                            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            finally:
                del model, checkpoint_holder
                gc.collect()
                torch.cuda.empty_cache()
            write_exclusive(destination / "probes" / model_id / f"trial_{trial_index:03d}.json", cell)
            print(json.dumps({"worker": worker["worker_id"], "model": model_id,
                              "trial": trial_index, "status": cell["status"]}), flush=True)
        if model_id == "H2GCN":
            del h2
            h2 = None
            gc.collect()
            torch.cuda.empty_cache()
        del local_x, local_edges, local_train, local_labels, retained_checkpoints
        if "state" in locals():
            del state
        gc.collect()
        torch.cuda.empty_cache()
    write_exclusive(destination / "worker_complete.json", {"phase": "preflight_only", "probes": 28,
                    "run_id": config["run_id"], "config_sha256": json_hash(config), "worker_id": worker["worker_id"],
                    "failed_probes": failures, "seconds": time.perf_counter() - started,
                    "validation_evaluations": 0, "test_evaluations": 0})


def summarize(config, directory, jobs):
    expected = workers(config)
    budget = resolved_budget(config)
    failures, comparisons, timings = [], [], []
    environments = set()
    forbidden_metrics = {"validation_accuracy", "validation_loss", "test_accuracy", "test_loss",
                         "selected_trial_id", "regret", "regret_pp", "target"}
    setup_estimate = 0.0
    expected_job_ids = {worker["worker_id"] for worker in expected}
    if len(jobs) != len(expected) or {job["worker_id"] for job in jobs} != expected_job_ids:
        failures.append({"reason": "missing, unexpected or duplicate worker jobs"})
    job_lookup = {job["worker_id"]: job for job in jobs}
    for dataset in config["datasets"]:
        for condition in config["conditions"]:
            group = [w for w in expected if w["dataset"] == dataset and w["condition"] == condition]
            headers = []
            worker_compute = {worker["worker_id"]: 0.0 for worker in group}
            h2_setup = 0.0
            for worker in group:
                worker_dir = directory / "workers" / worker["worker_id"]
                path = worker_dir / "worker.json"
                if path.exists():
                    header = json.loads(path.read_text())
                    identity = {**worker, "phase": "preflight_only", "analysis_status": config["analysis_status"],
                                "run_id": config["run_id"], "config_sha256": json_hash(config),
                                "validation_evaluations": 0, "test_evaluations": 0}
                    if any(header.get(key) != value for key, value in identity.items()):
                        failures.append({"worker": worker["worker_id"], "reason": "worker identity, phase, config or evaluation-count mismatch"})
                    if forbidden_metrics & set(header):
                        failures.append({"worker": worker["worker_id"], "reason": "forbidden held-out or selected-model metrics in preflight"})
                    environments.add(json_hash(header.get("environment")))
                    headers.append(header)
                completion = worker_dir / "worker_complete.json"
                required_complete = {"phase": "preflight_only", "probes": 28, "failed_probes": 0,
                                     "run_id": config["run_id"], "config_sha256": json_hash(config),
                                     "worker_id": worker["worker_id"], "validation_evaluations": 0, "test_evaluations": 0}
                complete = json.loads(completion.read_text()) if completion.exists() else {}
                if any(complete.get(key) != value for key, value in required_complete.items()):
                    failures.append({"worker": worker["worker_id"], "reason": "missing, failed or mismatched worker completion"})
                required_probes = {f"{model}/trial_{index:03d}.json" for model in config["models"] for index in range(4)}
                actual_probes = {path.relative_to(worker_dir / "probes").as_posix() for path in (worker_dir / "probes").rglob("*.json")}
                if required_probes != actual_probes:
                    failures.append({"worker": worker["worker_id"], "reason": "missing or unexpected probe files"})
            if len(headers) != 2 or any(headers[0].get(key) != headers[1].get(key)
                    for key in ("input_sha256", "transformed_sha256", "split_id", "data_provenance", "transform_metadata", "environment")):
                failures.append({"dataset": dataset, "condition": condition, "reason": "missing or mismatched worker inputs/runtime"})
            for model_id in config["models"]:
                model_seconds = 0.0
                for trial_id in range(4):
                    paths = [directory / "workers" / w["worker_id"] / "probes" / model_id / f"trial_{trial_id:03d}.json" for w in group]
                    rows = []
                    for path, worker in zip(paths, group):
                        if not path.exists():
                            continue
                        row = json.loads(path.read_text())
                        identity = {**worker, "model": model_id, "trial_id": f"trial_{trial_id:03d}",
                                    "configuration": config["training"]["trials"][trial_id],
                                    "seed": config["preflight"]["seed"], "phase": "preflight_only",
                                    "validation_evaluations": 0, "test_evaluations": 0,
                                    "run_id": config["run_id"], "config_sha256": json_hash(config),
                                    "device": config["execution"]["model_devices"][model_id]}
                        if any(row.get(key) != value for key, value in identity.items()):
                            failures.append({"worker": worker["worker_id"], "model": model_id, "trial_id": trial_id,
                                             "reason": "probe identity, phase, config or evaluation-count mismatch"})
                        if forbidden_metrics & set(row):
                            failures.append({"worker": worker["worker_id"], "reason": "forbidden held-out or selected-model metrics in preflight"})
                        if row.get("status") == "success":
                            if (row.get("training_steps") != 3 or len(row.get("trajectory", [])) != 3
                                    or len(row.get("epoch_seconds", [])) != 3
                                    or row.get("retained_checkpoint_count") != trial_id + 1):
                                failures.append({"worker": worker["worker_id"], "model": model_id,
                                                 "reason": "probe trajectory/retained-state scope mismatch"})
                            if (row.get("peak_cuda_reserved_bytes", float("inf")) > budget["max_cuda_reserved_bytes"]
                                    or row.get("peak_cuda_allocated_bytes", float("inf")) > budget["max_cuda_allocated_bytes"]
                                    or row.get("rss_bytes", float("inf")) > budget["max_worker_rss_bytes"]):
                                failures.append({"worker": worker["worker_id"], "model": model_id, "reason": "probe memory cap exceeded"})
                            durations = row.get("epoch_seconds", [])
                            if not durations or any(not isinstance(value, (int, float)) or not 0 < value < float("inf") for value in durations):
                                failures.append({"worker": worker["worker_id"], "reason": "invalid measured durations"})
                            else:
                                worker_compute[worker["worker_id"]] += sum(durations)
                            h2_setup = max(h2_setup, row.get("h2_preparation_seconds", 0.0))
                        rows.append(row)
                    success = len(rows) == 2 and all(r.get("status") == "success" and len(r.get("epoch_seconds", [])) == 3 for r in rows)
                    identical = success and all(rows[0][key] == rows[1][key] for key in ("initial_state", "trajectory"))
                    entry = {"dataset": dataset, "condition": condition, "model": model_id, "trial_id": f"trial_{trial_id:03d}",
                             "successful_repeats": sum(r.get("status") == "success" for r in rows), "bitwise_repeat_match": identical}
                    if not identical:
                        entry["failures"] = [{key: row.get(key) for key in ("exception_type", "exception")} for row in rows if row.get("status") != "success"]
                        failures.append(entry)
                    else:
                        # Maximum repeat mean includes first-step setup and all full-node inference.
                        seconds = max(sum(row["epoch_seconds"]) / len(row["epoch_seconds"]) for row in rows)
                        entry["measured_seconds_per_epoch"] = seconds
                        entry["peak_cuda_reserved_bytes"] = max(row["peak_cuda_reserved_bytes"] for row in rows)
                        entry["peak_cuda_allocated_bytes"] = max(row["peak_cuda_allocated_bytes"] for row in rows)
                        entry["four_checkpoints_bytes"] = max(row["four_checkpoints_bytes"] for row in rows)
                        model_seconds += seconds * config["training"]["max_epochs"]
                        timings.append(seconds)
                    comparisons.append(entry)
                if model_seconds * 2 > budget["formal_model_unit_wall_seconds"]:
                    failures.append({"dataset": dataset, "condition": condition, "model": model_id,
                                     "reason": "estimated four-trial unit exceeds wall cap"})
            if len(headers) == 2:
                transform_setup = max(header.get("transform_seconds", 0.0) for header in headers)
                # Residual includes process imports, hashing, checkpoint copies,
                # GC and preparation. Charge it again per future model unit;
                # this intentionally double-counts setup for a conservative plan.
                overhead = max(max(0.0, job_lookup.get(worker["worker_id"], {}).get("wall_seconds", 0.0)
                                   - worker_compute[worker["worker_id"]]) for worker in group)
                setup_estimate += (transform_setup + overhead) * 70 + h2_setup * 10
    for job in jobs:
        if job["returncode"] != 0 or job.get("stop_reason"):
            failures.append({"worker": job["worker_id"], "reason": job.get("stop_reason") or "worker failed"})
        if job.get("peak_rss_bytes", float("inf")) > budget["max_worker_rss_bytes"]:
            failures.append({"worker": job["worker_id"], "reason": "monitored RSS exceeded cap"})
    if len(jobs) != 44 or len(environments) != 1:
        failures.append({"reason": "incomplete worker coverage or mixed environments"})
    complete_timing = len(timings) == 616
    estimate = 2 * (sum(timings) * 500 * 10 + setup_estimate) if complete_timing else None
    if estimate is not None and estimate > budget["formal_total_wall_seconds"]:
        failures.append({"reason": "conservative full-run estimate exceeds total wall cap"})
    return {"schema_version": "1.0", "phase": "preflight_only", "analysis_status": config["analysis_status"],
            "run_id": config["run_id"], "config_sha256": json_hash(config),
            "expected_workers": 44, "observed_workers": len(jobs), "expected_paired_cells": 616,
            "bitwise_matching_paired_cells": sum(c["bitwise_repeat_match"] for c in comparisons),
            "validation_evaluations": 0, "test_evaluations": 0, "formal_records": 0,
            "resource_and_short_repeatability_passed": not failures,
            "formal_launch_authorized": False,
            "formal_launch_note": "Preparation only; complete formal runner/validator, committed binding and passing CI are separate launch gates.",
            "runtime_estimate": {"conservative_seconds": estimate,
                                 "setup_allowance_before_safety_factor_seconds": setup_estimate,
                                 "formula": "2 * (sum(max_repeat_mean_epoch_seconds) * 500 * 10 + sum(max_transform_setup + max_worker_nontraining_overhead) * 70 + sum(max_H2_preparation) * 10)",
                                 "limitations": "Three-step seed-0 probe, not a long-run guarantee; safety factor covers unmeasured selection/checkpoint/diagnostic overhead."},
            "limitations": ["Short repeats do not certify every seed or every training epoch.",
                            "Bitwise agreement is within this fixed runtime/hardware only.",
                            "Preflight records cannot enter accuracy, regret or calibration analyses."],
            "comparisons": comparisons, "failures": failures, "jobs": jobs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    if args.worker_id:
        manifest = json.loads((args.output_root / "preflight_manifest.json").read_text())
        verify_source_binding(manifest, config, args.binding)
        worker = next(w for w in workers(config) if w["worker_id"] == args.worker_id)
        run_worker(config, worker, args.data_root, binding, args.output_root / "workers" / worker["worker_id"])
        return
    import psutil
    if (binding.get("status") != "passed" or binding.get("dataset_count") != 11
            or binding.get("bound_split_count") != 110 or set(binding["datasets"]) != set(config["datasets"])
            or binding.get("run_id") != config["run_id"] or binding.get("analysis_status") != config["analysis_status"]
            or any(row.get("all_original_diagnostics_exact") is not True or len(row.get("full_diagnostic_checks", [])) != 10
                   for row in binding["datasets"].values())):
        raise ValueError("all eleven datasets and 110 split bindings must pass before compute probes")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError("preflight output directory must be new; preserve failed attempts")
    output.mkdir(parents=True)
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    source_commit = subprocess.check_output([*git, "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output([*git, "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError("commit and test preflight source before measurement")
    manifest = {"phase": "preflight_only", "analysis_status": config["analysis_status"], "run_id": config["run_id"],
                "source_commit": source_commit, "source_files": {p: file_hash(ROOT / p) for p in SOURCE_FILES},
                "config": config, "config_sha256": json_hash(config), "binding_sha256": file_hash(args.binding),
                "hardware": {"physical_memory_bytes": psutil.virtual_memory().total,
                             "available_memory_bytes": psutil.virtual_memory().available,
                             "logical_cpu_count": psutil.cpu_count(), "free_disk_bytes": shutil.disk_usage(output).free},
                "validation_evaluations": 0, "test_evaluations": 0, "formal_records": 0}
    write_exclusive(output / "preflight_manifest.json", manifest)
    jobs, started = [], time.perf_counter()
    budget = resolved_budget(config)
    for worker in workers(config):
        if time.perf_counter() - started > budget["preflight_total_wall_seconds"]:
            break
        if psutil.virtual_memory().available < budget["min_system_available_bytes"] or shutil.disk_usage(output).free < budget["min_disk_free_bytes"]:
            break
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
                   "--binding", str(args.binding.resolve()), "--data-root", str(args.data_root.resolve()),
                   "--output-root", str(output), "--worker-id", worker["worker_id"]]
        log_path = output / "logs" / f"{worker['worker_id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        worker_started, peak_rss, stop_reason = time.perf_counter(), 0, None
        with log_path.open("x", encoding="utf-8") as log:
            verify_source_binding(manifest, config, args.binding)
            process = subprocess.Popen(command, cwd=ROOT, env=backend_environment(), stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    memory = monitored.memory_info()
                    peak_rss = max(peak_rss, memory.rss, getattr(memory, "peak_wset", 0))
                except psutil.NoSuchProcess:
                    break
                if peak_rss > budget["max_worker_rss_bytes"]:
                    stop_reason = "worker RSS ceiling exceeded"
                elif psutil.virtual_memory().available < budget["min_system_available_bytes"]:
                    stop_reason = "system available-memory reserve crossed"
                elif time.perf_counter() - worker_started > budget["preflight_worker_wall_seconds"]:
                    stop_reason = "preflight worker wall cap exceeded"
                elif time.perf_counter() - started > budget["preflight_total_wall_seconds"]:
                    stop_reason = "preflight total wall cap exceeded"
                if stop_reason:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                time.sleep(0.25)
            returncode = process.wait()
        job = {**worker, "returncode": returncode, "stop_reason": stop_reason,
               "wall_seconds": time.perf_counter() - worker_started, "peak_rss_bytes": peak_rss}
        jobs.append(job)
        write_exclusive(output / "jobs" / f"{worker['worker_id']}.json", job)
        print(json.dumps(job), flush=True)
    result = summarize(config, output, jobs)
    try:
        verify_source_binding(manifest, config, args.binding)
    except (OSError, ValueError) as exc:
        result["failures"].append({"reason": str(exc)})
        result["resource_and_short_repeatability_passed"] = False
    result["preflight_wall_seconds"] = time.perf_counter() - started
    result["source_commit"] = source_commit
    result["binding_sha256"] = manifest["binding_sha256"]
    result["preflight_manifest_sha256"] = file_hash(output / "preflight_manifest.json")
    write_exclusive(output / "preflight_summary.json", result)
    write_exclusive(output / "preflight_inventory.json", {"phase": "preflight_only",
                    "files": {p.relative_to(output).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size}
                              for p in sorted(output.rglob("*")) if p.is_file()}})
    print(json.dumps({"preflight_passed": result["resource_and_short_repeatability_passed"],
                      "paired_cells_passed": result["bitwise_matching_paired_cells"],
                      "formal_records": 0, "summary": str(output / "preflight_summary.json")}), flush=True)
    if not result["resource_and_short_repeatability_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
