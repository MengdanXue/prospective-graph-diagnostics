"""One fixed, train-label-only CPU performance round; no formal training option.

Each worker has a fresh interpreter. Original H2GCN operations are unchanged;
only the CPU thread count and independently audited checkpoint storage vary.
All timings, failed attempts and numerical comparison evidence are retained.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.preflight_input_robustness_11 import (
    file_hash, json_hash, process_tree_memory, state_hash, tensor_hash,
    terminate_process_tree, write_exclusive,
)

DEFAULT_CONFIG = ROOT / "configs/h2gcn_performance_diagnostic_v1.json"
SOURCE_FILES = (
    "scripts/diagnose_h2gcn_performance.py",
    "scripts/h2gcn_checkpoint_diagnostic.py",
    "scripts/preflight_input_robustness_11.py",
    "scripts/input_robustness_data.py",
    "scripts/run_mlp_optimization_diagnostic.py",
    "scripts/run_preprocessing_sensitivity.py",
    "experiments/prospective_models.py",
    "experiments/prospective_data.py",
    "experiments/run_prospective_benchmark.py",
)
GIB = 1024**3
SCORE_FIELDS = {"validation_accuracy", "validation_loss", "test_accuracy", "test_loss",
                "selected_trial_id", "regret", "regret_pp", "selection_accuracy"}
EXCLUSIVE_STEP_COMPONENTS = (
    "rng_capture", "zero_grad", "train_forward", "train_loss", "backward", "gradient_finite_check",
    "optimizer_update", "unscored_eval_forward", "output_finite_check", "checkpoint_total",
    "audit_array_copy", "correctness_hash_instrumentation", "json_serialization",
    "exclusive_write_and_fsync",
)
COMPUTE_COMPONENTS = ("zero_grad", "train_forward", "train_loss", "backward",
                      "optimizer_update", "unscored_eval_forward")


def no_scores(value):
    if isinstance(value, dict):
        if SCORE_FIELDS.intersection(value):
            raise ValueError("held-out scores or formal selection found in performance evidence")
        for name in ("validation_evaluations", "test_evaluations", "formal_records"):
            if name in value and value[name] != 0:
                raise ValueError("performance evidence must have zero held-out/formal counters")
        for item in value.values():
            no_scores(item)
    elif isinstance(value, list):
        for item in value:
            no_scores(item)


def validate_config(config):
    parent = json.loads((ROOT / config["parent_config"]).read_text())
    profile_path = ROOT / config["resource_profile"]
    profile = json.loads(profile_path.read_text())
    if json_hash(parent) != config["parent_config_sha256"] or file_hash(profile_path) != config["resource_profile_sha256"]:
        raise ValueError("parent scientific configuration or v2 resource evidence changed")
    if file_hash(ROOT / config["data_binding"]) != config["data_binding_sha256"]:
        raise ValueError("input/split binding changed")
    if parent["formal_training_enabled"] is not False or config["formal_training_enabled"] is not False:
        raise ValueError("formal training must remain disabled")
    if (config["run_id"], config["phase"], config["analysis_status"], config["model"]) != (
            "h2gcn_cpu_performance_v1", "performance_diagnostic_only", "post_hoc_training_recipe_robustness", "H2GCN"):
        raise ValueError("wrong diagnostic identity")
    h2 = [row for row in profile["comparisons"] if row["model"] == "H2GCN"]
    costs = {name: sum(r["measured_seconds_per_epoch"] for r in h2 if r["dataset"] == name)
             for name in parent["datasets"]}
    chosen = sorted(costs, key=lambda name: (-costs[name], name))[:2]
    if config["selection"]["datasets"] != chosen or chosen != ["Squirrel", "Actor"]:
        raise ValueError("timing-based dataset selection changed")
    for name in chosen:
        trials = {f"trial_{index:03d}": sum(r["measured_seconds_per_epoch"] for r in h2
                  if r["dataset"] == name and r["trial_id"] == f"trial_{index:03d}") for index in range(4)}
        trial_id = sorted(trials, key=lambda item: (-trials[item], item))[0]
        if config["selection"]["trial_ids"][name] != trial_id:
            raise ValueError("timing-based trial selection changed")
        if config["selection"]["trials"][name] != parent["training"]["trials"][int(trial_id[-3:])]:
            raise ValueError("original trial configuration changed")
    expected = {"cpu_threads": [1, 4, 8], "checkpoint_modes": ["full_state_copy", "static_adjacency_once"],
                "conditions": parent["conditions"], "seed": 0, "repeats": 2, "expected_workers": 48,
                "expected_optimization_steps": 480, "expected_fresh_repeat_pairs": 24}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("fixed performance scope changed")
    if config["steps"] != {"total": 10, "cold": [0], "warmup": [1, 2], "steady": list(range(3, 10))}:
        raise ValueError("cold/warm/steady step assignment changed")
    if config["worker_order"]["cpu_thread_order"] != [4, 1, 8]:
        raise ValueError("fixed counterbalanced worker order changed")
    if config["correctness"]["rtol"] != 1e-5 or config["correctness"]["atol"] != 1e-6:
        raise ValueError("fixed numerical tolerance changed")
    if (config["qualification"]["minimum_geometric_speed_ratio"] != 1.10
            or config["qualification"]["maximum_fixture_time_ratio"] != 1.10):
        raise ValueError("fixed performance qualification changed")
    strict = {"device": "cpu", "cuda_visible_devices": "-1", "workers": 1,
              "torch_num_interop_threads": 1, "deterministic_algorithms": True, "deterministic_warn_only": False}
    if any(config["execution"].get(key) != value for key, value in strict.items()):
        raise ValueError("CPU deterministic execution contract changed")
    limits = {"total_wall_seconds": 7200, "worker_wall_seconds": 900, "max_worker_rss_gib": 8,
              "min_system_available_gib": 4, "min_disk_free_gib": 20, "max_cuda_allocated_bytes": 0}
    if any(config["resource_budget"].get(key) != value for key, value in limits.items()):
        raise ValueError("fixed diagnostic resource budget changed")
    if (config["failure_and_resume"]["rounds"] != 1 or config["failure_and_resume"]["automatic_retries"] != 0
            or config["training"]["retained_checkpoint_slots"] != 4 or config["training"]["checkpoint_prefills"] != 0
            or config["training"]["hidden_channels"] != parent["training"]["hidden_channels"]
            or config["training"]["weight_decay"] != parent["training"]["weight_decay"]):
        raise ValueError("checkpoint, trial or one-round contract changed")
    no_scores(config)
    return parent


def workers(config):
    first = [{"dataset": dataset, "condition": condition, "cpu_threads": threads, "checkpoint_mode": mode}
             for dataset in config["selection"]["datasets"] for condition in config["conditions"]
             for threads in config["worker_order"]["cpu_thread_order"] for mode in config["checkpoint_modes"]]
    result = []
    for repeat, sequence in enumerate((first, list(reversed(first)))):
        for setting in sequence:
            row = {**setting, "repeat": repeat, "trial_id": config["selection"]["trial_ids"][setting["dataset"]]}
            row["worker_id"] = f"{row['dataset']}__{row['condition']}__threads{row['cpu_threads']}__{row['checkpoint_mode']}__repeat{repeat}"
            result.append(row)
    return result


def worker_environment(threads):
    return {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            **{name: str(threads) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def configure_cpu(threads):
    if "torch" in sys.modules:
        raise RuntimeError("CPU backend must be configured in a fresh interpreter")
    expected = worker_environment(threads)
    for name in ("CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != expected[name]:
            raise RuntimeError(f"worker environment mismatch: {name}")
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must be hidden for this CPU-only diagnostic")
    return torch


def write_bytes_exclusive(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_ci_receipt(receipt, commit):
    if (not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit)
            or type(receipt.get("run_id")) is not int or receipt["run_id"] <= 0):
        raise ValueError("CI receipt requires explicit valid commit and run identity")
    if receipt.get("commit") != commit or receipt.get("status") != "completed" or receipt.get("conclusion") != "success":
        raise ValueError("CI receipt must show success for this exact committed source")
    required = {"lightweight-verification", "full-protocol-verification", "manuscript-build"}
    jobs = receipt.get("jobs", [])
    if len(jobs) != 3 or {job["name"] for job in jobs} != required:
        raise ValueError("CI receipt must contain the complete three-job run")
    if any(job.get("status") != "completed" or job.get("conclusion") != "success"
           or job.get("run_id") != receipt.get("run_id") for job in jobs):
        raise ValueError("CI jobs must all succeed in the same run")


def verify_source(manifest, config):
    if manifest.get("phase") != "performance_diagnostic_only" or manifest.get("config_sha256") != json_hash(config):
        raise ValueError("diagnostic manifest/config mismatch")
    if set(manifest["source_files"]) != set(SOURCE_FILES):
        raise ValueError("incomplete executable source binding")
    for relative, digest in manifest["source_files"].items():
        if file_hash(ROOT / relative) != digest:
            raise ValueError(f"executable source changed: {relative}")
    for field in ("data_binding", "resource_profile"):
        if file_hash(ROOT / config[field]) != config[field + "_sha256"]:
            raise ValueError(f"bound evidence changed: {field}")
    if json_hash(json.loads((ROOT / config["parent_config"]).read_text())) != config["parent_config_sha256"]:
        raise ValueError("frozen scientific configuration changed")
    validate_ci_receipt(manifest["ci_receipt"], manifest["source_commit"])


def pack_dense_evidence(model, optimizer, logits, inferred, loss, rng_before, torch):
    """Independent copies: no NumPy view may follow later parameter updates."""
    import numpy as np
    parameters, gradients, moments = [], [], []
    for name, parameter in model.named_parameters():
        parameters.append(parameter.detach().cpu().numpy().reshape(-1).copy())
        if parameter.grad is None:
            raise RuntimeError(f"missing original H2GCN gradient: {name}")
        gradients.append(parameter.grad.detach().cpu().numpy().reshape(-1).copy())
        state = optimizer.state[parameter]
        for key in ("step", "exp_avg", "exp_avg_sq"):
            value = state[key]
            moments.append(value.detach().cpu().numpy().reshape(-1).copy())
    result = {"parameters": np.concatenate(parameters), "gradients": np.concatenate(gradients),
            "optimizer_state": np.concatenate(moments),
            "train_logits": logits.detach().cpu().numpy().copy(),
            "unscored_logits": inferred.detach().cpu().numpy().copy(),
            "train_loss": loss.detach().cpu().numpy().copy(),
            "rng_before": rng_before.cpu().numpy().copy(),
            "rng_after": torch.get_rng_state().cpu().numpy().copy()}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("nonfinite dense correctness evidence")
    return result


def model_fingerprints(model):
    """Hash each state tensor once; separately time the static-buffer scan."""
    from scripts.h2gcn_checkpoint_diagnostic import BUFFER_NAMES
    state, dense_seconds, static_seconds = {}, 0.0, 0.0
    for name, value in sorted(model.state_dict().items()):
        start = time.perf_counter()
        state[name] = tensor_hash(value)
        duration = time.perf_counter() - start
        if name in BUFFER_NAMES:
            static_seconds += duration
        else:
            dense_seconds += duration
    return ({"full_state": json_hash(state), "static_adjacency": json_hash({name: state[name] for name in BUFFER_NAMES})},
            {"static_buffer_hash": static_seconds, "dense_state_hash": dense_seconds})


def measure_trajectory(*, model, x, edge_index, train_indices, train_labels, trial, weight_decay,
                       steps, checkpoint_store, destination, identity, torch):
    """Only train labels are accepted. No selection, validation or test scores."""
    import numpy as np
    import torch.nn.functional as F
    if any(value.is_cuda for value in (x, edge_index, train_indices, train_labels)):
        raise ValueError("CPU-only performance diagnostic")
    setup_started = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=trial["learning_rate"], weight_decay=weight_decay)
    adam_setup = time.perf_counter() - setup_started
    initial_started = time.perf_counter()
    initial_model, initial_hash_parts = model_fingerprints(model)
    initial = {**initial_model, "rng": tensor_hash(torch.get_rng_state()),
               "store_static_fingerprint": checkpoint_store.static_fingerprint}
    initial_hash_seconds = time.perf_counter() - initial_started
    records, arrays, checkpoint_expected = [], {}, {}
    for step in range(steps):
        timing = {}
        step_started = time.perf_counter()
        start = time.perf_counter()
        rng_before = torch.get_rng_state().clone()
        timing["rng_capture"] = time.perf_counter() - start
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        timing["zero_grad"] = time.perf_counter() - start
        start = time.perf_counter()
        logits = model(x, edge_index)
        timing["train_forward"] = time.perf_counter() - start
        start = time.perf_counter()
        loss = F.cross_entropy(logits[train_indices], train_labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        timing["train_loss"] = time.perf_counter() - start
        start = time.perf_counter()
        loss.backward()
        timing["backward"] = time.perf_counter() - start
        start = time.perf_counter()
        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise FloatingPointError("missing or nonfinite gradient")
        timing["gradient_finite_check"] = time.perf_counter() - start
        start = time.perf_counter()
        optimizer.step()
        timing["optimizer_update"] = time.perf_counter() - start
        start = time.perf_counter()
        model.eval()
        with torch.no_grad():
            inferred = model(x, edge_index)
        timing["unscored_eval_forward"] = time.perf_counter() - start
        start = time.perf_counter()
        if not torch.isfinite(inferred).all() or any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("nonfinite unscored output or parameter")
        timing["output_finite_check"] = time.perf_counter() - start
        start = time.perf_counter()
        copy_cost = checkpoint_store.capture(model, step % 4)
        timing["checkpoint_total"] = time.perf_counter() - start
        timing["checkpoint_parameter_copy"] = copy_cost["parameter_copy_seconds"]
        timing["checkpoint_static_copy"] = copy_cost["static_buffer_copy_seconds"]
        start = time.perf_counter()
        dense = pack_dense_evidence(model, optimizer, logits, inferred, loss, rng_before, torch)
        arrays.update({f"step{step:03d}.{key}": value for key, value in dense.items()})
        timing["audit_array_copy"] = time.perf_counter() - start
        start = time.perf_counter()
        # Scan the complete original state equally in both modes. This expensive
        # diagnostic operation is reported separately from checkpoint copying.
        model_hashes, hash_parts = model_fingerprints(model)
        if model_hashes["static_adjacency"] != initial["static_adjacency"]:
            raise ValueError("static adjacency content changed during the trajectory")
        fingerprint = {**model_hashes,
                       "arrays": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in dense.items()}}
        timing["correctness_hash_instrumentation"] = time.perf_counter() - start
        timing.update(hash_parts)
        checkpoint_expected[step % 4] = (fingerprint["full_state"], dense["unscored_logits"])
        row = {**identity, "step": step, "fingerprint": fingerprint, "timings": dict(timing),
               "retained_slots": min(step + 1, 4), "copy_metrics": copy_cost,
               "timing_completion_note": "serialization/write/total times are retained in worker_summary.json"}
        start = time.perf_counter()
        serialized = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
        timing["json_serialization"] = time.perf_counter() - start
        start = time.perf_counter()
        write_bytes_exclusive(destination / "steps" / f"step_{step:03d}.json", serialized)
        timing["exclusive_write_and_fsync"] = time.perf_counter() - start
        timing["end_to_end"] = time.perf_counter() - step_started
        timing["compute_only"] = sum(timing[name] for name in COMPUTE_COMPONENTS)
        timing["accounted_exclusive"] = sum(timing[name] for name in EXCLUSIVE_STEP_COMPONENTS)
        timing["unattributed"] = timing["end_to_end"] - timing["accounted_exclusive"]
        row["timings"] = timing
        records.append(row)
        del logits, inferred, loss, dense, rng_before
    final_rng = tensor_hash(torch.get_rng_state())
    start = time.perf_counter()
    checkpoint_store.check_static(model)
    static_validation_seconds = time.perf_counter() - start
    storage = checkpoint_store.storage_metrics()
    start = time.perf_counter()
    restoration = []
    for slot, (expected_hash, expected_logits) in sorted(checkpoint_expected.items()):
        checkpoint_store.restore(model, slot)
        model.eval()
        with torch.no_grad():
            restored_logits = model(x, edge_index).detach().cpu().numpy()
        exact = state_hash(model) == expected_hash and np.array_equal(restored_logits, expected_logits)
        restoration.append({"slot": slot, "complete_state_and_unscored_output_exact": exact})
        if not exact:
            raise ValueError("checkpoint restoration changed full model state or output")
    if final_rng != tensor_hash(torch.get_rng_state()):
        raise ValueError("post-trajectory checkpoint audit changed the CPU RNG state")
    restoration_seconds = time.perf_counter() - start
    start = time.perf_counter()
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    audit_serialization_seconds = time.perf_counter() - start
    start = time.perf_counter()
    write_bytes_exclusive(destination / "numeric_evidence.npz", buffer.getvalue())
    audit_write_seconds = time.perf_counter() - start
    start = time.perf_counter()
    numeric_digest = file_hash(destination / "numeric_evidence.npz")
    numeric_hash_seconds = time.perf_counter() - start
    metadata = {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()}
    return {"initial_fingerprint": initial, "steps": records, "restoration": restoration,
            "checkpoint_storage": storage, "adam_setup_seconds": adam_setup,
            "initial_full_state_hash_seconds": initial_hash_seconds,
            "initial_full_state_hash_parts": initial_hash_parts,
            "final_static_validation_seconds": static_validation_seconds,
            "checkpoint_restore_validation_seconds": restoration_seconds,
            "cross_thread_audit_serialization_seconds": audit_serialization_seconds,
            "cross_thread_audit_write_seconds": audit_write_seconds,
            "numeric_evidence_sha256": numeric_digest, "numeric_evidence_hash_seconds": numeric_hash_seconds,
            "numeric_evidence_arrays": metadata,
            "final_rng": final_rng, "cpu_rng_unchanged_by_restoration": True}


def run_worker(config, worker, data_root, destination):
    torch = configure_cpu(worker["cpu_threads"])
    import psutil
    from experiments.prospective_models import build_model, prepare_h2_adjacencies
    from experiments.run_prospective_benchmark import environment_snapshot, seed_everything
    from scripts.input_robustness_data import load_bound_dataset, fit_transform_features_bounded, array_sha256
    from scripts.h2gcn_checkpoint_diagnostic import CheckpointStore
    started, setup = time.perf_counter(), {}
    identity = {**worker, "phase": config["phase"], "run_id": config["run_id"],
                "analysis_status": config["analysis_status"], "config_sha256": json_hash(config),
                "seed": config["seed"], "validation_evaluations": 0, "test_evaluations": 0, "formal_records": 0}
    binding = json.loads((ROOT / config["data_binding"]).read_text())
    start = time.perf_counter()
    raw_x, labels, edges, splits, metadata = load_bound_dataset(worker["dataset"], data_root, binding)
    split, split_id = splits[config["seed"]]
    train_indices = torch.as_tensor(split["train"], dtype=torch.long)
    train_labels = labels[train_indices].clone()
    classes = int(train_labels.max()) + 1
    del labels, splits, split
    setup["input_load_and_binding"] = time.perf_counter() - start
    start = time.perf_counter()
    x, fitted = fit_transform_features_bounded(raw_x, train_indices, worker["condition"])
    setup["feature_transform"] = time.perf_counter() - start
    start = time.perf_counter()
    candidate = next(row for row in metadata["candidate_checks"] if row["seed"] == config["seed"])
    expected = metadata["normalize_features_sha256"] if worker["condition"] == "normalize_features" else candidate["transformed_feature_sha256"]
    if array_sha256(x) != expected or fitted != {**candidate["fit_metadata"], "condition": worker["condition"]}:
        raise ValueError("transformed input/statistics changed from sealed binding")
    if not torch.isfinite(x).all():
        raise FloatingPointError("nonfinite input")
    setup["transform_binding_verification"] = time.perf_counter() - start
    del raw_x, metadata, fitted, binding
    gc.collect()
    start = time.perf_counter()
    adjacency = prepare_h2_adjacencies(edges, num_nodes=x.shape[0])
    setup["adjacency_preparation"] = time.perf_counter() - start
    seed_everything(config["seed"])
    start = time.perf_counter()
    model = build_model("H2GCN", num_nodes=x.shape[0], in_channels=x.shape[1],
                        hidden_channels=config["training"]["hidden_channels"], out_channels=classes,
                        dropout=config["selection"]["trials"][worker["dataset"]]["dropout"],
                        edge_index=edges, h2_adjacencies=adjacency).to("cpu")
    setup["model_setup"] = time.perf_counter() - start
    start = time.perf_counter()
    store = CheckpointStore(model, worker["checkpoint_mode"])
    setup["checkpoint_store_setup"] = time.perf_counter() - start
    setup["static_adjacency_snapshot"] = store.initial_static_copy_seconds
    setup["initial_static_hash"] = store.initial_static_hash_seconds
    setup["initial_static_boundary_validation"] = store.initial_boundary_validation_seconds
    setup["initial_checkpoint_schema_validation"] = store.initial_schema_validation_seconds
    environment = {**environment_snapshot(torch.device("cpu")), "psutil": psutil.__version__,
                   "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                   "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                   "cudnn_benchmark": torch.backends.cudnn.benchmark,
                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                   "float32_matmul_precision": torch.get_float32_matmul_precision(),
                   "cuda_initialized": torch.cuda.is_initialized(),
                   "thread_environment": {key: os.environ[key] for key in (
                       "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES")}}
    header = {**identity, "environment": environment, "split_id": split_id, "transformed_feature_sha256": expected,
              "trial": config["selection"]["trials"][worker["dataset"]], "setup_timings": setup,
              "dense_parameter_count": sum(p.numel() for p in model.parameters()),
              "parameter_tensor_count": len(list(model.parameters())), "unscored_logits_shape": [int(x.shape[0]), classes]}
    write_exclusive(destination / "worker.json", header)
    result = measure_trajectory(model=model, x=x, edge_index=edges, train_indices=train_indices,
                                train_labels=train_labels, trial=header["trial"],
                                weight_decay=config["training"]["weight_decay"], steps=config["steps"]["total"],
                                checkpoint_store=store, destination=destination, identity=identity, torch=torch)
    summary = {**header, **result, "status": "success", "optimization_steps": config["steps"]["total"],
               "worker_measured_seconds_before_final_write": time.perf_counter() - started,
               "peak_worker_rss_bytes": max(psutil.Process().memory_info().rss,
                                            getattr(psutil.Process().memory_info(), "peak_wset", 0)),
               "cuda_allocated_bytes": 0, "cuda_initialized_after_probe": torch.cuda.is_initialized()}
    summary["accounted_nonoverlapping_worker_seconds"] = (
        sum(setup[name] for name in ("input_load_and_binding", "feature_transform", "transform_binding_verification",
                                    "adjacency_preparation", "model_setup", "checkpoint_store_setup"))
        + sum(result[name] for name in ("adam_setup_seconds", "initial_full_state_hash_seconds",
                  "final_static_validation_seconds", "checkpoint_restore_validation_seconds",
                  "cross_thread_audit_serialization_seconds", "cross_thread_audit_write_seconds", "numeric_evidence_hash_seconds"))
        + sum(row["timings"]["end_to_end"] for row in result["steps"]))
    summary["unattributed_worker_seconds_before_final_write"] = (
        summary["worker_measured_seconds_before_final_write"] - summary["accounted_nonoverlapping_worker_seconds"])
    no_scores(summary)
    start = time.perf_counter()
    write_exclusive(destination / "worker_summary.json", summary)
    final_write_seconds = time.perf_counter() - start
    write_exclusive(destination / "worker_complete.json", {**identity, "status": "success",
                    "optimization_steps": config["steps"]["total"], "final_record_write_and_fsync_seconds": final_write_seconds,
                    "worker_summary_sha256": file_hash(destination / "worker_summary.json"),
                    "worker_total_seconds": time.perf_counter() - started})
    print(json.dumps({"worker_id": worker["worker_id"], "status": "success"}), flush=True)


def compare_numeric_files(left_path, right_path, *, rtol, atol, exact):
    import numpy as np
    mismatch, maximum_abs, maximum_scaled = [], 0.0, 0.0
    with np.load(left_path, allow_pickle=False) as left, np.load(right_path, allow_pickle=False) as right:
        if not left.files or set(left.files) != set(right.files):
            return {"passed": False, "reason": "numeric evidence key mismatch"}
        for name in sorted(left.files):
            a, b = left[name], right[name]
            if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
                mismatch.append(name)
                continue
            exact_required = exact or "rng_" in name
            passed = np.array_equal(a, b) if exact_required else np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
            if not passed:
                mismatch.append(name)
            if a.size and a.dtype.kind == "f":
                difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
                maximum_abs = max(maximum_abs, float(difference.max()))
                maximum_scaled = max(maximum_scaled, float((difference / (atol + rtol * np.abs(b))).max()))
    return {"passed": not mismatch, "mismatched_arrays": mismatch, "maximum_absolute_difference": maximum_abs,
            "maximum_tolerance_fraction": maximum_scaled, "comparison": "exact" if exact else "fixed_allclose_with_exact_rng"}


def validate_numeric_evidence(row, path):
    import numpy as np
    n, p = row["dense_parameter_count"], row["parameter_tensor_count"]
    shape = tuple(row["unscored_logits_shape"])
    expected_shapes = {"parameters": (n,), "gradients": (n,), "optimizer_state": (2 * n + p,),
                       "train_logits": shape, "unscored_logits": shape, "train_loss": ()}
    types = {**{name: "float32" for name in expected_shapes}, "rng_before": "uint8", "rng_after": "uint8"}
    expected = {f"step{step:03d}.{name}" for step in range(10) for name in types}
    with np.load(path, allow_pickle=False) as evidence:
        if set(evidence.files) != expected or set(row["numeric_evidence_arrays"]) != expected:
            raise ValueError("missing, duplicate or unexpected numeric evidence arrays")
        if len(evidence.files) != len(expected):
            raise ValueError("duplicate numeric evidence entries")
        for key in sorted(expected):
            step_name, name = key.split(".")
            step = int(step_name[4:])
            value = evidence[key]
            metadata = row["numeric_evidence_arrays"][key]
            if (str(value.dtype) != types[name] or list(value.shape) != metadata["shape"]
                    or str(value.dtype) != metadata["dtype"] or not np.isfinite(value).all()):
                raise ValueError("invalid numeric evidence shape/type/value metadata")
            if name in expected_shapes and value.shape != expected_shapes[name]:
                raise ValueError("numeric evidence differs from original model tensor scope")
            if name.startswith("rng_") and (value.ndim != 1 or value.size < 1):
                raise ValueError("missing CPU RNG state")
            if hashlib.sha256(value.tobytes()).hexdigest() != row["steps"][step]["fingerprint"]["arrays"][name]:
                raise ValueError("numeric evidence and step fingerprints differ")


def qualify_settings(config, fixtures, invalid_settings, complete):
    """One global CPU/storage choice; selection uses no outcome or score."""
    candidates = []
    for threads, mode in itertools.product(config["cpu_threads"], config["checkpoint_modes"]):
        setting = (threads, mode)
        ratios, peak_rss = [], 0
        for dataset, condition in itertools.product(config["selection"]["datasets"], config["conditions"]):
            reference = fixtures.get((dataset, condition, 4, "full_state_copy"))
            candidate = fixtures.get((dataset, condition, threads, mode))
            if reference is None or candidate is None:
                continue
            ratios.append(candidate["slower_repeat_steady_median_seconds"] / reference["slower_repeat_steady_median_seconds"])
            peak_rss = max(peak_rss, candidate["maximum_process_tree_rss_bytes"])
        speed = math.exp(sum(math.log(1 / value) for value in ratios) / len(ratios)) if ratios else None
        eligible = (complete and len(ratios) == 4 and setting not in invalid_settings
                    and (4, "full_state_copy") not in invalid_settings
                    and speed >= config["qualification"]["minimum_geometric_speed_ratio"]
                    and max(ratios) <= config["qualification"]["maximum_fixture_time_ratio"])
        candidates.append({"cpu_threads": threads, "checkpoint_mode": mode, "qualifies": eligible,
                           "geometric_speed_ratio": speed, "fixture_time_ratios": ratios,
                           "maximum_process_tree_rss_bytes": peak_rss,
                           "correctness_or_resource_rejected": setting in invalid_settings})
    passing = sorted([row for row in candidates if row["qualifies"]], key=lambda row: (
        -row["geometric_speed_ratio"], row["maximum_process_tree_rss_bytes"], row["cpu_threads"], row["checkpoint_mode"]))
    return {"settings": candidates, "recommended_setting": passing[0] if passing else None,
            "recommendation": "adopt_verified_optimization_in_measured_scope" if passing else "retain_current_implementation_and_revise_budget",
            "formal_use_authorized": False, "budget_change_authorized": False,
            "scope_limit": "Two timing-selected fixtures; no full-eleven speedup extrapolation. User budget confirmation and later complete acceptance remain required."}


def summarize(config, output, jobs):
    expected = workers(config)
    failures, comparisons, summaries, invalid = [], [], {}, set()
    job_map = {job["worker_id"]: job for job in jobs}
    if len(jobs) != 48 or set(job_map) != {row["worker_id"] for row in expected}:
        failures.append({"reason": "incomplete or duplicate worker scope"})
    for worker in expected:
        worker_id = worker["worker_id"]
        location = output / "workers" / worker_id
        job = job_map.get(worker_id)
        try:
            row = json.loads((location / "worker_summary.json").read_text())
            complete = json.loads((location / "worker_complete.json").read_text())
            no_scores(row)
            no_scores(complete)
            identity = {**worker, "phase": config["phase"], "run_id": config["run_id"], "config_sha256": json_hash(config),
                        "analysis_status": config["analysis_status"], "seed": config["seed"],
                        "validation_evaluations": 0, "test_evaluations": 0, "formal_records": 0,
                        "status": "success", "optimization_steps": 10}
            if any(row.get(key) != value or complete.get(key) != value for key, value in identity.items()):
                raise ValueError("worker/completion identity mismatch")
            if file_hash(location / "worker_summary.json") != complete["worker_summary_sha256"]:
                raise ValueError("worker summary changed after completion")
            if file_hash(location / "numeric_evidence.npz") != row["numeric_evidence_sha256"]:
                raise ValueError("numeric audit evidence changed")
            if len(row["steps"]) != 10 or [item["step"] for item in row["steps"]] != list(range(10)):
                raise ValueError("incomplete measured trajectory")
            if len(list((location / "steps").glob("*.json"))) != 10:
                raise ValueError("incomplete step record files")
            validate_numeric_evidence(row, location / "numeric_evidence.npz")
            for index, item in enumerate(row["steps"]):
                saved = json.loads((location / "steps" / f"step_{index:03d}.json").read_text())
                no_scores(saved)
                if (any(value != item.get(key) for key, value in saved.items() if key != "timings")
                        or any(value != item["timings"].get(key) for key, value in saved["timings"].items())
                        or saved["worker_id"] != worker_id
                        or any(item.get(key) != value for key, value in identity.items() if key not in ("status", "optimization_steps"))
                        or item["fingerprint"]["static_adjacency"] != row["initial_fingerprint"]["static_adjacency"]
                        or item["retained_slots"] != min(index + 1, 4)
                        or item["copy_metrics"]["slot"] != index % 4
                        or item["copy_metrics"]["retained_checkpoint_count"] != min(index + 1, 4)):
                    raise ValueError("step record and summary differ")
                timings = item["timings"]
                if (any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                        for value in timings.values()) or timings["end_to_end"] <= 0):
                    raise ValueError("invalid or unreconciled component timing")
                if not math.isclose(sum(timings[name] for name in EXCLUSIVE_STEP_COMPONENTS), timings["accounted_exclusive"], rel_tol=1e-12):
                    raise ValueError("component timing does not reconcile")
                if (not math.isclose(timings["end_to_end"], timings["accounted_exclusive"] + timings["unattributed"], rel_tol=1e-12)
                        or not math.isclose(timings["compute_only"], sum(timings[name] for name in COMPUTE_COMPONENTS), rel_tol=1e-12)
                        or timings["checkpoint_parameter_copy"] + timings["checkpoint_static_copy"] > timings["checkpoint_total"]
                        or timings["static_buffer_hash"] + timings["dense_state_hash"] > timings["correctness_hash_instrumentation"]):
                    raise ValueError("total, nested or compute-only timing does not reconcile")
            expected_environment = {key: worker_environment(worker["cpu_threads"])[key] for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES")}
            if (row["cuda_allocated_bytes"] != 0 or row["cuda_initialized_after_probe"]
                    or row["environment"]["cuda_initialized"] or row["environment"]["cuda_available"]
                    or row["environment"]["torch_threads"] != worker["cpu_threads"]
                    or row["environment"]["interop_threads"] != 1
                    or row["environment"]["thread_environment"] != expected_environment
                    or row["environment"]["cuda_matmul_allow_tf32"] or row["environment"]["cudnn_allow_tf32"]
                    or row["environment"]["cudnn_benchmark"] or not row["environment"]["cudnn_deterministic"]
                    or row["environment"]["float32_matmul_precision"] != "highest"
                    or not row["environment"]["deterministic_algorithms"] or row["environment"]["deterministic_warn_only"]
                    or len(row["restoration"]) != 4 or {item["slot"] for item in row["restoration"]} != {0, 1, 2, 3}
                    or row["checkpoint_storage"]["slots"] != [0, 1, 2, 3]
                    or row["checkpoint_storage"]["static_buffer_replicas"] != (4 if worker["checkpoint_mode"] == "full_state_copy" else 1)
                    or not all(item["complete_state_and_unscored_output_exact"] for item in row["restoration"])):
                raise ValueError("backend or checkpoint-restoration contract failed")
            if (not job or job["returncode"] != 0 or job.get("stop_reason")
                    or job.get("memory_scope") != "owned_worker_process_tree_sum_including_windows_high_water"
                    or not job.get("observed_process_ids")
                    or job["peak_rss_bytes"] > config["resource_budget"]["max_worker_rss_gib"] * GIB
                    or row["peak_worker_rss_bytes"] > config["resource_budget"]["max_worker_rss_gib"] * GIB
                    or job["wall_seconds"] > config["resource_budget"]["worker_wall_seconds"]
                    or row["unattributed_worker_seconds_before_final_write"] < 0):
                raise ValueError("worker execution/resource gate failed")
            row["coordinator_job_wall_seconds"] = job["wall_seconds"]
            row["final_record_write_and_fsync_seconds"] = complete["final_record_write_and_fsync_seconds"]
            row["launch_import_completion_and_coordinator_seconds"] = max(0.0, job["wall_seconds"] - complete["worker_total_seconds"])
            summaries[worker_id] = row
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            failures.append({"worker_id": worker_id, "reason": str(exc)})
            invalid.add((worker["cpu_threads"], worker["checkpoint_mode"]))
    lookup = {(row["dataset"], row["condition"], row["cpu_threads"], row["checkpoint_mode"], row["repeat"]): row for row in expected}

    def compare(left_worker, right_worker, kind, exact):
        left_id, right_id = left_worker["worker_id"], right_worker["worker_id"]
        left, right = summaries.get(left_id), summaries.get(right_id)
        result = {"kind": kind, "left": left_id, "right": right_id, "passed": False}
        if left is not None and right is not None:
            result.update(compare_numeric_files(output / "workers" / left_id / "numeric_evidence.npz",
                                                output / "workers" / right_id / "numeric_evidence.npz",
                                                rtol=config["correctness"]["rtol"], atol=config["correctness"]["atol"], exact=exact))
            invariant = all(left.get(key) == right.get(key) for key in ("split_id", "transformed_feature_sha256", "trial"))
            invariant = invariant and left["initial_fingerprint"] == right["initial_fingerprint"]
            left_environment = {key: value for key, value in left["environment"].items() if key not in ("torch_threads", "thread_environment")}
            right_environment = {key: value for key, value in right["environment"].items() if key not in ("torch_threads", "thread_environment")}
            invariant = invariant and left_environment == right_environment
            if exact:
                invariant = invariant and [s["fingerprint"] for s in left["steps"]] == [s["fingerprint"] for s in right["steps"]]
            result["passed"] = result["passed"] and invariant
            if not invariant:
                result["invariant_failure"] = "initial state, input, RNG or exact trajectory mismatch"
        comparisons.append(result)
        if not result["passed"]:
            failures.append({"reason": "correctness comparison failed", **result})
            invalid.add((left_worker["cpu_threads"], left_worker["checkpoint_mode"]))
            if kind != "cross_thread":
                invalid.add((right_worker["cpu_threads"], right_worker["checkpoint_mode"]))

    for dataset, condition in itertools.product(config["selection"]["datasets"], config["conditions"]):
        for threads, mode in itertools.product(config["cpu_threads"], config["checkpoint_modes"]):
            compare(lookup[dataset, condition, threads, mode, 0], lookup[dataset, condition, threads, mode, 1], "fresh_repeat", True)
        for threads, repeat in itertools.product(config["cpu_threads"], range(2)):
            compare(lookup[dataset, condition, threads, "static_adjacency_once", repeat],
                    lookup[dataset, condition, threads, "full_state_copy", repeat], "checkpoint_mode", True)
        for threads, mode, repeat in itertools.product((1, 8), config["checkpoint_modes"], range(2)):
            compare(lookup[dataset, condition, threads, mode, repeat], lookup[dataset, condition, 4, mode, repeat], "cross_thread", False)
    fixtures = {}
    for dataset, condition, threads, mode in itertools.product(config["selection"]["datasets"], config["conditions"], config["cpu_threads"], config["checkpoint_modes"]):
        group = [summaries.get(lookup[dataset, condition, threads, mode, repeat]["worker_id"]) for repeat in range(2)]
        if any(row is None for row in group):
            continue
        summaries_by_repeat = []
        for row in group:
            parts = {phase: {name: statistics.median(row["steps"][step]["timings"][name] for step in config["steps"][phase])
                             for name in row["steps"][0]["timings"]} for phase in ("cold", "warmup", "steady")}
            summaries_by_repeat.append({"repeat": row["repeat"], "phase_medians": parts,
                                        "setup_timings": row["setup_timings"], "adam_setup_seconds": row["adam_setup_seconds"],
                                        "initial_full_state_hash_seconds": row["initial_full_state_hash_seconds"],
                                        "final_static_validation_seconds": row["final_static_validation_seconds"],
                                        "checkpoint_restore_validation_seconds": row["checkpoint_restore_validation_seconds"],
                                        "cross_thread_audit_serialization_seconds": row["cross_thread_audit_serialization_seconds"],
                                        "cross_thread_audit_write_seconds": row["cross_thread_audit_write_seconds"],
                                        "numeric_evidence_hash_seconds": row["numeric_evidence_hash_seconds"],
                                        "coordinator_job_wall_seconds": row["coordinator_job_wall_seconds"],
                                        "accounted_nonoverlapping_worker_seconds": row["accounted_nonoverlapping_worker_seconds"],
                                        "unattributed_worker_seconds_before_final_write": row["unattributed_worker_seconds_before_final_write"],
                                        "final_record_write_and_fsync_seconds": row["final_record_write_and_fsync_seconds"],
                                        "launch_import_completion_and_coordinator_seconds": row["launch_import_completion_and_coordinator_seconds"],
                                        "checkpoint_storage": row["checkpoint_storage"]})
        fixtures[dataset, condition, threads, mode] = {"dataset": dataset, "condition": condition, "cpu_threads": threads,
                "checkpoint_mode": mode, "repeats": summaries_by_repeat,
                "slower_repeat_steady_median_seconds": max(row["phase_medians"]["steady"]["end_to_end"] for row in summaries_by_repeat),
                "maximum_process_tree_rss_bytes": max(job_map[row["worker_id"]]["peak_rss_bytes"] for row in group)}
    scope_complete = len(summaries) == 48 and len(jobs) == 48 and len(job_map) == 48
    recommendation = qualify_settings(config, fixtures, invalid, scope_complete)
    return {"schema_version": "1.0", "run_id": config["run_id"], "phase": config["phase"],
            "analysis_status": config["analysis_status"], "config_sha256": json_hash(config),
            "expected_workers": 48, "observed_jobs": len(jobs), "successful_workers": len(summaries),
            "observed_optimization_steps": sum(len(row["steps"]) for row in summaries.values()),
            "fixed_scope_complete": scope_complete, "comparisons": comparisons,
            "fresh_repeat_pairs_passed": sum(row["passed"] for row in comparisons if row["kind"] == "fresh_repeat"),
            "checkpoint_mode_pairs_passed": sum(row["passed"] for row in comparisons if row["kind"] == "checkpoint_mode"),
            "cross_thread_pairs_passed": sum(row["passed"] for row in comparisons if row["kind"] == "cross_thread"),
            "fixtures": list(fixtures.values()), "qualification": recommendation, "jobs": jobs, "failures": failures,
            "formal_training_enabled": False, "formal_records": 0, "validation_evaluations": 0, "test_evaluations": 0,
            "budget_proposal_requires_user_confirmation": True,
            "limitations": ["One timing-ranked trial on two datasets; no full-grid or all-eleven performance guarantee.",
                            "Ten steps at seed zero; fixed allclose is numerical evidence, not identity of arbitrary long training.",
                            "Each step copies a checkpoint and hashes full buffers; this stress/audit load differs from score-triggered production saves.",
                            "Old v2 resource and scientific configurations are unchanged; no formal launch is authorized."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ci-receipt", type=Path, required=True)
    parser.add_argument("--worker-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    output = args.output_root.resolve()
    if args.worker_id:
        manifest = json.loads((output / "diagnostic_manifest.json").read_text())
        verify_source(manifest, config)
        worker = next(row for row in workers(config) if row["worker_id"] == args.worker_id)
        destination = output / "workers" / worker["worker_id"]
        try:
            run_worker(config, worker, args.data_root, destination)
        except Exception as exc:
            write_exclusive(destination / "worker_failure.json", {**worker, "phase": config["phase"], "run_id": config["run_id"],
                            "validation_evaluations": 0, "test_evaluations": 0, "formal_records": 0,
                            "exception_type": type(exc).__name__, "exception": str(exc),
                            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))})
            raise
        return
    import psutil
    if output.exists():
        raise FileExistsError("preserve all prior attempts; diagnostic output must be new")
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output([*git, "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("commit and synchronize the complete diagnostic source before measurement")
    receipt = json.loads(args.ci_receipt.read_text())
    validate_ci_receipt(receipt, commit)
    output.mkdir(parents=True)
    manifest = {"phase": config["phase"], "run_id": config["run_id"], "analysis_status": config["analysis_status"],
                "source_commit": commit, "source_files": {name: file_hash(ROOT / name) for name in SOURCE_FILES},
                "config": config, "config_sha256": json_hash(config), "ci_receipt": receipt,
                "ci_receipt_sha256": file_hash(args.ci_receipt),
                "hardware": {"cpu_logical_count": psutil.cpu_count(), "cpu_physical_count": psutil.cpu_count(logical=False),
                             "memory_bytes": psutil.virtual_memory().total, "available_bytes": psutil.virtual_memory().available,
                             "disk_free_bytes": shutil.disk_usage(output).free},
                "validation_evaluations": 0, "test_evaluations": 0, "formal_records": 0}
    write_exclusive(output / "diagnostic_manifest.json", manifest)
    budget, jobs, started, coordinator_stop = config["resource_budget"], [], time.perf_counter(), None
    for worker in workers(config):
        if time.perf_counter() - started > budget["total_wall_seconds"]:
            coordinator_stop = "diagnostic total wall cap exceeded"
            break
        if psutil.virtual_memory().available < budget["min_system_available_gib"] * GIB or shutil.disk_usage(output).free < budget["min_disk_free_gib"] * GIB:
            coordinator_stop = "system memory or disk reserve crossed"
            break
        verify_source(manifest, config)
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
                   "--data-root", str(args.data_root.resolve()), "--output-root", str(output),
                   "--ci-receipt", str(args.ci_receipt.resolve()), "--worker-id", worker["worker_id"]]
        log_path = output / "logs" / f"{worker['worker_id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        worker_started, peak_rss, stop_reason, pids = time.perf_counter(), 0, None, set()
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=worker_environment(worker["cpu_threads"]),
                                       stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                memory = process_tree_memory(monitored, psutil)
                peak_rss = max(peak_rss, memory["rss_bytes"], memory["high_water_bytes"])
                pids.update(memory["process_ids"])
                if peak_rss > budget["max_worker_rss_gib"] * GIB:
                    stop_reason = "worker process-tree RSS ceiling exceeded"
                elif psutil.virtual_memory().available < budget["min_system_available_gib"] * GIB:
                    stop_reason = "system available-memory reserve crossed"
                elif shutil.disk_usage(output).free < budget["min_disk_free_gib"] * GIB:
                    stop_reason = "free disk reserve crossed"
                elif time.perf_counter() - worker_started > budget["worker_wall_seconds"]:
                    stop_reason = "worker wall cap exceeded"
                elif time.perf_counter() - started > budget["total_wall_seconds"]:
                    stop_reason = "diagnostic total wall cap exceeded"
                if stop_reason:
                    terminate_process_tree(process, psutil)
                    break
                time.sleep(0.25)
            returncode = process.wait()
        job = {**worker, "returncode": returncode, "stop_reason": stop_reason,
               "wall_seconds": time.perf_counter() - worker_started, "peak_rss_bytes": peak_rss,
               "memory_scope": "owned_worker_process_tree_sum_including_windows_high_water",
               "observed_process_ids": sorted(pids)}
        jobs.append(job)
        write_exclusive(output / "jobs" / f"{worker['worker_id']}.json", job)
        print(json.dumps(job), flush=True)
        if stop_reason:
            coordinator_stop = stop_reason
            break
    result = summarize(config, output, jobs)
    result.update(source_commit=commit, diagnostic_wall_seconds=time.perf_counter() - started,
                  diagnostic_manifest_sha256=file_hash(output / "diagnostic_manifest.json"),
                  coordinator_stop_reason=coordinator_stop)
    verify_source(manifest, config)
    if result["diagnostic_wall_seconds"] > budget["total_wall_seconds"] or coordinator_stop:
        result["failures"].append({"reason": coordinator_stop or "diagnostic total wall cap exceeded"})
        result["qualification"] = qualify_settings(config, {}, set(), False)
    no_scores(result)
    write_exclusive(output / "diagnostic_summary.json", result)
    write_exclusive(output / "diagnostic_inventory.json", {"phase": config["phase"], "run_id": config["run_id"],
                    "files": {path.relative_to(output).as_posix(): {"sha256": file_hash(path), "bytes": path.stat().st_size}
                              for path in sorted(output.rglob("*")) if path.is_file()}})
    print(json.dumps({"fixed_scope_complete": result["fixed_scope_complete"], "successful_workers": result["successful_workers"],
                      "recommendation": result["qualification"]["recommendation"], "failures": len(result["failures"]),
                      "formal_records": 0}), flush=True)
    if result["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
