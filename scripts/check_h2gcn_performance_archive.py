"""Independent post-completion evidence audit; never imports training code.

Run only after the coordinator has exited and the owner confirms stability.
All outputs are exclusive and outside the measured evidence directory.
Sparse tensors/checkpoints are not stored in this archive: their recorded
fingerprint and restoration consistency can be checked, not rerun here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess
import zipfile

import numpy as np

COMMIT = "d2c6b71c27c9f9931dfd98faf94e142a106632c6"
CI_RUN = 34775366549
CONFIG_HASH = "fac9d9ab2bd4d4bf58a704c6c800aa0af0fa26227a2e47864aeb93d36c398aae"
DATASETS = ("Squirrel", "Actor")
CONDITIONS = ("normalize_features", "normalize_centered_scaled")
THREADS = (1, 4, 8)
MODES = ("full_state_copy", "static_adjacency_once")
FIELDS = ("parameters", "gradients", "optimizer_state", "train_logits", "unscored_logits", "train_loss", "rng_before", "rng_after")
SCORE_FIELDS = {"validation_accuracy", "validation_loss", "test_accuracy", "test_loss", "selected_trial_id", "regret", "regret_pp", "selection_accuracy"}
EXCLUSIVE = ("rng_capture", "zero_grad", "train_forward", "train_loss", "backward", "gradient_finite_check", "optimizer_update", "unscored_eval_forward", "output_finite_check", "checkpoint_total", "audit_array_copy", "correctness_hash_instrumentation", "json_serialization", "exclusive_write_and_fsync")
COMPUTE = ("zero_grad", "train_forward", "train_loss", "backward", "optimizer_update", "unscored_eval_forward")
SOURCE_FILES = {"scripts/diagnose_h2gcn_performance.py", "scripts/h2gcn_checkpoint_diagnostic.py", "scripts/preflight_input_robustness_11.py", "scripts/input_robustness_data.py", "scripts/run_mlp_optimization_diagnostic.py", "scripts/run_preprocessing_sensitivity.py", "experiments/prospective_models.py", "experiments/prospective_data.py", "experiments/run_prospective_benchmark.py"}
GIB = 1024 ** 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def text_digest_matches(path, expected):
    """Allow only Git's LF/CRLF checkout conversion for bound text files."""
    raw = Path(path).read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return expected in {hashlib.sha256(value).hexdigest() for value in (raw, lf, lf.replace(b"\n", b"\r\n"))}


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def zero_scores(value, location="root"):
    if isinstance(value, dict):
        require(not SCORE_FIELDS.intersection(value), f"held-out score field: {location}")
        for key, item in value.items():
            if key in ("validation_evaluations", "test_evaluations", "formal_records"):
                require(item == 0, f"nonzero forbidden counter: {location}.{key}")
            if key in ("formal_training_enabled", "formal_use_authorized", "budget_change_authorized"):
                require(item is False, f"unauthorized formal/budget switch: {location}.{key}")
            zero_scores(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            zero_scores(item, f"{location}[{index}]")


def close(actual, expected, context):
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), f"numeric summary mismatch: {context}: {actual} != {expected}")


def same_numeric_tree(actual, expected, context):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f"mapping keys differ: {context}")
        for key in expected:
            same_numeric_tree(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f"list length differs: {context}")
        for index, (a, b) in enumerate(zip(actual, expected)):
            same_numeric_tree(a, b, f"{context}[{index}]")
    elif isinstance(expected, float):
        close(actual, expected, context)
    else:
        require(actual == expected, f"value differs: {context}")


def worker_id(dataset, condition, threads, mode, repeat):
    return f"{dataset}__{condition}__threads{threads}__{mode}__repeat{repeat}"


def expected_workers():
    first = list(itertools.product(DATASETS, CONDITIONS, (4, 1, 8), MODES))
    rows = []
    for repeat, sequence in enumerate((first, first[::-1])):
        for dataset, condition, threads, mode in sequence:
            rows.append({"dataset": dataset, "condition": condition, "cpu_threads": threads,
                         "checkpoint_mode": mode, "repeat": repeat,
                         "trial_id": "trial_000" if dataset == "Squirrel" else "trial_001",
                         "worker_id": worker_id(dataset, condition, threads, mode, repeat)})
    return rows


def tensor_rng_hash(value):
    digest = hashlib.sha256()
    digest.update(b"torch.uint8")
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def inspect_inventory(evidence):
    inventory_path = evidence / "diagnostic_inventory.json"
    inventory = read_json(inventory_path)
    require(inventory["phase"] == "performance_diagnostic_only" and inventory["run_id"] == "h2gcn_cpu_performance_v1", "inventory identity")
    files = {p.relative_to(evidence).as_posix(): p for p in evidence.rglob("*") if p.is_file()}
    require(set(inventory["files"]) == set(files) - {"diagnostic_inventory.json"}, "inventory coverage excludes files or has stale paths")
    snapshots = {}
    for relative, path in sorted(files.items()):
        require(not path.is_symlink(), f"symlink in evidence: {relative}")
        metadata = {"sha256": sha(path), "bytes": path.stat().st_size}
        if relative != "diagnostic_inventory.json":
            require(metadata == inventory["files"][relative], f"inventory hash/size mismatch: {relative}")
        snapshots[relative] = metadata
        if path.suffix == ".json":
            zero_scores(read_json(path), relative)
    return snapshots


def inspect_binding(repo, evidence, receipt_path):
    manifest = read_json(evidence / "diagnostic_manifest.json")
    config = manifest["config"]
    require(manifest["source_commit"] == COMMIT, "unexpected measured source commit")
    require(manifest["config_sha256"] == json_sha(config) == CONFIG_HASH, "config identity mismatch")
    require(set(manifest["source_files"]) == SOURCE_FILES, "executable source coverage")
    git = ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo)]
    for relative, expected in manifest["source_files"].items():
        path = repo / relative
        require(text_digest_matches(path, expected), f"bound executable source changed: {relative}")
        committed = subprocess.check_output([*git, "show", f"{COMMIT}:{relative}"])
        require(committed.replace(b"\r\n", b"\n") == path.read_bytes().replace(b"\r\n", b"\n"), f"source is not the CI commit content: {relative}")
    committed_config = json.loads(subprocess.check_output([*git, "show", f"{COMMIT}:configs/h2gcn_performance_diagnostic_v1.json"]))
    require(committed_config == config, "measured config differs from committed config")
    require(read_json(repo / "configs/h2gcn_performance_diagnostic_v1.json") == config, "current frozen config changed")
    for field in ("data_binding", "resource_profile"):
        bound_path = repo / config[field]
        require(text_digest_matches(bound_path, config[field + "_sha256"]), f"bound evidence changed: {field}")
        committed = subprocess.check_output([*git, "show", f"{COMMIT}:{config[field]}"])
        require(committed.replace(b"\r\n", b"\n") == bound_path.read_bytes().replace(b"\r\n", b"\n"), f"bound evidence differs from measured commit: {field}")
    parent = read_json(repo / config["parent_config"])
    require(json_sha(parent) == config["parent_config_sha256"], "scientific parent configuration changed")
    require(parent["formal_training_enabled"] is False, "parent formal training enabled")
    receipt = read_json(receipt_path)
    require(receipt == manifest["ci_receipt"] and text_digest_matches(receipt_path, manifest["ci_receipt_sha256"]), "CI receipt byte/content binding")
    require(receipt["commit"] == COMMIT and receipt["run_id"] == CI_RUN, "CI/source run identity")
    require(receipt["status"] == "completed" and receipt["conclusion"] == "success", "CI did not pass")
    require(len(receipt["jobs"]) == 3 and {j["name"] for j in receipt["jobs"]} == {"lightweight-verification", "full-protocol-verification", "manuscript-build"}, "CI job scope")
    require(all(j["run_id"] == CI_RUN and j["status"] == "completed" and j["conclusion"] == "success" for j in receipt["jobs"]), "CI job conclusion")
    return manifest, config


def inspect_worker(evidence, config, expected, job):
    identity = {**expected, "phase": "performance_diagnostic_only", "run_id": "h2gcn_cpu_performance_v1",
                "analysis_status": "post_hoc_training_recipe_robustness", "config_sha256": CONFIG_HASH,
                "seed": 0, "validation_evaluations": 0, "test_evaluations": 0, "formal_records": 0}
    location = evidence / "workers" / expected["worker_id"]
    row = read_json(location / "worker_summary.json")
    complete = read_json(location / "worker_complete.json")
    header = read_json(location / "worker.json")
    for value in (row, complete, header):
        require(all(value.get(k) == v for k, v in identity.items()), f"worker identity: {expected['worker_id']}")
    require(all(row.get(k) == v for k, v in header.items()), "worker header changed in summary")
    for value in (row, complete):
        require(value["status"] == "success" and value["optimization_steps"] == 10, "worker completion scope")
    require(sha(location / "worker_summary.json") == complete["worker_summary_sha256"], "worker summary completion hash")
    require(sha(location / "numeric_evidence.npz") == row["numeric_evidence_sha256"], "numeric file hash")
    require(len(row["steps"]) == 10 and [s["step"] for s in row["steps"]] == list(range(10)), "worker ten-step trajectory")
    require({p.name for p in (location / "steps").iterdir()} == {f"step_{i:03d}.json" for i in range(10)}, "worker step file scope")
    require(row["parameter_tensor_count"] == 4, "original H2GCN parameter tensor scope")
    n, nodes = {"Squirrel": (136005, 5201), "Actor": (61957, 7600)}[expected["dataset"]]
    require(row["dense_parameter_count"] == n and row["unscored_logits_shape"] == [nodes, 5], "original H2GCN parameter/output dimensions")
    shapes = {"parameters": (n,), "gradients": (n,), "optimizer_state": (2 * n + 4,), "train_logits": (nodes, 5), "unscored_logits": (nodes, 5), "train_loss": ()}
    keys = {f"step{i:03d}.{field}" for i in range(10) for field in FIELDS}
    with np.load(location / "numeric_evidence.npz", allow_pickle=False) as arrays:
        require(len(arrays.files) == 80 and set(arrays.files) == keys == set(row["numeric_evidence_arrays"]), "numeric evidence scope")
        previous_rng = None
        for index, step in enumerate(row["steps"]):
            require(all(step.get(k) == v for k, v in identity.items()), "step identity")
            saved = read_json(location / "steps" / f"step_{index:03d}.json")
            require(set(saved) == set(step), "step record field scope")
            require(all(step[k] == v for k, v in saved.items() if k != "timings"), "step file/summary content")
            require(all(step["timings"].get(k) == v for k, v in saved["timings"].items()), "step file/summary timings")
            require(set(step["fingerprint"]["arrays"]) == set(FIELDS), "step fingerprint array coverage")
            for field in FIELDS:
                name = f"step{index:03d}.{field}"
                value = arrays[name]
                dtype = "uint8" if field.startswith("rng_") else "float32"
                require(str(value.dtype) == dtype and np.isfinite(value).all(), f"nonfinite/type evidence: {name}")
                require(row["numeric_evidence_arrays"][name] == {"shape": list(value.shape), "dtype": dtype}, "numeric metadata binding")
                if field in shapes:
                    require(value.shape == shapes[field], "numeric model shape")
                else:
                    require(value.ndim == 1 and value.size > 0, "missing CPU RNG evidence")
                require(hashlib.sha256(value.tobytes()).hexdigest() == step["fingerprint"]["arrays"][field], "raw numeric hash mismatch")
            before, after = arrays[f"step{index:03d}.rng_before"], arrays[f"step{index:03d}.rng_after"]
            if index == 0:
                require(tensor_rng_hash(before) == row["initial_fingerprint"]["rng"], "initial CPU RNG hash mismatch")
            if previous_rng is not None:
                require(np.array_equal(previous_rng, before), "CPU RNG discontinuity between steps")
            previous_rng = after.copy()
            require(step["fingerprint"]["static_adjacency"] == row["initial_fingerprint"]["static_adjacency"], "static fingerprint changed during training")
            require(step["retained_slots"] == min(index + 1, 4), "retained slots progression")
            metrics = step["copy_metrics"]
            require(metrics["mode"] == expected["checkpoint_mode"] and metrics["slot"] == index % 4 and metrics["retained_checkpoint_count"] == min(index + 1, 4), "checkpoint progression")
            require(metrics["replacement"] == (index >= 4), "checkpoint replacement schedule")
            timing = step["timings"]
            require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in timing.values()), "nonfinite/negative timing")
            require(timing["end_to_end"] > 0, "nonpositive measured step")
            close(timing["accounted_exclusive"], sum(timing[k] for k in EXCLUSIVE), "exclusive component sum")
            close(timing["end_to_end"], timing["accounted_exclusive"] + timing["unattributed"], "end-to-end reconciliation")
            close(timing["compute_only"], sum(timing[k] for k in COMPUTE), "compute-only reconciliation")
            require(timing["checkpoint_parameter_copy"] + timing["checkpoint_static_copy"] <= timing["checkpoint_total"], "nested copy timings")
            require(timing["static_buffer_hash"] + timing["dense_state_hash"] <= timing["correctness_hash_instrumentation"], "nested hash timings")
            close(timing["checkpoint_parameter_copy"], metrics["parameter_copy_seconds"], "parameter timing metric")
            close(timing["checkpoint_static_copy"], metrics["static_buffer_copy_seconds"], "static timing metric")
        require(tensor_rng_hash(previous_rng) == row["final_rng"] and row["cpu_rng_unchanged_by_restoration"] is True, "final CPU RNG/restoration evidence")
    env = row["environment"]
    thread_env = {"OMP_NUM_THREADS": str(expected["cpu_threads"]), "MKL_NUM_THREADS": str(expected["cpu_threads"]), "OPENBLAS_NUM_THREADS": str(expected["cpu_threads"]), "PYTHONHASHSEED": "0", "CUDA_VISIBLE_DEVICES": "-1"}
    require(env["torch_threads"] == expected["cpu_threads"] and env["interop_threads"] == 1 and env["thread_environment"] == thread_env, "CPU runtime settings")
    require(all(env[k] is False for k in ("cuda_initialized", "cuda_available", "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark", "deterministic_warn_only")), "forbidden backend settings")
    require(env["deterministic_algorithms"] is True and env["cudnn_deterministic"] is True and env["float32_matmul_precision"] == "highest", "strict deterministic backend")
    require(row["cuda_allocated_bytes"] == 0 and row["cuda_initialized_after_probe"] is False, "CUDA activity")
    require(len(row["restoration"]) == 4 and {v["slot"] for v in row["restoration"]} == {0, 1, 2, 3} and all(v["complete_state_and_unscored_output_exact"] is True for v in row["restoration"]), "four stored-slot restoration attestations")
    storage = row["checkpoint_storage"]
    require(storage["slots"] == [0, 1, 2, 3] and storage["retained_checkpoint_count"] == 4 and storage["static_buffer_replicas"] == (4 if expected["checkpoint_mode"] == MODES[0] else 1), "retained checkpoint storage scope")
    require(storage["retained_parameter_bytes"] == 4 * n * 4, "retained dense parameter bytes")
    require(storage["total_unique_tensor_bytes"] == storage["retained_parameter_bytes"] + storage["retained_static_buffer_bytes"], "unique checkpoint storage accounting")
    require(job["returncode"] == 0 and job["stop_reason"] is None and job["observed_process_ids"], "failed/missing worker process")
    require(job["memory_scope"] == "owned_worker_process_tree_sum_including_windows_high_water", "worker memory scope")
    require(job["peak_rss_bytes"] <= 8 * GIB and row["peak_worker_rss_bytes"] <= 8 * GIB and job["wall_seconds"] <= 900, "worker resource caps")
    setup = row["setup_timings"]
    initial_parts = ("static_adjacency_snapshot", "initial_static_hash", "initial_checkpoint_schema_validation", "initial_static_boundary_validation")
    require(sum(setup[k] for k in initial_parts) <= setup["checkpoint_store_setup"], "initial setup timing overlap")
    accounted = sum(setup[k] for k in ("input_load_and_binding", "feature_transform", "transform_binding_verification", "adjacency_preparation", "model_setup", "checkpoint_store_setup"))
    accounted += sum(row[k] for k in ("adam_setup_seconds", "initial_full_state_hash_seconds", "final_static_validation_seconds", "checkpoint_restore_validation_seconds", "cross_thread_audit_serialization_seconds", "cross_thread_audit_write_seconds", "numeric_evidence_hash_seconds"))
    accounted += sum(step["timings"]["end_to_end"] for step in row["steps"])
    close(row["accounted_nonoverlapping_worker_seconds"], accounted, "whole-worker accounting")
    close(row["worker_measured_seconds_before_final_write"], accounted + row["unattributed_worker_seconds_before_final_write"], "whole-worker residual")
    require(row["unattributed_worker_seconds_before_final_write"] >= 0, "negative worker residual")
    return row


def comparison_plan():
    result = []
    for dataset, condition in itertools.product(DATASETS, CONDITIONS):
        for threads, mode in itertools.product(THREADS, MODES):
            result.append(("fresh_repeat", worker_id(dataset, condition, threads, mode, 0), worker_id(dataset, condition, threads, mode, 1), True))
        for threads, repeat in itertools.product(THREADS, range(2)):
            result.append(("checkpoint_mode", worker_id(dataset, condition, threads, MODES[1], repeat), worker_id(dataset, condition, threads, MODES[0], repeat), True))
        for threads, mode, repeat in itertools.product((1, 8), MODES, range(2)):
            result.append(("cross_thread", worker_id(dataset, condition, threads, mode, repeat), worker_id(dataset, condition, 4, mode, repeat), False))
    return result


def classify_comparison(left_complete, right_complete, numerical_passed=None):
    """Unavailable evidence is distinct from a measured numeric failure."""
    if not left_complete or not right_complete:
        return "unavailable_incomplete_scope"
    require(type(numerical_passed) is bool, "available comparison requires an explicit measured verdict")
    return "passed" if numerical_passed else "measured_mismatch"


def setting_qualifies(*, complete, resource_stopped, ratios, speed, candidate_rejected, reference_rejected):
    return bool(complete and not resource_stopped and len(ratios) == 4
                and not candidate_rejected and not reference_rejected and speed is not None
                and speed >= 1.10 and max(ratios) <= 1.10)


def inspect_partial_worker(evidence, expected, job, complete_rows):
    """Verify durable partial records without inventing unavailable arrays."""
    wid = expected["worker_id"]
    location = evidence / "workers" / wid
    require(job["returncode"] != 0 and job["stop_reason"], "incomplete worker lacks a recorded failed attempt")
    require(not (location / "worker_complete.json").exists(), "incomplete worker unexpectedly has completion marker")
    require(not (location / "worker_summary.json").exists(), "unexpected unaccepted worker summary")
    require(not (location / "numeric_evidence.npz").exists(), "partial worker unexpectedly serialized a complete numeric audit")
    header = read_json(location / "worker.json")
    require(all(header.get(k) == v for k, v in expected.items()), "partial worker identity")
    require(header["config_sha256"] == CONFIG_HASH and header["seed"] == 0, "partial worker configuration/seed")
    steps = sorted((location / "steps").glob("*.json"))
    require([p.name for p in steps] == [f"step_{i:03d}.json" for i in range(len(steps))] and len(steps) < 10, "partial step sequence")
    static = {row["initial_fingerprint"]["static_adjacency"] for row in complete_rows.values() if row["dataset"] == expected["dataset"]}
    partial_times = []
    for index, path in enumerate(steps):
        step = read_json(path)
        require(step["step"] == index and all(step.get(k) == v for k, v in expected.items()), "partial step identity")
        require(step["config_sha256"] == CONFIG_HASH and step["seed"] == 0, "partial step configuration/seed")
        require(step["retained_slots"] == min(index + 1, 4) and step["copy_metrics"]["slot"] == index % 4, "partial checkpoint progression")
        require(step["copy_metrics"]["retained_checkpoint_count"] == min(index + 1, 4) and step["copy_metrics"]["replacement"] == (index >= 4), "partial retention progression")
        require(set(step["fingerprint"]["arrays"]) == set(FIELDS), "partial numeric fingerprint scope")
        require(step["fingerprint"]["static_adjacency"] in static, "partial static hash differs from completed dataset evidence")
        timing = step["timings"]
        require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in timing.values()), "invalid partial timing")
        require("end_to_end" not in timing, "partial on-disk record unexpectedly contains completed total timing")
        require(timing["checkpoint_parameter_copy"] + timing["checkpoint_static_copy"] <= timing["checkpoint_total"], "partial nested copy timing")
        require(timing["static_buffer_hash"] + timing["dense_state_hash"] <= timing["correctness_hash_instrumentation"], "partial nested hash timing")
        partial_times.append({"step": index, "last_write_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(), "saved_nonoverlapping_component_seconds": sum(timing[k] for k in EXCLUSIVE if k in timing)})
    return {"worker_id": wid, "status": "resource_stopped_incomplete", "returncode": job["returncode"], "stop_reason": job["stop_reason"], "wall_seconds": job["wall_seconds"], "peak_rss_bytes": job["peak_rss_bytes"], "durable_step_records": len(steps), "partial_step_records": partial_times, "numeric_arrays_available": False, "restoration_audit_available": False,
            "limitation": "Saved step fingerprints and component timings are preserved. No numeric trajectory archive or completion marker exists, so independent numerical or restoration comparisons are unavailable for this worker."}


def inspect_comparisons(evidence, summary, rows):
    plan = comparison_plan()
    actual = {(v["kind"], v["left"], v["right"]): v for v in summary["comparisons"]}
    require(len(summary["comparisons"]) == len(actual) == 80 and set(actual) == {(k, l, r) for k, l, r, _ in plan}, "comparison scope/duplicates")
    verified, invalid = [], set()
    expected_by_id = {worker["worker_id"]: worker for worker in expected_workers()}
    for kind, left_id, right_id, exact in plan:
        observed = actual[kind, left_id, right_id]
        if left_id not in rows or right_id not in rows:
            require(observed["passed"] is False and "comparison" not in observed, "missing evidence was represented as a measured comparison")
            verified.append({"kind": kind, "left": left_id, "right": right_id, "passed": False,
                             "classification": classify_comparison(left_id in rows, right_id in rows),
                             "missing_complete_workers": [wid for wid in (left_id, right_id) if wid not in rows]})
            left, right = expected_by_id[left_id], expected_by_id[right_id]
            invalid.add((left["cpu_threads"], left["checkpoint_mode"]))
            if kind != "cross_thread":
                invalid.add((right["cpu_threads"], right["checkpoint_mode"]))
            continue
        left, right = rows[left_id], rows[right_id]
        mismatch, maximum_abs, maximum_scaled = [], 0.0, 0.0
        with np.load(evidence / "workers" / left_id / "numeric_evidence.npz", allow_pickle=False) as a_file, np.load(evidence / "workers" / right_id / "numeric_evidence.npz", allow_pickle=False) as b_file:
            for name in sorted(a_file.files):
                a, b = a_file[name], b_file[name]
                passed = np.array_equal(a, b) if exact or ".rng_" in name else np.allclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=False)
                if not passed:
                    mismatch.append(name)
                if a.size and a.dtype.kind == "f":
                    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
                    maximum_abs = max(maximum_abs, float(difference.max()))
                    maximum_scaled = max(maximum_scaled, float((difference / (1e-6 + 1e-5 * np.abs(b))).max()))
        invariant = all(left[k] == right[k] for k in ("split_id", "transformed_feature_sha256", "trial", "initial_fingerprint"))
        invariant = invariant and {k: v for k, v in left["environment"].items() if k not in ("torch_threads", "thread_environment")} == {k: v for k, v in right["environment"].items() if k not in ("torch_threads", "thread_environment")}
        if exact:
            invariant = invariant and [v["fingerprint"] for v in left["steps"]] == [v["fingerprint"] for v in right["steps"]]
        require(observed["passed"] == (not mismatch and invariant), f"comparison verdict differs: {kind}:{left_id}")
        require(observed["mismatched_arrays"] == mismatch, "comparison mismatch list differs")
        close(observed["maximum_absolute_difference"], maximum_abs, "maximum absolute difference")
        close(observed["maximum_tolerance_fraction"], maximum_scaled, "maximum tolerance fraction")
        require(observed["comparison"] == ("exact" if exact else "fixed_allclose_with_exact_rng"), "numeric comparison type")
        verified.append({"kind": kind, "left": left_id, "right": right_id, "passed": observed["passed"], "classification": classify_comparison(True, True, observed["passed"]), "maximum_absolute_difference": maximum_abs, "maximum_tolerance_fraction": maximum_scaled, "mismatched_arrays": mismatch, "invariants_passed": invariant})
        if not observed["passed"]:
            invalid.add((left["cpu_threads"], left["checkpoint_mode"]))
            if kind != "cross_thread":
                invalid.add((right["cpu_threads"], right["checkpoint_mode"]))
    for kind, field in (("fresh_repeat", "fresh_repeat_pairs_passed"), ("checkpoint_mode", "checkpoint_mode_pairs_passed"), ("cross_thread", "cross_thread_pairs_passed")):
        require(summary[field] == sum(v["passed"] for v in verified if v["kind"] == kind), "comparison count summary")
    return verified, invalid


def inspect_qualification(config, summary, rows, jobs, invalid):
    reported = {(r["dataset"], r["condition"], r["cpu_threads"], r["checkpoint_mode"]): r for r in summary["fixtures"]}
    expected_fixture_keys = {key for key in itertools.product(DATASETS, CONDITIONS, THREADS, MODES) if all(worker_id(*key, repeat) in rows for repeat in range(2))}
    require(len(reported) == len(summary["fixtures"]) and set(reported) == expected_fixture_keys, "available fixture coverage")
    fixtures, fixture_metrics = {}, []
    for dataset, condition, threads, mode in itertools.product(DATASETS, CONDITIONS, THREADS, MODES):
        key = dataset, condition, threads, mode
        if key not in expected_fixture_keys:
            continue
        workers = [rows[worker_id(*key, repeat)] for repeat in range(2)]
        repeats = []
        for worker in workers:
            medians = {phase: {name: statistics.median(worker["steps"][i]["timings"][name] for i in config["steps"][phase]) for name in worker["steps"][0]["timings"]} for phase in ("cold", "warmup", "steady")}
            actual_repeat = next(v for v in reported[key]["repeats"] if v["repeat"] == worker["repeat"])
            same_numeric_tree(actual_repeat["phase_medians"], medians, "phase medians")
            repeats.append(medians)
        value = max(repeat["steady"]["end_to_end"] for repeat in repeats)
        peak = max(jobs[w["worker_id"]]["peak_rss_bytes"] for w in workers)
        close(reported[key]["slower_repeat_steady_median_seconds"], value, "slower repeat median")
        require(reported[key]["maximum_process_tree_rss_bytes"] == peak, "fixture peak RSS")
        fixtures[key] = value, peak
        fixture_metrics.append({"dataset": dataset, "condition": condition, "cpu_threads": threads, "checkpoint_mode": mode, "slower_repeat_steady_median_seconds": value, "maximum_process_tree_rss_bytes": peak})
    settings = []
    overridden = bool(summary["coordinator_stop_reason"]) or summary["diagnostic_wall_seconds"] > 7200
    for threads, mode in itertools.product(THREADS, MODES):
        pairs = [(d, c) for d, c in itertools.product(DATASETS, CONDITIONS) if (d, c, threads, mode) in fixtures and (d, c, 4, MODES[0]) in fixtures]
        ratios = [] if overridden else [fixtures[d, c, threads, mode][0] / fixtures[d, c, 4, MODES[0]][0] for d, c in pairs]
        peak = 0 if overridden else max((fixtures[d, c, threads, mode][1] for d, c in pairs), default=0)
        speed = math.exp(sum(math.log(1 / ratio) for ratio in ratios) / len(ratios)) if ratios else None
        rejected = False if overridden else (threads, mode) in invalid
        qualifies = setting_qualifies(complete=summary["fixed_scope_complete"], resource_stopped=overridden, ratios=ratios, speed=speed, candidate_rejected=rejected, reference_rejected=(4, MODES[0]) in invalid)
        settings.append({"cpu_threads": threads, "checkpoint_mode": mode, "qualifies": qualifies, "geometric_speed_ratio": speed, "fixture_time_ratios": ratios, "maximum_process_tree_rss_bytes": peak, "correctness_or_resource_rejected": rejected})
    same_numeric_tree(summary["qualification"]["settings"], settings, "qualification settings")
    passing = sorted((v for v in settings if v["qualifies"]), key=lambda v: (-v["geometric_speed_ratio"], v["maximum_process_tree_rss_bytes"], v["cpu_threads"], v["checkpoint_mode"]))
    recommended = passing[0] if passing else None
    same_numeric_tree(summary["qualification"]["recommended_setting"], recommended, "recommended setting")
    require(summary["qualification"]["recommendation"] == ("adopt_verified_optimization_in_measured_scope" if passing else "retain_current_implementation_and_revise_budget"), "recommendation label")
    return settings, fixture_metrics


def write_exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def inspect_owned_processes(evidence):
    """Inspect only command lines matching this run; never terminate anything."""
    import psutil
    matches = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info["cmdline"] or []
            if not any(Path(argument).name == "diagnose_h2gcn_performance.py" for argument in command):
                continue
            if "--output-root" in command:
                output_index = command.index("--output-root") + 1
                if output_index < len(command) and Path(command[output_index]).resolve() == evidence:
                    matches.append(process.info["pid"])
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError, ValueError):
            continue
    require(not matches, f"measured coordinator/workers still active: {matches}")
    return {"checked_utc": datetime.now(timezone.utc).isoformat(), "matching_active_processes": 0,
            "scope": "live command lines for diagnose_h2gcn_performance.py with this exact output-root; no process was stopped by the verifier"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", "--repo", dest="repo", type=Path, required=True)
    parser.add_argument("--archive-root", "--evidence", dest="evidence", type=Path, required=True, help="preserved run directory, or the directory extracted from its ZIP")
    parser.add_argument("--ci-receipt", type=Path, required=True)
    parser.add_argument("--output", "--report", dest="report", type=Path, required=True)
    parser.add_argument("--archive-output", "--archive", dest="archive", type=Path, help="optional new ZIP; never overwrites an existing archive")
    parser.add_argument("--stable-confirmed", action="store_true", help="explicitly assert that the measured coordinator has finished and the input directory is immutable")
    args = parser.parse_args()
    require(args.stable_confirmed, "owner must confirm coordinator completion and stable evidence before audit/archive")
    evidence, repo = args.evidence.resolve(), args.repo.resolve()
    require(args.report.resolve() != evidence and evidence not in args.report.resolve().parents, "verification output must be outside immutable evidence")
    require(not args.report.exists(), "verification report already exists; preserve it")
    if args.archive:
        require(not args.archive.exists() and evidence not in args.archive.resolve().parents, "archive must be new and outside immutable evidence")
    process_check_before = inspect_owned_processes(evidence)
    snapshots = inspect_inventory(evidence)
    manifest, config = inspect_binding(repo, evidence, args.ci_receipt)
    summary = read_json(evidence / "diagnostic_summary.json")
    require(summary["source_commit"] == COMMIT and summary["config_sha256"] == CONFIG_HASH and summary["diagnostic_manifest_sha256"] == sha(evidence / "diagnostic_manifest.json"), "final summary provenance")
    expected = expected_workers()
    require(0 < len(summary["jobs"]) <= 48 and [j["worker_id"] for j in summary["jobs"]] == [r["worker_id"] for r in expected[:len(summary["jobs"])]], "fresh worker scope/order or forbidden retry")
    jobs = {j["worker_id"]: j for j in summary["jobs"]}
    require({p.name for p in (evidence / "workers").iterdir()} == set(jobs), "extra or missing worker directories")
    require({p.name for p in (evidence / "jobs").iterdir()} == {f"{wid}.json" for wid in jobs}, "extra or missing job record files")
    require({p.name for p in (evidence / "logs").iterdir()} == {f"{wid}.log" for wid in jobs}, "extra or missing worker logs")
    rows, partial, unlaunched = {}, [], []
    for worker in expected:
        wid = worker["worker_id"]
        if wid not in jobs:
            require(not (evidence / "workers" / wid).exists(), "never-launched worker has unexplained files")
            unlaunched.append(wid)
            continue
        require(read_json(evidence / "jobs" / f"{wid}.json") == jobs[wid], "job file/summary differs")
        require(all(jobs[wid][k] == v for k, v in worker.items()), "job worker identity")
        if jobs[wid]["returncode"] == 0 and jobs[wid]["stop_reason"] is None:
            rows[wid] = inspect_worker(evidence, config, worker, jobs[wid])
        else:
            partial.append(inspect_partial_worker(evidence, worker, jobs[wid], rows))
    complete = len(rows) == len(jobs) == 48
    require(summary["fixed_scope_complete"] is complete and summary["successful_workers"] == len(rows) and summary["observed_optimization_steps"] == 10 * len(rows) and summary["observed_jobs"] == len(jobs) and summary["expected_workers"] == 48, "actual/summary fixed scope differs")
    common_environments = {json_sha({key: value for key, value in row["environment"].items() if key not in ("torch_threads", "thread_environment")}) for row in rows.values()}
    require(len(common_environments) == 1, "runtime environment changed across dataset/condition fixtures")
    require(sum(job["wall_seconds"] for job in jobs.values()) <= summary["diagnostic_wall_seconds"], "coordinator time omits worker execution")
    for dataset in DATASETS:
        group = [r for r in rows.values() if r["dataset"] == dataset]
        if not group:
            continue
        require(len({r["initial_fingerprint"]["static_adjacency"] for r in group}) == 1, "dataset static adjacency fingerprint differs by condition/thread/mode/repeat")
        require(len({r["initial_fingerprint"]["store_static_fingerprint"] for r in group}) == 1, "dataset checkpoint-bank fingerprint differs")
    comparisons, invalid = inspect_comparisons(evidence, summary, rows)
    settings, fixtures = inspect_qualification(config, summary, rows, jobs, invalid)
    comparison_failures = [r for r in summary["failures"] if r.get("reason") == "correctness comparison failed"]
    require({(r["kind"], r["left"], r["right"]) for r in comparison_failures} == {(r["kind"], r["left"], r["right"]) for r in comparisons if not r["passed"]}, "failed comparison report differs")
    require(len(comparison_failures) == sum(not row["passed"] for row in comparisons), "duplicate failed comparison reports")
    resource_override = bool(summary["coordinator_stop_reason"]) or summary["diagnostic_wall_seconds"] > 7200
    missing_worker_failures = [row for row in summary["failures"] if row.get("worker_id") and row.get("reason") != "correctness comparison failed"]
    require(len(missing_worker_failures) == 48 - len(rows) and {row["worker_id"] for row in missing_worker_failures} == {row["worker_id"] for row in expected} - set(rows), "missing worker failure classification")
    scope_failure_count = sum(row.get("reason") == "incomplete or duplicate worker scope" for row in summary["failures"])
    require(scope_failure_count == int(len(jobs) != 48), "incomplete scope failure summary")
    require(len(summary["failures"]) == len(comparison_failures) + len(missing_worker_failures) + scope_failure_count + int(resource_override), "unexplained or omitted final failures")
    if partial:
        require(summary["coordinator_stop_reason"] == partial[-1]["stop_reason"] and jobs[summary["jobs"][-1]["worker_id"]]["stop_reason"], "resource stop did not end the launch sequence")
    # Rehash all files after numeric verification. A changed or newly added file
    # prevents archive creation even if the earlier inventory was valid.
    require(inspect_inventory(evidence) == snapshots, "evidence changed during independent verification")
    archive = None
    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as packed:
            for relative in sorted(snapshots):
                packed.write(evidence / relative, arcname=relative)
        with zipfile.ZipFile(args.archive, "r") as packed:
            require(packed.testzip() is None, "archive CRC check failed")
            require(len(packed.namelist()) == len(snapshots) and set(packed.namelist()) == set(snapshots), "archive coverage")
            for info in packed.infolist():
                digest = hashlib.sha256()
                with packed.open(info) as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                require(info.file_size == snapshots[info.filename]["bytes"] and digest.hexdigest() == snapshots[info.filename]["sha256"], f"archive content mismatch: {info.filename}")
        require(inspect_inventory(evidence) == snapshots, "evidence changed while archiving")
        archive = {"path": str(args.archive.resolve()), "sha256": sha(args.archive), "bytes": args.archive.stat().st_size, "files": len(snapshots), "all_crcs_and_member_sha256_verified": True}
    process_check_after = inspect_owned_processes(evidence)
    measured_status = "complete" if complete and not summary["failures"] else "incomplete_resource_stopped" if resource_override and not complete else "correctness_or_resource_failed"
    worker_metrics = []
    for wid, row in rows.items():
        worker_metrics.append({"worker_id": wid, "dataset": row["dataset"], "condition": row["condition"], "cpu_threads": row["cpu_threads"], "checkpoint_mode": row["checkpoint_mode"], "repeat": row["repeat"], "coordinator_wall_seconds": jobs[wid]["wall_seconds"], "peak_process_tree_rss_bytes": jobs[wid]["peak_rss_bytes"], "phase_medians": {phase: {name: statistics.median(row["steps"][index]["timings"][name] for index in config["steps"][phase]) for name in row["steps"][0]["timings"]} for phase in ("cold", "warmup", "steady")}, "setup_timings": row["setup_timings"], "checkpoint_storage": row["checkpoint_storage"]})
    report = {"schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(), "verification_status": "passed", "diagnostic_status": measured_status, "diagnostic_succeeded": measured_status == "complete", "scope": "independent archive integrity and numerical evidence audit, without training or raw adjacency reconstruction", "source_commit": COMMIT, "ci_run_id": CI_RUN, "config_sha256": CONFIG_HASH,
              "evidence_directory": str(evidence), "inventory_sha256": snapshots["diagnostic_inventory.json"]["sha256"], "inventory_entries_verified": len(snapshots) - 1, "evidence_files_verified": len(snapshots), "uncompressed_evidence_bytes": sum(v["bytes"] for v in snapshots.values()),
              "text_source_binding_policy": "Bound source/metadata/CI text digests allow only LF/CRLF checkout conversion; executable and scientific source content also matches the measured Git commit. Archived evidence files are always checked byte for byte.",
              "expected_workers": 48, "launched_workers": len(jobs), "successful_workers": len(rows), "incomplete_attempts": partial, "never_launched_workers": unlaunched, "never_launched_count": len(unlaunched), "fixed_scope_complete": complete,
              "optimization_steps_in_complete_trajectories": 10 * len(rows), "durable_partial_step_records": sum(row["durable_step_records"] for row in partial), "durable_step_records": 10 * len(rows) + sum(row["durable_step_records"] for row in partial), "numeric_arrays_verified": len(rows) * 80, "worker_rng_continuity_checks": len(rows) * 9, "initial_and_final_rng_hash_checks": len(rows) * 2,
              "comparison_scope": dict(Counter(r["kind"] for r in comparisons)), "comparisons_passed": {kind: sum(r["passed"] for r in comparisons if r["kind"] == kind) for kind in ("fresh_repeat", "checkpoint_mode", "cross_thread")}, "comparisons": comparisons,
              "comparison_classification": dict(Counter(row["classification"] for row in comparisons)), "available_comparisons_by_kind": {kind: sum(row["classification"] != "unavailable_incomplete_scope" for row in comparisons if row["kind"] == kind) for kind in ("fresh_repeat", "checkpoint_mode", "cross_thread")},
              "qualification_independently_recomputed": settings, "fixtures_independently_recomputed": fixtures, "recommended_setting": summary["qualification"]["recommended_setting"], "recorded_diagnostic_failure_count": len(summary["failures"]), "diagnostic_wall_seconds": summary["diagnostic_wall_seconds"], "maximum_process_tree_rss_bytes": max(j["peak_rss_bytes"] for j in jobs.values()),
              "completed_worker_metrics": worker_metrics, "owned_process_check_before": process_check_before, "owned_process_check_after": process_check_after,
              "formal_records": 0, "validation_evaluations": 0, "test_evaluations": 0, "formal_training_enabled": False, "archive": archive,
              "limitations": ["Static adjacency content and retained checkpoint tensors are not stored in this archive. This audit verifies their recorded hash invariants, source binding and four-slot restoration attestations; it does not independently reconstruct or restore those tensors.", "Unavailable comparisons marked passed=false by the coordinator are explicitly classified as unavailable here, not as measured numerical mismatches.", "A passed archive audit means preserved evidence is internally consistent; it does not mean the incomplete, resource-stopped diagnostic passed.", "Wall-clock time is not a measurement of active CPU compute time. This verifier does not infer the cause of the delayed stop from run files alone.", "Only the fixed CPU performance diagnostic is assessed; no accuracy conclusion, universal speedup or full-eleven runtime extrapolation follows.", "The CI receipt is independently hash-bound to the measured commit and its recorded three successful jobs; this offline audit does not query the CI service again."],
              "validator_sha256": sha(Path(__file__))}
    write_exclusive(args.report, report)
    print(json.dumps({key: report[key] for key in ("verification_status", "diagnostic_status", "launched_workers", "successful_workers", "never_launched_count", "durable_step_records", "numeric_arrays_verified", "comparison_scope", "comparison_classification", "comparisons_passed", "recommended_setting", "archive")}, sort_keys=True))


if __name__ == "__main__":
    main()
