"""Run a frozen, validation-only independent-process repeatability audit.

The coordinator deliberately imports only the standard library. Each worker
configures its environment before importing Torch. Existing artifacts are never
overwritten. Instrumented training mirrors the frozen trial's stopping rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "scripts/audit_training_reproducibility.py",
    "experiments/prospective_models.py",
    "experiments/run_prospective_benchmark.py",
    "experiments/prospective_data.py",
    "scripts/run_mlp_optimization_diagnostic.py",
    "scripts/run_preprocessing_sensitivity.py",
    "results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json",
)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def tensor_hash(tensor):
    if tensor is None:
        return None
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def state_fingerprint(state):
    tensors = {name: tensor_hash(value) for name, value in sorted(state.items())}
    return {"sha256": json_hash(tensors), "tensors": tensors}


def expand_workers(config):
    workers = []
    for group in config["groups"]:
        for repeat in range(group["repeats"]):
            workers.append({**group, "repeat": repeat, "worker_id": f"{group['group_id']}_repeat{repeat:02d}"})
    if len(workers) != config["expected_workers"] or len({w["worker_id"] for w in workers}) != len(workers):
        raise ValueError("invalid frozen worker count or identifiers")
    return workers


def configure_backend(backend):
    # Called before Torch (or any repository modules) is imported by a worker.
    if "torch" in sys.modules:
        raise RuntimeError("backend must be configured before importing Torch")
    if backend == "cuda_deterministic":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    import torch
    if backend == "cuda_deterministic":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.use_deterministic_algorithms(False)
    return torch


def rng_fingerprint(torch):
    import random
    import numpy as np
    numpy_state = np.random.get_state()
    return {
        "python": json_hash(random.getstate()),
        "numpy": json_hash([numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]),
        "torch_cpu": tensor_hash(torch.get_rng_state()),
        "torch_cuda": [tensor_hash(s) for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def train_instrumented(*, model, x, y, edges, train, validation, training, trial, early_epochs, torch):
    """Frozen accuracy/loss/earliest-epoch selection, with non-RNG-consuming hashes."""
    import torch.nn.functional as F
    optimizer = torch.optim.Adam(model.parameters(), lr=trial["learning_rate"], weight_decay=training["weight_decay"])
    history, early = [], []
    best_key, best_state, stale = None, None, 0
    for epoch in range(training["max_epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        snapshot = {"epoch": epoch, "rng_before_train": rng_fingerprint(torch)} if epoch in early_epochs else None
        logits = model(x, edges)
        loss = F.cross_entropy(logits[train], y[train])
        loss.backward()
        if snapshot is not None:
            snapshot["train_logits_sha256"] = tensor_hash(logits)
            snapshot["gradients"] = state_fingerprint({n: p.grad for n, p in model.named_parameters()})
            snapshot["gradient_l2_by_parameter"] = {n: float(p.grad.detach().double().norm().cpu()) if p.grad is not None else None for n, p in model.named_parameters()}
            snapshot["rng_after_backward"] = rng_fingerprint(torch)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits = model(x, edges)
            validation_loss = float(F.cross_entropy(validation_logits[validation], y[validation]).item())
            validation_accuracy = float((validation_logits[validation].argmax(1) == y[validation]).float().mean().item())
        entry = {"epoch": epoch, "train_loss": float(loss.item()), "validation_loss": validation_loss, "validation_accuracy": validation_accuracy}
        history.append(entry)
        if snapshot is not None:
            snapshot["post_step_state"] = state_fingerprint(model.state_dict())
            snapshot["validation_logits_sha256"] = tensor_hash(validation_logits)
            snapshot["rng_after_eval"] = rng_fingerprint(torch)
            early.append(snapshot)
        key = (-validation_accuracy, validation_loss, epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= training["patience"]:
            break
    if best_state is None:
        raise RuntimeError("no selected checkpoint")
    return {"history": history, "history_sha256": json_hash(history), "early_trajectory": early,
            "best_epoch": best_key[2], "validation_accuracy": -best_key[0], "validation_loss": best_key[1],
            "epochs_completed": len(history)}, best_state


def worker_run(config, worker, output_root, data_root, torch):
    sys.path.insert(0, str(ROOT))
    import torch.nn.functional as F
    from experiments.prospective_models import build_model
    from experiments.run_prospective_benchmark import environment_snapshot, seed_everything
    from scripts.run_preprocessing_sensitivity import load_inputs
    from scripts.run_mlp_optimization_diagnostic import fit_transform_features
    manifest = json.loads((output_root / "run_manifest.json").read_text())
    if {name: file_hash(ROOT / name) for name in SOURCES} != manifest["source_files_sha256"]:
        raise RuntimeError("audited source changed after manifest freeze")
    torch.set_num_threads(config["execution"]["torch_num_threads"])
    device = torch.device("cpu" if worker["backend"] == "cpu" else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    audit = json.loads((ROOT / SOURCES[-1]).read_text())
    x, y, edges, splits = load_inputs(data_root, config["dataset"], config["dataset_specification"], audit)
    split, split_id = splits[worker["seed"]]
    train = torch.as_tensor(split["train"], dtype=torch.long)
    validation = torch.as_tensor(split["validation"], dtype=torch.long)
    x, transform = fit_transform_features(x, train, worker["condition"])
    input_hashes = {"features": tensor_hash(x), "labels": tensor_hash(y), "edges": tensor_hash(edges), "train_indices": tensor_hash(train), "validation_indices": tensor_hash(validation)}
    trial = config["trials"][worker["trial_id"]]
    training = {**config["training"], "max_epochs": worker.get("max_epochs", config["training"]["max_epochs"])}
    seed_everything(worker["seed"])
    model_args = {"num_nodes": len(y), "in_channels": x.size(1), "hidden_channels": training["hidden_channels"], "out_channels": int(y.max()) + 1, "dropout": trial["dropout"], "edge_index": edges}
    model = build_model(worker["model"], **model_args).to(device)
    initial = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
    initial_rng = rng_fingerprint(torch)
    folder = output_root / "workers" / worker["worker_id"]
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / "initial_state.pt").open("xb") as handle:
        torch.save(initial, handle)
    x, y, edges, train, validation = [t.to(device) for t in (x, y, edges, train, validation)]
    started = time.perf_counter()
    trajectory, selected = train_instrumented(model=model, x=x, y=y, edges=edges, train=train, validation=validation, training=training, trial=trial, early_epochs=config["instrumentation"]["early_epochs"], torch=torch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    with (folder / "selected_state.pt").open("xb") as handle:
        torch.save(selected, handle)
    model.load_state_dict(selected)
    model.eval()
    forwards = []
    with torch.no_grad():
        reference = model(x, edges).detach().cpu()
        for _ in range(config["instrumentation"]["fixed_checkpoint_forward_repeats"]):
            logits = model(x, edges).detach().cpu()
            forwards.append({"logits_sha256": tensor_hash(logits), "max_abs_logit_difference": float((logits - reference).abs().max()), "prediction_disagreements": int((logits.argmax(1) != reference.argmax(1)).sum())})
    seed_everything(worker["seed"])
    rebuilt = build_model(worker["model"], **model_args).to(device)
    rebuilt.load_state_dict(selected)
    rebuilt.eval()
    with torch.no_grad():
        rebuilt_logits = rebuilt(x, edges).detach().cpu()
    with (folder / "selected_logits.pt").open("xb") as handle:
        torch.save(reference, handle)
    partitions = {}
    for name, indices in (("train", train.cpu()), ("validation", validation.cpu())):
        local_y = y.cpu()[indices]
        local_logits = reference[indices]
        partitions[name] = {"n": len(indices), "loss": float(F.cross_entropy(local_logits, local_y)), "accuracy": float((local_logits.argmax(1) == local_y).float().mean()), "prediction_counts": torch.bincount(local_logits.argmax(1), minlength=model_args["out_channels"]).tolist()}
    result = {"status": "success", "run_id": config["run_id"], "worker": worker, "pid": os.getpid(), "dataset": config["dataset"], "split_id": split_id,
        "training": training, "trial": trial, "input_sha256": input_hashes, "transform_metadata": transform,
        "environment": {**environment_snapshot(device), "torch_file": torch.__file__, "torch_geometric_file": __import__("torch_geometric").__file__, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(), "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"), "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic, "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "float32_matmul_precision": torch.get_float32_matmul_precision(), "torch_threads": torch.get_num_threads()},
        "initial_state": state_fingerprint(initial), "initial_rng": initial_rng, "selected_state": state_fingerprint(selected), "parameter_names": list(dict(model.named_parameters())), "buffer_names": list(dict(model.named_buffers())),
        **trajectory, "training_seconds": training_seconds, "partitions": partitions, "fixed_checkpoint_reference_logits_sha256": tensor_hash(reference), "fixed_checkpoint_forwards": forwards,
        "rebuilt_checkpoint": {"logits_sha256": tensor_hash(rebuilt_logits), "max_abs_logit_difference": float((rebuilt_logits-reference).abs().max()), "prediction_disagreements": int((rebuilt_logits.argmax(1) != reference.argmax(1)).sum())},
        "artifacts_sha256": {p.name: file_hash(p) for p in folder.glob("*.pt")}, "source_files_sha256": manifest["source_files_sha256"], "test_evaluations": 0}
    write_exclusive(folder / "record.json", result)
    return result


def summarize_records(config, records):
    groups = []
    for group in config["groups"]:
        rows = [r for r in records if r["worker"]["group_id"] == group["group_id"]]
        success = [r for r in rows if r["status"] == "success"]
        entry = {**group, "success_count": len(success), "failure_count": len(rows)-len(success)}
        if success:
            entry.update({"validation_accuracy": [r["validation_accuracy"] for r in success], "validation_loss": [r["validation_loss"] for r in success], "best_epoch": [r["best_epoch"] for r in success], "epochs_completed": [r["epochs_completed"] for r in success], "accuracy_range": max(r["validation_accuracy"] for r in success)-min(r["validation_accuracy"] for r in success), "identical_initial_states": len({r["initial_state"]["sha256"] for r in success}) == 1, "identical_initial_rng": len({json_hash(r["initial_rng"]) for r in success}) == 1, "identical_selected_states": len({r["selected_state"]["sha256"] for r in success}) == 1, "identical_histories": len({r["history_sha256"] for r in success}) == 1, "max_fixed_checkpoint_logit_difference": max(v["max_abs_logit_difference"] for r in success for v in r["fixed_checkpoint_forwards"]), "max_fixed_checkpoint_prediction_disagreements": max(v["prediction_disagreements"] for r in success for v in r["fixed_checkpoint_forwards"])})
            divergence = {}
            for key in ("rng_before_train", "train_logits_sha256", "gradients", "post_step_state", "validation_logits_sha256"):
                epochs = set.intersection(*(set(e["epoch"] for e in r["early_trajectory"]) for r in success))
                divergence[key] = next((e for e in sorted(epochs) if len({json_hash(next(s[key] for s in r["early_trajectory"] if s["epoch"] == e)) for r in success}) > 1), None)
            entry["first_observed_early_divergence_epoch"] = divergence
        groups.append(entry)
    return {"run_id": config["run_id"], "config": config, "worker_count": len(records), "success_count": sum(r["status"] == "success" for r in records), "groups": groups, "test_evaluations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training_reproducibility_audit_v1.json")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-id")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    workers = expand_workers(config)
    if args.worker_id:
        worker = next(w for w in workers if w["worker_id"] == args.worker_id)
        try:
            torch = configure_backend(worker["backend"])
            result = worker_run(config, worker, args.output_root, args.data_root, torch)
            print(json.dumps({"worker_id": args.worker_id, "validation_accuracy": result["validation_accuracy"], "best_epoch": result["best_epoch"]}), flush=True)
        except Exception as exc:
            write_exclusive(args.output_root / "failures" / f"{args.worker_id}.json", {"status": "failure", "worker": worker, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
            raise
        return
    args.output_root.mkdir(parents=True, exist_ok=False)
    source_hashes = {name: file_hash(ROOT / name) for name in SOURCES}
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT)]
    commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run([*git, "status", "--short"], capture_output=True, text=True, check=True).stdout.strip()
    manifest = {"run_id": config["run_id"], "config": config, "config_sha256": json_hash(config), "source_commit": commit, "source_worktree_status": dirty, "source_files_sha256": source_hashes, "python_executable": sys.executable, "argv": sys.argv, "started_unix": time.time(), "workers": workers, "test_evaluations": 0}
    write_exclusive(args.output_root / "run_manifest.json", manifest)
    frozen_config = args.output_root / "frozen_config.json"
    write_exclusive(frozen_config, config)
    records = []
    for worker in workers:
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(frozen_config.resolve()), "--data-root", str(args.data_root.resolve()), "--output-root", str(args.output_root.resolve()), "--worker-id", worker["worker_id"]]
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(worker["seed"])
        if worker["backend"] == "cuda_deterministic":
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        else:
            env.pop("CUBLAS_WORKSPACE_CONFIG", None)
        logs = args.output_root / "logs"
        logs.mkdir(exist_ok=True)
        with (logs / f"{worker['worker_id']}.log").open("x", encoding="utf-8") as handle:
            process = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        record_path = args.output_root / "workers" / worker["worker_id"] / "record.json"
        failure_path = args.output_root / "failures" / f"{worker['worker_id']}.json"
        if record_path.exists():
            result = json.loads(record_path.read_text())
        elif failure_path.exists():
            result = json.loads(failure_path.read_text())
        else:
            result = {"status": "failure", "worker": worker, "error": "worker exited without record", "returncode": process.returncode}
            write_exclusive(failure_path, result)
        records.append(result)
        print(json.dumps({"completed": len(records), "expected": len(workers), "worker": worker["worker_id"], "status": result["status"], "validation_accuracy": result.get("validation_accuracy")}), flush=True)
    summary = summarize_records(config, records)
    summary["manifest"] = manifest
    summary["records"] = records
    write_exclusive(args.output_root / "audit_summary.json", summary)
    checksum = {str(p.relative_to(args.output_root)): file_hash(p) for p in sorted(args.output_root.rglob("*")) if p.is_file()}
    write_exclusive(args.output_root / "complete.json", {"run_id": config["run_id"], "status": "complete" if summary["success_count"] == len(workers) else "completed_with_failures", "worker_count": len(records), "success_count": summary["success_count"], "finished_unix": time.time(), "files_sha256": checksum, "test_evaluations": 0})


if __name__ == "__main__":
    main()
