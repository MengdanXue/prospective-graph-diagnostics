#!/usr/bin/env python3
"""Generate immutable dataset/model/seed records for the frozen benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy
import sklearn
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import (
    Actor,
    Coauthor,
    HeterophilousGraphDataset,
    Planetoid,
    WebKB,
    WikipediaNetwork,
)
from torch_geometric.transforms import NormalizeFeatures

from experiments.prospective_data import (
    canonical_undirected_edges,
    dataset_specifications,
    make_stratified_split,
    split_identifier,
    train_only_diagnostics,
    validate_dataset_eligibility,
)
from experiments.prospective_models import build_model, prepare_h2_adjacencies


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "prospective_benchmark_v1.json"


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_stage_failure(
    path: Path,
    *,
    stage: str,
    exception: Exception,
    context: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    write_json_exclusive(
        path,
        {
            "schema_version": "1.0",
            **context,
            "status": "error",
            "stage": stage,
            "exception_type": type(exception).__name__,
            "exception": str(exception),
            "traceback": "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            ),
            "elapsed_seconds": float(elapsed_seconds),
        },
    )


def make_split_with_failure(
    *,
    labels: np.ndarray,
    seed: int,
    failure_path: Path,
    context: dict[str, Any],
    started_at: float,
) -> dict[str, np.ndarray]:
    try:
        return make_stratified_split(labels, seed=seed)
    except Exception as exc:
        write_stage_failure(
            failure_path,
            stage="split_construction",
            exception=exc,
            context=context,
            elapsed_seconds=time.perf_counter() - started_at,
        )
        raise


def validate_trial_audit(record: dict[str, Any], training: dict[str, Any]) -> None:
    trials = record.get("trials") or []
    expected_trials = training["trials"]
    if len(trials) != len(expected_trials):
        raise ValueError("trial count mismatch")
    for index, (row, configuration) in enumerate(zip(trials, expected_trials)):
        if row.get("trial_id") != f"trial_{index:03d}":
            raise ValueError("trial identifier mismatch")
        if row.get("configuration") != configuration:
            raise ValueError("trial configuration mismatch")
    selected = select_trial(trials)
    if record.get("selected_trial_id") != selected["trial_id"]:
        raise ValueError("selected trial mismatch")
    for key in ("validation_accuracy", "validation_loss"):
        if not math.isclose(float(record[key]), float(selected[key]), abs_tol=1e-12):
            raise ValueError(f"selected {key} mismatch")
    if record.get("test_evaluations_after_selection") != 1:
        raise ValueError("test evaluation count mismatch")


def validate_resume_record(
    path: Path,
    expected: dict[str, Any],
    training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("status") != "success":
        raise RuntimeError(f"cannot resume past non-success record: {path}")
    mismatches = [key for key, value in expected.items() if record.get(key) != value]
    if mismatches:
        raise RuntimeError(f"stale resume record {path}; mismatched fields: {mismatches}")
    if training is not None:
        try:
            validate_trial_audit(record, training)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid trial audit in resume record {path}: {exc}") from exc
    return record


def validate_output_root_ownership(output_root: Path, run_id: str) -> None:
    output_root = Path(output_root)
    if not output_root.exists():
        return
    for path in sorted(output_root.rglob("*.json")):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot establish output-root ownership from {path}: {exc}"
            ) from exc
        artifact_run_id = artifact.get("run_id")
        if artifact_run_id is None:
            raise RuntimeError(f"cannot establish output-root ownership from {path}")
        if artifact_run_id != run_id:
            raise RuntimeError(
                f"output root belongs to another run: {path} has {artifact_run_id!r}, "
                f"expected {run_id!r}"
            )


def select_trial(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("trial list must not be empty")
    return sorted(
        rows,
        key=lambda row: (
            -float(row["validation_accuracy"]),
            float(row["validation_loss"]),
            str(row["trial_id"]),
        ),
    )[0]


def _accuracy(logits: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> float:
    predictions = logits[indices].argmax(dim=1)
    return float((predictions == labels[indices]).float().mean().item())


def _train_trial(
    *,
    model_id: str,
    seed: int,
    trial_id: str,
    trial: dict[str, float],
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    hidden_channels: int,
    max_epochs: int,
    patience: int,
    weight_decay: float,
    device: torch.device,
    h2_adjacencies: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    seed_everything(seed)
    model = build_model(
        model_id,
        num_nodes=x.size(0),
        in_channels=x.size(1),
        hidden_channels=hidden_channels,
        out_channels=int(y.max().item()) + 1,
        dropout=float(trial["dropout"]),
        edge_index=edge_index,
        h2_adjacencies=h2_adjacencies,
    ).to(device)
    local_x = x.to(device)
    local_y = y.to(device)
    local_edges = edge_index.to(device)
    local_train = train_indices.to(device)
    local_validation = validation_indices.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(trial["learning_rate"]),
        weight_decay=weight_decay,
    )
    best_key: tuple[float, float, int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_accuracy = -1.0
    best_loss = float("inf")
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(local_x, local_edges)
        train_loss = F.cross_entropy(logits[local_train], local_y[local_train])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(local_x, local_edges)
            validation_loss = float(
                F.cross_entropy(
                    validation_logits[local_validation], local_y[local_validation]
                ).item()
            )
            validation_accuracy = _accuracy(
                validation_logits, local_y, local_validation
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.item()),
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        candidate = (-validation_accuracy, validation_loss, epoch)
        if best_key is None or candidate < best_key:
            best_key = candidate
            best_epoch = epoch
            best_accuracy = validation_accuracy
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    summary = {
        "trial_id": trial_id,
        "configuration": dict(trial),
        "validation_accuracy": best_accuracy,
        "validation_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "history": history,
        "duration_seconds": time.perf_counter() - started,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, best_state


def run_model_unit(
    *,
    run_id: str,
    dataset: str,
    model_id: str,
    seed: int,
    split_id: str,
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    test_indices: torch.Tensor,
    training: dict[str, Any],
    source_commit: str,
    environment: dict[str, Any],
    data_provenance: dict[str, Any],
    device: torch.device,
    h2_adjacencies: tuple[torch.Tensor, torch.Tensor] | None = None,
    config_sha256: str = "test-only",
    frozen_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    trial_rows = []
    states: dict[str, dict[str, torch.Tensor]] = {}
    for index, trial in enumerate(training["trials"]):
        trial_id = f"trial_{index:03d}"
        row, state = _train_trial(
            model_id=model_id,
            seed=seed,
            trial_id=trial_id,
            trial=trial,
            x=x,
            y=y,
            edge_index=edge_index,
            train_indices=train_indices,
            validation_indices=validation_indices,
            hidden_channels=int(training["hidden_channels"]),
            max_epochs=int(training["max_epochs"]),
            patience=int(training["patience"]),
            weight_decay=float(training["weight_decay"]),
            device=device,
            h2_adjacencies=h2_adjacencies,
        )
        trial_rows.append(row)
        states[trial_id] = state
    selected = select_trial(trial_rows)
    selected_configuration = selected["configuration"]
    seed_everything(seed)
    model = build_model(
        model_id,
        num_nodes=x.size(0),
        in_channels=x.size(1),
        hidden_channels=int(training["hidden_channels"]),
        out_channels=int(y.max().item()) + 1,
        dropout=float(selected_configuration["dropout"]),
        edge_index=edge_index,
        h2_adjacencies=h2_adjacencies,
    ).to(device)
    model.load_state_dict(states[selected["trial_id"]])
    model.eval()
    with torch.no_grad():
        test_logits = model(x.to(device), edge_index.to(device))
        test_accuracy = _accuracy(test_logits, y.to(device), test_indices.to(device))
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "success",
        "dataset": dataset,
        "model": model_id,
        "family": "mlp" if model_id == "MLP" else "graph",
        "seed": int(seed),
        "split_id": split_id,
        "validation_accuracy": selected["validation_accuracy"],
        "validation_loss": selected["validation_loss"],
        "test_accuracy": test_accuracy,
        "test_evaluations_after_selection": 1,
        "selected_trial_id": selected["trial_id"],
        "trials": trial_rows,
        "training_configuration": dict(training),
        "source_commit": source_commit,
        "environment": environment,
        "data_provenance": data_provenance,
        "config_sha256": config_sha256,
        "frozen_config": frozen_config,
        "duration_seconds": time.perf_counter() - started,
    }


def environment_snapshot(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }


def _source_commit() -> str:
    for arguments in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        completed = subprocess.run(arguments, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError("refusing to run experiments from a dirty tracked worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def config_sha256(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_dataset(name: str, data_root: Path):
    classes = {
        "Planetoid": Planetoid,
        "WebKB": WebKB,
        "WikipediaNetwork": WikipediaNetwork,
        "Actor": Actor,
        "HeterophilousGraphDataset": HeterophilousGraphDataset,
        "Coauthor": Coauthor,
    }
    specification = dataset_specifications()[name]
    dataset_class = classes[specification["class"]]
    kwargs = {key: value for key, value in specification.items() if key != "class"}
    probe = object.__new__(dataset_class)
    probe.root = str(data_root)
    if dataset_class is Planetoid:
        probe.name = kwargs["name"]
        probe.split = "public"
    elif dataset_class is WebKB:
        probe.name = kwargs["name"].lower()
    elif dataset_class is WikipediaNetwork:
        probe.name = kwargs["name"].lower()
        probe.geom_gcn_preprocess = True
    elif dataset_class is HeterophilousGraphDataset:
        probe.name = kwargs["name"].lower().replace("-", "_")
    elif dataset_class is Coauthor:
        probe.name = kwargs["name"]
    required_paths = [Path(path) for path in probe.raw_paths + probe.processed_paths]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"read-only cache preflight failed for {name}; missing files: {missing}"
        )
    before = {str(path.resolve()): (path.stat().st_size, path.stat().st_mtime_ns) for path in required_paths}
    dataset = dataset_class(
        root=str(data_root),
        transform=NormalizeFeatures(),
        **kwargs,
    )
    after = {str(path.resolve()): (path.stat().st_size, path.stat().st_mtime_ns) for path in required_paths}
    if before != after:
        raise RuntimeError(f"dataset cache changed during read-only load: {name}")
    return dataset


def _processed_provenance(dataset) -> dict[str, Any]:
    paths = [Path(path) for path in dataset.processed_paths]
    records = []
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append(
            {
                "path": path.name,
                "root": str(path.parent.resolve()),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return {"processed_files": records}


def _bidirectional_edge_index(edges: np.ndarray) -> torch.Tensor:
    both = np.concatenate((edges, edges[:, ::-1]), axis=0)
    order = np.lexsort((both[:, 1], both[:, 0]))
    return torch.from_numpy(both[order].T.copy()).long()


def _indices_to_mask(indices: np.ndarray, node_count: int) -> np.ndarray:
    mask = np.zeros(node_count, dtype=bool)
    mask[indices] = True
    return mask


def _selected(values: list[str] | None, frozen: list[str], label: str) -> list[str]:
    if not values:
        return frozen
    unknown = sorted(set(values) - set(frozen))
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}")
    return [value for value in frozen if value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> int:
    run_started = time.perf_counter()
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_output_root_ownership(args.output_root, config["run_id"])
    config_digest = config_sha256(config)
    datasets = _selected(args.datasets, config["datasets"], "datasets")
    models = _selected(args.models, config["models"], "models")
    seeds = _selected(
        [str(seed) for seed in args.seeds] if args.seeds else None,
        [str(seed) for seed in config["seeds"]],
        "seeds",
    )
    seeds = [int(seed) for seed in seeds]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    environment = environment_snapshot(device)
    commit = _source_commit()

    for dataset_name in datasets:
        failure_root = args.output_root / "failures" / dataset_name
        if args.resume and failure_root.exists() and any(failure_root.rglob("*.json")):
            raise RuntimeError(f"existing failure blocks resume: {failure_root}")
        try:
            dataset = _load_dataset(dataset_name, args.data_root)
            data = dataset[0]
            x = data.x.float().cpu()
            y = data.y.long().view(-1).cpu()
            canonical = canonical_undirected_edges(
                data.edge_index.cpu().numpy(), node_count=data.num_nodes
            )
            edge_index = _bidirectional_edge_index(canonical)
            provenance = _processed_provenance(dataset)
        except Exception as exc:
            write_json_exclusive(
                args.output_root / "failures" / dataset_name / "dataset_setup.json",
                {
                    "schema_version": "1.0",
                    "run_id": config["run_id"],
                    "status": "error",
                    "dataset": dataset_name,
                    "stage": "dataset_setup",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                    "source_commit": commit,
                    "config_sha256": config_digest,
                    "frozen_config": config,
                    "environment": environment,
                    "command": sys.argv,
                    "elapsed_seconds": time.perf_counter() - run_started,
                },
            )
            return 1
        eligibility = config.get("eligibility")
        if eligibility is not None:
            minimum_class_count = int(eligibility["minimum_class_count"])
            try:
                class_counts = validate_dataset_eligibility(
                    y.numpy(), minimum_class_count=minimum_class_count
                )
            except Exception as exc:
                write_stage_failure(
                    failure_root / "dataset_eligibility.json",
                    stage="dataset_eligibility",
                    exception=exc,
                    context={
                        "run_id": config["run_id"],
                        "dataset": dataset_name,
                        "source_commit": commit,
                        "config_sha256": config_digest,
                        "frozen_config": config,
                        "environment": environment,
                        "data_provenance": provenance,
                        "command": sys.argv,
                    },
                    elapsed_seconds=time.perf_counter() - run_started,
                )
                return 1
            provenance = {
                **provenance,
                "eligibility": {
                    "eligible": True,
                    "minimum_class_count": minimum_class_count,
                    "class_counts": class_counts,
                },
            }
        h2 = None
        for seed in seeds:
            split_failure_path = failure_root / f"seed_{seed:03d}_split.json"
            try:
                split = make_split_with_failure(
                    labels=y.numpy(),
                    seed=seed,
                    failure_path=split_failure_path,
                    context={
                        "run_id": config["run_id"],
                        "dataset": dataset_name,
                        "seed": seed,
                        "source_commit": commit,
                        "config_sha256": config_digest,
                        "frozen_config": config,
                        "environment": environment,
                        "data_provenance": provenance,
                        "command": sys.argv,
                    },
                    started_at=run_started,
                )
            except Exception:
                return 1
            split_id = split_identifier(split)
            train_mask = _indices_to_mask(split["train"], data.num_nodes)
            try:
                diagnostics = train_only_diagnostics(
                    canonical,
                    y.numpy(),
                    train_mask,
                    sample_count=int(config["diagnostics"]["two_hop_walks"]),
                    seed=seed + 200_000,
                )
            except Exception as exc:
                write_json_exclusive(
                    args.output_root
                    / "failures"
                    / dataset_name
                    / "diagnostics"
                    / f"seed_{seed:03d}.json",
                    {
                        "schema_version": "1.0",
                        "run_id": config["run_id"],
                        "status": "error",
                        "dataset": dataset_name,
                        "seed": seed,
                        "split_id": split_id,
                        "stage": "train_only_diagnostics",
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "traceback": traceback.format_exc(),
                        "source_commit": commit,
                        "config_sha256": config_digest,
                        "frozen_config": config,
                        "environment": environment,
                        "data_provenance": provenance,
                        "command": sys.argv,
                        "elapsed_seconds": time.perf_counter() - run_started,
                    },
                )
                return 1
            diagnostic_path = (
                args.output_root
                / "diagnostics"
                / dataset_name
                / f"seed_{seed:03d}.json"
            )
            diagnostic_record = {
                "schema_version": "1.0",
                "run_id": config["run_id"],
                "status": "success",
                "dataset": dataset_name,
                "seed": seed,
                "split_id": split_id,
                "homophily": diagnostics["homophily"],
                "mean_degree": diagnostics["mean_degree"],
                "delta_h": diagnostics["delta_h"],
                "details": diagnostics,
                "provenance": {
                    "uses_test_labels": False,
                    "homophily_label_scope": (
                        "train_only" if diagnostics["homophily"] is not None else "unavailable"
                    ),
                    "two_hop_label_scope": (
                        "train_only" if diagnostics["delta_h"] is not None else "unavailable"
                    ),
                    "source_commit": commit,
                    "environment": environment,
                    "data": provenance,
                    "config_sha256": config_digest,
                    "frozen_config": config,
                },
            }
            if not diagnostic_path.exists():
                write_json_exclusive(diagnostic_path, diagnostic_record)
            elif args.resume:
                validate_resume_record(
                    diagnostic_path,
                    {
                        "run_id": config["run_id"],
                        "dataset": dataset_name,
                        "seed": seed,
                        "split_id": split_id,
                        "provenance": diagnostic_record["provenance"],
                    },
                )
            else:
                raise FileExistsError(diagnostic_path)

            for model_id in models:
                record_path = (
                    args.output_root
                    / "records"
                    / dataset_name
                    / model_id
                    / f"seed_{seed:03d}.json"
                )
                failure_path = (
                    args.output_root
                    / "failures"
                    / dataset_name
                    / model_id
                    / f"seed_{seed:03d}.json"
                )
                if failure_path.exists():
                    raise RuntimeError(f"existing failure blocks resume: {failure_path}")
                if record_path.exists():
                    if args.resume:
                        validate_resume_record(
                            record_path,
                            {
                                "run_id": config["run_id"],
                                "dataset": dataset_name,
                                "model": model_id,
                                "seed": seed,
                                "split_id": split_id,
                                "source_commit": commit,
                                "config_sha256": config_digest,
                                "data_provenance": provenance,
                                "test_evaluations_after_selection": 1,
                            },
                            config["training"],
                        )
                        continue
                    raise FileExistsError(record_path)
                try:
                    if model_id == "H2GCN" and h2 is None:
                        h2 = prepare_h2_adjacencies(
                            edge_index, num_nodes=data.num_nodes
                        )
                    record = run_model_unit(
                        run_id=config["run_id"],
                        dataset=dataset_name,
                        model_id=model_id,
                        seed=seed,
                        split_id=split_id,
                        x=x,
                        y=y,
                        edge_index=edge_index,
                        train_indices=torch.from_numpy(split["train"]),
                        validation_indices=torch.from_numpy(split["validation"]),
                        test_indices=torch.from_numpy(split["test"]),
                        training=config["training"],
                        source_commit=commit,
                        environment=environment,
                        data_provenance=provenance,
                        device=device,
                        h2_adjacencies=h2,
                        config_sha256=config_digest,
                        frozen_config=config,
                    )
                    write_json_exclusive(record_path, record)
                    print(
                        json.dumps(
                            {
                                "dataset": dataset_name,
                                "model": model_id,
                                "seed": seed,
                                "status": "success",
                                "output": str(record_path),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                except Exception as exc:
                    failure = {
                        "schema_version": "1.0",
                        "run_id": config["run_id"],
                        "status": "error",
                        "dataset": dataset_name,
                        "model": model_id,
                        "seed": seed,
                        "split_id": split_id,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "traceback": traceback.format_exc(),
                        "source_commit": commit,
                        "environment": environment,
                        "data_provenance": provenance,
                        "config_sha256": config_digest,
                        "frozen_config": config,
                        "command": sys.argv,
                        "elapsed_seconds": time.perf_counter() - run_started,
                    }
                    write_json_exclusive(failure_path, failure)
                    print(json.dumps(failure, sort_keys=True), flush=True)
                    return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
