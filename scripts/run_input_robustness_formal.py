"""Execute the approved input-robustness run through the guarded formal entry.

This is a thin dispatch adapter.  It only builds real units from the frozen
configuration and read-only bound data, then delegates execution, accounting,
checkpointing, validation, and analysis to the existing formal entry.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch

from experiments.prospective_models import prepare_h2_adjacencies
from experiments.prospective_data import canonical_undirected_edges
from experiments.run_prospective_benchmark import environment_snapshot
from scripts.input_robustness_analysis_adapter import adapt_validated_root
from scripts.input_robustness_budget import (
    CAPS_SECONDS, CI_JOBS, LOCAL_VERIFICATION_JOBS, BudgetLedger, canonical_hash, inspect_ledger,
)
from scripts.input_robustness_data import fit_transform_features_bounded, load_bound_dataset
from scripts.input_robustness_formal_entry import FormalExecutionController
from scripts.input_robustness_formal_records import (
    CONDITIONS, DATASETS, MODELS, FormalRecordError, FormalRecordWriter,
    digest, expected_keys, file_digest, validate_record,
)
from scripts.validate_input_robustness_formal_records import validate_complete_run


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FormalRecordError(f"JSON object required: {path}")
    return value


def _source_commit() -> str:
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0:
        raise FormalRecordError("formal source worktree is dirty")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0:
        raise FormalRecordError("formal source index is dirty")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _configure_runtime(config: Mapping[str, Any]) -> None:
    execution = config["execution"]
    torch.set_num_threads(int(execution["torch_num_threads"]))
    torch.set_num_interop_threads(int(execution.get("torch_num_interop_threads", 1)))
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = bool(execution.get("allow_tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(execution.get("allow_tf32", False))
    torch.backends.cudnn.benchmark = bool(execution.get("cudnn_benchmark", False))
    torch.backends.cudnn.deterministic = bool(execution.get("cudnn_deterministic", True))


def _ci_gate(ci_path: Path, commit: str) -> dict[str, Any]:
    ci = _read(ci_path)
    payload = ci.get("ci", {})
    jobs = payload.get("jobs", [])
    names = {row.get("name") for row in jobs}
    required = (LOCAL_VERIFICATION_JOBS if payload.get("kind") == "local_verification" else CI_JOBS)
    if (ci.get("status") != "passed" or ci.get("commit") != commit
            or payload.get("status") != "completed" or payload.get("conclusion") != "success"
            or payload.get("commit") != commit or len(jobs) != len(required) or names != required
            or any(row.get("run_id") != payload.get("run_id")
                   or row.get("status") != "completed"
                   or row.get("conclusion") != "success"
                   or row.get("commit") != commit for row in jobs)):
        raise FormalRecordError("same-SHA CI receipt is incomplete or does not bind this source")
    return payload


def _phase_gate(*, acceptance_path: Path, ci_path: Path, commit: str,
                config_sha: str, binding_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    acceptance = _read(acceptance_path)
    if acceptance.get("execution_commit") != commit or acceptance.get("technical_gates_passed") is not True:
        raise FormalRecordError("formal launch acceptance does not bind the current source")
    ci = _ci_gate(ci_path, commit)
    source_files = {}
    for relative in (
        "scripts/input_robustness_formal_entry.py",
        "scripts/input_robustness_formal_records.py",
        "scripts/input_robustness_formal_worker.py",
        "scripts/input_robustness_training_core.py",
        "scripts/input_robustness_checkpoint_store.py",
        "scripts/input_robustness_runtime.py",
        "scripts/input_robustness_budget.py",
        "scripts/validate_input_robustness_formal_records.py",
        "scripts/input_robustness_analysis_adapter.py",
        "scripts/run_input_robustness_formal.py",
    ):
        source_files[relative] = file_digest(ROOT / relative)
    provenance = {
        "source_commit": commit, "config_sha256": config_sha,
        "data_binding_sha256": binding_sha, "source_files": source_files,
        "environment": _environment_binding(),
    }
    receipt = {
        "run_id": ci["run_id"], "status": ci["status"], "conclusion": ci["conclusion"],
        "commit": ci["commit"],
        "jobs": [{"name": row["name"], "run_id": row["run_id"],
                  "status": row["status"], "conclusion": row["conclusion"]} for row in ci["jobs"]],
    }
    if ci.get("kind") is not None:
        receipt["kind"] = ci["kind"]
    gate = {
        "ci_receipt_file_sha256": file_digest(ci_path),
        "review_record_sha256": file_digest(acceptance_path),
        "ci_receipt": receipt,
    }
    return provenance, gate


def _environment_binding() -> dict[str, Any]:
    """Runtime identity that a resumed segment must reproduce exactly."""
    environment: dict[str, Any] = {
        "python": sys.version.split()[0], "torch": torch.__version__, "device_policy": "config-bound",
        "torch_cuda": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        environment["cuda_device"] = torch.cuda.get_device_name(0)
        try:
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True, timeout=30,
            ).strip().splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError, IndexError) as exc:
            raise FormalRecordError("CUDA driver version is required to bind a resumable run") from exc
        environment["cuda_driver"] = driver
    return environment


# ---------------------------------------------------------------------------
# Pause and resume.  A run is split into ledger segments: every invocation
# creates a fresh append-only ledger segment, while earlier segments are only
# inspected.  An attempt left open by a sleep, crash or power loss therefore
# stays open (and conservatively charged) in its own segment instead of
# blocking the next invocation.  Records remain immutable; a resumed
# invocation only adds the missing ones.

PAUSE_FILE = "PAUSE_REQUESTED"
_SEGMENT_PATTERN = re.compile(r"^segment_(\d{3})$")
_ATTEMPT_PATTERN = re.compile(r"attempt_(\d{6})")


def _segments(parent: Path) -> list[Path]:
    if not parent.exists():
        return []
    return sorted(path for path in parent.iterdir() if path.is_dir() and _SEGMENT_PATTERN.match(path.name))


def _next_segment(parent: Path) -> Path:
    existing = _segments(parent)
    number = int(_SEGMENT_PATTERN.match(existing[-1].name).group(1)) + 1 if existing else 1
    return parent / f"segment_{number:03d}"


def _prior_segment_usage(parent: Path, authority: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only, conservative charge of all earlier segments."""
    rows = []
    totals = {"formal": 0.0, "total": 0.0}
    for segment in _segments(parent):
        inspection = inspect_ledger(segment, authority=dict(authority))
        charged = inspection["conservative_chargeable_seconds_through_inspection"]
        rows.append({"segment": segment.name, "head": inspection["head"],
                     "open_attempts": [row["attempt_id"] for row in inspection["open_attempts"]],
                     "formal_seconds": charged["formal"], "total_seconds": charged["total"]})
        totals["formal"] += charged["formal"]
        totals["total"] += charged["total"]
    return {"segments": rows, "formal_seconds": totals["formal"], "total_seconds": totals["total"]}


def _next_attempt_number(root: Path) -> int:
    """Attempt identities never repeat within one output root."""
    numbers = [-1]
    for sub in ("workers", "failures", "checkpoints"):
        base = root / sub
        if base.exists():
            numbers.extend(int(match.group(1)) for path in base.rglob("*")
                           for match in [_ATTEMPT_PATTERN.search(path.name)] if match)
    return max(numbers) + 1


def _completed_keys(writer: FormalRecordWriter) -> set[tuple[str, str, str, int]]:
    """Validated identities already present; every stored record must validate."""
    completed = set()
    for path in sorted((writer.root / "records").rglob("*.json")) if (writer.root / "records").exists() else []:
        record = _read(path)
        validate_record(record, expected=writer.manifest, synthetic=False)
        key = (record["dataset"], record["condition"], record["model"], int(record["seed"]))
        expected_path = writer.root / "records" / key[1] / key[0] / key[2] / f"seed_{key[3]:03d}.json"
        if path.resolve() != expected_path.resolve():
            raise FormalRecordError(f"record stored under a different identity: {path}")
        completed.add(key)
    return completed


def _pausable(units: Iterable[dict[str, Any]], root: Path,
              controller: Callable[[], Any]) -> Iterable[dict[str, Any]]:
    """Stop yielding at a whole-unit boundary once the pause file exists."""
    for unit in units:
        if (root / PAUSE_FILE).exists():
            controller().request_normal_pause()
            return
        yield unit


def _units(*, config: Mapping[str, Any], binding_path: Path, data_root: Path,
           output_root: Path, commit: str, config_sha: str, binding_sha: str,
           datasets: Iterable[str], seeds: Iterable[int], models: Iterable[str],
           skip: frozenset[tuple[str, str, str, int]] | set[tuple[str, str, str, int]] = frozenset(),
           first_attempt: int = 0) -> Iterable[dict[str, Any]]:
    binding = _read(binding_path)
    training = dict(config["training"])
    estimate = float(config["budget_accounting"]["formal_model_unit_all_attempts_seconds"]
                     if "formal_model_unit_all_attempts_seconds" in config["budget_accounting"]
                     else config["budget_accounting"]["four_trial_unit_all_attempts_seconds"])
    # The approved cap is per complete model unit.  A model unit contains both
    # paired conditions only in the batch accounting; each attempt remains one
    # condition and four trials, so use the explicit unit cap.
    estimate = float(config.get("resource_budget", {}).get("formal_model_unit_wall_seconds", 28800))
    counter = int(first_attempt)
    for dataset in datasets:
        if all((dataset, condition, model, int(seed)) in skip
               for seed in seeds for model in models for condition in CONDITIONS):
            continue
        raw_x, y, edge_index, splits, entry = load_bound_dataset(dataset, data_root, binding)
        h2 = None
        for seed in seeds:
            split, split_id = splits[int(seed)]
            train = torch.from_numpy(split["train"])
            diagnostics = next(row["diagnostics"] for row in entry["full_diagnostic_checks"]
                               if int(row["seed"]) == int(seed))
            for model in models:
                device_name = config["execution"]["model_devices"][model]
                device = torch.device(device_name)
                for condition in CONDITIONS:
                    if (dataset, condition, model, int(seed)) in skip:
                        continue
                    if model == "H2GCN" and h2 is None:
                        h2 = prepare_h2_adjacencies(edge_index, num_nodes=int(raw_x.size(0)))
                    transformed, fit_metadata = fit_transform_features_bounded(raw_x, train, condition)
                    candidate = next((row for row in entry.get("candidate_checks", [])
                                      if int(row.get("seed", -1)) == int(seed)
                                      and row.get("fit_metadata", {}).get("condition") == condition), None)
                    transform_sha = (entry["normalize_features_sha256"] if condition == CONDITIONS[0]
                                     else candidate["transformed_feature_sha256"])
                    transform_binding = {
                        "dataset": dataset, "condition": condition, "seed": int(seed),
                        "split_id": split_id, "transformed_feature_sha256": transform_sha,
                        "fit_statistics_sha256": digest(fit_metadata),
                    }
                    batch = f"pair_{dataset}_seed_{int(seed):03d}_{model}"
                    unit = {
                        "run_id": config["run_id"], "dataset": dataset, "condition": condition,
                        "model_id": model, "seed": int(seed), "split_id": split_id,
                        "x": transformed, "y": y, "edge_index": edge_index,
                        "train_indices": train, "validation_indices": torch.from_numpy(split["validation"]),
                        "test_indices": torch.from_numpy(split["test"]), "training": training,
                        "source_commit": commit, "environment": environment_snapshot(device),
                        "data_provenance": {"binding_path": str(binding_path.resolve()),
                                            "dataset_binding_sha256": digest(entry),
                                            "materialized_sha256": entry["materialized_sha256"]},
                        "device": device, "h2_adjacencies": h2,
                        "config_sha256": config_sha, "frozen_config": dict(config),
                        "data_binding_sha256": binding_sha, "transform_binding": transform_binding,
                        "diagnostics": diagnostics, "estimated_seconds": estimate,
                        "attempt_id": f"attempt_{counter:06d}",
                        "unit_id": f"unit_{dataset}_{condition}_{model}_{int(seed):03d}",
                        "batch_id": batch,
                    }
                    counter += 1
                    yield unit
                    del transformed
        del raw_x, y, edge_index


def _pair_check(writer: FormalRecordWriter, config: Mapping[str, Any], binding_path: Path,
                first_pair: list[Mapping[str, Any]]) -> dict[str, Any]:
    scope = expected_keys(datasets=[first_pair[0]["dataset"]], conditions=CONDITIONS,
                          models=[first_pair[0]["model_id"]], seeds=[first_pair[0]["seed"]])
    seen = set()
    for path in sorted((writer.root / "records").rglob("*.json")):
        record = _read(path)
        validate_record(record, expected=writer.manifest, synthetic=False)
        seen.add((record["dataset"], record["condition"], record["model"], int(record["seed"])))
    expected = {(first_pair[0]["dataset"], condition, first_pair[0]["model_id"], int(first_pair[0]["seed"]))
                for condition in CONDITIONS}
    if seen != expected or len(seen) != 2:
        raise FormalRecordError("first paired batch did not produce exactly both conditions")
    return {"status": "passed", "batch_id": first_pair[0]["batch_id"],
            "identities": sorted([list(key) for key in seen]), "config_sha256": digest(config),
            "data_binding_sha256": file_digest(binding_path)}


def _write_state(root: Path, payload: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run-state.json"
    if path.exists():
        return
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True,
                                 ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_segment_state(root: Path, segment: str, payload: Mapping[str, Any]) -> Path:
    """One immutable end-of-invocation record per ledger segment."""
    directory = root / "segments"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{segment}.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), indent=2, sort_keys=True, default=str,
                                ensure_ascii=False, allow_nan=False) + "\n")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path, required=True,
                        help="parent directory of append-only ledger segments (segment_NNN)")
    parser.add_argument("--acceptance-record", type=Path, required=True)
    parser.add_argument("--ci-receipt", type=Path, required=True)
    parser.add_argument("--launch-authorized", action="store_true")
    parser.add_argument("--formal-training-enabled", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="continue an incomplete output root in a new ledger segment")
    return parser.parse_args()


def _ledger_authority() -> dict[str, Any]:
    authority_file = ROOT / "results/diagnostic/posthoc_input_robustness_11_v2/budget_authorization.json"
    authority = _read(authority_file)
    return {"run_id": authority["run_id"], "authorization_sha256": file_digest(authority_file),
            "proposal_sha256": authority["approved_proposal"]["sha256"],
            "scope_sha256": canonical_hash(authority["scope"])}


def _check_resume_shape(*, resume: bool, output_root: Path, ledger_parent: Path) -> None:
    has_segments = bool(_segments(ledger_parent))
    if resume:
        if not (output_root / "manifest.json").is_file() or not has_segments:
            raise FormalRecordError("--resume requires an existing output root and ledger segments")
    else:
        if output_root.exists() or ledger_parent.exists():
            raise FormalRecordError("a fresh run requires a new output root and a new ledger parent; use --resume to continue")


def main() -> int:
    args = _parse_args()
    if not args.launch_authorized or not args.formal_training_enabled:
        raise SystemExit("formal launch is locked; both explicit authorization flags are required")
    config = _read(args.config)
    if config.get("formal_training_enabled") is not False:
        raise SystemExit("authoritative config must remain unchanged with formal_training_enabled=false")
    commit = _source_commit()
    config_sha, binding_sha = digest(config), file_digest(args.binding)
    _configure_runtime(config)
    provenance, gate = _phase_gate(acceptance_path=args.acceptance_record, ci_path=args.ci_receipt,
                                   commit=commit, config_sha=config_sha, binding_sha=binding_sha)
    ledger_authority = _ledger_authority()
    ledger_parent = Path(args.budget_ledger)
    _check_resume_shape(resume=args.resume, output_root=Path(args.output_root), ledger_parent=ledger_parent)
    prior = _prior_segment_usage(ledger_parent, ledger_authority)
    if prior["formal_seconds"] >= CAPS_SECONDS["formal"] or prior["total_seconds"] >= CAPS_SECONDS["total"]:
        raise FormalRecordError(f"cumulative budget across ledger segments is exhausted: {prior}")
    ledger_parent.mkdir(parents=True, exist_ok=True)
    segment_path = _next_segment(ledger_parent)
    segment = segment_path.name
    ledger = None
    writer = None
    pair_result = None
    final = None
    removed_pause_file = False
    try:
        ledger = BudgetLedger.create(segment_path, authority=ledger_authority)
        phase_id = f"formal_input_robustness_{commit[:12]}"
        ledger.register_phase(phase_id, provenance=provenance, gate_receipt=gate)
        scope = {"datasets": list(config["datasets"]), "conditions": list(config["conditions"]),
                 "models": list(config["models"]), "seeds": list(config["seeds"])}
        manifest = {"run_id": config["run_id"], "source_commit": commit,
                    "config_sha256": config_sha, "data_binding_sha256": binding_sha,
                    "environment": provenance["environment"], "scope": scope,
                    "expected_records": config["expected_records"], "expected_trials": config["expected_trials"]}
        if args.resume:
            writer = FormalRecordWriter.reopen(args.output_root, manifest=manifest,
                                               config_path=args.config, data_binding_path=args.binding)
            if (writer.root / PAUSE_FILE).exists():
                (writer.root / PAUSE_FILE).unlink()
                removed_pause_file = True
        else:
            writer = FormalRecordWriter(args.output_root, manifest=manifest,
                                        config_path=args.config, data_binding_path=args.binding)
        completed = _completed_keys(writer) if args.resume else set()
        first_attempt = _next_attempt_number(writer.root)
        units = _units(config=config, binding_path=args.binding, data_root=args.data_root,
                       output_root=args.output_root, commit=commit, config_sha=config_sha,
                       binding_sha=binding_sha, datasets=config["datasets"], seeds=config["seeds"],
                       models=config["models"], skip=completed, first_attempt=first_attempt)
        first_keys = {(config["datasets"][0], condition, config["models"][0], int(config["seeds"][0]))
                      for condition in CONDITIONS}
        missing_first = first_keys - completed
        if missing_first:
            # The paired gate still runs before any other unit, including on a
            # resume that was interrupted inside the first pair.
            first_pair = list(itertools.islice(units, len(missing_first)))
            got = {(u["dataset"], u["condition"], u["model_id"], int(u["seed"])) for u in first_pair}
            if got != missing_first:
                raise FormalRecordError("frozen dispatch did not start with the first paired conditions")
            gate_pair = [dict(first_pair[0])]
            first = FormalExecutionController(writer=writer, ledger=ledger, phase_id=phase_id)
            pair_result = first.run_units(first_pair, launch_authorized=True,
                                          formal_training_enabled=True, finalize=False,
                                          post_stage=lambda: _pair_check(writer, config, args.binding, gate_pair))
            if pair_result["status"] != "completed" or pair_result["stage_result"] is None:
                raise FormalRecordError(f"first paired batch did not complete and validate: {pair_result['status']}")
        holder: list[FormalExecutionController] = []
        remaining = FormalExecutionController(writer=writer, ledger=ledger, phase_id=phase_id)
        holder.append(remaining)
        final = remaining.run_units(_pausable(units, writer.root, lambda: holder[0]), launch_authorized=True,
                                    formal_training_enabled=True,
                                    post_stage=lambda: adapt_validated_root(
                                        writer.root, config_path=args.config, data_binding_path=args.binding))
        if final["status"] == "paused":
            state = {"status": "paused", "segment": segment, "commit": commit,
                     "records_before_segment": len(completed), "first_attempt": first_attempt,
                     "removed_pause_file_at_start": removed_pause_file, "prior_segments": prior,
                     "units_completed_in_segment": len(final["units"]) + (len(pair_result["units"]) if pair_result else 0),
                     "ledger_snapshot": ledger.snapshot()}
            _write_segment_state(writer.root, segment, state)
            print(json.dumps({"status": "paused", "segment": segment}, sort_keys=True))
            return 0
        if final["status"] != "completed" or final["complete"] is None:
            raise FormalRecordError(f"formal run did not produce a complete record marker: {final['status']}")
        summary = validate_complete_run(writer.root, expected_keys=expected_keys(**scope), synthetic=False,
                                        config_path=args.config, data_binding_path=args.binding)
        result = {"status": "passed", "commit": commit, "record_root": str(writer.root),
                  "record_count": summary["record_count"], "record_digest": summary["record_digest"],
                  "first_pair": pair_result["stage_result"] if pair_result else "completed_in_earlier_segment",
                  "final_segment": segment, "prior_segments": prior}
        _write_segment_state(writer.root, segment, {**result, "ledger_snapshot": ledger.snapshot()})
        _write_state(writer.root, result)
        print(json.dumps(result, sort_keys=True, default=str))
        return 0
    except BaseException as exc:
        interrupted = any(r is not None and r.get("status") == "emergency_stopped" for r in (pair_result, final))
        state = {"status": "stopped" if isinstance(exc, KeyboardInterrupt) or interrupted else "failed",
                 "segment": segment, "commit": commit, "config": str(args.config.resolve()),
                 "binding": str(args.binding.resolve()), "exception": repr(exc),
                 "prior_segments": prior, "pair_result": pair_result, "final_result": final}
        if ledger is not None:
            state["ledger_snapshot"] = ledger.snapshot()
        if writer is not None:
            _write_segment_state(writer.root, segment, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
