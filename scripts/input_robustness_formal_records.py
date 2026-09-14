"""Guarded formal execution and immutable per-model-unit records.

The module deliberately keeps execution separate from the preflight outputs.  A
formal caller must opt in with both launch flags, provide the same provenance
used by the approved configuration, and write each unit exactly once.  The
small record contract is also the input contract for the post-hoc analysis
adapter; a successful training process without these records is not a
successful formal unit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


DATASETS = (
    "Cora", "CiteSeer", "PubMed", "Wisconsin", "Cornell", "Chameleon",
    "Squirrel", "Actor", "Roman-empire", "Amazon-ratings", "Coauthor-CS",
)
CONDITIONS = ("normalize_features", "normalize_centered_scaled")
MODELS = ("MLP", "GCN", "GAT", "GraphSAGE", "H2GCN", "LINKX", "GPR-GNN")
GRAPH_MODELS = tuple(model for model in MODELS if model != "MLP")
TRIAL_IDS = tuple(f"trial_{index:03d}" for index in range(4))
STRATEGIES = (
    "always_graph", "always_mlp", "degree_only", "historical_combined",
    "homophily_only", "homophily_plus_degree", "random_50_50",
    "two_hop_only", "validation_selection",
)
THRESHOLDS = (-1.0,) + tuple(index / 100 for index in range(101)) + (2.0,)
EXPECTED_RECORDS = len(DATASETS) * len(CONDITIONS) * len(MODELS) * 10
EXPECTED_TRIALS = EXPECTED_RECORDS * len(TRIAL_IDS)


class FormalRecordError(ValueError):
    """A record violates the formal execution contract."""


class DuplicateRecordError(FormalRecordError):
    """A unit or manifest would overwrite an existing immutable artifact."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_key(record: Mapping[str, Any]) -> tuple[str, str, str, int]:
    try:
        return (str(record["dataset"]), str(record["condition"]),
                str(record["model"]), int(record["seed"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise FormalRecordError(f"record has no valid unit identity: {exc}") from exc


def expected_keys(*, datasets: Sequence[str] = DATASETS,
                  conditions: Sequence[str] = CONDITIONS,
                  models: Sequence[str] = MODELS,
                  seeds: Sequence[int] = range(10)) -> set[tuple[str, str, str, int]]:
    return {(str(dataset), str(condition), str(model), int(seed))
            for dataset in datasets for condition in conditions
            for model in models for seed in seeds}


def select_trial(trials: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    rows = list(trials)
    if not rows:
        raise FormalRecordError("four trial rows are required")
    try:
        return sorted(rows, key=lambda row: (
            -float(row["validation_accuracy"]), float(row["validation_loss"]),
            str(row["trial_id"]),
        ))[0]
    except (KeyError, TypeError, ValueError) as exc:
        raise FormalRecordError(f"invalid validation selection values: {exc}") from exc


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormalRecordError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FormalRecordError(f"{label} must be finite")
    return result


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise FormalRecordError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise FormalRecordError(f"{label} must be a SHA-256 hex digest") from exc
    return value


def validate_record(record: Mapping[str, Any], *, expected: Mapping[str, Any] | None = None,
                    synthetic: bool = False) -> None:
    """Validate one complete unit without reading or mutating the output tree."""
    required = {
        "schema_version", "run_id", "status", "dataset", "condition", "model", "seed",
        "split_id", "source_commit", "config_sha256", "data_binding_sha256", "environment",
        "transform_binding", "trials", "selected_trial_id", "validation_accuracy",
        "validation_loss", "validation_evaluations", "test_accuracy",
        "test_evaluations_after_selection", "test_evaluation", "checkpoint_mode",
        "checkpoint_manifest", "checkpoint_complete", "record_complete", "diagnostics",
        "execution_mode",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise FormalRecordError(f"record missing fields: {missing}")
    if record["schema_version"] != "1.0" or record["status"] != "success":
        raise FormalRecordError("only schema 1.0 successful records are accepted")
    if record["execution_mode"] not in ("formal", "synthetic_rehearsal"):
        raise FormalRecordError("unknown execution mode")
    if record["execution_mode"] == "synthetic_rehearsal" and not synthetic:
        raise FormalRecordError("synthetic record supplied to the formal validator")
    if not synthetic and (record["execution_mode"] != "formal" or record.get("formal_training_enabled") is not True):
        raise FormalRecordError("formal records must prove that formal training was enabled")
    if expected is not None:
        for field in ("run_id", "source_commit", "config_sha256", "data_binding_sha256"):
            if field in expected and record.get(field) != expected[field]:
                raise FormalRecordError(f"{field} provenance mismatch")
    dataset, condition, model, seed = record_key(record)
    if dataset not in DATASETS or condition not in CONDITIONS or model not in MODELS:
        raise FormalRecordError("unit identity is outside the approved 11-dataset scope")
    if seed < 0 or seed > 9:
        raise FormalRecordError("seed is outside the approved 0..9 scope")
    if not isinstance(record["split_id"], str) or not record["split_id"]:
        raise FormalRecordError("split_id is required")
    if not isinstance(record["source_commit"], str) or len(record["source_commit"]) != 40:
        raise FormalRecordError("source_commit must be a full git SHA")
    _require_hash(record["config_sha256"], "config_sha256")
    _require_hash(record["data_binding_sha256"], "data_binding_sha256")
    if not isinstance(record["environment"], Mapping) or not record["environment"]:
        raise FormalRecordError("execution environment is required")

    binding = record["transform_binding"]
    if not isinstance(binding, Mapping):
        raise FormalRecordError("transform_binding is required")
    for field, value in (("dataset", dataset), ("condition", condition), ("seed", seed),
                         ("split_id", record["split_id"])):
        if binding.get(field) != value:
            raise FormalRecordError(f"transform binding {field} mismatch")
    _require_hash(binding.get("transformed_feature_sha256"), "transformed_feature_sha256")
    _require_hash(binding.get("fit_statistics_sha256"), "fit_statistics_sha256")

    trials = record["trials"]
    if not isinstance(trials, list) or len(trials) != 4:
        raise FormalRecordError("exactly four tuning trials are required")
    seen_trials: set[str] = set()
    configurations = record.get("training_configuration", {}).get("trials")
    for index, trial in enumerate(trials):
        if not isinstance(trial, Mapping):
            raise FormalRecordError("trial row must be an object")
        trial_id = trial.get("trial_id")
        if trial_id != TRIAL_IDS[index] or trial_id in seen_trials:
            raise FormalRecordError("trial identifiers must be trial_000..trial_003 once each")
        seen_trials.add(trial_id)
        _finite_number(trial.get("validation_accuracy"), f"{trial_id}.validation_accuracy")
        _finite_number(trial.get("validation_loss"), f"{trial_id}.validation_loss")
        if any(key.startswith("test_") for key in trial):
            raise FormalRecordError("test results must not be present before selection")
        if configurations is not None and trial.get("configuration") != configurations[index]:
            raise FormalRecordError("trial configuration differs from the frozen four-trial grid")
    selected = select_trial(trials)
    if record["selected_trial_id"] != selected["trial_id"]:
        raise FormalRecordError("selected_trial_id does not follow validation tie rule")
    if not math.isclose(float(record["validation_accuracy"]), float(selected["validation_accuracy"]), abs_tol=1e-12):
        raise FormalRecordError("selected validation accuracy is not copied from the selected trial")
    if not math.isclose(float(record["validation_loss"]), float(selected["validation_loss"]), abs_tol=1e-12):
        raise FormalRecordError("selected validation loss is not copied from the selected trial")
    if record["validation_evaluations"] != 4:
        raise FormalRecordError("four validation-selection evaluations are required")
    if record["test_evaluations_after_selection"] != 1:
        raise FormalRecordError("exactly one post-selection test evaluation is required")
    if (record["test_evaluation"].get("selected_trial_id") != record["selected_trial_id"]
            or record["test_evaluation"].get("count") != 1):
        raise FormalRecordError("test evaluation is not bound to selected trial")
    _finite_number(record["test_accuracy"], "test_accuracy")

    checkpoints = record["checkpoint_manifest"]
    if record["checkpoint_mode"] != "original_full_state_copy" or not record["checkpoint_complete"]:
        raise FormalRecordError("complete original checkpoint copies are required")
    if not isinstance(checkpoints, list) or len(checkpoints) != 4:
        raise FormalRecordError("four checkpoint manifests are required")
    checkpoint_ids = []
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping) or checkpoint.get("saved") is not True:
            raise FormalRecordError("checkpoint is not marked saved")
        if checkpoint.get("trial_id") not in TRIAL_IDS:
            raise FormalRecordError("checkpoint trial identity mismatch")
        if not isinstance(checkpoint.get("path"), str) or not checkpoint["path"]:
            raise FormalRecordError("checkpoint path is required for complete persistence")
        checkpoint_ids.append(checkpoint["trial_id"])
        _require_hash(checkpoint.get("sha256"), "checkpoint.sha256")
        if isinstance(checkpoint.get("bytes"), bool) or not isinstance(checkpoint.get("bytes"), int) or checkpoint["bytes"] <= 0:
            raise FormalRecordError("checkpoint byte count must be positive")
    if tuple(sorted(checkpoint_ids)) != TRIAL_IDS:
        raise FormalRecordError("checkpoint manifest must contain each trial exactly once")
    if record["record_complete"] is not True:
        raise FormalRecordError("record_complete marker is required")
    diagnostics = record["diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise FormalRecordError("train-only diagnostics are required")
    for field in ("homophily", "mean_degree", "two_hop_agreement"):
        _finite_number(diagnostics.get(field), f"diagnostics.{field}")


def build_record(*, run_id: str, dataset: str, condition: str, model: str, seed: int,
                 split_id: str, source_commit: str, config_sha256: str,
                 data_binding_sha256: str, environment: Mapping[str, Any],
                 transform_binding: Mapping[str, Any], training_configuration: Mapping[str, Any],
                 trials: Sequence[Mapping[str, Any]], test_accuracy: float,
                 checkpoint_manifest: Sequence[Mapping[str, Any]], diagnostics: Mapping[str, Any],
                 synthetic: bool = False, duration_seconds: float = 0.0) -> dict[str, Any]:
    """Construct the immutable envelope around core training output."""
    selected = select_trial(trials)
    record = {
        "schema_version": "1.0", "run_id": run_id, "status": "success",
        "dataset": dataset, "condition": condition, "model": model, "seed": int(seed),
        "split_id": split_id, "source_commit": source_commit,
        "config_sha256": config_sha256, "data_binding_sha256": data_binding_sha256,
        "environment": dict(environment), "transform_binding": dict(transform_binding),
        "training_configuration": dict(training_configuration), "trials": [dict(row) for row in trials],
        "selected_trial_id": selected["trial_id"],
        "validation_accuracy": float(selected["validation_accuracy"]),
        "validation_loss": float(selected["validation_loss"]), "validation_evaluations": 4,
        "test_accuracy": float(test_accuracy), "test_evaluations_after_selection": 1,
        "test_evaluation": {"selected_trial_id": selected["trial_id"], "count": 1},
        "checkpoint_mode": "original_full_state_copy", "checkpoint_manifest": [dict(row) for row in checkpoint_manifest],
        "checkpoint_complete": True, "record_complete": True, "diagnostics": dict(diagnostics),
        "execution_mode": "synthetic_rehearsal" if synthetic else "formal",
        "formal_training_enabled": not synthetic, "duration_seconds": float(duration_seconds),
    }
    validate_record(record, synthetic=synthetic)
    return record


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise DuplicateRecordError(f"immutable artifact already exists: {path}") from exc
    try:
        data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        written = 0
        while written < len(data):
            count = os.write(fd, data[written:])
            if count <= 0:
                raise OSError("exclusive JSON write made no progress")
            written += count
        os.fsync(fd)
    finally:
        os.close(fd)


class FormalRecordWriter:
    """Exclusive writer for one formal or synthetic rehearsal output root."""

    def __init__(self, root: Path, *, manifest: Mapping[str, Any], synthetic: bool = False):
        self.root = Path(root)
        self.synthetic = bool(synthetic)
        if self.root.exists():
            raise DuplicateRecordError(f"formal output root must be new: {self.root}")
        self.root.mkdir(parents=True)
        self.manifest = dict(manifest)
        self.manifest.setdefault("schema_version", "1.0")
        self.manifest.setdefault("execution_mode", "synthetic_rehearsal" if synthetic else "formal")
        _exclusive_json(self.root / "manifest.json", self.manifest)

    def write_record(self, record: Mapping[str, Any]) -> Path:
        validate_record(record, expected=self.manifest, synthetic=self.synthetic)
        dataset, condition, model, seed = record_key(record)
        path = self.root / "records" / condition / dataset / model / f"seed_{seed:03d}.json"
        _exclusive_json(path, record)
        return path

    def write_failure(self, failure: Mapping[str, Any], *, identity: str) -> Path:
        """Preserve a failed attempt without making it part of accepted records."""
        path = self.root / "failures" / f"{identity}.json"
        _exclusive_json(path, dict(failure))
        return path

    def finalize(self, *, expected: set[tuple[str, str, str, int]] | None = None) -> dict[str, Any]:
        from scripts.validate_input_robustness_formal_records import validate_run
        # The complete marker is the final artifact, so validate the records
        # first and then bind that marker to the returned digest.
        result = validate_run(self.root, expected_keys=expected, synthetic=self.synthetic,
                              require_complete=False)
        complete = {
            "schema_version": "1.0", "status": "complete", "run_id": self.manifest.get("run_id"),
            "expected_records": result["record_count"], "expected_trials": result["trial_count"],
            "record_digest": result["record_digest"], "synthetic": self.synthetic,
        }
        _exclusive_json(self.root / "complete.json", complete)
        return complete


def run_formal_model_unit(*, writer: FormalRecordWriter, core_runner: Callable[..., Mapping[str, Any]] | None = None,
                          launch_authorized: bool, formal_training_enabled: bool,
                          checkpoint_manifest: Sequence[Mapping[str, Any]], **runner_kwargs: Any) -> dict[str, Any]:
    """Run one existing-core unit only after the explicit formal launch gate.

    ``core_runner`` is the existing ``run_model_unit`` in production and a
    fixed synthetic function in tests.  The caller supplies the four persisted
    checkpoint manifests produced by its checkpoint sink; the wrapper then
    binds them to the selected/test record and writes it exclusively.
    """
    if launch_authorized is not True or formal_training_enabled is not True:
        raise FormalRecordError("formal model execution is locked until formal launch authorization")
    if core_runner is None:
        # Keep the established training implementation as the production
        # default; tests inject a fixed runner and therefore never train.
        from experiments.run_prospective_benchmark import run_model_unit
        core_runner = run_model_unit
    record = dict(core_runner(**runner_kwargs))
    record["checkpoint_manifest"] = [dict(item) for item in checkpoint_manifest]
    record["checkpoint_mode"] = "original_full_state_copy"
    record["checkpoint_complete"] = True
    record["record_complete"] = True
    record["validation_evaluations"] = 4
    record["test_evaluation"] = {"selected_trial_id": record.get("selected_trial_id"), "count": 1}
    record["execution_mode"] = "formal"
    record["formal_training_enabled"] = True
    writer.write_record(record)
    return record


class FormalUnitRunner:
    """Connect one existing-core unit to the shared cumulative budget ledger.

    The ledger remains the owner of clocks, caps, pause boundaries and
    cross-attempt accounting.  This adapter only supplies the unit identity and
    binds a successful exclusive record hash to ``attempt_closed``; it never
    enables the formal switch itself.
    """

    def __init__(self, *, writer: FormalRecordWriter, ledger: Any, phase_id: str,
                 activity_id: str = "input_robustness_formal", batch_id: str | None = None):
        self.writer, self.ledger, self.phase_id = writer, ledger, phase_id
        self.activity_id, self.batch_id = activity_id, batch_id

    def run(self, *, attempt_id: str, unit_id: str, estimated_seconds: float,
            launch_authorized: bool, formal_training_enabled: bool,
            core_runner: Callable[..., Mapping[str, Any]] | None = None,
            checkpoint_manifest: Sequence[Mapping[str, Any]], **runner_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if launch_authorized is not True or formal_training_enabled is not True:
            raise FormalRecordError("formal model execution is locked until formal launch authorization")
        self.ledger.begin_attempt(
            attempt_id, activity_id=self.activity_id, phase_id=self.phase_id,
            budget_group="formal", estimated_seconds=estimated_seconds,
            unit_id=unit_id, batch_id=self.batch_id,
        )
        try:
            record = run_formal_model_unit(
                writer=self.writer, core_runner=core_runner,
                launch_authorized=launch_authorized,
                formal_training_enabled=formal_training_enabled,
                checkpoint_manifest=checkpoint_manifest, **runner_kwargs,
            )
            path = self.writer.root / "records" / record["condition"] / record["dataset"] / record["model"] / f"seed_{int(record['seed']):03d}.json"
            snapshot = self.ledger.close_attempt(attempt_id, outcome="completed", record_sha256=file_digest(path))
            return record, snapshot
        except KeyboardInterrupt:
            # Emergency stop deliberately leaves the open attempt for the
            # existing ledger recovery/review path; no fake completion is made.
            raise
        except BaseException as exc:
            self.writer.write_failure({"status": "failed", "attempt_id": attempt_id,
                                       "unit_id": unit_id, "reason": repr(exc)}, identity=attempt_id)
            snapshot = self.ledger.close_attempt(attempt_id, outcome="failed", reason=repr(exc))
            raise
