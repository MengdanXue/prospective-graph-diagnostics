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


class FormalEmergencyStop(KeyboardInterrupt):
    """Emergency stop raised by the formal scheduler at a safe poll boundary."""


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


def _optional_finite_number(value: Any, label: str) -> None:
    if value is not None:
        _finite_number(value, label)


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
    for field in ("homophily", "mean_degree", "delta_h"):
        _optional_finite_number(diagnostics.get(field), f"diagnostics.{field}")


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

    def __init__(self, root: Path, *, manifest: Mapping[str, Any], synthetic: bool = False,
                 config_path: Path | None = None, data_binding_path: Path | None = None):
        self.root = Path(root)
        self.synthetic = bool(synthetic)
        self.config_path = Path(config_path) if config_path is not None else None
        self.data_binding_path = Path(data_binding_path) if data_binding_path is not None else None
        if not self.synthetic and (self.config_path is None or self.data_binding_path is None):
            raise FormalRecordError(
                "formal record writers require the authoritative config and data-binding paths"
            )
        if not self.synthetic:
            if not self.config_path.is_file() or not self.data_binding_path.is_file():
                raise FormalRecordError("authoritative config and data-binding files must exist")
        if self.root.exists():
            raise DuplicateRecordError(f"formal output root must be new: {self.root}")
        self.root.mkdir(parents=True)
        self.manifest = dict(manifest)
        self.manifest.setdefault("schema_version", "1.0")
        self.manifest.setdefault("execution_mode", "synthetic_rehearsal" if synthetic else "formal")
        if not self.synthetic:
            self.manifest.setdefault("authoritative_config", str(self.config_path.resolve()))
            self.manifest.setdefault("authoritative_data_binding", str(self.data_binding_path.resolve()))
        _exclusive_json(self.root / "manifest.json", self.manifest)

    def write_record(self, record: Mapping[str, Any]) -> Path:
        validate_record(record, expected=self.manifest, synthetic=self.synthetic)
        if not self.synthetic:
            from scripts.input_robustness_checkpoint_store import verify_checkpoint
            for checkpoint in record["checkpoint_manifest"]:
                verify_checkpoint(self.root, checkpoint)
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
        result = validate_run(
            self.root, expected_keys=expected, synthetic=self.synthetic,
            require_complete=False, config_path=self.config_path,
            data_binding_path=self.data_binding_path,
        )
        complete = {
            "schema_version": "1.0", "status": "complete", "run_id": self.manifest.get("run_id"),
            "expected_records": result["record_count"], "expected_trials": result["trial_count"],
            "record_digest": result["record_digest"], "synthetic": self.synthetic,
        }
        _exclusive_json(self.root / "complete.json", complete)
        return complete


def run_formal_model_unit(*, writer: FormalRecordWriter, core_runner: Callable[..., Mapping[str, Any]] | None = None,
                          launch_authorized: bool, formal_training_enabled: bool,
                          checkpoint_manifest: Sequence[Mapping[str, Any]] | None = None,
                          transform_binding: Mapping[str, Any] | None = None,
                          diagnostics: Mapping[str, Any] | None = None,
                          monitor_callback: Callable[[], Any] | None = None,
                          **runner_kwargs: Any) -> dict[str, Any]:
    """Run one existing-core unit only after the explicit formal launch gate.

    ``core_runner`` is the existing ``run_model_unit`` in production and a
    fixed synthetic function in tests.  The caller supplies the four persisted
    checkpoint manifests produced by its checkpoint sink; the wrapper then
    binds them to the selected/test record and writes it exclusively.
    """
    if launch_authorized is not True or formal_training_enabled is not True:
        raise FormalRecordError("formal model execution is locked until formal launch authorization")
    using_default_core = core_runner is None
    if using_default_core:
        # Keep the public benchmark runner frozen; the formal adapter calls
        # the same trial/model primitives through its checkpoint-aware layer.
        from scripts.input_robustness_training_core import run_model_unit_with_checkpoints
        core_runner = run_model_unit_with_checkpoints
    if "checkpoint_dir" not in runner_kwargs and all(key in runner_kwargs for key in ("condition", "dataset", "model_id", "seed")):
        runner_kwargs["checkpoint_dir"] = (
            writer.root.resolve() / "checkpoints" / str(runner_kwargs["condition"]) /
            str(runner_kwargs["dataset"]) / str(runner_kwargs["model_id"]) /
            f"seed_{int(runner_kwargs['seed']):03d}"
        )
    call_kwargs = dict(runner_kwargs)
    if using_default_core:
        call_kwargs.pop("condition", None)
        call_kwargs.pop("data_binding_sha256", None)
        call_kwargs.pop("transform_binding", None)
        call_kwargs.pop("diagnostics", None)
        if monitor_callback is not None:
            call_kwargs["monitor"] = monitor_callback
    elif monitor_callback is not None:
        # Custom cores used by callers may opt into the same heartbeat hook;
        # the hook is kept out of the formal record and is therefore explicit.
        call_kwargs.setdefault("monitor", monitor_callback)
    record = dict(core_runner(**call_kwargs))
    if "condition" not in record and "condition" in runner_kwargs:
        record["condition"] = runner_kwargs["condition"]
    if "data_binding_sha256" not in record and "data_binding_sha256" in runner_kwargs:
        record["data_binding_sha256"] = runner_kwargs["data_binding_sha256"]
    if transform_binding is not None:
        record["transform_binding"] = dict(transform_binding)
    elif "transform_binding" in runner_kwargs:
        record["transform_binding"] = dict(runner_kwargs["transform_binding"])
    if diagnostics is not None:
        record["diagnostics"] = dict(diagnostics)
    elif "diagnostics" in runner_kwargs:
        record["diagnostics"] = dict(runner_kwargs["diagnostics"])
    manifests = checkpoint_manifest if checkpoint_manifest is not None else record.get("checkpoint_manifest")
    if not isinstance(manifests, Sequence):
        raise FormalRecordError("formal core must return four checkpoint manifests")
    normalized = []
    checkpoint_dir = runner_kwargs.get("checkpoint_dir")
    checkpoint_prefix = Path()
    if checkpoint_dir is not None:
        try:
            checkpoint_prefix = Path(checkpoint_dir).resolve().relative_to(writer.root.resolve())
        except ValueError as exc:
            raise FormalRecordError("checkpoint_dir must be inside the formal output root") from exc
    for item in manifests:
        row = dict(item)
        path = row.get("path")
        if isinstance(path, str):
            path_obj = Path(path)
            if checkpoint_prefix and not path_obj.is_absolute():
                row["path"] = (checkpoint_prefix / path_obj).as_posix()
            elif not path_obj.is_absolute() and not path.replace("\\", "/").startswith("checkpoints/"):
                row["path"] = f"checkpoints/{path}"
        normalized.append(row)
    record["checkpoint_manifest"] = normalized
    record["checkpoint_mode"] = "original_full_state_copy"
    record["checkpoint_complete"] = True
    record["record_complete"] = True
    record["validation_evaluations"] = 4
    record["test_evaluation"] = {"selected_trial_id": record.get("selected_trial_id"), "count": 1}
    record["execution_mode"] = "formal"
    record["formal_training_enabled"] = True
    writer.write_record(record)
    return record


def run_rehearsal_model_unit(*, writer: FormalRecordWriter,
                             core_runner: Callable[..., Mapping[str, Any]] | None = None,
                             condition: str, diagnostics: Mapping[str, Any],
                             **runner_kwargs: Any) -> dict[str, Any]:
    """Execute the real training core for a synthetic fixture in isolation.

    This path never sets either formal launch flag and writes only to a writer
    explicitly created with ``synthetic=True``.  It still performs all four
    trials, persists/reloads checkpoints through the core's checkpoint sink,
    and exercises record validation and summary generation.
    """
    if not writer.synthetic:
        raise FormalRecordError("rehearsal output must use a synthetic writer")
    if condition not in CONDITIONS:
        raise FormalRecordError("unknown approved condition")
    if core_runner is None:
        from scripts.input_robustness_training_core import run_model_unit_with_checkpoints
        core_runner = run_model_unit_with_checkpoints
    runner_kwargs = dict(runner_kwargs)
    runner_kwargs.setdefault("checkpoint_dir", writer.root / "checkpoints" / condition /
                            str(runner_kwargs.get("dataset")) / str(runner_kwargs.get("model_id")) /
                            f"seed_{int(runner_kwargs.get('seed', 0)):03d}")
    call_kwargs = dict(runner_kwargs)
    call_kwargs.pop("condition", None)
    call_kwargs.pop("data_binding_sha256", None)
    call_kwargs.pop("diagnostics", None)
    core_record = dict(core_runner(**call_kwargs))
    manifests = core_record.get("checkpoint_manifest")
    if not isinstance(manifests, Sequence) or len(manifests) != 4:
        raise FormalRecordError("real core did not return four persisted checkpoints")
    prefix = Path(runner_kwargs["checkpoint_dir"]).resolve().relative_to(writer.root.resolve())
    manifests = [{**dict(item), "path": (prefix / str(item["path"])).as_posix()} for item in manifests]
    record = build_record(
        run_id=str(core_record["run_id"]), dataset=str(core_record["dataset"]), condition=condition,
        model=str(core_record["model"]), seed=int(core_record["seed"]), split_id=str(core_record["split_id"]),
        source_commit=str(core_record["source_commit"]), config_sha256=str(core_record["config_sha256"]),
        data_binding_sha256=str(core_record.get("data_binding_sha256", "c" * 64)),
        environment=core_record["environment"], transform_binding={
            "dataset": str(core_record["dataset"]), "condition": condition, "seed": int(core_record["seed"]),
            "split_id": str(core_record["split_id"]), "transformed_feature_sha256": "d" * 64,
            "fit_statistics_sha256": "e" * 64,
        }, training_configuration=core_record["training_configuration"], trials=core_record["trials"],
        test_accuracy=float(core_record["test_accuracy"]), checkpoint_manifest=manifests,
        diagnostics=diagnostics, synthetic=True, duration_seconds=float(core_record.get("duration_seconds", 0.0)),
    )
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
            checkpoint_manifest: Sequence[Mapping[str, Any]] | None = None,
            monitor_callback: Callable[[], Any] | None = None,
            **runner_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if launch_authorized is not True or formal_training_enabled is not True:
            raise FormalRecordError("formal model execution is locked until formal launch authorization")
        if monitor_callback is not None:
            monitor_callback()
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
                checkpoint_manifest=checkpoint_manifest,
                monitor_callback=monitor_callback, **runner_kwargs,
            )
            path = self.writer.root / "records" / record["condition"] / record["dataset"] / record["model"] / f"seed_{int(record['seed']):03d}.json"
            snapshot = self.ledger.close_attempt(attempt_id, outcome="completed", record_sha256=file_digest(path))
            return record, snapshot
        except FormalEmergencyStop as exc:
            # Preserve the unfinished attempt for the ledger's external review
            # path.  A failure artifact records why dispatch stopped, while no
            # close event is fabricated for a partially executed unit.
            self.writer.write_failure({"status": "external_interruption",
                                       "attempt_id": attempt_id,
                                       "unit_id": unit_id,
                                       "reason": repr(exc)}, identity=attempt_id)
            raise
        except KeyboardInterrupt:
            # Emergency stop deliberately leaves the open attempt for the
            # existing ledger recovery/review path; no fake completion is made.
            raise
        except BaseException as exc:
            self.writer.write_failure({"status": "failed", "attempt_id": attempt_id,
                                       "unit_id": unit_id, "reason": repr(exc)}, identity=attempt_id)
            snapshot = self.ledger.close_attempt(attempt_id, outcome="failed", reason=repr(exc))
            raise
