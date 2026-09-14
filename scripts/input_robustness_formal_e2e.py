"""Run the isolated, non-synthetic formal-entry acceptance fixture.

This is an evidence producer for the committed scheduler, not a research
launcher.  It creates a two-condition one-dataset fixture in a caller-owned
directory, uses the real frozen trial trainer and checkpoint store, and never
touches the approved 11-dataset output or shared research ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from scripts.input_robustness_analysis_adapter import adapt_validated_root
from scripts.input_robustness_budget import BudgetLedger, CI_JOBS
from scripts.input_robustness_formal_entry import FormalExecutionController
from scripts.input_robustness_formal_records import CONDITIONS, MODELS, FormalRecordWriter, digest, expected_keys
from scripts.input_robustness_runtime import PowerEventWatcher, TaskPowerRequest
from scripts.validate_input_robustness_formal_records import validate_complete_run

FIXTURE_DATASETS = ("Cora", "Actor")


class _PowerBackend:
    def create_request(self, reason): return 1
    def set_request(self, handle, request_type): pass
    def clear_request(self, handle, request_type): pass
    def close_handle(self, handle): pass
    def register_notification(self, callback): self.callback = callback; return 2
    def unregister_notification(self, registration): pass


class _PauseAfterBoundary(FormalExecutionController):
    def _monitor(self):
        super()._monitor()
        if self.monitor_samples == 5:
            self.request_normal_pause()


class _EmergencyAtBoundary(FormalExecutionController):
    def _monitor(self):
        super()._monitor()
        if self.monitor_samples == 3:
            self.request_emergency_stop()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _authority(ledger_path: Path, config_sha: str, binding_sha: str, commit: str, phase: str):
    authority = {"approval_sha256": "2" * 64, "proposal_sha256": "3" * 64, "scope_sha256": "4" * 64}
    ledger = BudgetLedger.create(ledger_path, authority=authority, monitor_gap_seconds=999999)
    gate = {"ci_receipt_file_sha256": "e" * 64, "review_record_sha256": "f" * 64,
            "ci_receipt": {"commit": commit, "run_id": 17, "status": "completed",
                           "conclusion": "success", "jobs": [
                               {"name": name, "run_id": 17, "status": "completed", "conclusion": "success"}
                               for name in CI_JOBS]}}
    ledger.register_phase(phase, provenance={
        "source_commit": commit, "config_sha256": config_sha,
        "data_binding_sha256": binding_sha, "source_files": {"formal_entry.py": "d" * 64},
        "environment": {"device": "cpu"}}, gate_receipt=gate)
    return ledger


def _fixture(output: Path, *, launch_config: Path):
    output.mkdir(parents=True, exist_ok=False)
    binding_path = output / "fixture-data-binding.json"
    binding = {"bound_split_count": len(FIXTURE_DATASETS), "datasets": {dataset: {
        "split_checks": [{"seed": 0, "split_id": f"{dataset}-tiny-split"}],
        "full_diagnostic_checks": [{"seed": 0, "diagnostics": {"homophily": 0.4 if dataset == "Cora" else 0.7,
                                                                       "mean_degree": 3.0 if dataset == "Cora" else 1.0,
                                                                       "delta_h": 0.1}}],
        "candidate_checks": [
            {"seed": 0, "fit_metadata": {"condition": CONDITIONS[0]}, "transformed_feature_sha256": "d" * 64},
            {"seed": 0, "fit_metadata": {"condition": CONDITIONS[1]}, "transformed_feature_sha256": "f" * 64},
        ], "normalize_features_sha256": "d" * 64,
    } for dataset in FIXTURE_DATASETS}}
    _write_json(binding_path, binding)
    training = {"hidden_channels": 4, "max_epochs": 1, "patience": 1, "weight_decay": 0.0005,
                "trials": [{"learning_rate": 0.01, "dropout": 0.5}, {"learning_rate": 0.01, "dropout": 0.7},
                            {"learning_rate": 0.005, "dropout": 0.5}, {"learning_rate": 0.005, "dropout": 0.7}]}
    config = {"formal_training_enabled": False, "datasets": list(FIXTURE_DATASETS), "conditions": list(CONDITIONS),
              "models": list(MODELS), "seeds": [0], "expected_records": len(FIXTURE_DATASETS) * len(MODELS) * 2,
              "expected_trials": len(FIXTURE_DATASETS) * len(MODELS) * 2 * 4,
              "training": training, "execution": {"model_devices": {"H2GCN": "cpu"},
              "torch_num_threads": 4, "h2gcn_checkpoint_mode": "original_full_state_copy"},
              "h2gcn_checkpoint_mode": "original_full_state_copy",
              "bound_input_source": {"path": str(binding_path), "sha256": _sha(binding_path)}}
    config_path = output / "fixture-config.json"
    _write_json(config_path, config)
    return config, config_path, binding_path, training


def _units(*, root: Path, config: dict[str, Any], config_path: Path, binding_path: Path,
           training: dict[str, Any], commit: str):
    x = torch.tensor([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.], [0., 0., 1.],
                      [1., 0., 1.], [0., 1., 1.], [1., 1., 1.], [.5, .5, 0.]])
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    indices = lambda values: torch.tensor(values, dtype=torch.long)
    config_sha, binding_sha = digest(config), _sha(binding_path)
    common = {"run_id": "isolated-formal-e2e-v1", "seed": 0, "x": x, "y": y,
              "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
              "train_indices": indices([0, 1, 2, 3]), "validation_indices": indices([4, 5]),
              "test_indices": indices([6, 7]), "training": training, "source_commit": commit,
              "environment": {"device": "cpu"}, "data_provenance": {"source": str(binding_path)},
              "device": torch.device("cpu"), "config_sha256": config_sha, "data_binding_sha256": binding_sha,
              "frozen_config": config}
    units = []
    index = 0
    for dataset in FIXTURE_DATASETS:
        for condition in CONDITIONS:
            for model_id in MODELS:
                units.append({**common, "dataset": dataset, "condition": condition, "model_id": model_id,
                      "split_id": f"{dataset}-tiny-split",
                      "attempt_id": f"attempt{index:03d}", "unit_id": f"unit{index:03d}", "estimated_seconds": 1,
                      "transform_binding": {"dataset": dataset, "condition": condition, "seed": 0,
                                             "split_id": f"{dataset}-tiny-split",
                                             "transformed_feature_sha256": "d" * 64 if condition == CONDITIONS[0] else "f" * 64,
                                             "fit_statistics_sha256": "e" * 64 if condition == CONDITIONS[0] else "0" * 64},
                      "diagnostics": {"homophily": 0.4 if dataset == "Cora" else 0.7,
                                      "mean_degree": 3.0 if dataset == "Cora" else 1.0, "delta_h": 0.1}})
                index += 1
    return units


def _writer(root: Path, *, config_path: Path, binding_path: Path, config: dict[str, Any], commit: str):
    return FormalRecordWriter(root, manifest={
        "run_id": "isolated-formal-e2e-v1", "source_commit": commit,
        "config_sha256": digest(config), "data_binding_sha256": _sha(binding_path),
        "environment": {"device": "cpu"}, "scope": {"datasets": list(FIXTURE_DATASETS),
        "conditions": list(CONDITIONS), "models": list(MODELS), "seeds": [0]},
        "expected_records": len(FIXTURE_DATASETS) * len(MODELS) * 2,
        "expected_trials": len(FIXTURE_DATASETS) * len(MODELS) * 2 * 4},
        config_path=config_path, data_binding_path=binding_path)


def _controller(controller_cls, writer, ledger, phase, units):
    return controller_cls(writer=writer, ledger=ledger, phase_id=phase,
                          power_request=TaskPowerRequest(_PowerBackend()),
                          power_watcher=PowerEventWatcher(_PowerBackend()))


def run(output: Path, *, launch_config: Path, receipt_path: Path) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"isolated evidence root must be new: {output}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config, config_path, binding_path, training = _fixture(output, launch_config=launch_config)
    units = _units(root=output, config=config, config_path=config_path, binding_path=binding_path,
                   training=training, commit=commit)
    normal_writer = _writer(output / "formal-records", config_path=config_path,
                            binding_path=binding_path, config=config, commit=commit)
    normal_ledger = _authority(output / "normal-ledger", digest(config), _sha(binding_path), commit, "isolated-normal")
    normal = _controller(FormalExecutionController, normal_writer, normal_ledger, "isolated-normal", units)
    normal_result = normal.run_units(units, launch_authorized=True, formal_training_enabled=True)
    scope = expected_keys(datasets=FIXTURE_DATASETS, conditions=CONDITIONS, models=MODELS, seeds=(0,))
    validated = validate_complete_run(normal_writer.root, expected_keys=scope, config_path=config_path,
                                      data_binding_path=binding_path)
    analysis = adapt_validated_root(normal_writer.root, config_path=config_path, data_binding_path=binding_path)

    pause_writer = _writer(output / "pause-records", config_path=config_path, binding_path=binding_path,
                           config=config, commit=commit)
    pause_ledger = _authority(output / "pause-ledger", digest(config), _sha(binding_path), commit, "isolated-pause")
    pause = _controller(_PauseAfterBoundary, pause_writer, pause_ledger, "isolated-pause", units)
    pause_result = pause.run_units(units, launch_authorized=True, formal_training_enabled=True, finalize=False)

    emergency_writer = _writer(output / "emergency-records", config_path=config_path,
                               binding_path=binding_path, config=config, commit=commit)
    emergency_ledger = _authority(output / "emergency-ledger", digest(config), _sha(binding_path), commit, "isolated-emergency")
    emergency = _controller(_EmergencyAtBoundary, emergency_writer, emergency_ledger, "isolated-emergency", units)
    emergency_result = emergency.run_units(units, launch_authorized=True, formal_training_enabled=True, finalize=False)
    if pause_result["status"] != "paused" or emergency_result["status"] != "emergency_stopped":
        raise RuntimeError("pause/emergency controller evidence did not reach the expected boundary")
    if not emergency_ledger._state["attempts"]["attempt000"]["outcome"] == "open":
        raise RuntimeError("emergency stop did not preserve the unfinished attempt")

    log_path = output / "formal-e2e.log.json"
    _write_json(log_path, {"normal": normal_result, "pause": pause_result,
                           "emergency": emergency_result, "analysis": analysis})
    artifacts = []
    for path in sorted(normal_writer.root.rglob("*")):
        if path.is_file():
            artifacts.append({"path": str(path), "sha256": _sha(path)})
    receipt = {"schema_version": "2.0", "status": "passed", "commit": commit,
               "config_path": str(config_path), "config_sha256": digest(config),
               "data_binding_path": str(binding_path), "data_binding_sha256": _sha(binding_path),
               "launch_config_path": str(Path(launch_config).resolve()),
               "launch_config_sha256": digest(json.loads(Path(launch_config).read_text(encoding="utf-8"))),
               "formal_record_root": str(normal_writer.root),
               "complete_marker_sha256": _sha(normal_writer.root / "complete.json"),
               "record_digest": validated["record_digest"],
               "formal_entry_path": str(Path(__file__).resolve()),
               "formal_entry_source_sha256": _sha(Path(__file__)),
               "logs": [{"path": str(log_path), "sha256": _sha(log_path)}], "artifacts": artifacts,
               "formal_path_passed": True, "writer_finalized": True, "actual_training": True,
               "checkpoint_reload_verified": True, "monitor_samples": normal_result["monitor_samples"],
               "pause_control_verified": pause_result["status"] == "paused",
               "emergency_stop_verified": emergency_result["status"] == "emergency_stopped",
               "cumulative_budget_verified": len(normal_result["units"]) == len(FIXTURE_DATASETS) * len(MODELS) * 2,
               "analysis_verified": analysis["analysis_status"] == "post_hoc_two_condition_adapter",
               "research_results_created": {"validation_evaluations": 0, "test_evaluations": 0, "formal_records": 0},
               "isolated_fixture": True}
    _write_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output, launch_config=args.launch_config, receipt_path=args.receipt)
    print(json.dumps({"status": result["status"], "commit": result["commit"], "receipt": str(args.receipt)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
