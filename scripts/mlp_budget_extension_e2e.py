"""Persistent CPU-only acceptance through the actual guarded MLP24 entry.

The source fixture generates all seven original models on two tiny graphs.
No research dataset is opened. Native power handles and injected control
events are reported separately; no synthetic flag bypasses record validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from scripts.input_robustness_budget import BudgetLedger, inspect_ledger
from scripts.input_robustness_formal_records import _exclusive_json, digest, file_digest
from scripts.input_robustness_runtime import PowerEventWatcher, TaskPowerRequest
from scripts.mlp_budget_extension import ExtensionRecordWriter, read_json, validate_extension_run
from scripts.mlp_budget_extension_entry import ACCEPTANCE_SOURCES, ExtensionExecutionController, validate_cumulative_attempts
from scripts.supplement_budget_handoff import stream_snapshot
from tests.test_input_robustness_budget import AUTHORITY, gate, provenance
from tests.test_input_robustness_formal_entry import _PowerBackend
from tests.test_mlp_budget_extension import build_fixture

ROOT = Path(__file__).resolve().parents[1]


def run_acceptance(output: Path, *, launch_config_path: Path, native_power: bool = False,
                   commit: str | None = None):
    if output.exists():
        raise FileExistsError("acceptance output must be new; preserve prior evidence")
    output.mkdir(parents=True)
    fixture = build_fixture(output, full_portfolio=True)
    commit = commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    fixture_provenance = provenance(commit)
    fixture_provenance.update(config_sha256=digest(fixture["config"]),
                              data_binding_sha256=file_digest(fixture["binding_path"]))

    def ledger_for(name):
        value = BudgetLedger.create(output / f"ledger_{name}", authority=AUTHORITY, monitor_gap_seconds=5)
        value.register_phase("mlp24_fixture", provenance=fixture_provenance, gate_receipt=gate(commit))
        return value

    def writer_for(name, *, resume=False):
        return ExtensionRecordWriter(output / name, config_path=fixture["config_path"],
            base_evidence=fixture["evidence"], source_commit=commit, resume=resume)

    def controller_for(writer, ledger, mode=None):
        power = TaskPowerRequest() if native_power else TaskPowerRequest(_PowerBackend())
        watcher = PowerEventWatcher() if native_power else PowerEventWatcher(_PowerBackend())
        class InjectedController(ExtensionExecutionController):
            def _supervisor_poll(self, process, elapsed):
                if elapsed >= .75:
                    if mode == "pause":
                        self.request_normal_pause()
                    elif mode == "emergency":
                        self.request_emergency_stop()
                    elif mode == "ledger_stop":
                        if not getattr(self, "_injected_ledger_stop", False):
                            original_poll = self.ledger.poll
                            self.ledger.poll = lambda force=False: {
                                **original_poll(force=force), "must_stop": True,
                                "stop_reasons": ["injected ledger must_stop"]}
                            self._injected_ledger_stop = True
                    elif mode == "suspend":
                        self.power_watcher._callback(4)
                return super()._supervisor_poll(process, elapsed)
        return InjectedController(writer=writer, ledger=ledger, phase_id="mlp24_fixture",
                                  power_request=power, power_watcher=watcher)

    def unit(dataset, condition, attempt, delay=0):
        return {"dataset": dataset, "condition": condition, "seed": 0,
                "attempt_id": attempt, "worker_sleep_seconds": delay}

    conditions = fixture["config"]["conditions"]
    complete_writer = writer_for("complete_extension")
    complete_ledger = ledger_for("complete")
    units = [unit(dataset, condition, f"complete_{index:03d}", 6 if index == 0 else 0)
             for index, (dataset, condition) in enumerate(
                 (d, c) for d in fixture["config"]["datasets"] for c in conditions)]
    complete = controller_for(complete_writer, complete_ledger).run_units(
        units, launch_authorized=True, formal_training_enabled=True, final_attempt_id="complete_finalize")
    _exclusive_json(output / "completed_entry.json", complete)
    if complete["status"] != "completed":
        raise RuntimeError(f"complete CPU entry failed: {complete['failure']}")
    validated = validate_extension_run(complete_writer.root, config_path=fixture["config_path"],
        base_root=fixture["base_root"], base_config_path=fixture["base_config_path"], binding_path=fixture["binding_path"])
    long_path = complete_writer.root / "workers" / "complete_000.supervision.json"
    continuous = {"status": complete["status"], "monitor_samples": complete["monitor_samples"],
                  "monitor_gap_seconds": 5, "supervision": {"path": str(long_path), "sha256": file_digest(long_path)}}
    _exclusive_json(output / "continuous_monitoring.json", continuous)

    pause_writer, pause_ledger = writer_for("pause_extension"), ledger_for("pause")
    paused = controller_for(pause_writer, pause_ledger, "pause").run_units(
        [unit("Cora", conditions[0], "pause_000"), unit("Cora", conditions[1], "pause_001")],
        launch_authorized=True, formal_training_enabled=True, finalize=False)
    _exclusive_json(output / "normal_pause.json", paused)
    if paused["status"] != "paused" or len(paused["units"]) != 1:
        raise RuntimeError("normal pause did not retain exactly one complete unit")

    interrupted_writer, interrupted_ledger = writer_for("interrupted_extension"), ledger_for("interrupted")
    stopped = controller_for(interrupted_writer, interrupted_ledger, "emergency").run_units(
        [unit("Cora", conditions[0], "interrupt_000", 6)],
        launch_authorized=True, formal_training_enabled=True, finalize=False)
    stopped_path = interrupted_writer.root / "workers" / "interrupt_000.supervision.json"
    stopped["supervision"] = {"path": str(stopped_path), "sha256": file_digest(stopped_path)}
    _exclusive_json(output / "emergency_stop.json", stopped)
    if stopped["status"] != "emergency_stopped" or stopped["units"]:
        raise RuntimeError("emergency stop failed")
    # Review the actual interruption and reopen this same ledger and model unit.
    unit_id = f"unit_Cora_{conditions[0]}_MLP_000"
    activity_id = f"mlp24_{unit_id}"
    failure_path = interrupted_writer.root / "failures" / "interrupt_000.json"
    review = {"all_existing_records_validated": True, "external_interruption_cause_confirmed": True,
              "no_unresolved_failure_artifacts": True, "failure_sha256": file_digest(failure_path),
              "supervision_sha256": file_digest(stopped_path)}
    _exclusive_json(output / "interruption_review.json", review)
    interrupted_ledger.review_external_interruption(activity_id, provenance=fixture_provenance,
        review_receipt={**review, "review_record_sha256": file_digest(output / "interruption_review.json")})
    interrupted_ledger.poll(force=True)
    reopened = BudgetLedger.open(interrupted_ledger.path, authority=AUTHORITY, expected_head=interrupted_ledger.head)
    resumed = controller_for(writer_for("interrupted_extension", resume=True), reopened).run_units(
        [unit("Cora", conditions[0], "interrupt_001")],
        launch_authorized=True, formal_training_enabled=True, finalize=False)
    _exclusive_json(output / "resumed_entry.json", resumed)
    if resumed["status"] != "completed" or len(resumed["units"]) != 1:
        raise RuntimeError("reviewed same-unit resume failed")
    reopened.poll(force=True)
    checked = stream_snapshot(reopened.path, authority=AUTHORITY, expected_head=reopened.head)
    exact = validate_cumulative_attempts(checked, unit_id=unit_id, attempt_ids=["interrupt_000", "interrupt_001"])
    cumulative = {"ledger_path": str(reopened.path), "authority": AUTHORITY, "head": reopened.head,
                  "unit_id": unit_id, "attempt_ids": ["interrupt_000", "interrupt_001"],
                  **exact}
    _exclusive_json(output / "cumulative_budget.json", cumulative)
    injections = {}
    for mode in ("ledger_stop", "suspend"):
        writer, ledger = writer_for(f"{mode}_extension"), ledger_for(mode)
        result = controller_for(writer, ledger, mode).run_units(
            [unit("Cora", conditions[0], f"{mode}_000", 6)],
            launch_authorized=True, formal_training_enabled=True, finalize=False)
        supervision_path = writer.root / "workers" / f"{mode}_000.supervision.json"
        result["supervision"] = {"path": str(supervision_path), "sha256": file_digest(supervision_path)}
        _exclusive_json(output / f"{mode}.json", result)
        if result["status"] != "emergency_stopped" or result["units"]:
            raise RuntimeError(f"{mode} injection did not stop an active worker")
        injections[mode] = {"path": str(output / f"{mode}.json"), "sha256": file_digest(output / f"{mode}.json")}
    receipt = {"status": "passed", "source_commit": commit,
        "config_sha256": digest(read_json(launch_config_path)),
        "entry_source_sha256": file_digest(ROOT / "scripts/mlp_budget_extension_entry.py"),
        "fixture": {"root": str(complete_writer.root), "config_path": str(fixture["config_path"]),
                    "base_root": str(fixture["base_root"]), "base_config_path": str(fixture["base_config_path"]),
                    "binding_path": str(fixture["binding_path"]), "record_digest": validated["record_digest"]},
        "logs": [{"path": str(path), "sha256": file_digest(path)} for path in sorted(output.rglob("*.log"))],
        "checks": {name: {"path": str(output / f"{name}.json"), "sha256": file_digest(output / f"{name}.json")}
                   for name in ("normal_pause", "emergency_stop", "cumulative_budget", "continuous_monitoring")},
        "native_power_interface": native_power, "simulated_injections": injections,
        "native_power_evidence": {"path": str(output / "completed_entry.json"), "sha256": file_digest(output / "completed_entry.json")},
        "research_training_started": False,
        "source_files": {name: file_digest(ROOT / name) for name in ACCEPTANCE_SOURCES}}
    _exclusive_json(output / "acceptance.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--native-power", action="store_true")
    args = parser.parse_args()
    run_acceptance(args.output.resolve(), launch_config_path=args.launch_config.resolve(), native_power=args.native_power)
    print(str(args.output / "acceptance.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
