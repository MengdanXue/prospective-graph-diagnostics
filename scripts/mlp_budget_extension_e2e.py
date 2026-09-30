"""Persistent CPU-only acceptance through the actual guarded MLP24 entry.

The source fixture generates all seven original models on two tiny graphs.
No research dataset is opened. Native power handles and injected control
events are reported separately; no synthetic flag bypasses record validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from scripts import mlp_budget_extension_entry as entry_module
from scripts import supplement_budget_handoff as handoff
from scripts import terminal_failure_review as failure_review
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
ENVIRONMENTAL_FAILURE = "raise AssertionError('Torch not compiled with CUDA enabled')"


def _reference(path: Path) -> dict:
    return {"path": str(path), "sha256": file_digest(path)}


def _handoff_spec(base: Path, commit: str, config_sha: str, roots: list[str], sources: list[dict]) -> dict:
    spec = {"schema_version": handoff.SCHEMA, "authority": AUTHORITY, "source_commit": commit,
            "config_sha256": config_sha, "inventory_roots": roots, "sources": sources}
    head = handoff._increment_index(base)[1]
    if head["event_count"]:
        spec["increment_index_head"] = head
    return spec


def _brief(result: dict) -> dict:
    return {key: result[key] for key in ("must_stop", "historical_stop_reasons", "current_blocking_stop_reasons",
                                         "waived_by_review", "failure_retry_authorizations")} | {
        "failure_reviews": [row["failure_reviews"] for row in result["sources"]]}


def run_environmental_failure_retry(output: Path, *, fixture, fixture_provenance, commit, controller_for,
                                    writer_for, conditions) -> dict:
    """Isolated rehearsal of the reviewed environmental-failure amendment.

    The failed worker is a real supervised child process that exits with the
    original CUDA-build error (a simulated environment); the retry runs the
    actual CPU fixture worker. A fixture reviewer signs only this isolated
    fixture review. No research ledger or dataset is opened.
    """
    base = output / "environmental_failure_retry"
    base.mkdir()
    config_sha = digest(fixture["config"])
    unit_id = f"unit_Cora_{conditions[0]}_MLP_000"
    identity = {"dataset": "Cora", "condition": conditions[0], "model": "MLP", "seed": 0,
                "trials": "trial_000-trial_023"}
    proof = {"simulated_injection": "environmental worker failure (CPU-only CUDA build)", "unit_id": unit_id}

    # 1. A real supervised worker fails before research computation.
    original = BudgetLedger.create(base / "original", authority=AUTHORITY, monitor_gap_seconds=5)
    original.register_phase("mlp24_fixture", provenance=fixture_provenance, gate_receipt=gate(commit))
    failed_writer = writer_for("environmental_failure_extension")
    real_supervisor = entry_module.supervise_process

    def failing_worker(command, **kwargs):
        return real_supervisor([sys.executable, "-c", ENVIRONMENTAL_FAILURE], **kwargs)

    with patch.object(entry_module, "supervise_process", side_effect=failing_worker):
        failed = controller_for(failed_writer, original).run_units(
            [{"dataset": "Cora", "condition": conditions[0], "seed": 0, "attempt_id": "envfail_000"}],
            launch_authorized=True, formal_training_enabled=True, finalize=False)
    original_head = original.head
    receipt_path = base / "original_terminal_receipt.json"
    _exclusive_json(receipt_path, {"head": original_head})
    evidence = {"failure_record": failed_writer.root / "failures/envfail_000.json",
                "worker_log": failed_writer.root / "workers/envfail_000.log",
                "supervision": failed_writer.root / "workers/envfail_000.supervision.json",
                "worker_request": failed_writer.root / "workers/envfail_000.pt"}
    preserved = [base / "original/budget_events.jsonl", receipt_path, *evidence.values()]
    before = {str(path): file_digest(path) for path in preserved}
    proof["failed_run"] = {"status": failed["status"], "failure": failed["failure"], "records": sorted(
        str(p) for p in (failed_writer.root / "records").rglob("*.json")) if (failed_writer.root / "records").exists() else []}

    # 2. Review plan and records: an unsigned draft, then a fixture sign-off.
    plan = failure_review.build_review_plan(
        ledger_path=base / "original", terminal_receipt_path=receipt_path, attempt_id="envfail_000",
        authority=AUTHORITY, research_identity=identity, evidence_paths=evidence,
        classification=failure_review.ENVIRONMENTAL,
        findings={"research_computation_started": False, "checkpoints": 0, "records": 0, "test_evaluations": 0,
                  "test_once_consumed": False, "blocking_stop_reasons": []},
        disposition={"retry_allowed": True, "reason": "fixture worker ran a CPU-only build before training",
                     "max_retries": 1, "must_be_new_segment": True, "new_attempt_id_required": True,
                     "retry_scope": {"unit_id": unit_id, "batch_id": "pair_Cora_seed_000_MLP",
                                     "activity_id": f"mlp24_{unit_id}", "research_identity": identity},
                     "required_environment": dict(fixture_provenance["environment"])})
    plan_path = base / "review_plan.json"
    _exclusive_json(plan_path, plan)
    draft = failure_review.draft_review_record(plan_path, source_commit=commit)
    draft_path = base / "review_record_draft.json"
    _exclusive_json(draft_path, draft)
    signed_path = base / "review_record_fixture_signed.json"
    _exclusive_json(signed_path, {**draft, "status": failure_review.SIGNED_STATUS, "decision": failure_review.APPROVE,
                                  "signoff": {"reviewer": "isolated acceptance fixture",
                                              "signed_at_utc": "fixture"}})
    source = {"ledger_path": str(base / "original"), "terminal_receipt": _reference(receipt_path)}

    def inspect(spec):
        return handoff.inspect_handoff(spec, base_root=base, process_probe=lambda: [])

    proof["draft_handoff"] = _brief(inspect(_handoff_spec(base, commit, config_sha, ["original"], [
        {**source, "failure_reviews": [_reference(draft_path)]}])))
    signed_spec = _handoff_spec(base, commit, config_sha, ["original"], [
        {**source, "failure_reviews": [_reference(signed_path)]}])
    proof["signed_handoff"] = _brief(inspect(signed_spec))

    # 3. New inherited segment: phase, one-time authorization, then the real retry.
    spec_path = base / "handoff_signed.json"
    spec_path.write_text(json.dumps(signed_spec), encoding="utf-8")
    segment = base / "segment_retry"
    adapter = handoff.open_inherited_ledger(spec_path, segment, base_root=base, expected_config_sha=config_sha,
                                            expected_source_commit=commit, required_inventory_roots=["original"])
    adapter.register_phase("mlp24_fixture", provenance=fixture_provenance, gate_receipt=gate(commit))
    prefix = entry_module.attempt_prefix(adapter)
    retry_id = entry_module.unit_attempt_id(prefix, 0, retry_ordinal=1)
    granted = adapter.pending_failure_retries()
    try:
        adapter.begin_attempt(f"{prefix}_u000000", activity_id=f"mlp24_{unit_id}", phase_id="mlp24_fixture",
                              budget_group="formal", estimated_seconds=1, unit_id=unit_id,
                              batch_id="pair_Cora_seed_000_MLP")
        unauthorized = "admitted"
    except handoff.budget.BudgetError as exc:
        unauthorized = str(exc)
    adapter.authorize_failure_retry(granted[0]["review_record_sha256"], retry_attempt_id=retry_id,
                                    phase_id="mlp24_fixture")
    retry_writer = writer_for("environmental_retry_extension")
    retried = controller_for(retry_writer, adapter).run_units(
        [{"dataset": "Cora", "condition": conditions[0], "seed": 0, "attempt_id": retry_id}],
        launch_authorized=True, formal_training_enabled=True, finalize=False)
    segment_head = adapter.head
    adapter.close()
    proof["retry"] = {"segment_path": str(segment), "segment_head": segment_head, "retry_attempt_id": retry_id,
                      "original_attempt_id": "envfail_000", "granted_before_authorization": len(granted),
                      "unauthorized_unit_attempt": unauthorized, "status": retried["status"],
                      "failure": retried["failure"], "units": retried["units"]}

    # 4. The same review cannot be consumed again.
    seg_receipt = sorted(segment.glob("terminal_receipt*.json"))[-1]
    second_spec = _handoff_spec(base, commit, config_sha, ["original"], [
        {**source, "failure_reviews": [_reference(signed_path)]},
        {"ledger_path": str(segment), "terminal_receipt": _reference(seg_receipt)}])
    proof["second_handoff"] = _brief(inspect(second_spec))
    second_path = base / "handoff_second.json"
    second_path.write_text(json.dumps(second_spec), encoding="utf-8")
    second = handoff.open_inherited_ledger(second_path, base / "segment_second", base_root=base,
                                           expected_config_sha=config_sha, expected_source_commit=commit,
                                           required_inventory_roots=["original"])
    second.register_phase("mlp24_fixture", provenance=fixture_provenance, gate_receipt=gate(commit))
    refusals = {"pending_after_consumption": len(second.pending_failure_retries())}
    for name, action in (
            ("authorize_again", lambda: second.authorize_failure_retry(
                granted[0]["review_record_sha256"], retry_attempt_id="again", phase_id="mlp24_fixture")),
            ("forged_index_consumption", lambda: handoff._register_increment(
                base, (base / "forged").resolve(), "9" * 64,
                consumed_failure_reviews=[granted[0]["review_record_sha256"]]))):
        try:
            action()
            refusals[name] = "accepted"
        except handoff.budget.BudgetError as exc:
            refusals[name] = str(exc)
    second.close()
    proof["second_consumption"] = refusals

    # 5. An independent unreviewed failure still blocks.
    independent = BudgetLedger.create(base / "independent", authority=AUTHORITY, monitor_gap_seconds=5)
    independent.register_phase("mlp24_fixture", provenance=fixture_provenance, gate_receipt=gate(commit))
    independent.begin_attempt("independent_000", activity_id="independent_unit", phase_id="mlp24_fixture",
                              budget_group="formal", estimated_seconds=0, unit_id="unit_CiteSeer_independent",
                              batch_id="pair_CiteSeer_independent")
    independent.close_attempt("independent_000", outcome="failed", reason="isolated independent failure")
    independent_receipt = base / "independent_terminal_receipt.json"
    _exclusive_json(independent_receipt, {"head": independent.head})
    second_receipt = sorted((base / "segment_second").glob("terminal_receipt*.json"))[-1]
    proof["independent_handoff"] = _brief(inspect(_handoff_spec(base, commit, config_sha, ["original", "independent"], [
        {**source, "failure_reviews": [_reference(signed_path)]},
        {"ledger_path": str(segment), "terminal_receipt": _reference(seg_receipt)},
        {"ledger_path": str(base / "segment_second"), "terminal_receipt": _reference(second_receipt)},
        {"ledger_path": str(base / "independent"), "terminal_receipt": _reference(independent_receipt)}])))

    # 6. Environment gate: a CPU-only probe is refused; this interpreter passes the fixture policy.
    cpu_only = {"sys_executable": "simulated", "python": "simulated", "torch": "2.9.1+cpu", "torch_cuda": None,
                "cuda_available": False, "device_count": 0, "cuda_device": None, "environment_snapshot": None}
    try:
        entry_module.formal_environment_preflight({"execution": {"model_devices": {"MLP": "cuda"}}},
                                                  fixture["base_root"], probe=lambda device: cpu_only)
        refused = {"status": "passed"}
    except entry_module.EnvironmentPreflightError as exc:
        refused = exc.evidence
    passed = entry_module.formal_environment_preflight(fixture["config"], fixture["base_root"])
    proof["environment_preflight"] = {"simulated_cpu_only_for_cuda_policy": refused, "actual_interpreter": passed}
    proof["original"] = {"ledger_path": str(base / "original"), "head": original_head, "attempt_id": "envfail_000",
                         "preserved_sha256_before": before,
                         "preserved_sha256_after": {str(path): file_digest(path) for path in preserved}}
    proof["review"] = {"plan": _reference(plan_path), "draft": _reference(draft_path),
                       "fixture_signed": _reference(signed_path)}
    _exclusive_json(output / "environmental_failure_retry.json", proof)
    return proof


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
    run_environmental_failure_retry(output, fixture=fixture, fixture_provenance=fixture_provenance, commit=commit,
                                    controller_for=controller_for, writer_for=writer_for, conditions=conditions)
    receipt = {"status": "passed", "source_commit": commit,
        "config_sha256": digest(read_json(launch_config_path)),
        "entry_source_sha256": file_digest(ROOT / "scripts/mlp_budget_extension_entry.py"),
        "fixture": {"root": str(complete_writer.root), "config_path": str(fixture["config_path"]),
                    "base_root": str(fixture["base_root"]), "base_config_path": str(fixture["base_config_path"]),
                    "binding_path": str(fixture["binding_path"]), "record_digest": validated["record_digest"]},
        "logs": [{"path": str(path), "sha256": file_digest(path)} for path in sorted(output.rglob("*.log"))],
        "checks": {name: {"path": str(output / f"{name}.json"), "sha256": file_digest(output / f"{name}.json")}
                   for name in ("normal_pause", "emergency_stop", "cumulative_budget", "continuous_monitoring",
                                "environmental_failure_retry")},
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
