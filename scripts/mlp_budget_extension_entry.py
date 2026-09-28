"""Guarded MLP-budget extension dispatch using the existing power/ledger guardian.

No ledger is created here.  Execution requires the cumulative handoff wrapper;
the default command only reports preparation gates and never starts training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import torch

from scripts.accept_input_robustness_resources import supervise_process
from scripts.input_robustness_formal_entry import FormalExecutionController
from scripts.input_robustness_formal_records import MODELS, FormalEmergencyStop, FormalRecordError, _exclusive_json, digest, file_digest
from scripts.input_robustness_budget import CI_JOBS
from scripts.mlp_budget_extension import (
    ExtensionRecordWriter, identity, load_base_evidence, read_json, validate_extension_run,
)
from scripts.run_input_robustness_formal import _ci_gate, _environment_binding, _source_commit

ROOT = Path(__file__).resolve().parents[1]
PAUSE_FILE = "PAUSE_REQUESTED"
STOP_FILE = "EMERGENCY_STOP_REQUESTED"
ACCEPTANCE_SOURCES = (
    "scripts/mlp_budget_extension.py", "scripts/mlp_budget_extension_entry.py",
    "scripts/mlp_budget_extension_worker.py", "scripts/mlp_budget_extension_e2e.py",
    "scripts/analyze_published_diagnostics.py", "scripts/published_graph_diagnostics.py",
    "tests/test_mlp_budget_extension.py",
)


def validate_cumulative_attempts(snapshot: Mapping[str, Any], *, unit_id: str,
                                 attempt_ids: list[str]) -> dict[str, Any]:
    """Check the journal's integer nanoseconds; seconds are display-only."""
    attempts = [snapshot["attempts"][key] for key in attempt_ids]
    charges = [row["charged_ns"] for row in attempts]
    unit_ns = snapshot["charged_ns"]["units"][unit_id]
    if (len(attempts) != 2 or any(row["unit_id"] != unit_id for row in attempts)
            or attempts[0]["outcome"] != "external_interruption" or attempts[1]["outcome"] != "completed"
            or any(type(value) is not int or value <= 0 for value in charges)
            or type(unit_ns) is not int or unit_ns != sum(charges)):
        raise FormalRecordError(
            f"same-unit cumulative accounting mismatch: unit={unit_id!r}, unit_ns={unit_ns!r}, "
            f"attempt_ids={attempt_ids!r}, attempt_charges_ns={charges!r}, sum_ns={sum(charges)!r}")
    return {"unit_charged_ns": unit_ns, "attempt_charges_ns": charges,
            "unit_charged_seconds": unit_ns / 1_000_000_000,
            "attempt_charges_seconds": [value / 1_000_000_000 for value in charges]}


class ExtensionExecutionController(FormalExecutionController):
    """Retain the existing continuous guardian and append-only accounting API."""

    def _monitor(self) -> None:
        if (self.writer.root / PAUSE_FILE).exists():
            self.request_normal_pause()
        if (self.writer.root / STOP_FILE).exists():
            self.request_emergency_stop()
        super()._monitor()

    def _child(self, request: Mapping[str, Any], *, attempt_id: str,
               cap_seconds: float = 28800) -> dict[str, Any]:
        directory = self.writer.root / "workers"
        directory.mkdir(parents=True, exist_ok=True)
        request_path = directory / f"{attempt_id}.pt"
        result_path = directory / f"{attempt_id}.json"
        if request_path.exists() or result_path.exists():
            raise FormalRecordError("worker attempt identity already exists")
        torch.save(dict(request), request_path)
        command = [sys.executable, "-m", "scripts.mlp_budget_extension_worker", "--request",
                   str(request_path.resolve()), "--result", str(result_path.resolve())]
        environment = os.environ.copy()
        environment["CUBLAS_WORKSPACE_CONFIG"] = self.writer.config["execution"].get("cublas_workspace_config", ":4096:8")
        if "seed" in request:
            environment["PYTHONHASHSEED"] = str(request["seed"])
        result = supervise_process(
            command, cwd=ROOT, environment=environment,
            log_path=directory / f"{attempt_id}.log", worker_cap_seconds=float(cap_seconds),
            poll_seconds=0.25, on_poll=self._supervisor_poll,
        )
        _exclusive_json(directory / f"{attempt_id}.supervision.json", result)
        if result.get("stop_reasons"):
            self.monitor_stop_reasons.extend(result["stop_reasons"])
            raise FormalEmergencyStop("extension guardian stopped worker: " + "; ".join(result["stop_reasons"]))
        if result.get("returncode") != 0 or not result_path.is_file():
            raise FormalRecordError("extension worker failed without a valid result")
        return read_json(result_path)

    def run_units(self, units: Iterable[Mapping[str, Any]], *, launch_authorized: bool,
                  formal_training_enabled: bool, finalize: bool = True,
                  final_attempt_id: str | None = None,
                  analysis_data_root: Path | None = None) -> dict[str, Any]:
        if launch_authorized is not True or formal_training_enabled is not True:
            raise FormalRecordError("extension execution requires explicit launch authorization")
        self.power_request.acquire()
        self.power_watcher.start()
        status, complete = "completed", None
        try:
            self._monitor()
            for parameters in units:
                if self.pause_requested:
                    status = "paused"
                    break
                request = dict(parameters)
                attempt = request.pop("attempt_id")
                dataset, seed, condition = request["dataset"], int(request["seed"]), request["condition"]
                unit_id = f"unit_{dataset}_{condition}_MLP_{seed:03d}"
                batch_id = f"pair_{dataset}_seed_{seed:03d}_MLP"
                self.ledger.begin_attempt(
                    attempt, activity_id=f"mlp24_{unit_id}", phase_id=self.runner.phase_id,
                    budget_group="formal", estimated_seconds=self.writer.evidence["unit_estimates_seconds"][
                        identity(dataset, condition, seed)], unit_id=unit_id, batch_id=batch_id,
                )
                try:
                    saved = self._child({
                        **request, "operation": "unit", "config_path": str(self.writer.config_path.resolve()),
                        "output_root": str(self.writer.root.resolve()), "base_evidence": self.writer.evidence,
                        "source_commit": self.writer.manifest["source_commit"], "attempt_id": attempt,
                    }, attempt_id=attempt)
                    path = Path(saved["record_path"])
                    expected = self.writer.root / "records" / condition / dataset / f"seed_{seed:03d}.json"
                    if path.resolve() != expected.resolve() or file_digest(path) != saved["record_sha256"]:
                        raise FormalRecordError("supervised saved record path/hash mismatch")
                    self._monitor()
                    snapshot = self.ledger.close_attempt(attempt, outcome="completed", record_sha256=file_digest(path))
                except (FormalEmergencyStop, KeyboardInterrupt) as exc:
                    self.writer.write_failure({"status": "external_interruption", "attempt_id": attempt,
                                               "unit_id": unit_id, "reason": repr(exc)}, identity=attempt)
                    supervision_path = self.writer.root / "workers" / f"{attempt}.supervision.json"
                    if supervision_path.is_file():
                        stopped = read_json(supervision_path)
                        if stopped.get("returncode") is not None and not stopped.get("owned_processes_remaining"):
                            self.ledger.close_attempt(attempt, outcome="external_interruption", reason=repr(exc))
                    raise
                except Exception as exc:
                    self.writer.write_failure({"status": "failed", "attempt_id": attempt,
                                               "unit_id": unit_id, "reason": repr(exc)}, identity=attempt)
                    self.ledger.close_attempt(attempt, outcome="failed", reason=repr(exc))
                    raise
                self.unit_results.append({"record_path": str(path), "record_sha256": file_digest(path),
                                          "unit_id": unit_id, "batch_id": batch_id, "ledger": snapshot})
                if self.pause_requested:
                    status = "paused"
                    break
            if self.pause_requested:
                status = "paused"
            if status == "completed" and finalize:
                if not final_attempt_id:
                    raise FormalRecordError("final validation requires a unique accounted attempt identity")
                self.ledger.begin_attempt(final_attempt_id, activity_id=f"{final_attempt_id}_validation",
                    phase_id=self.runner.phase_id, budget_group="control", estimated_seconds=0.0)
                try:
                    complete = self._child({
                        "operation": "finalize", "config_path": str(self.writer.config_path.resolve()),
                        "output_root": str(self.writer.root.resolve()), "base_evidence": self.writer.evidence,
                        "source_commit": self.writer.manifest["source_commit"],
                        "analysis_data_root": str(analysis_data_root.resolve()) if analysis_data_root else None,
                    }, attempt_id=final_attempt_id, cap_seconds=7200)
                    self._monitor()
                    self.ledger.close_attempt(final_attempt_id, outcome="completed",
                                               record_sha256=file_digest(self.writer.root / "run_complete.json"))
                except (FormalEmergencyStop, KeyboardInterrupt):
                    raise
                except Exception as exc:
                    self.ledger.close_attempt(final_attempt_id, outcome="failed", reason=repr(exc))
                    raise
        except (FormalEmergencyStop, KeyboardInterrupt) as exc:
            status, self.failure = "emergency_stopped", repr(exc)
        except Exception as exc:
            status, self.failure = "failed", repr(exc)
        finally:
            try:
                self.power_watcher.stop()
            finally:
                self.power_request.release()
        return {"status": status, "failure": self.failure, "complete": complete,
                "monitor_samples": self.monitor_samples, "units": self.unit_results,
                "monitor_gap_seconds": 5, "power_request": self.power_request.snapshot(),
                "power_watcher": self.power_watcher.snapshot(), "ledger": self.ledger.snapshot()}


def strict_ci_gate(path: Path, commit: str) -> dict[str, Any]:
    receipt = _ci_gate(path, commit)
    job_ids = [row.get("id") for row in receipt["jobs"]]
    if (receipt.get("kind") != "github_actions" or {row["name"] for row in receipt["jobs"]} != CI_JOBS
            or type(receipt.get("run_id")) is not int or receipt["run_id"] <= 0
            or any(type(value) is not int or value <= 0 for value in job_ids)
            or len(set(job_ids)) != len(CI_JOBS)):
        raise FormalRecordError("research extension requires the three remote CI jobs")
    return receipt


def validate_acceptance(path: Path, *, config_path: Path, commit: str,
                        require_native_power: bool = True) -> dict[str, Any]:
    """Bind the actual rehearsal tree, source and logs, never a bare passed flag."""
    receipt = read_json(path)
    if receipt.get("source_commit") != commit or receipt.get("config_sha256") != digest(read_json(config_path)):
        raise FormalRecordError("extension acceptance has stale source/config identity")
    if receipt.get("entry_source_sha256") != file_digest(Path(__file__)):
        raise FormalRecordError("extension acceptance entry source changed")
    if set(receipt.get("source_files", {})) != set(ACCEPTANCE_SOURCES) or any(
            file_digest(ROOT / name) != value for name, value in receipt["source_files"].items()):
        raise FormalRecordError("extension acceptance source file bindings changed")
    native = receipt.get("native_power_evidence", {})
    if not native.get("path") or file_digest(Path(native["path"])) != native.get("sha256"):
        raise FormalRecordError("extension acceptance lacks actual power lifecycle evidence")
    native_result = read_json(Path(native["path"]))
    power, watcher = native_result.get("power_request", {}), native_result.get("power_watcher", {})
    if (require_native_power and receipt.get("native_power_interface") is not True) or any((
            power.get("acquired") is not True, power.get("released") is not True,
            power.get("handle_closed") is not True, bool(power.get("active_request_types")),
            bool(power.get("cleanup_errors")), watcher.get("registered") is not True,
            watcher.get("stopped") is not True, bool(watcher.get("subscription_active")),
            bool(watcher.get("cleanup_errors")))):
        raise FormalRecordError("native power acquisition/cleanup is not proved")
    fixture = receipt.get("fixture", {})
    fixture_config = read_json(Path(fixture["config_path"]))
    if fixture_config.get("execution_mode") != "isolated_fixture":
        raise FormalRecordError("acceptance must use isolated CPU fixtures")
    result = validate_extension_run(Path(fixture["root"]), config_path=Path(fixture["config_path"]),
        base_root=Path(fixture["base_root"]), base_config_path=Path(fixture["base_config_path"]),
        binding_path=Path(fixture["binding_path"]))
    if result["record_digest"] != fixture.get("record_digest") or result["source_commit"] != commit:
        raise FormalRecordError("acceptance fixture digest/source mismatch")
    fixture_base_config = read_json(Path(fixture["base_config_path"]))
    if (result["record_count"] < 4 or len(fixture_config["datasets"]) < 2
            or set(fixture_base_config["models"]) != set(MODELS)
            or set(fixture_config["conditions"]) != set(read_json(config_path)["conditions"])):
        raise FormalRecordError("acceptance requires two datasets, both conditions and the full seven-model base")
    completion = read_json(Path(fixture["root"]) / "run_complete.json")
    if (completion.get("analysis_sha256") != file_digest(Path(fixture["root"]) / "analysis.json")
            or completion.get("records_complete_sha256") != file_digest(Path(fixture["root"]) / "complete.json")
            or completion.get("record_digest") != result["record_digest"]
            or completion.get("source_commit") != commit):
        raise FormalRecordError("acceptance has no bound completed analysis")
    analysis = read_json(Path(fixture["root"]) / "analysis.json")
    if analysis.get("mlp_budgets") != [4, 24] or analysis.get("portfolio_count") != 63:
        raise FormalRecordError("acceptance analysis scope is incomplete")
    from scripts.mlp_budget_extension_worker import analyze_fixture_extension
    reconstructed = analyze_fixture_extension(
        evidence=read_json(Path(fixture["root"]) / "manifest.json")["base_evidence"],
        output=Path(fixture["root"]), config_path=Path(fixture["config_path"]))
    if digest(reconstructed) != digest(analysis):
        raise FormalRecordError("acceptance analysis differs from actual record reconstruction")
    logs = receipt.get("logs", [])
    if not logs:
        raise FormalRecordError("acceptance must include real rehearsal logs")
    for row in logs:
        if file_digest(Path(row["path"])) != row["sha256"]:
            raise FormalRecordError("acceptance log hash mismatch")
    checks = receipt.get("checks", {})
    for name in ("normal_pause", "emergency_stop", "cumulative_budget", "continuous_monitoring"):
        evidence = checks.get(name, {})
        if not evidence.get("path") or file_digest(Path(evidence["path"])) != evidence.get("sha256"):
            raise FormalRecordError(f"acceptance missing bound {name} evidence")
        proof = read_json(Path(evidence["path"]))
        if name == "normal_pause":
            if proof.get("status") != "paused" or len(proof.get("units", [])) != 1 or proof.get("complete") is not None:
                raise FormalRecordError("normal pause did not stop after one complete unit")
            for unit in proof["units"]:
                if file_digest(Path(unit["record_path"])) != unit["record_sha256"]:
                    raise FormalRecordError("pause record hash differs")
        elif name in {"emergency_stop", "continuous_monitoring"}:
            bound = proof.get("supervision", {})
            if not bound.get("path") or file_digest(Path(bound["path"])) != bound.get("sha256"):
                raise FormalRecordError("missing actual owned-worker supervision evidence")
            observed = read_json(Path(bound["path"]))
            if observed.get("owned_processes_remaining") or not observed.get("observed_process_ids"):
                raise FormalRecordError("worker cleanup or actual process identity is not proved")
            if name == "emergency_stop" and (proof.get("status") != "emergency_stopped"
                    or not observed.get("stop_reasons") or observed.get("returncode") in (None, 0)):
                raise FormalRecordError("emergency stop did not terminate an active worker")
            if name == "continuous_monitoring" and (observed.get("wall_seconds", 0) <= 5
                    or proof.get("monitor_gap_seconds") != 5 or proof.get("monitor_samples", 0) < 20
                    or observed.get("stop_reasons")
                    or observed.get("clock_guard", {}).get("statistics", {}).get("sample_count", 0) < 20
                    or observed.get("clock_guard", {}).get("statistics", {}).get("maximum_poll_gap_ns", 5_000_000_001) > 5_000_000_000):
                raise FormalRecordError("continuous monitoring did not cover a >5 second live workload")
        else:
            from scripts.supplement_budget_handoff import stream_snapshot
            ledger = stream_snapshot(Path(proof["ledger_path"]), authority=proof["authority"], expected_head=proof["head"])
            exact = validate_cumulative_attempts(ledger, unit_id=proof["unit_id"], attempt_ids=proof["attempt_ids"])
            if any(proof.get(field) != exact[field] for field in ("unit_charged_ns", "attempt_charges_ns")):
                raise FormalRecordError("cumulative receipt integer charges differ from the journal")
    for name, reason in (("ledger_stop", "injected ledger must_stop"), ("suspend", "suspend/resume")):
        bound = receipt.get("simulated_injections", {}).get(name, {})
        if not bound.get("path") or file_digest(Path(bound["path"])) != bound.get("sha256"):
            raise FormalRecordError(f"acceptance lacks bound {name} injection")
        injected = read_json(Path(bound["path"]))
        supervision_ref = injected.get("supervision", {})
        if not supervision_ref.get("path") or file_digest(Path(supervision_ref["path"])) != supervision_ref.get("sha256"):
            raise FormalRecordError(f"{name} injection lacks actual worker supervision")
        supervision = read_json(Path(supervision_ref["path"]))
        if (injected.get("status") != "emergency_stopped" or injected.get("units")
                or supervision.get("returncode") in (None, 0) or supervision.get("owned_processes_remaining")
                or not any(reason in item for item in supervision.get("stop_reasons", []))):
            raise FormalRecordError(f"{name} injection did not stop the active owned worker")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--budget-handoff", type=Path)
    parser.add_argument("--budget-ledger", type=Path)
    parser.add_argument("--ci-receipt", type=Path)
    parser.add_argument("--acceptance-record", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    # Preparation reports real missing gates; it cannot turn on either switch.
    errors, evidence = [], None
    if not args.execute:
        try:
            from scripts.mlp_budget_extension import validate_extension_config
            validate_extension_config(config, read_json(args.base_config))
            if not (args.base_root / "complete.json").is_file():
                raise FormalRecordError("base complete marker is missing")
            evidence = {"marker": read_json(args.base_root / "complete.json")}
        except Exception as exc:
            errors.append(f"base metadata: {exc}")
    if config.get("execution_mode", "research") not in {"formal", "research"}:
        errors.append("the production command cannot execute an isolated fixture configuration")
    for name in ("budget_handoff", "budget_ledger", "ci_receipt", "acceptance_record", "data_root", "output_root"):
        if getattr(args, name) is None:
            errors.append(f"missing --{name.replace('_', '-')}")
    if not args.execute:
        print(json.dumps({"status": "preparation_only", "launch_enabled": False,
                          "base_metadata_present": evidence is not None, "strict_base_validation": "pending_guarded_prepare",
                          "missing_gates": errors}, sort_keys=True))
        return 0
    if errors:
        raise SystemExit("extension launch refused: " + "; ".join(errors))
    from scripts.supplement_budget_handoff import open_inherited_ledger
    commit = _source_commit()
    ci = strict_ci_gate(args.ci_receipt, commit)
    # Only small metadata are read before the accounted, supervised preparation.
    if (read_json(args.acceptance_record).get("source_commit") != commit
            or read_json(args.acceptance_record).get("config_sha256") != digest(config)):
        raise FormalRecordError("acceptance source/config differs before preparation")
    with open_inherited_ledger(args.budget_handoff, args.budget_ledger, base_root=args.base_root.parent,
                              expected_config_sha=digest(config), expected_source_commit=commit,
                              required_inventory_roots=config["budget"]["inventory_roots"], resume=args.resume) as ledger:
        phase = "mlp24_extension"
        ledger.register_phase(phase, provenance={
            "source_commit": commit, "config_sha256": digest(config),
            "data_binding_sha256": file_digest(args.binding),
            "source_files": {f"scripts/{name}.py": file_digest(ROOT / "scripts" / f"{name}.py")
                             for name in ("mlp_budget_extension", "mlp_budget_extension_entry", "mlp_budget_extension_worker")},
            "environment": _environment_binding(),
        }, gate_receipt={"ci_receipt_file_sha256": file_digest(args.ci_receipt),
                        "review_record_sha256": file_digest(args.acceptance_record), "ci_receipt": ci})
        prefix = f"mlp24_{ledger.head['event_count']}"
        scratch = args.output_root.parent / f"{args.output_root.name}_preparation_{prefix}"
        proxy = SimpleNamespace(root=scratch, config=config, synthetic=False)
        preparer = ExtensionExecutionController(writer=proxy, ledger=ledger, phase_id=phase)
        preparer.power_request.acquire()
        preparer.power_watcher.start()
        attempt = f"{prefix}_prepare"
        try:
            ledger.begin_attempt(attempt, activity_id=f"{prefix}_preparation", phase_id=phase,
                                 budget_group="control", estimated_seconds=0.0)
            prepared = preparer._child({"operation": "prepare", "config_path": str(args.config.resolve()),
                "base_root": str(args.base_root.resolve()), "base_config_path": str(args.base_config.resolve()),
                "binding_path": str(args.binding.resolve()), "acceptance_path": str(args.acceptance_record.resolve()),
                "resume": args.resume, "output_root": str(args.output_root.resolve()),
                "source_commit": commit}, attempt_id=attempt, cap_seconds=7200)
            evidence = prepared["base_evidence"]
            ledger.close_attempt(attempt, outcome="completed", record_sha256=digest(prepared))
        except (FormalEmergencyStop, KeyboardInterrupt):
            raise
        except Exception as exc:
            ledger.close_attempt(attempt, outcome="failed", reason=repr(exc))
            raise
        finally:
            preparer.power_watcher.stop()
            preparer.power_request.release()
        writer = ExtensionRecordWriter(args.output_root, config_path=args.config,
            base_evidence=evidence, source_commit=commit, resume=args.resume)
        existing = {tuple(key) for key in prepared["completed_keys"]}
        # Ledger event sequence disambiguates attempts across reviewed resumes.
        units = [{"dataset": dataset, "seed": seed, "condition": condition,
                  "attempt_id": f"{prefix}_{index:06d}", "data_root": str(args.data_root.resolve())}
                 for index, (dataset, seed, condition) in enumerate(
                    (d, s, c) for d in config["datasets"] for s in config["seeds"] for c in config["conditions"])
                 if (dataset, condition, seed) not in existing]
        controller = ExtensionExecutionController(writer=writer, ledger=ledger, phase_id=phase)
        result = controller.run_units(units, launch_authorized=True, formal_training_enabled=True,
                                      final_attempt_id=f"{prefix}_finalize", analysis_data_root=args.data_root)
        _exclusive_json(writer.root / "attempt_summaries" / f"{prefix}.json", result)
        print(json.dumps({"status": result["status"], "completed_units": len(result["units"]),
                          "failure": result["failure"]}, sort_keys=True))
        return 0 if result["status"] in {"completed", "paused"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
