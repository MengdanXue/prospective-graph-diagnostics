"""Committed, guarded formal scheduler for input-robustness model units.

The scheduler is deliberately small: the existing ledger owns admission,
clocks and cumulative charging; the existing power classes own the Windows
request and suspend/resume subscription.  This module joins those components
to the immutable formal record writer.  It has no default authorization and
there is no path here that turns on the research configuration implicitly.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import sys
from typing import Any, Callable, Iterable, Mapping

import torch

from scripts.input_robustness_formal_records import (
    FormalEmergencyStop, FormalRecordError, FormalRecordWriter, FormalUnitRunner,
    file_digest, run_formal_model_unit,
    expected_keys,
)
from scripts.input_robustness_runtime import PowerEventWatcher, TaskPowerRequest


def run_formal_model_unit_from_core(*, writer: FormalRecordWriter,
                                    core_record: Mapping[str, Any],
                                    launch_authorized: bool,
                                    formal_training_enabled: bool,
                                    monitor_callback: Callable[[], Any] | None = None,
                                    **runner_kwargs: Any) -> dict[str, Any]:
    """Bind a guardian-approved child result through the normal record path."""
    return run_formal_model_unit(
        writer=writer, core_runner=lambda **_: dict(core_record),
        launch_authorized=launch_authorized,
        formal_training_enabled=formal_training_enabled,
        monitor_callback=monitor_callback, **runner_kwargs,
    )


class FormalExecutionController:
    """Run explicitly authorized units with live polling and task-scoped power controls."""

    def __init__(self, *, writer: FormalRecordWriter, ledger: Any, phase_id: str,
                 power_request: TaskPowerRequest | None = None,
                 power_watcher: PowerEventWatcher | None = None,
                 activity_id: str = "input_robustness_formal",
                 batch_id: str | None = None,
                 isolated_process: bool = True):
        if writer.synthetic:
            raise FormalRecordError("formal scheduler requires a non-synthetic writer")
        self.writer = writer
        self.ledger = ledger
        self.power_request = power_request or TaskPowerRequest()
        self.power_watcher = power_watcher or PowerEventWatcher()
        # The budget contract requires every formal attempt to belong to a
        # paired batch.  If production does not provide one, FormalUnitRunner
        # derives it from dataset/seed/model, so paired conditions share only
        # their own model-unit batch.
        self.runner = FormalUnitRunner(writer=writer, ledger=ledger, phase_id=phase_id,
                                       activity_id=activity_id, batch_id=batch_id)
        self.isolated_process = bool(isolated_process)
        self.pause_requested = False
        self.emergency_requested = False
        self.monitor_samples = 0
        self.unit_results: list[dict[str, Any]] = []
        self.failure: str | None = None
        self.monitor_stop_reasons: list[str] = []
        self.last_ledger_snapshot: dict[str, Any] | None = None

    def request_normal_pause(self) -> None:
        """Stop dispatch after the current model unit reaches its boundary."""
        self.pause_requested = True

    def request_emergency_stop(self) -> None:
        """Stop at the next monitor boundary and preserve the open attempt."""
        self.emergency_requested = True

    def _monitor(self) -> None:
        self.monitor_samples += 1
        snapshot = self.ledger.poll(force=True)
        self.last_ledger_snapshot = snapshot
        reasons = list(snapshot.get("stop_reasons", []))
        if snapshot.get("must_stop") and not reasons:
            reasons.append("ledger must_stop")
        if self.power_watcher.events:
            reasons.append("suspend/resume notification or callback failure")
        if reasons:
            self.monitor_stop_reasons = list(dict.fromkeys(reasons))
            raise FormalEmergencyStop("outer monitor stop: " + "; ".join(self.monitor_stop_reasons))
        if self.emergency_requested:
            raise FormalEmergencyStop("emergency stop requested by formal scheduler")

    def _supervisor_poll(self, process: Any, elapsed: float) -> list[str]:
        del process, elapsed
        try:
            self._monitor()
        except FormalEmergencyStop as exc:
            self.monitor_stop_reasons = list(dict.fromkeys(self.monitor_stop_reasons + [str(exc)]))
            return list(self.monitor_stop_reasons)
        return []

    def _run_supervised_core(self, *, attempt_id: str, unit_id: str,
                             runner_kwargs: Mapping[str, Any], estimated_seconds: float) -> dict[str, Any]:
        """Run the frozen core in an owned child under the existing guardian."""
        from scripts.accept_input_robustness_resources import supervise_process

        worker_root = self.writer.root / "workers"
        worker_root.mkdir(parents=True, exist_ok=True)
        request_path = worker_root / f"{attempt_id}.pt"
        result_path = worker_root / f"{attempt_id}.json"
        log_path = worker_root / f"{attempt_id}.log"
        core_kwargs = dict(runner_kwargs)
        sleep_seconds = float(core_kwargs.pop("worker_sleep_seconds", 0.0))
        # The child owns the core only.  Record binding and immutable writing
        # remain in the parent after the guardian has accepted the result.
        for key in ("condition", "data_binding_sha256", "transform_binding", "diagnostics"):
            core_kwargs.pop(key, None)
        torch.save({"core_kwargs": core_kwargs, "worker_sleep_seconds": sleep_seconds}, request_path)
        command = [sys.executable, "-m", "scripts.input_robustness_formal_worker",
                   "--request", str(request_path.resolve()), "--result", str(result_path.resolve())]
        supervision = supervise_process(
            command, cwd=Path(__file__).resolve().parents[1], environment=os.environ.copy(),
            log_path=log_path, worker_cap_seconds=max(30.0, float(estimated_seconds) * 2.0),
            poll_seconds=0.25, on_poll=self._supervisor_poll,
        )
        if supervision.get("stop_reasons"):
            self.monitor_stop_reasons = list(dict.fromkeys(
                self.monitor_stop_reasons + list(supervision["stop_reasons"])))
            raise FormalEmergencyStop("outer guardian stopped worker: " + "; ".join(self.monitor_stop_reasons))
        if supervision.get("returncode") != 0 or not result_path.is_file():
            raise FormalRecordError("formal worker exited without a complete core result")
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FormalRecordError("formal worker result is unreadable") from exc

    def run_units(self, units: Iterable[Mapping[str, Any]], *,
                  launch_authorized: bool, formal_training_enabled: bool,
                  finalize: bool = True,
                  core_runner: Callable[..., Mapping[str, Any]] | None = None,
                  post_stage: Callable[[], Any] | None = None) -> dict[str, Any]:
        if launch_authorized is not True or formal_training_enabled is not True:
            raise FormalRecordError("formal scheduler is locked until both launch flags are true")
        self.power_request.acquire()
        self.power_watcher.start()
        status = "completed"
        complete = None
        stage_result = None
        try:
            # Preparation and dispatch are inside the same monitored lifetime
            # as the worker.  This catches a pre-existing ledger veto before
            # any request envelope or model unit is created.
            self._monitor()
            for unit in units:
                if self.pause_requested:
                    status = "paused"
                    break
                params = dict(unit)
                attempt_id = str(params.pop("attempt_id"))
                unit_id = str(params.pop("unit_id"))
                estimated_seconds = float(params.pop("estimated_seconds"))
                # The append-only budget ledger binds an activity descriptor
                # (including unit identity) immutably.  Consecutive units
                # therefore receive distinct activity ids while retaining the
                # same approved phase and paired batch.
                unit_runner = FormalUnitRunner(
                    writer=self.writer, ledger=self.ledger, phase_id=self.runner.phase_id,
                    activity_id=f"{self.runner.activity_id}_{unit_id}",
                    batch_id=self.runner.batch_id,
                )
                if unit_runner.batch_id is None:
                    try:
                        unit_runner.batch_id = (
                            f"pair_{params['dataset']}_seed_{int(params['seed']):03d}_"
                            f"{params['model_id']}"
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        raise FormalRecordError(
                            "formal units require dataset, seed and model_id to derive a paired batch id"
                        ) from exc
                if self.isolated_process and core_runner is None:
                    if "checkpoint_dir" not in params and all(
                            key in params for key in ("condition", "dataset", "model_id", "seed")):
                        params["checkpoint_dir"] = (
                            self.writer.root.resolve() / "checkpoints" / str(params["condition"])
                            / str(params["dataset"]) / str(params["model_id"])
                            / f"seed_{int(params['seed']):03d}" / str(attempt_id)
                        )
                    self.ledger.begin_attempt(
                        attempt_id, activity_id=f"{unit_runner.activity_id}", phase_id=unit_runner.phase_id,
                        budget_group="formal", estimated_seconds=estimated_seconds,
                        unit_id=unit_id, batch_id=unit_runner.batch_id,
                    )
                    try:
                        core_record = self._run_supervised_core(
                            attempt_id=attempt_id, unit_id=unit_id,
                            runner_kwargs=params, estimated_seconds=estimated_seconds,
                        )
                        record = run_formal_model_unit_from_core(
                            writer=self.writer, core_record=core_record,
                            launch_authorized=launch_authorized,
                            formal_training_enabled=formal_training_enabled,
                            monitor_callback=self._monitor, **params,
                        )
                        self._monitor()
                        path = self.writer.root / "records" / record["condition"] / record["dataset"] / record["model"] / f"seed_{int(record['seed']):03d}.json"
                        snapshot = self.ledger.close_attempt(attempt_id, outcome="completed",
                                                             record_sha256=file_digest(path))
                    except FormalEmergencyStop:
                        self.writer.write_failure({"status": "external_interruption", "attempt_id": attempt_id,
                                                   "unit_id": unit_id, "reason": "; ".join(self.monitor_stop_reasons)},
                                                  identity=attempt_id)
                        raise
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        self.writer.write_failure({"status": "failed", "attempt_id": attempt_id,
                                                   "unit_id": unit_id, "reason": repr(exc)}, identity=attempt_id)
                        snapshot = self.ledger.close_attempt(attempt_id, outcome="failed", reason=repr(exc))
                        raise
                else:
                    record, snapshot = unit_runner.run(
                        attempt_id=attempt_id, unit_id=unit_id,
                        estimated_seconds=estimated_seconds,
                        launch_authorized=launch_authorized,
                        formal_training_enabled=formal_training_enabled,
                        monitor_callback=self._monitor,
                        core_runner=core_runner, **params,
                    )
                self.unit_results.append({"record": record, "ledger": snapshot})
                if self.pause_requested:
                    status = "paused"
                    break
            if status == "completed" and finalize:
                self._monitor()
                scope = self.writer.manifest.get("scope", {})
                complete = self.writer.finalize(expected=expected_keys(
                    datasets=scope.get("datasets", ()), conditions=scope.get("conditions", ()),
                    models=scope.get("models", ()), seeds=scope.get("seeds", ()),
                ))
                self._monitor()
            if status == "completed" and post_stage is not None:
                self._monitor()
                stage_result = post_stage()
                self._monitor()
        except FormalEmergencyStop as exc:
            self.failure = repr(exc)
            status = "emergency_stopped"
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self.failure = repr(exc)
            status = "failed"
        finally:
            # Both components are task scoped.  Cleanup errors propagate and
            # therefore cannot be mistaken for a successful formal run.
            watcher_error = None
            power_error = None
            try:
                self.power_watcher.stop()
            except BaseException as exc:  # pragma: no cover - backend failures are injected in tests
                watcher_error = repr(exc)
            try:
                self.power_request.release()
            except BaseException as exc:  # pragma: no cover - backend failures are injected in tests
                power_error = repr(exc)
            if watcher_error or power_error:
                status = "failed"
                self.failure = repr({"watcher": watcher_error, "power": power_error})

        return {
            "status": status,
            "monitor_samples": self.monitor_samples,
            "power_request": self.power_request.snapshot(),
            "power_watcher": self.power_watcher.snapshot(),
            "units": self.unit_results,
            "complete": complete,
            "stage_result": stage_result,
            "failure": self.failure,
            "formal_record_root": str(self.writer.root),
        }


def run_formal_entry(*, writer: FormalRecordWriter, ledger: Any,
                     phase_id: str, units: Iterable[Mapping[str, Any]],
                     launch_authorized: bool, formal_training_enabled: bool,
                     power_request: TaskPowerRequest | None = None,
                     power_watcher: PowerEventWatcher | None = None,
                     finalize: bool = True,
                     post_stage: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Explicit entry point used by production dispatch and isolated tests."""
    controller = FormalExecutionController(
        writer=writer, ledger=ledger, phase_id=phase_id,
        power_request=power_request, power_watcher=power_watcher,
    )
    return controller.run_units(
        units, launch_authorized=launch_authorized,
        formal_training_enabled=formal_training_enabled, finalize=finalize,
        post_stage=post_stage,
    )
