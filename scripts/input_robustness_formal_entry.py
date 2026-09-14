"""Committed, guarded formal scheduler for input-robustness model units.

The scheduler is deliberately small: the existing ledger owns admission,
clocks and cumulative charging; the existing power classes own the Windows
request and suspend/resume subscription.  This module joins those components
to the immutable formal record writer.  It has no default authorization and
there is no path here that turns on the research configuration implicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from scripts.input_robustness_formal_records import (
    FormalEmergencyStop, FormalRecordError, FormalRecordWriter, FormalUnitRunner,
    expected_keys,
)
from scripts.input_robustness_runtime import PowerEventWatcher, TaskPowerRequest


class FormalExecutionController:
    """Run explicitly authorized units with live polling and task-scoped power controls."""

    def __init__(self, *, writer: FormalRecordWriter, ledger: Any, phase_id: str,
                 power_request: TaskPowerRequest | None = None,
                 power_watcher: PowerEventWatcher | None = None,
                 activity_id: str = "input_robustness_formal",
                 batch_id: str | None = None):
        if writer.synthetic:
            raise FormalRecordError("formal scheduler requires a non-synthetic writer")
        self.writer = writer
        self.ledger = ledger
        self.power_request = power_request or TaskPowerRequest()
        self.power_watcher = power_watcher or PowerEventWatcher()
        # The budget contract requires every formal attempt to belong to a
        # paired batch.  Production dispatch supplies its approved batch id;
        # the stable default is only for an explicitly isolated entry.
        batch_id = batch_id or "formal_batch"
        self.runner = FormalUnitRunner(writer=writer, ledger=ledger, phase_id=phase_id,
                                       activity_id=activity_id, batch_id=batch_id)
        self.pause_requested = False
        self.emergency_requested = False
        self.monitor_samples = 0
        self.unit_results: list[dict[str, Any]] = []
        self.failure: str | None = None

    def request_normal_pause(self) -> None:
        """Stop dispatch after the current model unit reaches its boundary."""
        self.pause_requested = True

    def request_emergency_stop(self) -> None:
        """Stop at the next monitor boundary and preserve the open attempt."""
        self.emergency_requested = True

    def _monitor(self) -> None:
        self.monitor_samples += 1
        self.ledger.poll(force=True)
        if self.emergency_requested:
            raise FormalEmergencyStop("emergency stop requested by formal scheduler")

    def run_units(self, units: Iterable[Mapping[str, Any]], *,
                  launch_authorized: bool, formal_training_enabled: bool,
                  finalize: bool = True) -> dict[str, Any]:
        if launch_authorized is not True or formal_training_enabled is not True:
            raise FormalRecordError("formal scheduler is locked until both launch flags are true")
        self.power_request.acquire()
        self.power_watcher.start()
        status = "completed"
        try:
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
                record, snapshot = unit_runner.run(
                    attempt_id=attempt_id, unit_id=unit_id,
                    estimated_seconds=estimated_seconds,
                    launch_authorized=launch_authorized,
                    formal_training_enabled=formal_training_enabled,
                    monitor_callback=self._monitor,
                    **params,
                )
                self.unit_results.append({"record": record, "ledger": snapshot})
                if self.pause_requested:
                    status = "paused"
                    break
        except FormalEmergencyStop as exc:
            self.failure = repr(exc)
            status = "emergency_stopped"
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

        complete = None
        if status == "completed" and finalize:
            scope = self.writer.manifest.get("scope", {})
            complete = self.writer.finalize(expected=expected_keys(
                datasets=scope.get("datasets", ()), conditions=scope.get("conditions", ()),
                models=scope.get("models", ()), seeds=scope.get("seeds", ()),
            ))
        return {
            "status": status,
            "monitor_samples": self.monitor_samples,
            "power_request": self.power_request.snapshot(),
            "power_watcher": self.power_watcher.snapshot(),
            "units": self.unit_results,
            "complete": complete,
            "failure": self.failure,
            "formal_record_root": str(self.writer.root),
        }


def run_formal_entry(*, writer: FormalRecordWriter, ledger: Any,
                     phase_id: str, units: Iterable[Mapping[str, Any]],
                     launch_authorized: bool, formal_training_enabled: bool,
                     power_request: TaskPowerRequest | None = None,
                     power_watcher: PowerEventWatcher | None = None,
                     finalize: bool = True) -> dict[str, Any]:
    """Explicit entry point used by production dispatch and isolated tests."""
    controller = FormalExecutionController(
        writer=writer, ledger=ledger, phase_id=phase_id,
        power_request=power_request, power_watcher=power_watcher,
    )
    return controller.run_units(
        units, launch_authorized=launch_authorized,
        formal_training_enabled=formal_training_enabled, finalize=finalize,
    )
