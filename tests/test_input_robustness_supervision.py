"""Process-supervision fault injection with fake clocks and fake owned PIDs.

No real process, power request, sleep transition, GPU or research model runs.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import psutil

from scripts import accept_input_robustness_resources as acceptance


class FakeClock:
    def __init__(self):
        self.elapsed = 0.0

    def time(self):
        return 1700000000.0 + self.elapsed

    def monotonic(self):
        return 1000.0 + self.elapsed

    def advance(self, seconds):
        self.elapsed += seconds


class FakeNode:
    def __init__(self, harness, pid, *, created=10.0, root=False):
        self.harness, self.pid, self.created, self.root = harness, pid, created, root
        self.alive, self.resists_kill, self.kill_error = True, False, None
        self.kill_delay = 0.0

    def create_time(self):
        return self.created

    def is_running(self):
        self.harness.refresh()
        return self.alive

    def status(self):
        return "running" if self.is_running() else psutil.STATUS_ZOMBIE

    def children(self, recursive=False):
        if not self.root or not self.is_running():
            return []
        return [node for node in self.harness.nodes.values() if not node.root and node.alive]

    def memory_info(self):
        return SimpleNamespace(rss=64, peak_wset=128)

    def suspend(self):
        self.harness.suspended.append(self.pid)

    def kill(self):
        self.harness.kill_attempts.append(self.pid)
        self.harness.clock.advance(self.kill_delay)
        self.kill_delay = 0.0
        if self.kill_error:
            raise self.kill_error
        if not self.resists_kill:
            self.alive = False
            if self.root:
                self.harness.returncode = self.harness.kill_exit_code


class FakePopen:
    pid = 11001

    def __init__(self, harness):
        self.harness = harness

    def poll(self):
        self.harness.refresh()
        return self.harness.returncode

    def wait(self, timeout=None):
        self.harness.wait_calls += 1
        self.harness.clock.advance(self.harness.wait_delay)
        self.harness.wait_delay = 0.0
        self.harness.refresh()
        if self.harness.returncode is None:
            raise subprocess.TimeoutExpired("fake-owned-worker", timeout)
        return self.harness.returncode


class SupervisorFixture:
    def __init__(self, *, lifetime=0.25, exit_code=0):
        self.clock = FakeClock()
        self.lifetime, self.exit_code, self.returncode = lifetime, exit_code, None
        self.kill_exit_code = -9
        self.wait_delay, self.wait_calls = 0.0, 0
        self.kill_attempts, self.suspended = [], []
        self.stack = ExitStack()
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.nodes = {11001: FakeNode(self, 11001, root=True),
                      22002: FakeNode(self, 22002, created=20.0)}
        self.process = FakePopen(self)
        self.memory_calls, self.memory_error_at, self.memory_error = 0, None, None
        self.memory_override = None
        self.virtual_error = None
        self.available, self.disk_free = 1024, 1024
        self.limits = {"max_worker_rss_bytes": 1024, "min_system_available_bytes": 128,
                       "min_disk_free_bytes": 128}
        self.on_natural_exit = None

    def refresh(self):
        if self.returncode is None and self.lifetime is not None and self.clock.elapsed >= self.lifetime:
            self.returncode = self.exit_code
            self.nodes[11001].alive = False
            if self.on_natural_exit:
                callback, self.on_natural_exit = self.on_natural_exit, None
                callback()

    def lookup(self, pid):
        self.refresh()
        node = self.nodes.get(pid)
        if node is None or not node.alive:
            raise psutil.NoSuchProcess(pid)
        return node

    def memory(self, process, psutil_module):
        self.memory_calls += 1
        if self.memory_calls == self.memory_error_at:
            raise self.memory_error
        if self.memory_override is not None:
            return self.memory_override
        selected = [process, *process.children(recursive=True)]
        return {"rss_bytes": 64 * len(selected), "high_water_bytes": 128 * len(selected),
                "process_ids": [node.pid for node in selected]}

    def virtual_memory(self):
        if self.virtual_error:
            raise self.virtual_error
        return SimpleNamespace(available=self.available)

    def __enter__(self):
        self.stack.enter_context(patch.object(acceptance.time, "time", self.clock.time))
        self.stack.enter_context(patch.object(acceptance.time, "monotonic", self.clock.monotonic))
        self.stack.enter_context(patch.object(acceptance.time, "time_ns", lambda: round(self.clock.time() * 10**9)))
        self.stack.enter_context(patch.object(acceptance.time, "monotonic_ns", lambda: round(self.clock.monotonic() * 10**9)))
        self.stack.enter_context(patch.object(acceptance.time, "sleep", self.clock.advance))
        self.launch = self.stack.enter_context(patch.object(acceptance.subprocess, "Popen", return_value=self.process))
        self.stack.enter_context(patch.object(psutil, "Process", side_effect=self.lookup))
        self.stack.enter_context(patch.object(psutil, "virtual_memory", self.virtual_memory))
        self.stack.enter_context(patch.object(psutil, "wait_procs", side_effect=lambda nodes, timeout:
            ([node for node in nodes if not node.alive], [node for node in nodes if node.alive])))
        self.stack.enter_context(patch.object(acceptance.preflight, "process_tree_memory", self.memory))
        self.stack.enter_context(patch.object(acceptance.shutil, "disk_usage",
                                              side_effect=lambda path: SimpleNamespace(free=self.disk_free)))
        return self

    def __exit__(self, *exception):
        return self.stack.__exit__(*exception)

    def run(self, **options):
        arguments = {"cwd": self.directory, "environment": {},
                     "log_path": self.directory / "worker.log", "worker_cap_seconds": 30,
                     "memory_limits": self.limits}
        arguments.update(options)
        return acceptance.supervise_process(["fake-owned-worker"], **arguments)


class SupervisionTests(unittest.TestCase):
    def assert_failed(self, result):
        self.assertTrue(result["stop_reasons"] or result["returncode"] != 0)

    def test_clean_exit_records_process_tree_and_stops_a_known_orphan(self):
        with SupervisorFixture() as fixture:
            result = fixture.run()
            self.assertEqual(result["returncode"], 0)
            self.assertIsNone(result["stop_reason"])
            self.assertEqual(result["observed_process_ids"], [11001, 22002])
            self.assertEqual(result["peak_rss_bytes"], 256)
            self.assertEqual(result["owned_processes_remaining"], [])
            self.assertFalse(fixture.nodes[22002].alive)
            self.assertIn(22002, fixture.kill_attempts)
            fixture.launch.assert_called_once()

    def test_postexit_injected_monitor_gap_vetoes_zero_returncode(self):
        with SupervisorFixture(lifetime=0.0) as fixture:
            result = fixture.run(injected_postexit_gap_seconds=6.0)
            self.assertEqual(result["returncode"], 0)
            self.assertIn("monitor_gap", " ".join(result["stop_reasons"]))
            self.assertGreaterEqual(result["wall_seconds"], 6.0)

    def test_final_wait_gap_is_checked_after_exit(self):
        with SupervisorFixture() as fixture:
            fixture.wait_delay = 6.0
            result = fixture.run()
            self.assertIn("monitor_gap", " ".join(result["stop_reasons"]))
            self.assertGreaterEqual(result["wall_seconds"], 6.25)

    def test_cleanup_gap_is_included_in_final_monitor_verdict_and_wall_time(self):
        with SupervisorFixture() as fixture:
            fixture.nodes[22002].kill_delay = 6.0
            result = fixture.run()
            self.assertIn("monitor_gap", " ".join(result["stop_reasons"]))
            self.assertGreaterEqual(result["wall_seconds"], 6.25)

    def test_active_gap_stops_owned_processes(self):
        with SupervisorFixture(lifetime=None) as fixture:
            result = fixture.run(injected_gap_seconds=6.0)
            self.assertIn("monitor_gap", " ".join(result["stop_reasons"]))
            self.assertTrue(all(not node.alive for node in fixture.nodes.values()))

    def test_process_memory_exception_fails_closed_and_cleans_known_processes(self):
        with SupervisorFixture(lifetime=None) as fixture:
            fixture.memory_error_at = 2
            fixture.memory_error = psutil.AccessDenied(pid=11001)
            try:
                result = fixture.run()
            except psutil.AccessDenied:
                pass
            else:
                self.assert_failed(result)
            self.assertTrue(all(not node.alive for node in fixture.nodes.values()))
            self.assertIn(11001, fixture.kill_attempts)
            self.assertIn(22002, fixture.kill_attempts)

    def test_system_memory_exception_fails_closed_and_cleans_known_processes(self):
        with SupervisorFixture(lifetime=None) as fixture:
            fixture.virtual_error = OSError("synthetic unavailable memory sample")
            try:
                result = fixture.run()
            except OSError:
                pass
            else:
                self.assert_failed(result)
            self.assertTrue(all(not node.alive for node in fixture.nodes.values()))

    def test_malformed_memory_sample_cannot_be_treated_as_zero_usage(self):
        for value in (float("nan"), -1.0):
            with self.subTest(field="owned_memory", value=value), SupervisorFixture() as fixture:
                fixture.memory_override = {"rss_bytes": value, "high_water_bytes": value,
                                           "process_ids": [11001, 22002]}
                try:
                    result = fixture.run()
                except ValueError:
                    pass
                else:
                    self.assert_failed(result)
                self.assertTrue(all(not node.alive for node in fixture.nodes.values()))
        for field in ("available", "disk_free"):
            with self.subTest(field=field), SupervisorFixture() as fixture:
                setattr(fixture, field, float("nan"))
                try:
                    result = fixture.run()
                except ValueError:
                    pass
                else:
                    self.assert_failed(result)
                self.assertTrue(all(not node.alive for node in fixture.nodes.values()))

    def test_resistant_owned_descendant_is_reported_as_incomplete_cleanup(self):
        with SupervisorFixture() as fixture:
            fixture.nodes[22002].resists_kill = True
            result = fixture.run()
            self.assertIn("owned descendants remain alive", result["stop_reason"])
            self.assertEqual(result["owned_processes_remaining"], [{"pid": 22002, "create_time": 20.0}])

    def test_one_cleanup_error_does_not_skip_other_known_owned_processes(self):
        with SupervisorFixture(lifetime=None) as fixture:
            fixture.nodes[33003] = FakeNode(fixture, 33003, created=30.0)
            fixture.nodes[33003].kill_error = psutil.AccessDenied(pid=33003)
            emergency = fixture.directory / "emergency_stop.json"
            emergency.write_text("{}", encoding="utf-8")
            try:
                result = fixture.run(emergency_path=emergency)
            except psutil.AccessDenied:
                pass
            else:
                self.assert_failed(result)
            self.assertFalse(fixture.nodes[11001].alive)
            self.assertFalse(fixture.nodes[22002].alive)
            self.assertIn(11001, fixture.kill_attempts)
            self.assertIn(22002, fixture.kill_attempts)

    def test_reused_pid_is_not_treated_as_the_original_owned_descendant(self):
        with SupervisorFixture() as fixture:
            original_child = fixture.nodes[22002]

            def replace_pid():
                original_child.alive = False
                fixture.nodes[22002] = FakeNode(fixture, 22002, created=999.0)

            fixture.on_natural_exit = replace_pid
            result = fixture.run()
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(result["owned_processes_remaining"], [])
            self.assertTrue(fixture.nodes[22002].alive)
            self.assertNotIn(22002, fixture.kill_attempts)

    def test_normal_pause_returncode_is_preserved_without_emergency_reason(self):
        with SupervisorFixture(exit_code=acceptance.PAUSE_EXIT) as fixture:
            result = fixture.run()
            self.assertEqual(result["returncode"], acceptance.PAUSE_EXIT)
            self.assertIsNone(result["stop_reason"])
            self.assertEqual(result["owned_processes_remaining"], [])

    def test_emergency_reason_cannot_be_reclassified_as_normal_pause_by_exit_code(self):
        with SupervisorFixture(lifetime=None) as fixture:
            fixture.kill_exit_code = acceptance.PAUSE_EXIT
            emergency = fixture.directory / "emergency_stop.json"
            emergency.write_text("{}", encoding="utf-8")
            result = fixture.run(emergency_path=emergency)
            self.assertEqual(result["returncode"], acceptance.PAUSE_EXIT)
            self.assertIn("emergency stop requested", result["stop_reasons"])
            self.assertTrue(all(not node.alive for node in fixture.nodes.values()))

    def test_numeric_worker_failure_returncode_is_never_changed_to_normal_pause(self):
        with SupervisorFixture(exit_code=2) as fixture:
            result = fixture.run()
            self.assertEqual(result["returncode"], 2)
            self.assertNotEqual(result["returncode"], acceptance.PAUSE_EXIT)

    def test_lost_guardian_stops_the_owned_tree(self):
        with SupervisorFixture(lifetime=None) as fixture:
            result = fixture.run(guardian_identity={"pid": 99009, "create_time": 90.0})
            self.assertIn("outer guardian is no longer alive", result["stop_reasons"])
            self.assertTrue(all(not node.alive for node in fixture.nodes.values()))


if __name__ == "__main__":
    unittest.main()
