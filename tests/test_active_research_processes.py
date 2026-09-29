"""Real OS process-tree coverage for active_research_processes().

Spawns short-lived Python (and shell) subprocesses to exercise the actual
psutil-based process tree the function walks. No research code, ledger, or
training is touched; every spawned process is a throwaway script under a
TemporaryDirectory.
"""
import json
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "scripts.mlp_budget_extension_entry"

CHECKER_TEMPLATE = textwrap.dedent("""\
    import sys, json
    sys.path.insert(0, {root!r})
    from scripts import supplement_budget_handoff as sbh
    result = sbh.active_research_processes()
    with open({result_path!r}, "w", encoding="utf-8") as stream:
        json.dump(result, stream)
""")


class ActiveResearchProcessesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self._procs = []
        self.addCleanup(self._reap)

    def _reap(self):
        for proc in self._procs:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass

    def _spawn(self, argv):
        proc = subprocess.Popen(argv)
        self._procs.append(proc)
        return proc

    def _checker_script(self, name):
        result_path = self.dir / f"{name}.result.json"
        script_path = self.dir / f"{name}.py"
        script_path.write_text(
            CHECKER_TEMPLATE.format(root=str(ROOT), result_path=str(result_path)),
            encoding="utf-8")
        return script_path, result_path

    def _read_result(self, result_path, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if result_path.is_file():
                text = result_path.read_text(encoding="utf-8")
                if text:
                    return json.loads(text)
            time.sleep(0.05)
        raise TimeoutError(f"checker never wrote a result to {result_path}")

    # 1. self launcher false positive -------------------------------------
    def test_self_launcher_chain_is_not_a_false_positive(self):
        """launcher(python, marker in its own argv) -> checker(python, os.getpid()).

        Reproduces the real venv Scripts/python.exe -> base-interpreter
        structure measured on the target machine (launcher/interpreter
        create_time gap ~0.035s). Must FAIL against the unfixed code: the
        launcher is an ancestor, ancestors are never excluded pre-fix.
        """
        checker_script, result_path = self._checker_script("self_launcher_checker")
        launcher_script = self.dir / "self_launcher.py"
        launcher_script.write_text(textwrap.dedent(f"""\
            import subprocess, sys
            child = subprocess.Popen([sys.executable, {str(checker_script)!r}])
            child.wait()
        """), encoding="utf-8")
        launcher = self._spawn([sys.executable, str(launcher_script), MARKER])
        result = self._read_result(result_path)
        launcher.wait(timeout=15)
        self.assertEqual(
            result, [],
            f"self launcher ancestor was misidentified as an existing research process: {result}")

    # 2. independent existing research process must still be caught -------
    def test_independent_existing_process_is_still_detected(self):
        """A sibling process (no ancestor/descendant relation) with the marker."""
        checker_script, result_path = self._checker_script("independent_checker")
        holder_script = self.dir / "marker_holder.py"
        holder_script.write_text("import time\ntime.sleep(10)\n", encoding="utf-8")
        holder = self._spawn([sys.executable, str(holder_script), MARKER])
        time.sleep(0.3)  # let the holder register in the OS process table
        checker = self._spawn([sys.executable, str(checker_script)])
        checker.wait(timeout=15)
        result = self._read_result(result_path)
        pids = {entry["pid"] for entry in result}
        self.assertIn(
            holder.pid, pids,
            f"an unrelated, independent process with the marker was not detected: {result}")

    # 3. an old ancestor must not be blanket-excluded ----------------------
    def test_old_unrelated_python_ancestor_is_still_detected(self):
        """A python ancestor whose create_time predates this invocation by 5s.

        5s is chosen to sit far outside the fix's launcher-chain tolerance
        (set from the measured ~0.035s real launcher/interpreter gap), so
        this cannot flake into "close enough" by coincidence.
        """
        checker_script, result_path = self._checker_script("old_ancestor_checker")
        old_launcher_script = self.dir / "old_launcher.py"
        old_launcher_script.write_text(textwrap.dedent(f"""\
            import subprocess, sys, time
            time.sleep(5)
            child = subprocess.Popen([sys.executable, {str(checker_script)!r}])
            child.wait()
        """), encoding="utf-8")
        old_ancestor = self._spawn([sys.executable, str(old_launcher_script), MARKER])
        result = self._read_result(result_path, timeout=20)
        old_ancestor.wait(timeout=15)
        pids = {entry["pid"] for entry in result}
        self.assertIn(
            old_ancestor.pid, pids,
            "an old ancestor (5s create_time gap) with the marker was incorrectly "
            f"excluded just for being an ancestor: {result}")

    # 4. a non-python ancestor must not be leapfrogged ---------------------
    def test_non_python_ancestor_blocks_chain_walk(self):
        """grandparent(python, marker) -> parent(shell, non-python) -> checker(python).

        The walk must stop at the first non-python ancestor (the shell) and
        must not jump over it to exempt the python grandparent.
        """
        checker_script, result_path = self._checker_script("nonpython_ancestor_checker")
        if sys.platform == "win32":
            # Pass argv elements separately so Python's own list2cmdline
            # quotes each one; a pre-joined quoted string here gets
            # re-quoted as a single doubly-quoted token and cmd.exe fails
            # to resolve it as a program name.
            shell_argv = ["cmd.exe", "/c", sys.executable, str(checker_script)]
        else:
            # "& wait" forces sh to fork a persisting child instead of
            # exec-replacing itself into the python process.
            shell_command = f'"{sys.executable}" "{checker_script}" & wait'
            shell_argv = ["/bin/sh", "-c", shell_command]
        grandparent_script = self.dir / "nonpython_grandparent.py"
        grandparent_script.write_text(textwrap.dedent(f"""\
            import subprocess
            child = subprocess.Popen({shell_argv!r})
            child.wait()
        """), encoding="utf-8")
        grandparent = self._spawn([sys.executable, str(grandparent_script), MARKER])
        result = self._read_result(result_path, timeout=15)
        grandparent.wait(timeout=15)
        pids = {entry["pid"] for entry in result}
        self.assertIn(
            grandparent.pid, pids,
            "a python grandparent with the marker, separated from the checker by a "
            f"non-python shell hop, must still be detected, not leapfrogged: {result}")


if __name__ == "__main__":
    unittest.main()
