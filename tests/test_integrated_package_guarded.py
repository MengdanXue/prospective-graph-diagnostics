"""Lightweight boundary tests; no actual controller, dataset, ledger or training."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import verify_integrated_package_guarded as reader


class GuardedReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "package"
        self.package.mkdir()
        for name in (reader.READER, reader.VERIFIER):
            (self.package / name).write_text("# fixed synthetic fixture\n", encoding="utf-8")
        files = {p.name: {"sha256": reader.fsha(p), "bytes": p.stat().st_size}
                 for p in self.package.iterdir()}
        self.manifest = self.package / "PACKAGE_MANIFEST.json"
        self.manifest.write_text(json.dumps({"files": files}), encoding="utf-8")
        self.sha = reader.fsha(self.manifest)
        self.output_root = self.root / "output"

    def arguments(self):
        values = {"controller-root": self.root / "controller", "package-root": self.package,
                  "handoff": self.root / "handoff.json", "ledger-root": self.root / "ledger",
                  "output-root": self.output_root, "controller-ci-receipt": self.root / "controller-ci.json",
                  "wrapper-ci-receipt": self.root / "wrapper-ci.json", "package-manifest-sha256": self.sha}
        return [s for k, v in values.items() for s in ("--" + k, str(v))]

    def test_external_manifest_anchor_refuses_refreshed_forgery(self):
        (self.package / reader.READER).write_text("# replacement\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "file hash"):
            reader.package_binding(self.package, self.sha)
        item = json.loads(self.manifest.read_text())
        item["files"][reader.READER] = {"sha256": reader.fsha(self.package / reader.READER),
                                       "bytes": (self.package / reader.READER).stat().st_size}
        self.manifest.write_text(json.dumps(item), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "external binding"):
            reader.package_binding(self.package, self.sha)

    def test_unlisted_python_shadow_is_rejected(self):
        (self.package / "json.py").write_text("# shadow\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unlisted"):
            reader.package_binding(self.package, self.sha)

    def test_wrong_or_dirty_controller_source_is_rejected(self):
        with patch.object(reader.subprocess, "check_output", return_value="f" * 40):
            with self.assertRaisesRegex(ValueError, "wrong controller"):
                reader.source_identity(self.root, reader.CONTROLLER_SHA)
        with patch.object(reader.subprocess, "check_output", side_effect=[reader.CONTROLLER_SHA, " M controller.py\n"]):
            with self.assertRaisesRegex(ValueError, "clean committed"):
                reader.source_identity(self.root, reader.CONTROLLER_SHA)

    def test_default_preparation_is_read_only_and_never_imports_runtime(self):
        before = {p.relative_to(self.root).as_posix(): reader.fsha(p) for p in self.root.rglob("*") if p.is_file()}
        with patch.object(reader, "source_identity", return_value=reader.CONTROLLER_SHA), \
             patch.object(reader.importlib, "import_module", side_effect=AssertionError("no execution import")), \
             redirect_stdout(StringIO()) as output:
            self.assertEqual(reader.main(self.arguments()), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["budget_group"], "control")
        self.assertEqual(report["operation"], "journal_reader_reconstruction")
        self.assertFalse(report["launch_enabled"])
        self.assertFalse(self.output_root.exists())
        after = {p.relative_to(self.root).as_posix(): reader.fsha(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_cli_rejects_arbitrary_command_and_reader_path(self):
        for option in ("--command", "--reader"):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                reader.parser().parse_args(self.arguments() + [option, "anything"])
        command = reader.reader_command(self.package, self.root / "output.json")
        self.assertEqual(command[2], str((self.package / reader.READER).resolve()))
        self.assertEqual(command[3], "--output")
        self.assertEqual(len(command), 5)

    def test_only_child_is_specialized_control_admission_is_inherited(self):
        class AcceptedController:
            def run_preparation(self, request, *, attempt_id, activity_id):
                self.ledger.begin_attempt(attempt_id, budget_group="control")
                return self._child(request, attempt_id=attempt_id)

        calls = []
        def supervised(command, **kwargs):
            calls.append((command, kwargs))
            Path(command[-1]).write_text("{}", encoding="utf-8")
            return {"returncode": 0, "stop_reasons": [], "owned_processes_remaining": []}

        api = SimpleNamespace(ExtensionExecutionController=AcceptedController, supervise_process=supervised,
                              _exclusive_json=lambda path, result: None, FormalEmergencyStop=RuntimeError)
        output = self.root / "result.json"
        binding = reader.package_binding(self.package, self.sha)
        cls = reader.make_controller(api, self.package, output, binding)
        self.assertIs(cls.run_preparation, AcceptedController.run_preparation)
        instance = cls()
        instance.writer = SimpleNamespace(root=self.output_root)
        instance._supervisor_poll = lambda *_: []
        instance._monitor = lambda: None
        charged = []
        instance.ledger = SimpleNamespace(begin_attempt=lambda *a, **kw: charged.append(kw["budget_group"]))
        result = instance.run_preparation({"operation": reader.OPERATION}, attempt_id="fixture", activity_id="fixture")
        self.assertEqual(charged, ["control"])
        self.assertFalse(result["training_started"])
        self.assertEqual(calls[0][1]["poll_seconds"], 0.25)
        self.assertEqual(calls[0][1]["worker_cap_seconds"], 7200)
        self.assertNotIn("PYTHONPATH", calls[0][1]["environment"])
        with self.assertRaisesRegex(ValueError, "unsupported"):
            instance._child({"operation": "unit"}, attempt_id="wrong")
        (self.package / reader.READER).write_text("# swapped after binding\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "replaced"):
            instance._child({"operation": reader.OPERATION}, attempt_id="replacement")


if __name__ == "__main__":
    unittest.main()
