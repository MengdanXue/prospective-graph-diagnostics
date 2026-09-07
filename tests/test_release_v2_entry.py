import contextlib
import io
import unittest
from unittest import mock

from scripts import run_release_v2 as entry


class ReleaseV2EntryTests(unittest.TestCase):
    def test_every_workflow_explicitly_selects_v2_and_propagates_exit_status(self):
        for name, script in entry.COMMANDS.items():
            with self.subTest(command=name), mock.patch.object(entry.subprocess, "run") as run:
                run.return_value.returncode = 7
                self.assertEqual(entry.main([name, "--output-root", "a path with spaces"]), 7)
                args = run.call_args.args[0]
                self.assertEqual(args[1], str(entry.ROOT / script))
                self.assertEqual(args[2:4], ["--config", str(entry.CONFIG)])
                self.assertEqual(args[4:], ["--output-root", "a path with spaces"])
                self.assertEqual(entry.CONFIG.name, "prospective_benchmark_v2.json")

    def test_conflicting_configuration_is_rejected_before_dispatch(self):
        for options in (["--config", "v1.json"], ["--config=v1.json"]):
            with self.subTest(options=options), mock.patch.object(entry.subprocess, "run") as run:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    entry.main(["benchmark", *options])
                run.assert_not_called()
