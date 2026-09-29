"""Reject incomplete or stale CI evidence at the research entry."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.input_robustness_formal_records import FormalRecordError
from scripts.mlp_budget_extension_entry import strict_ci_gate


class ExtensionCIGateTests(unittest.TestCase):
    def test_only_complete_unique_same_commit_same_run_remote_jobs_pass(self):
        commit = "a" * 40
        jobs = [{"id": index + 10, "name": name, "run_id": 123, "commit": commit,
                 "status": "completed", "conclusion": "success"}
                for index, name in enumerate(("lightweight-verification", "full-protocol-verification", "manuscript-build"))]
        valid = {"status": "passed", "commit": commit, "ci": {
            "kind": "github_actions", "commit": commit, "run_id": 123,
            "status": "completed", "conclusion": "success", "jobs": jobs}}
        invalid = []
        row = copy.deepcopy(valid)
        row["ci"]["jobs"].pop()
        invalid.append(("missing job", row))
        row = copy.deepcopy(valid)
        row["ci"]["jobs"][2] = copy.deepcopy(row["ci"]["jobs"][0])
        invalid.append(("duplicate job", row))
        for key, value in (("commit", "b" * 40), ("run_id", 124), ("conclusion", "failure")):
            row = copy.deepcopy(valid)
            row["ci"]["jobs"][1][key] = value
            invalid.append((key, row))
        row = copy.deepcopy(valid)
        row["ci"]["kind"] = "local_verification"
        invalid.append(("local evidence", row))
        row = copy.deepcopy(valid)
        row["ci"]["run_id"] = None
        for job in row["ci"]["jobs"]:
            job["run_id"] = None
        invalid.append(("null shared run", row))
        for value in (None, 10):
            row = copy.deepcopy(valid)
            row["ci"]["jobs"][2]["id"] = value
            invalid.append((f"invalid job id {value}", row))
        invalid.append(("bare success", {"status": "passed", "commit": commit}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ci.json"
            path.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(strict_ci_gate(path, commit)["run_id"], 123)
            with self.assertRaises(FormalRecordError):
                strict_ci_gate(path, "b" * 40)
            for label, receipt in invalid:
                with self.subTest(label=label):
                    path.write_text(json.dumps(receipt), encoding="utf-8")
                    with self.assertRaises(FormalRecordError):
                        strict_ci_gate(path, commit)


if __name__ == "__main__":
    unittest.main()
