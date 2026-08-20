import json
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.summarize_reviewer_appendix import summarize_audit


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "diagnostic" / "route_a_prospective_v2" / "analysis"


class ReviewerAppendixSummaryTests(unittest.TestCase):
    def test_published_summary_reconstructs_from_the_published_audit(self):
        audit = json.loads((ANALYSIS / "diagnostic_audit.json").read_text(encoding="utf-8"))
        published = json.loads(
            (ANALYSIS / "reviewer_appendix_summary.json").read_text(encoding="utf-8")
        )
        metadata = {
            dataset: {
                key: row[key]
                for key in ("node_count", "edge_count", "class_count", "feature_count")
            }
            for dataset, row in published["datasets"].items()
        }
        rebuilt = summarize_audit(audit, metadata)
        self.assertEqual(published["record_counts"], rebuilt["record_counts"])
        self.assertEqual(published["overall"], rebuilt["overall"])
        self.assertEqual(published["fallback_sensitivity"], rebuilt["fallback_sensitivity"])
        self.assertEqual(published["datasets"], rebuilt["datasets"])

    def test_cli_help_is_available_without_loading_datasets(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "summarize_reviewer_appendix.py"), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("--data-root", completed.stdout)


if __name__ == "__main__":
    unittest.main()
