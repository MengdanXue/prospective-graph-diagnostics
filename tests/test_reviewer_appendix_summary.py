import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_reviewer_appendix import rebuild_from_retained_metadata, summarize_audit, verify_audit_binding


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "diagnostic" / "route_a_prospective_v2" / "analysis"
RETAINED = ROOT / "docs/applied-intelligence/provenance/reviewer_appendix_summary.pre-2026-09-28.json"


class ReviewerAppendixSummaryTests(unittest.TestCase):
    def test_published_summary_reconstructs_from_the_published_audit(self):
        audit_bytes = (ANALYSIS / "diagnostic_audit.json").read_bytes()
        audit = json.loads(audit_bytes)
        published = json.loads(
            (ANALYSIS / "reviewer_appendix_summary.json").read_text(encoding="utf-8")
        )
        verify_audit_binding(published, audit_bytes)
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

    def test_retained_summary_is_preserved_and_new_binding_reconstructs_exactly(self):
        retained_bytes = RETAINED.read_bytes()
        self.assertEqual(
            "d826f13959469402862fcd96c4e39b8255bc9a4b75a60192dd1395793d28e7d2",
            hashlib.sha256(retained_bytes).hexdigest(),
        )
        audit_bytes = (ANALYSIS / "diagnostic_audit.json").read_bytes()
        rebuilt = rebuild_from_retained_metadata(
            audit_bytes, retained_bytes, audit_name="diagnostic_audit.json",
            retained_summary_name=RETAINED.relative_to(ROOT).as_posix(),
        )
        published = json.loads((ANALYSIS / "reviewer_appendix_summary.json").read_bytes())
        self.assertEqual(published, rebuilt)
        with self.assertRaisesRegex(ValueError, "source audit SHA-256 mismatch"):
            verify_audit_binding(json.loads(retained_bytes), audit_bytes)
        # Even a semantically neutral byte change must invalidate byte provenance.
        with self.assertRaisesRegex(ValueError, "source audit SHA-256 mismatch"):
            verify_audit_binding(published, audit_bytes + b"\n")

    def test_retained_metadata_path_rejects_a_changed_scientific_result(self):
        audit_bytes = (ANALYSIS / "diagnostic_audit.json").read_bytes()
        altered = copy.deepcopy(json.loads(audit_bytes))
        altered["units"][0]["selected_graph_test"] += 0.125
        with self.assertRaisesRegex(ValueError, "scientific fields differ"):
            rebuild_from_retained_metadata(
                json.dumps(altered).encode(), RETAINED.read_bytes(),
                audit_name="diagnostic_audit.json", retained_summary_name=RETAINED.name,
            )

    def test_retained_metadata_cli_needs_no_dataset_cache_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "summary.json"
            command = [sys.executable, str(ROOT / "scripts/summarize_reviewer_appendix.py"),
                       "--audit", str(ANALYSIS / "diagnostic_audit.json"),
                       "--metadata-summary", RETAINED.relative_to(ROOT).as_posix(),
                       "--output", str(output)]
            first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(0, first.returncode, first.stderr)
            published = json.loads((ANALYSIS / "reviewer_appendix_summary.json").read_bytes())
            self.assertEqual(published, json.loads(output.read_bytes()))
            original_bytes = output.read_bytes()
            second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(0, second.returncode)
            self.assertEqual(original_bytes, output.read_bytes())

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
