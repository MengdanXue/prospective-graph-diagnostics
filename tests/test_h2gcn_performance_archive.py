"""Compact evidence-only checks; no graph data or training is executed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import check_h2gcn_performance_archive as audit


class H2PerformanceArchiveTests(unittest.TestCase):
    def test_inventory_detects_changed_bytes_and_unlisted_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "record.json"
            payload.write_text('{"formal_records": 0}\n', encoding="utf-8")
            inventory = {"phase": "performance_diagnostic_only", "run_id": "h2gcn_cpu_performance_v1",
                         "files": {payload.name: {"sha256": audit.sha(payload), "bytes": payload.stat().st_size}}}
            (root / "diagnostic_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
            self.assertEqual(len(audit.inspect_inventory(root)), 2)
            original = payload.read_bytes()
            payload.write_bytes(original.replace(b"0", b"1"))
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                audit.inspect_inventory(root)
            payload.write_bytes(original)
            (root / "unlisted.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage"):
                audit.inspect_inventory(root)

    def test_unavailable_evidence_is_not_a_measured_numerical_failure(self):
        self.assertEqual(audit.classify_comparison(False, True, False), "unavailable_incomplete_scope")
        self.assertEqual(audit.classify_comparison(True, False, False), "unavailable_incomplete_scope")
        self.assertEqual(audit.classify_comparison(True, True, False), "measured_mismatch")
        self.assertEqual(audit.classify_comparison(True, True, True), "passed")
        with self.assertRaisesRegex(ValueError, "explicit measured verdict"):
            audit.classify_comparison(True, True)

    def test_incomplete_or_resource_stopped_scope_never_qualifies(self):
        fields = {"complete": True, "resource_stopped": False, "ratios": [0.5] * 4,
                  "speed": 2.0, "candidate_rejected": False, "reference_rejected": False}
        self.assertTrue(audit.setting_qualifies(**fields))
        for override in ({"complete": False}, {"resource_stopped": True}, {"candidate_rejected": True},
                         {"reference_rejected": True}, {"ratios": [0.5] * 3}, {"speed": 1.099999},
                         {"ratios": [0.5, 0.5, 0.5, 1.100001]}):
            with self.subTest(override=override):
                self.assertFalse(audit.setting_qualifies(**{**fields, **override}))
        self.assertTrue(audit.setting_qualifies(**{**fields, "speed": 1.10, "ratios": [0.8, 0.8, 0.8, 1.10]}))

    def test_held_out_scores_and_formal_switches_are_rejected_recursively(self):
        audit.zero_scores({"steps": [{"train_loss": 0.5}], "formal_records": 0, "formal_training_enabled": False})
        for forbidden in ({"test_accuracy": 0.8}, {"validation_loss": 0.2}, {"test_evaluations": 1},
                          {"formal_records": 1}, {"formal_training_enabled": True}):
            with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                audit.zero_scores({"nested": [forbidden]})

    def test_bound_text_accepts_checkout_newlines_but_rejects_content_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.py"
            expected = hashlib.sha256(b"first\nsecond\n").hexdigest()
            path.write_bytes(b"first\r\nsecond\r\n")
            self.assertTrue(audit.text_digest_matches(path, expected))
            path.write_bytes(b"first\r\nchanged\r\n")
            self.assertFalse(audit.text_digest_matches(path, expected))

    def test_independent_numeric_comparison_preserves_rng_exactness_and_unavailable_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chosen = next(row for row in audit.comparison_plan() if row[0] == "cross_thread")
            kind, left_id, right_id, _ = chosen
            rows = {}
            for wid, threads, rng in ((left_id, 1, 7), (right_id, 4, 8)):
                destination = root / "workers" / wid
                destination.mkdir(parents=True)
                np.savez(destination / "numeric_evidence.npz", parameters=np.array([1.0], dtype=np.float32),
                         rng_before=np.array([rng], dtype=np.uint8))
                rows[wid] = {"cpu_threads": threads, "checkpoint_mode": "full_state_copy", "split_id": "fixed",
                             "transformed_feature_sha256": "fixed", "trial": {}, "initial_fingerprint": {}, "environment": {}, "steps": []}
            comparisons = [{"kind": kind, "left": left, "right": right, "passed": False}
                           for kind, left, right, _ in audit.comparison_plan()]
            measured = next(row for row in comparisons if (row["kind"], row["left"], row["right"]) == chosen[:3])
            measured.update(comparison="fixed_allclose_with_exact_rng", mismatched_arrays=["rng_before"],
                            maximum_absolute_difference=0.0, maximum_tolerance_fraction=0.0)
            summary = {"comparisons": comparisons, "fresh_repeat_pairs_passed": 0,
                       "checkpoint_mode_pairs_passed": 0, "cross_thread_pairs_passed": 0}
            verified, _ = audit.inspect_comparisons(root, summary, rows)
            self.assertEqual(sum(row["classification"] == "unavailable_incomplete_scope" for row in verified), 79)
            self.assertEqual(sum(row["classification"] == "measured_mismatch" for row in verified), 1)
            actual = next(row for row in verified if row["classification"] == "measured_mismatch")
            self.assertEqual(actual["mismatched_arrays"], ["rng_before"])
            self.assertTrue(actual["invariants_passed"])


if __name__ == "__main__":
    unittest.main()
