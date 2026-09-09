"""Verify the final evidence statement and anonymous-copy boundaries."""
import copy
import io
import json
from pathlib import Path
import unittest

from pypdf import PdfReader, PdfWriter
from scripts.build_tmlr_review_package import (
    anonymous_bytes, check_json_projection, redact_json, safe_name,
)
from scripts.check_tmlr_freeze import check


class TmlrFreezeTests(unittest.TestCase):
    def test_final_lodo_claim_is_bound_to_fold_records(self):
        report = check(Path(__file__).resolve().parents[1])
        self.assertEqual(report["constant_graph_folds"], 10)
        self.assertAlmostEqual(report["roman_share_of_calibrated_regret_pct"], 89.3697979456374)

    def test_json_redaction_keeps_scientific_leaves_and_external_names(self):
        original = {"environment": {"python": "C:/Users/example/project/python.exe"},
                    "author": "John Smith", "accuracy": 0.6619,
                    "history": [None, True, 1, "GPR-GNN"], "hash": "a" * 64}
        saved = copy.deepcopy(original)
        result = redact_json(original)
        self.assertEqual(check_json_projection(original, result), 1)
        self.assertEqual(original, saved)
        self.assertEqual(result["author"], "John Smith")
        self.assertEqual(result["accuracy"], 0.6619)
        corrupted = copy.deepcopy(result)
        corrupted["accuracy"] = 0.8
        with self.assertRaises(AssertionError):
            check_json_projection(original, corrupted)

    def test_unmodified_json_preserves_exact_bytes(self):
        raw = b'{"accuracy":0.6619, "dataset":"Roman-empire"}\n'
        delivered, edits = anonymous_bytes("record.json", raw)
        self.assertEqual(delivered, raw)
        self.assertEqual(edits, 0)

    def test_pdf_path_metadata_is_removed_without_losing_pages(self):
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=200)
        writer.add_metadata({"/Author": "C:/Users/example/research"})
        source = io.BytesIO()
        writer.write(source)
        delivered, edits = anonymous_bytes("figure.pdf", source.getvalue())
        reader = PdfReader(io.BytesIO(delivered))
        self.assertEqual(len(reader.pages), 1)
        self.assertEqual(reader.pages[0].mediabox.width, 100)
        self.assertNotIn("example", str(reader.metadata))
        self.assertEqual(edits, 1)

    def test_archive_names_reject_escape_and_platform_specific_paths(self):
        for name in ("../private.txt", "/absolute.txt", "C:/private.txt", r"a\b.txt", "a//b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_name(name)
        safe_name("records/graph/seed_000.json")


if __name__ == "__main__":
    unittest.main()
