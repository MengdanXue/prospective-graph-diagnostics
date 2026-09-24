"""Verify the final evidence statement and anonymous-copy boundaries."""
import copy
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
import zipfile

from pypdf import PdfReader, PdfWriter
from scripts.build_tmlr_review_package import (
    anonymous_bytes, check_json_projection, redact_json, redact_text, safe_name,
    ReviewAliases, add_review_file, assert_anonymous_text, canonical_digest,
    digest, prepare_review_inputs, write_review_zip,
    rebind_size_metadata,
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

    def test_review_alias_is_consistent_and_preserves_original_evidence(self):
        original = {"benchmark_id": "route_a_diagnostic_v1", "accuracy": 0.746,
                    "source_commit": "dca835a" + "a" * 33,
                    "source_sha256": "b" * 64, "attempts": [None, True, 4]}
        saved = copy.deepcopy(original)
        aliases = ReviewAliases.discover([("record.json", json.dumps(original).encode(), "fixture")])
        review = redact_json(original, aliases)
        self.assertEqual(original, saved)
        self.assertEqual(check_json_projection(original, review, aliases), 2)
        self.assertEqual(review["benchmark_id"], "benchmark_spec_v1")
        self.assertEqual(review["source_commit"], "source-revision-001")
        self.assertEqual(review["source_sha256"], original["source_sha256"])
        source = 'BENCHMARK_ID = "route_a_diagnostic_v1"'
        self.assertEqual(redact_text(source), 'BENCHMARK_ID = "benchmark_spec_v1"')
        self.assertEqual(redact_text(review["benchmark_id"]), review["benchmark_id"])
        corrupted = copy.deepcopy(review)
        corrupted["accuracy"] = 0.1
        with self.assertRaises(AssertionError):
            check_json_projection(original, corrupted, aliases)

    def test_review_presentation_removes_targeted_linkable_identifiers(self):
        source = r"\texttt{route\_a\_diagnostic\_v1} (dca835a)"
        review = redact_text(source)
        self.assertEqual(review, r"\texttt{benchmark\_spec\_v1} ([pre-freeze source revision])")
        self.assertEqual(redact_text(review), review)

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

    def test_revision_discovery_requires_semantics_and_preserves_data_hashes(self):
        sha = "1" * 40
        inputs = [("record.json", json.dumps({"source_commit": sha, "model_sha256": "2" * 64}).encode(), "fixture"),
                  (f"records/source_snapshot/{sha}/code.py", f'COMMIT = "{sha}"'.encode(), "fixture")]
        aliases = ReviewAliases.discover(inputs)
        self.assertEqual(redact_text(inputs[1][0], aliases), "records/source_snapshot/source-revision-001/code.py")
        result = redact_json(json.loads(inputs[0][1]), aliases)
        self.assertEqual(result["model_sha256"], "2" * 64)
        self.assertEqual(result["source_commit"], "source-revision-001")
        with self.assertRaisesRegex(ValueError, "unclassified 40-hex"):
            ReviewAliases.discover([("unknown.txt", ("3" * 40).encode(), "fixture")])

    def test_path_json_key_and_python_import_aliases_are_consistent(self):
        aliases = ReviewAliases(["4" * 40])
        original = {"results/route_a/record.json": {"source_commit": "4" * 40, "score": .6}}
        review = redact_json(original, aliases)
        self.assertEqual(review["results/benchmark/record.json"]["score"], .6)
        check_json_projection(original, review, aliases)
        self.assertEqual(redact_text("from scripts.audit_route_a_claims import main", aliases),
                         "from scripts.audit_benchmark_claims import main")
        with self.assertRaisesRegex(ValueError, "key collision"):
            redact_json({"route_a": 1, "benchmark": 2}, aliases)
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            manifest, private = {"files": {}}, {}
            add_review_file(stage, manifest, private, aliases, "route_a.json", b'{"score":1}', "fixture")
            with self.assertRaisesRegex(ValueError, "colliding"):
                add_review_file(stage, manifest, private, aliases, "benchmark.json", b'{"score":2}', "fixture")

    def test_derived_bindings_match_review_config_records_and_nested_manifest(self):
        config = {"run_id": "route_a_prospective_v2", "models": ["MLP"], "seeds": [0], "trials": 4}
        row = {"source_commit": "5" * 40, "config_sha256": canonical_digest(config),
               "frozen_config": config, "test_accuracy": .67, "raw_sha256": "6" * 64}
        row_bytes = (json.dumps(row, indent=2) + "\n").encode()
        manifest = {"archive_schema_version": "1.0", "files": [{"path": "prospective/records/x.json",
                    "group": "prospective_records", "public_sha256": digest(row_bytes), "public_bytes": len(row_bytes),
                    "source_sha256": "7" * 64, "source_bytes": 1000, "redactions": 1}]}
        license_bytes = b"MIT License\nCopyright MengdanXue\n"
        inputs = [("configs/c.json", (json.dumps(config, indent=2) + "\n").encode(), "fixture"),
                  ("records/frozen/prospective/records/x.json", row_bytes, "fixture"),
                  ("records/frozen/MANIFEST.json", json.dumps(manifest).encode(), "fixture"),
                  ("LICENSE", license_bytes, "fixture"),
                  ("old_manifest.json", json.dumps({"original_sha256": digest(license_bytes)}).encode(), "fixture")]
        saved = copy.deepcopy(inputs)
        aliases = ReviewAliases.discover(inputs)
        rendered, metadata = prepare_review_inputs(inputs, aliases)
        self.assertEqual(inputs, saved)
        review_config = json.loads(rendered["configs/c.json"][0])
        review_row = json.loads(rendered["records/frozen/prospective/records/x.json"][0])
        self.assertEqual(review_row["config_sha256"], canonical_digest(review_config))
        self.assertEqual(review_row["frozen_config"], review_config)
        self.assertEqual(review_row["test_accuracy"], row["test_accuracy"])
        self.assertEqual(review_row["raw_sha256"], row["raw_sha256"])
        public_entry = json.loads(rendered["records/frozen/MANIFEST.json"][0])["files"][0]
        self.assertEqual(public_entry["public_sha256"], digest(rendered["records/frozen/prospective/records/x.json"][0]))
        self.assertEqual(public_entry["public_bytes"], len(rendered["records/frozen/prospective/records/x.json"][0]))
        self.assertEqual(public_entry["source_bytes"], 1000)
        self.assertTrue(public_entry["source_sha256"].startswith("archived-byte-digest-"))
        for name, (data, _count) in rendered.items():
            self.assertNotIn(digest(license_bytes).encode(), data)
            assert_anonymous_text(redact_text(name, aliases), data.decode(), aliases)
        self.assertIn("records/frozen/MANIFEST.json", metadata)
        corrupt = copy.deepcopy(review_row)
        corrupt["test_accuracy"] = .8
        with self.assertRaises(AssertionError):
            check_json_projection(row, corrupt, aliases)

    def test_private_original_hashes_never_enter_review_manifest_or_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            stage = output / "anonymous_supplement"
            stage.mkdir()
            manifest, private = {"files": {}}, {}
            original = b"MIT License\nCopyright MengdanXue\n"
            add_review_file(stage, manifest, private, ReviewAliases(), "LICENSE", original, "fixture")
            self.assertNotIn("original_sha256", manifest["files"]["LICENSE"])
            self.assertEqual(private["LICENSE"]["original_sha256"], digest(original))
            (stage / "REVIEW_MANIFEST.json").write_text(json.dumps(manifest))
            (output / "author-only-provenance.json").write_text(json.dumps(private))
            zip_path = output / "supplement.zip"
            write_review_zip(stage, zip_path, manifest)
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(set(archive.namelist()), {"LICENSE", "REVIEW_MANIFEST.json"})
                self.assertNotIn(digest(original).encode(), archive.read("REVIEW_MANIFEST.json"))

    def test_nested_source_sizes_follow_bound_review_bytes_only(self):
        source = b'import scripts.audit_route_a_claims\n'
        source_sha = digest(source)
        other_sha = "f" * 64
        nested = {"audit": {"source_files": {"scripts/route_a.py": {
                    "sha256": source_sha, "size": len(source), "bytes": len(source), "score": .67}}},
                  "omitted_model": {"sha256": other_sha, "size": 3000}, "scientific_size": 10}
        inputs = [("scripts/route_a.py", source, "fixture"),
                  ("source_binding.json", json.dumps(nested).encode(), "fixture")]
        aliases = ReviewAliases.discover(inputs)
        rendered, changes = prepare_review_inputs(inputs, aliases)
        result = json.loads(rendered["source_binding.json"][0])
        bound = result["audit"]["source_files"]["scripts/benchmark.py"]
        self.assertEqual(bound["sha256"], digest(rendered["scripts/route_a.py"][0]))
        self.assertEqual(bound["size"], len(rendered["scripts/route_a.py"][0]))
        self.assertEqual(bound["bytes"], bound["size"])
        self.assertEqual(bound["score"], .67)
        self.assertEqual(result["omitted_model"], nested["omitted_model"])
        self.assertEqual(result["scientific_size"], 10)
        self.assertEqual(len(changes["source_binding.json"]), 2)
        false_size = {"sha256": source_sha, "size": 0}
        with self.assertRaisesRegex(ValueError, "original byte count"):
            rebind_size_metadata(false_size, copy.deepcopy(false_size),
                                 {source_sha: (len(source), rendered["scripts/route_a.py"][0])}, aliases)

    def test_review_validator_requires_exact_registered_alias_without_weakening_other_checks(self):
        source = (Path(__file__).resolve().parents[1] / "scripts/validate_preprocessing_records.py").read_bytes()
        aliases = ReviewAliases(["8" * 40, "9" * 40])
        rendered, _ = anonymous_bytes("scripts/validate_preprocessing_records.py", source, aliases)
        expected = redact_text(source.decode(), aliases)
        expected = expected.replace('_COMMIT = re.compile(r"[0-9a-f]{40}\\Z")',
                                    '_COMMIT = re.compile(r"(?:source\\-revision\\-001|source\\-revision\\-002)\\Z")')
        expected = expected.replace("source_commit must be a full 40-hex SHA", "source_commit must be a registered review revision alias")
        self.assertEqual(rendered.decode(), expected)
        pattern = re.search(r'^_COMMIT = re.compile\(r"(.*)"\)$', rendered.decode(), re.M).group(1)
        matcher = re.compile(pattern)
        self.assertIsNotNone(matcher.fullmatch("source-revision-001"))
        for invalid in ("source-revision-999", "8" * 40, "", "source-revision-001-extra"):
            self.assertIsNone(matcher.fullmatch(invalid))

    def test_scan_checks_paths_and_unknown_revisions_as_well_as_text(self):
        aliases = ReviewAliases(["a" * 40])
        for name, text in (("route_a/file.txt", "safe"), ("safe.txt", "a" * 40),
                           ("safe.txt", "b" * 40), ("safe.txt", r"route\_a")):
            with self.subTest(name=name, text=text), self.assertRaises(AssertionError):
                assert_anonymous_text(name, text, aliases)


if __name__ == "__main__":
    unittest.main()
