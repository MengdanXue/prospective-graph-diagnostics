"""Exercise integrity failures and immutable output behavior with small ZIPs."""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.package_diagnostic_artifacts import add_entry, readme, write_zip
from scripts.verify_diagnostic_artifacts import file_sha256, verify_archive


class DiagnosticArtifactPackagingTests(unittest.TestCase):
    def make_archive(self, root):
        path = root / "records.zip"
        write_zip(path, {"records/unit.json": b'{"accuracy": 0.5}\n'},
                  {"package_name": "fixture", "package_kind": "supplementary controls"})
        return path

    def repack(self, path, entries):
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in entries.items():
                archive.writestr(name, data)
        path.with_suffix(".zip.sha256").write_text(f"{file_sha256(path)}  {path.name}\n")

    def test_round_trip_covers_readme_and_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self.make_archive(root / "first")
            second = self.make_archive(root / "second")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            report = verify_archive(first)
            self.assertEqual(report["verified_entries"], 2)
            with zipfile.ZipFile(first) as archive:
                self.assertIn("README_rebuild.md", json.loads(archive.read("hash_manifest.json"))["files"])

    def test_existing_archive_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.make_archive(root)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self.make_archive(root)
            self.assertEqual(path.read_bytes(), before)

    def test_entry_tampering_fails_even_with_updated_zip_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_archive(Path(tmp))
            with zipfile.ZipFile(path) as archive:
                entries = {n: archive.read(n) for n in archive.namelist()}
            entries["records/unit.json"] = b'{"accuracy": 0.9}\n'
            self.repack(path, entries)
            with self.assertRaisesRegex(ValueError, "entry integrity mismatch"):
                verify_archive(path)

    def test_unlisted_entry_fails_even_with_updated_zip_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_archive(Path(tmp))
            with zipfile.ZipFile(path) as archive:
                entries = {n: archive.read(n) for n in archive.namelist()}
            entries["unlisted.txt"] = b"uncovered"
            self.repack(path, entries)
            with self.assertRaisesRegex(ValueError, "incomplete manifest coverage"):
                verify_archive(path)

    def test_unsafe_entry_paths_are_rejected(self):
        for name in ("../x", "/x", "C:/x", "a/../x", "a\\x"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                add_entry({}, name, b"data")

    def test_readmes_describe_the_actual_package_scope(self):
        controls = readme("controls", "supplementary controls")
        audit = readme("audit", "reliability audit")
        self.assertIn("280 preprocessing", controls)
        self.assertIn("one post-selection test evaluation", controls)
        self.assertIn("22 completed independent workers", audit)
        self.assertNotIn("420 completed", audit)


if __name__ == "__main__":
    unittest.main()
