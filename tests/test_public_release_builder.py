import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "build_public_artifact_release.py"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PublicReleaseBuilderTests(unittest.TestCase):
    def load_builder(self):
        self.assertTrue(BUILDER_PATH.is_file(), "public release builder is missing")
        spec = importlib.util.spec_from_file_location("public_release_builder", BUILDER_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def write_fixture(self, root: Path) -> dict[str, Path]:
        files = {
            "prospective": root
            / "route_a_prospective_v2"
            / "records"
            / "Cora"
            / "GCN"
            / "seed_000.json",
            "diagnostic": root
            / "route_a_prospective_v2"
            / "diagnostics"
            / "Cora"
            / "seed_000.json",
            "degree": root
            / "route_a_degree_matched_v1"
            / "records"
            / "Cora"
            / "seed_000.json",
        }
        provenance = {
            "processed_files": [
                {
                    "path": "data.pt",
                    "root": r"D:\\Users\\private\\Documents\\paper\\data\\cora\\processed",
                    "sha256": "a" * 64,
                    "size": 123,
                }
            ]
        }
        payloads = {
            "prospective": {
                "dataset": "Cora",
                "model": "GCN",
                "run_id": "route_a_prospective_v2",
                "source_commit": "source-commit",
                "status": "success",
                "test_accuracy": 0.75,
                "data_provenance": provenance,
            },
            "diagnostic": {
                "dataset": "Cora",
                "run_id": "route_a_prospective_v2",
                "status": "success",
                "homophily": 0.8,
                "provenance": {"data": provenance},
            },
            "degree": {
                "dataset": "Cora",
                "run_id": "route_a_degree_matched_v1",
                "status": "success",
                "paired_test_difference": -0.4,
                "data_provenance": provenance,
                "conditions": {
                    "original": {"data_provenance": provenance},
                    "randomized": {"data_provenance": provenance},
                },
            },
        }
        for name, path in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payloads[name], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        return files

    def test_build_release_redacts_only_cache_roots_and_records_both_hashes(self):
        builder = self.load_builder()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source"
            files = self.write_fixture(source)
            before = {name: path.read_bytes() for name, path in files.items()}
            output = tmp_path / "release.zip"

            builder.build_release(
                source,
                output,
                expected_counts={
                    "prospective_records": 1,
                    "prospective_diagnostics": 1,
                    "degree_records": 1,
                },
                release_commit="release-commit",
            )

            self.assertEqual(
                {name: path.read_bytes() for name, path in files.items()}, before
            )
            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("MANIFEST.json"))
                records = {
                    entry["group"]: entry for entry in manifest["files"]
                }
                prospective_bytes = archive.read(records["prospective_records"]["path"])
                prospective = json.loads(prospective_bytes)
                self.assertEqual(
                    prospective["data_provenance"]["processed_files"][0]["root"],
                    "data/Cora/processed",
                )
                self.assertEqual(prospective["test_accuracy"], 0.75)
                self.assertEqual(
                    records["prospective_records"]["source_sha256"],
                    sha256_bytes(before["prospective"]),
                )
                self.assertEqual(
                    records["prospective_records"]["public_sha256"],
                    sha256_bytes(prospective_bytes),
                )
                self.assertEqual(manifest["release_commit"], "release-commit")
                self.assertEqual(manifest["total_json_files"], 3)
                self.assertEqual(manifest["total_redactions"], 5)

    def test_release_is_deterministic_and_refuses_overwrite(self):
        builder = self.load_builder()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source"
            self.write_fixture(source)
            counts = {
                "prospective_records": 1,
                "prospective_diagnostics": 1,
                "degree_records": 1,
            }
            first = tmp_path / "first.zip"
            second = tmp_path / "second.zip"
            builder.build_release(
                source, first, expected_counts=counts, release_commit="commit"
            )
            builder.build_release(
                source, second, expected_counts=counts, release_commit="commit"
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with self.assertRaises(FileExistsError):
                builder.build_release(
                    source, first, expected_counts=counts, release_commit="commit"
                )


if __name__ == "__main__":
    unittest.main()
