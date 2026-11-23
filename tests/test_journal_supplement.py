"""Small synthetic archives exercise journal derivation integrity end to end.

No tensor, dataset, training process, or historical evidence is used or changed.
The successful fixture traverses the real outer verifier and record aggregator;
tampered fixtures refresh outer checksums so inner checks must detect the fault.
"""

import json
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

from scripts import verify_journal_supplement as verifier


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


class JournalSupplementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / verifier.DERIVED
        self.compact = self.root / "compact.json"
        self.entries, self.mapping = self.fixture()
        self.write_archive()

    def fixture(self):
        config = {"run_id": "synthetic-journal-test", "groups": [{"group_id": "fixture"}]}
        workers = [{"worker_id": f"fixture_{i:02d}", "group_id": "fixture"} for i in range(22)]
        source = b"# synthetic source identity only\n"
        manifest = {"workers": workers, "config": config,
                    "python_executable": verifier.REDACTED,
                    "source_files_sha256": {"fixture.py": verifier.sha(source)}}
        entries = {"audit/source_snapshot/synthetic/fixture.py": source}
        records = []
        for worker in workers:
            artifact = (worker["worker_id"] + " synthetic model bytes").encode()
            record = {"worker": worker, "status": "success", "validation_accuracy": 0.5,
                      "validation_loss": 0.7, "best_epoch": 1, "epochs_completed": 1,
                      "initial_state": {"sha256": "initial"}, "initial_rng": {"seed": 0},
                      "selected_state": {"sha256": "selected"}, "history_sha256": "history",
                      "fixed_checkpoint_forwards": [{"max_abs_logit_difference": 0,
                                                     "prediction_disagreements": 0}],
                      "early_trajectory": [], "artifacts_sha256": {"model.pt": verifier.sha(artifact)}}
            prefix = "audit/workers/" + worker["worker_id"] + "/"
            entries[prefix + "record.json"] = encoded(record)
            entries[prefix + "model.pt"] = artifact
            records.append(record)
        summary = {**verifier.summarize_records(config, records), "manifest": manifest, "records": records}
        entries["audit/run_manifest.json"] = encoded(manifest)
        entries["audit/audit_summary.json"] = encoded(summary)
        # The production inventory has 113 completion-bound files. Small log
        # payloads satisfy that inventory without allocating model tensors.
        for i in range(113 - len(entries)):
            entries[f"audit/logs/synthetic_{i:03d}.log"] = b"synthetic fixture\n"
        complete = {"status": "complete", "worker_count": 22, "success_count": 22,
                    "test_evaluations": 0,
                    "files_sha256": {name.removeprefix("audit/"): verifier.sha(data)
                                     for name, data in entries.items()}}
        entries["audit/complete.json"] = encoded(complete)
        mapping = {"original_archive_sha256": verifier.ORIGINAL_AUDIT,
                   "allowed_metadata_fields": sorted(verifier.PATH_KEYS), "files": {}}
        for name, data in entries.items():
            row = {"operation": "unchanged", "original_sha256": verifier.sha(data),
                   "derived_sha256": verifier.sha(data), "original_bytes": len(data),
                   "derived_bytes": len(data)}
            if name in {"audit/run_manifest.json", "audit/audit_summary.json"}:
                obj = json.loads(data)
                row.update(operation="metadata_paths_only",
                           scientific_sha256=verifier.sha(verifier.canonical(verifier.metadata_free(obj))),
                           redacted_json_pointers=["/python_executable" if name.endswith("run_manifest.json")
                                                   else "/manifest/python_executable"])
            elif name == "audit/complete.json":
                row.update(operation="completion_hash_rebinding",
                           scientific_sha256=verifier.sha(verifier.canonical(
                               {k: v for k, v in complete.items() if k != "files_sha256"})))
            mapping["files"][name] = row
        self.compact.write_bytes(encoded({"full_summary_sha256": verifier.sha(entries["audit/audit_summary.json"]),
                                         "full_summary_bytes": len(entries["audit/audit_summary.json"]),
                                         "groups": summary["groups"]}))
        return entries, mapping

    def write_archive(self):
        self.entries["JOURNAL_DERIVATION.json"] = encoded(self.mapping)
        package = {"files": {name: {"sha256": verifier.sha(data), "size": len(data)}
                             for name, data in self.entries.items()},
                   "audit": {"source_files": {"fixture.py": {
                       "mode": "synthetic", "sha256": verifier.sha(
                           self.entries["audit/source_snapshot/synthetic/fixture.py"])}}}}
        with zipfile.ZipFile(self.path, "w") as archive:
            for name, data in self.entries.items():
                archive.writestr(name, data)
            archive.writestr("hash_manifest.json", encoded(package))
        self.refresh_checksum()

    def refresh_checksum(self):
        self.path.with_suffix(".zip.sha256").write_text(
            f"{verifier.sha(self.path.read_bytes())}  {self.path.name}\n", encoding="ascii")

    def replace_and_rebind_bytes(self, name, obj):
        data = encoded(obj)
        self.entries[name] = data
        self.mapping["files"][name].update(derived_sha256=verifier.sha(data), derived_bytes=len(data))
        self.write_archive()

    def test_valid_synthetic_archive_reaggregates_records_and_checks_all_files(self):
        report = verifier.verify_derived(self.path, self.compact)
        self.assertEqual(report["workers_reaggregated"], 22)
        self.assertEqual(report["completion_files_hashed"], 113)
        self.assertFalse(report["original_bytes_compared"])

    def test_duplicate_zip_member_rejected_with_updated_outer_checksum(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.path, "a") as archive:
                archive.writestr("audit/run_manifest.json", self.entries["audit/run_manifest.json"])
        self.refresh_checksum()
        with self.assertRaisesRegex(ValueError, "duplicate ZIP"):
            verifier.verify_derived(self.path, self.compact)

    def test_unsafe_zip_member_rejected_with_updated_outer_checksum(self):
        # Windows zipfile normalizes backslashes while reading names. Check
        # the raw-name guard directly and exercise other attacks through ZIPs.
        with self.assertRaisesRegex(ValueError, "unsafe ZIP"):
            verifier.safe_names(["audit\\local.json"])
        for name in ("../outside.json", "/absolute.json", "C:/local.json", "a/./b"):
            with self.subTest(name=name):
                self.write_archive()
                with zipfile.ZipFile(self.path, "a") as archive:
                    archive.writestr(name, b"{}")
                self.refresh_checksum()
                with self.assertRaisesRegex(ValueError, "unsafe ZIP"):
                    verifier.verify_derived(self.path, self.compact)

    def test_redaction_rejects_local_paths_in_non_metadata_fields(self):
        for value in ({"record": {"checkpoint": "C:/private/model.pt"}},
                      {"notes": ["/home/person/data"]},
                      {"python_executable": {"arbitrary": "/Users/person/python"}}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "local path"):
                verifier.redact_paths(value)

    def test_allowed_redaction_retains_scientific_values_and_exact_pointers(self):
        original = {"worker/a": [{"python_executable": "C:/private/python.exe", "accuracy": 0.5}]}
        result, pointers = verifier.redact_paths(original)
        self.assertEqual(pointers, ["/worker~1a/0/python_executable"])
        self.assertEqual(verifier.pointer_value(result, pointers[0]), verifier.REDACTED)
        self.assertEqual(verifier.metadata_free(original), verifier.metadata_free(result))
        self.assertEqual(original["worker/a"][0]["python_executable"], "C:/private/python.exe")

    def test_scientific_tampering_rejected_after_refreshing_byte_checksums(self):
        name = "audit/audit_summary.json"
        summary = json.loads(self.entries[name])
        summary["groups"][0]["validation_accuracy"][0] = 0.99
        self.replace_and_rebind_bytes(name, summary)
        with self.assertRaisesRegex(ValueError, "scientific value changed"):
            verifier.verify_derived(self.path, self.compact)

    def test_internal_completion_hash_tampering_rejected_after_outer_rebinding(self):
        name = "audit/complete.json"
        complete = json.loads(self.entries[name])
        complete["files_sha256"]["workers/fixture_00/model.pt"] = "0" * 64
        self.replace_and_rebind_bytes(name, complete)
        with self.assertRaisesRegex(ValueError, "internal completion hash mismatch"):
            verifier.verify_derived(self.path, self.compact)

    def test_completion_scientific_fields_cannot_be_rebound_as_metadata(self):
        name = "audit/complete.json"
        complete = json.loads(self.entries[name])
        complete["test_evaluations"] = 1
        self.replace_and_rebind_bytes(name, complete)
        with self.assertRaisesRegex(ValueError, "completion science changed"):
            verifier.verify_derived(self.path, self.compact)


if __name__ == "__main__":
    unittest.main()
