"""Adversarial checks for post-hoc analysis input integrity and policy scoring."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.evaluate_diagnostics import audit_payload
from scripts.assemble_prospective_diagnostics import assemble_payload
from scripts.summarize_equal_budget_sensitivity import GRAPH_ARCHITECTURES, evaluate_portfolio, load_units
from tests import test_prospective_assembly as fixture_module


class EqualTrialSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.prospective = self.base / "prospective"
        self.records = self.prospective / "records"
        self.config = fixture_module.ProspectiveAssemblyTests().make_fixture(
            self.prospective, run_id="route_a_prospective_v2"
        )
        for model, accuracy in (("MLP", .7), ("GCN", .4), ("GAT", .8)):
            path = self.records / "toy" / model / "seed_000.json"
            self.mutate(path, "test_accuracy", accuracy)
        self.audit = audit_payload(assemble_payload(self.prospective, self.config))
        self.write_manifest()

    @staticmethod
    def mutate(path, field, value):
        record = json.loads(path.read_text())
        record[field] = value
        path.write_text(json.dumps(record), encoding="utf-8")

    def write_manifest(self):
        files = []
        for directory, group in (("records", "prospective_records"), ("diagnostics", "prospective_diagnostics")):
            for path in sorted((self.prospective / directory).rglob("*.json")):
                contents = path.read_bytes()
                files.append({"path": path.relative_to(self.base).as_posix(), "group": group,
                              "public_bytes": len(contents), "public_sha256": hashlib.sha256(contents).hexdigest()})
        (self.base / "MANIFEST.json").write_text(json.dumps({"archive_schema_version": "1.0", "files": files}))

    def load(self, audit=None):
        return load_units(self.records, config=self.config, audit=self.audit if audit is None else audit)

    def test_validated_records_show_a_portfolio_dependent_policy_reversal(self):
        units = self.load()
        full = evaluate_portfolio(self.audit, units, GRAPH_ARCHITECTURES)
        gcn = evaluate_portfolio(self.audit, units, ["GCN"])
        self.assertAlmostEqual(full["mean_regret_pp"]["always_graph"], 0)
        self.assertAlmostEqual(full["mean_regret_pp"]["combined"], 10)
        self.assertAlmostEqual(gcn["mean_regret_pp"]["always_graph"], 30)
        self.assertAlmostEqual(gcn["mean_regret_pp"]["combined"], 0)
        self.assertEqual((full["graph_targets"], gcn["graph_targets"]), (1, 0))

    def test_tampering_with_an_unselected_architecture_is_rejected(self):
        self.assertEqual(self.audit["units"][0]["selected_graph"], "GAT")
        self.mutate(self.records / "toy/GCN/seed_000.json", "test_accuracy", 1.0)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.load()

    def test_duplicate_records_are_not_silently_overwritten(self):
        original = self.records / "toy/GCN/seed_000.json"
        original.with_name("seed_000_copy.json").write_bytes(original.read_bytes())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load()

    def test_missing_record_is_rejected(self):
        (self.records / "toy/GCN/seed_000.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_rechecksummed_invalid_provenance_or_trials_are_rejected(self):
        path = self.records / "toy/GCN/seed_000.json"
        original = path.read_bytes()
        for field, value in (("source_commit", "different"), ("split_id", "different"),
                             ("config_sha256", "different"), ("run_id", "different"),
                             ("environment", {"python": "different"}),
                             ("selected_trial_id", "trial_003"), ("test_accuracy", float("nan"))):
            with self.subTest(field=field):
                path.write_bytes(original)
                self.mutate(path, field, value)
                self.write_manifest()
                with self.assertRaises(ValueError):
                    self.load()

    def test_audit_scope_split_actions_and_outcomes_must_match_replay(self):
        for field, value in (("split_id", "other"), ("dataset", "other"),
                             ("selected_graph_test", .99), ("homophily", .8)):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.audit)
                changed["units"][0][field] = value
                with self.assertRaisesRegex(ValueError, "audit value mismatch"):
                    self.load(changed)
        changed = copy.deepcopy(self.audit)
        changed["units"][0]["decisions"]["historical_combined"]["action"] = "graph"
        with self.assertRaises(ValueError):
            self.load(changed)
        changed = copy.deepcopy(self.audit)
        changed["units"].append(copy.deepcopy(changed["units"][0]))
        with self.assertRaises(ValueError):
            self.load(changed)

    def test_runtime_metadata_can_change_without_changing_results(self):
        changed = copy.deepcopy(self.audit)
        changed["environment"] = {"python": "other-runtime"}
        self.assertEqual(self.load(changed), self.load())

    def test_manifest_duplicate_and_path_escape_are_rejected(self):
        path = self.base / "MANIFEST.json"
        original = json.loads(path.read_text())
        for case in ("duplicate", "escape"):
            with self.subTest(case=case):
                manifest = copy.deepcopy(original)
                if case == "duplicate":
                    manifest["files"].append(manifest["files"][0])
                else:
                    manifest["files"][0]["path"] = "prospective/records/../../outside.json"
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "manifest path"):
                    self.load()


if __name__ == "__main__":
    unittest.main()
