"""The committed entry really trains, saves, analyzes and stops CPU workers."""
from pathlib import Path
import tempfile
import unittest

from scripts.mlp_budget_extension_e2e import run_acceptance
from scripts.input_robustness_formal_records import FormalRecordError
from scripts.mlp_budget_extension_entry import validate_acceptance, validate_cumulative_attempts


class MLPBudgetExtensionEntryTests(unittest.TestCase):
    def test_cumulative_invariant_uses_exact_nanoseconds_and_rejects_one_ns_loss(self):
        snapshot = {"charged_ns": {"units": {"unit": 300_000_000}}, "attempts": {
            "first": {"unit_id": "unit", "outcome": "external_interruption", "charged_ns": 100_000_000},
            "second": {"unit_id": "unit", "outcome": "completed", "charged_ns": 200_000_000}}}
        self.assertGreater(100_000_000 / 1e9 + 200_000_000 / 1e9, 300_000_000 / 1e9)
        exact = validate_cumulative_attempts(snapshot, unit_id="unit", attempt_ids=["first", "second"])
        self.assertEqual(exact["unit_charged_ns"], 300_000_000)
        snapshot["charged_ns"]["units"]["unit"] -= 1
        with self.assertRaisesRegex(FormalRecordError, "unit_ns=299999999.*sum_ns=300000000"):
            validate_cumulative_attempts(snapshot, unit_id="unit", attempt_ids=["first", "second"])

    def test_complete_guarded_path_pause_emergency_and_same_unit_resume(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs/diagnostics_mlp_budget_extension_v1.json"
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "e2e"
            receipt = run_acceptance(output, launch_config_path=config_path, commit="b" * 40)
            self.assertFalse(receipt["research_training_started"])
            validate_acceptance(output / "acceptance.json", config_path=config_path, commit="b" * 40,
                                require_native_power=False)


if __name__ == "__main__":
    unittest.main()
