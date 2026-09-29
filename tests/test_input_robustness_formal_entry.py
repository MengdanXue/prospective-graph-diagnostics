"""Isolated non-synthetic acceptance of the committed formal scheduler."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.input_robustness_budget import BudgetLedger, CI_JOBS
from scripts.input_robustness_formal_entry import FormalExecutionController
from scripts.input_robustness_formal_records import CONDITIONS, FormalEmergencyStop, FormalRecordWriter, digest, expected_keys
from scripts.input_robustness_runtime import PowerEventWatcher, TaskPowerRequest
from scripts.validate_input_robustness_formal_records import validate_complete_run


class _PowerBackend:
    def create_request(self, reason): return 1
    def set_request(self, handle, request_type): pass
    def clear_request(self, handle, request_type): pass
    def close_handle(self, handle): pass
    def register_notification(self, callback): self.callback = callback; return 2
    def unregister_notification(self, registration): pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FormalEntryTests(unittest.TestCase):
    def test_outer_monitor_vetoes_ledger_stop_and_power_event(self):
        class Ledger:
            def poll(self, force=False):
                return {"must_stop": True, "stop_reasons": []}
        class Watcher:
            events = [{"event": "suspend"}]
        class Writer:
            synthetic = False
        controller = FormalExecutionController(writer=Writer(), ledger=Ledger(), phase_id="phase",
                                               power_request=TaskPowerRequest(_PowerBackend()),
                                               power_watcher=Watcher())
        with self.assertRaisesRegex(FormalEmergencyStop, "ledger must_stop"):
            controller._monitor()

    def test_authoritative_finalize_and_monitored_two_condition_run(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            binding_path = base / "binding.json"
            training = {
                "hidden_channels": 4, "max_epochs": 1, "patience": 1, "weight_decay": 0.0005,
                "trials": [
                    {"learning_rate": 0.01, "dropout": 0.5}, {"learning_rate": 0.01, "dropout": 0.7},
                    {"learning_rate": 0.005, "dropout": 0.5}, {"learning_rate": 0.005, "dropout": 0.7},
                ],
            }
            binding = {"bound_split_count": 1, "datasets": {"Cora": {
                "split_checks": [{"seed": 0, "split_id": "tiny-split"}],
                "full_diagnostic_checks": [{"seed": 0, "diagnostics": {"homophily": 0.4, "mean_degree": 3.0, "delta_h": 0.1}}],
                "candidate_checks": [
                    {"seed": 0, "fit_metadata": {"condition": CONDITIONS[0]}, "transformed_feature_sha256": "d" * 64},
                    {"seed": 0, "fit_metadata": {"condition": CONDITIONS[1]}, "transformed_feature_sha256": "f" * 64},
                ],
                "normalize_features_sha256": "d" * 64,
            }}}
            binding_path.write_text(json.dumps(binding), encoding="utf-8")
            config = {
                "formal_training_enabled": False, "datasets": ["Cora"],
                "conditions": list(CONDITIONS), "models": ["MLP"], "seeds": [0],
                "expected_records": 2, "expected_trials": 8, "training": training,
                "execution": {"model_devices": {"H2GCN": "cpu"}, "torch_num_threads": 4,
                               "h2gcn_checkpoint_mode": "original_full_state_copy"},
                "h2gcn_checkpoint_mode": "original_full_state_copy",
                "bound_input_source": {"path": str(binding_path), "sha256": _sha(binding_path)},
            }
            config_path = base / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            config_sha = digest(config)
            binding_sha = _sha(binding_path)
            scope = expected_keys(datasets=("Cora",), conditions=CONDITIONS,
                                  models=("MLP",), seeds=(0,))
            writer = FormalRecordWriter(
                base / "formal-records",
                manifest={"run_id": "isolated-formal-v1", "source_commit": "a" * 40,
                          "config_sha256": config_sha, "data_binding_sha256": binding_sha,
                          "environment": {"device": "cpu"},
                          "scope": {"datasets": ["Cora"], "conditions": list(CONDITIONS),
                                    "models": ["MLP"], "seeds": [0]},
                          "expected_records": 2, "expected_trials": 8},
                config_path=config_path, data_binding_path=binding_path,
            )
            authority = {"approval_sha256": "2" * 64, "proposal_sha256": "3" * 64,
                         "scope_sha256": "4" * 64}
            ledger = BudgetLedger.create(base / "ledger", authority=authority,
                                         monitor_gap_seconds=5)
            gate = {"ci_receipt_file_sha256": "e" * 64, "review_record_sha256": "f" * 64,
                    "ci_receipt": {"commit": "a" * 40, "run_id": 17, "status": "completed",
                                   "conclusion": "success", "jobs": [
                                       {"name": name, "run_id": 17, "status": "completed", "conclusion": "success"}
                                       for name in CI_JOBS]}}
            ledger.register_phase("isolated-formal", provenance={
                "source_commit": "a" * 40, "config_sha256": config_sha,
                "data_binding_sha256": binding_sha, "source_files": {"formal_entry.py": "d" * 64},
                "environment": {"device": "cpu"}}, gate_receipt=gate)
            x = torch.tensor([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.], [0., 0., 1.],
                              [1., 0., 1.], [0., 1., 1.], [1., 1., 1.], [.5, .5, 0.]])
            y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
            indices = lambda values: torch.tensor(values, dtype=torch.long)
            common = {"run_id": "isolated-formal-v1", "dataset": "Cora", "model_id": "MLP",
                      "seed": 0, "split_id": "tiny-split", "x": x, "y": y,
                      "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                      "train_indices": indices([0, 1, 2, 3]), "validation_indices": indices([4, 5]),
                      "test_indices": indices([6, 7]), "training": training,
                      "source_commit": "a" * 40, "environment": {"device": "cpu"},
                      "data_provenance": {"source": str(binding_path)}, "device": torch.device("cpu"),
                      "config_sha256": config_sha, "frozen_config": config}
            units = []
            for index, condition in enumerate(CONDITIONS):
                units.append({**common, "condition": condition,
                              "data_binding_sha256": binding_sha,
                              "attempt_id": f"attempt{index:03d}", "unit_id": f"unit{index:03d}",
                              "estimated_seconds": 1,
                              "transform_binding": {"dataset": "Cora", "condition": condition, "seed": 0,
                                                     "split_id": "tiny-split",
                                                     "transformed_feature_sha256": "d" * 64 if index == 0 else "f" * 64,
                                                     "fit_statistics_sha256": "e" * 64 if index == 0 else "0" * 64},
                              "diagnostics": {"homophily": 0.4, "mean_degree": 3.0, "delta_h": 0.1}})
            controller = FormalExecutionController(
                writer=writer, ledger=ledger, phase_id="isolated-formal",
                power_request=TaskPowerRequest(_PowerBackend()),
                power_watcher=PowerEventWatcher(_PowerBackend()),
            )
            result = controller.run_units(units, launch_authorized=True,
                                          formal_training_enabled=True)
            self.assertEqual(result["status"], "completed")
            self.assertGreaterEqual(result["monitor_samples"], 10)
            self.assertEqual({attempt["batch_id"] for attempt in ledger._state["attempts"].values()},
                             {"pair_Cora_seed_000_MLP"})
            self.assertTrue(result["complete"])
            validated = validate_complete_run(writer.root, expected_keys=scope, synthetic=False,
                                              config_path=config_path, data_binding_path=binding_path)
            self.assertEqual(validated["record_count"], 2)
            self.assertTrue((writer.root / "complete.json").is_file())
