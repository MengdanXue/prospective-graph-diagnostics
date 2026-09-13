"""CPU toy checks for checkpoint-copy treatments; no research training/data."""
from __future__ import annotations

import pickle
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.prospective_models import H2GCNModel, prepare_h2_adjacencies
from scripts.h2gcn_checkpoint_diagnostic import (
    BUFFER_NAMES, MODES, PARAMETER_NAMES, CheckpointStore, state_fingerprint, tensor_fingerprint,
)
from scripts import h2gcn_checkpoint_diagnostic as checkpoint_module


def rng_state():
    return (random.getstate(), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone())


def tree_fingerprint(value):
    if isinstance(value, torch.Tensor):
        return tensor_fingerprint(value)
    if isinstance(value, dict):
        return {key: tree_fingerprint(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tree_fingerprint(item) for item in value]
    return value


class H2CheckpointDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
                                   [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]])
        self.x = torch.arange(24, dtype=torch.float32).view(6, 4) / 24

    def model(self):
        # Rebuild the original implementation; fresh buffers also exercise
        # registration of equivalent graph values on a new trial instance.
        return H2GCNModel(4, 3, 2, 0.5, prepare_h2_adjacencies(self.edges, num_nodes=6))

    def assert_rng_equal(self, before, after):
        self.assertEqual(before[:2], after[:2])
        self.assertTrue(torch.equal(before[2], after[2]))

    def test_both_modes_capture_exact_states_and_restore_exact_logits_for_four_slots(self):
        torch.manual_seed(7)
        model = self.model().eval()
        stores = [CheckpointStore(model, mode) for mode in MODES]
        expected = []
        for slot in range(4):
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(0.01 * (slot + 1))
                expected.append(model(self.x, self.edges).clone())
            for store in stores:
                result = store.capture(model, slot)
                self.assertEqual(result["retained_checkpoint_count"], slot + 1)
                self.assertFalse(result["replacement"])
                self.assertGreaterEqual(result["parameter_copy_seconds"], 0)
                if store.mode == "static_adjacency_once":
                    self.assertEqual(result["static_buffer_copy_seconds"], 0)
                    self.assertEqual(result["static_buffer_copy_bytes"], 0)
                else:
                    self.assertGreater(result["static_buffer_copy_bytes"], 0)
            self.assertEqual(state_fingerprint(stores[0].snapshot(slot)),
                             state_fingerprint(stores[1].snapshot(slot)))
        for slot in range(4):
            for store in stores:
                restored = self.model().eval()
                store.restore(restored, slot)
                with torch.no_grad():
                    actual = restored(self.x, self.edges)
                self.assertTrue(torch.equal(actual, expected[slot]))
        for store in stores:
            store.check_static(model)

    def test_capture_does_not_mutate_model_gradients_optimizer_mode_or_rng(self):
        torch.manual_seed(17)
        random.seed(17)
        np.random.seed(17)
        model = self.model().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        model(self.x, self.edges).square().sum().backward()
        optimizer.step()
        original_model = state_fingerprint(model.state_dict())
        original_gradients = state_fingerprint({name: parameter.grad for name, parameter in model.named_parameters()})
        original_optimizer = tree_fingerprint(optimizer.state_dict())
        original_modes = {name: module.training for name, module in model.named_modules()}
        original_rng = rng_state()
        for mode in MODES:
            store = CheckpointStore(model, mode)
            for slot in range(4):
                store.capture(model, slot)
            store.check_static(model)
            self.assertEqual(state_fingerprint(model.state_dict()), original_model)
            self.assertEqual(state_fingerprint({name: parameter.grad for name, parameter in model.named_parameters()}),
                             original_gradients)
            self.assertEqual(tree_fingerprint(optimizer.state_dict()), original_optimizer)
            self.assertEqual({name: module.training for name, module in model.named_modules()}, original_modes)
            self.assert_rng_equal(original_rng, rng_state())

    def test_parameter_snapshots_do_not_alias_later_training_updates(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                model = self.model()
                store = CheckpointStore(model, mode)
                store.capture(model, 0)
                original = state_fingerprint(store.snapshot(0))
                current = dict(model.named_parameters())
                for name in PARAMETER_NAMES:
                    self.assertNotEqual(store.snapshot(0)[name].data_ptr(), current[name].data_ptr())
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(10)
                self.assertEqual(state_fingerprint(store.snapshot(0)), original)
                with self.assertRaises(TypeError):
                    store.snapshot(0)["embedding.weight"] = torch.zeros(1)

    def test_storage_counts_include_four_slots_and_old_slot_during_replacement(self):
        model = self.model()
        baseline = CheckpointStore(model, "full_state_copy")
        candidate = CheckpointStore(model, "static_adjacency_once")
        self.assertEqual(baseline.initial_static_copy_seconds, 0)
        self.assertEqual(baseline.initial_static_copy_bytes, 0)
        buffer_bytes = candidate.initial_static_copy_bytes
        parameter_bytes = sum(value.untyped_storage().nbytes() for value in model.parameters())
        self.assertGreater(buffer_bytes, 0)
        self.assertEqual(candidate.storage_metrics()["total_unique_tensor_bytes"], buffer_bytes)
        for slot in range(4):
            baseline.capture(model, slot)
            candidate.capture(model, slot)
            count = slot + 1
            self.assertEqual(baseline.storage_metrics()["retained_parameter_bytes"], count * parameter_bytes)
            self.assertEqual(baseline.storage_metrics()["retained_static_buffer_bytes"], count * buffer_bytes)
            self.assertEqual(candidate.storage_metrics()["retained_parameter_bytes"], count * parameter_bytes)
            self.assertEqual(candidate.storage_metrics()["retained_static_buffer_bytes"], buffer_bytes)
        for store, expected_peak, expected_after in [
            (baseline, 5 * (parameter_bytes + buffer_bytes), 4 * (parameter_bytes + buffer_bytes)),
            (candidate, 5 * parameter_bytes + buffer_bytes, 4 * parameter_bytes + buffer_bytes),
        ]:
            result = store.capture(model, 3)
            self.assertTrue(result["replacement"])
            self.assertEqual(result["peak_retained_tensor_bytes"], expected_peak)
            self.assertEqual(result["post_capture_retained_tensor_bytes"], expected_after)
            self.assertEqual(store.storage_metrics()["last_capture_peak_unique_tensor_bytes"], expected_peak)
            self.assertEqual(store.storage_metrics()["retained_checkpoint_count"], 4)
        for name in BUFFER_NAMES:
            baseline_ptrs = {baseline.snapshot(slot)[name].values().data_ptr() for slot in range(4)}
            candidate_ptrs = {candidate.snapshot(slot)[name].values().data_ptr() for slot in range(4)}
            self.assertEqual(len(baseline_ptrs), 4)
            self.assertEqual(len(candidate_ptrs), 1)
            self.assertNotIn(dict(model.named_buffers())[name].values().data_ptr(), candidate_ptrs)

    def test_capture_does_not_hash_graph_and_empty_sparse_buffers_are_supported(self):
        model = H2GCNModel(4, 3, 2, 0.5, prepare_h2_adjacencies(
            torch.empty((2, 0), dtype=torch.long), num_nodes=6))
        for mode in MODES:
            with self.subTest(mode=mode):
                store = CheckpointStore(model, mode)
                with patch.object(checkpoint_module, "state_fingerprint", side_effect=AssertionError("hash in capture")), \
                        patch.object(checkpoint_module, "tensor_fingerprint", side_effect=AssertionError("hash in capture")):
                    for slot in range(4):
                        store.capture(model, slot)
                self.assertEqual(store.storage_metrics()["retained_static_buffer_bytes"], 0)
                store.check_static(model)

    def test_source_buffer_mutation_is_rejected_without_overwriting_checkpoint(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                model = self.model()
                store = CheckpointStore(model, mode)
                store.capture(model, 0)
                before = state_fingerprint(store.snapshot(0))
                model.one_hop._values().add_(1)
                with self.assertRaisesRegex(ValueError, "buffer identity/version"):
                    store.capture(model, 0)
                with self.assertRaisesRegex(ValueError, "source adjacency"):
                    store.check_static(model)
                self.assertEqual(state_fingerprint(store.snapshot(0)), before)

    def test_private_or_checkpoint_buffer_corruption_is_rejected_before_restore(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                model = self.model()
                store = CheckpointStore(model, mode)
                store.capture(model, 0)
                # Deliberate misuse of a read-only tensor view simulates bank
                # corruption; restore must reject before changing the target.
                store.snapshot(0)["two_hop"]._values().add_(0.5)
                target = self.model()
                target_before = state_fingerprint(target.state_dict())
                with self.assertRaisesRegex(ValueError, "corrupt|adjacency buffers changed"):
                    store.restore(target, 0)
                self.assertEqual(state_fingerprint(target.state_dict()), target_before)

    def test_new_trial_model_requires_explicit_static_validation(self):
        model = self.model()
        store = CheckpointStore(model, "static_adjacency_once")
        store.capture(model, 0)
        other = self.model()
        with self.assertRaisesRegex(ValueError, "check_static"):
            store.capture(other, 1)
        self.assertEqual(store.check_static(other), store.static_fingerprint)
        store.capture(other, 1)
        store.restore(other, 0)
        # Strict restoration changes buffer versions; successful validation
        # refreshes those tokens so a later capture can proceed safely.
        store.capture(other, 1)

    def test_unknown_names_modes_and_slots_are_rejected(self):
        model = self.model()
        with self.assertRaisesRegex(ValueError, "unknown checkpoint treatment"):
            CheckpointStore(model, "automatic")
        with self.assertRaises(ValueError):
            CheckpointStore(torch.nn.Linear(4, 2), "full_state_copy")
        store = CheckpointStore(model, "full_state_copy")
        for slot in (-1, 4, True, 0.0, "0"):
            with self.subTest(slot=slot), self.assertRaises(ValueError):
                store.capture(model, slot)
        with self.assertRaises(KeyError):
            store.snapshot(0)
        model.register_buffer("unplanned", torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "names changed"):
            store.capture(model, 0)
        with self.assertRaisesRegex(ValueError, "names changed"):
            CheckpointStore(model, "static_adjacency_once")

    def test_changed_model_or_checkpoint_metadata_and_keys_are_rejected(self):
        for change in ("shape", "dtype", "keys"):
            with self.subTest(change=change):
                model = self.model()
                store = CheckpointStore(model, "static_adjacency_once")
                store.capture(model, 0)
                if change == "shape":
                    store._slots[0]["embedding.weight"] = torch.zeros(1, 1)
                elif change == "dtype":
                    store._slots[0]["embedding.weight"] = store._slots[0]["embedding.weight"].double()
                else:
                    del store._slots[0]["embedding.bias"]
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    store.restore(self.model(), 0)
        model = self.model()
        store = CheckpointStore(model, "full_state_copy")
        model.double()
        with self.assertRaisesRegex(ValueError, "CPU float32"):
            store.capture(model, 0)


if __name__ == "__main__":
    unittest.main()
