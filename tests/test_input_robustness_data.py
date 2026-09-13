"""Binding corruption and numerical-identity tests for the post-hoc input study."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.prospective_data import make_stratified_split, split_identifier
from scripts.input_robustness_data import (
    CONDITIONS, DATASETS, array_sha256, check_original_binding,
    estimate_transform_peak_bytes, file_sha256, fit_transform_features_bounded,
    enrich_dataset_binding, load_bound_dataset, verify_release_manifest,
)
from scripts.run_mlp_optimization_diagnostic import fit_transform_features


class InputRobustnessDataTests(unittest.TestCase):
    def test_exact_original_scope_includes_coauthor_and_excludes_texas(self):
        self.assertEqual(len(DATASETS), 11)
        self.assertIn("Coauthor-CS", DATASETS)
        self.assertNotIn("Texas", DATASETS)
        self.assertEqual(CONDITIONS, ("normalize_features", "normalize_centered_scaled"))

    def test_bounded_transform_exactly_matches_historical_outputs_and_fit(self):
        rng = np.random.default_rng(629)
        matrices = [rng.normal(size=(41, 17)).astype(np.float32),
                    rng.integers(0, 2, size=(41, 17)).astype(np.float32),
                    rng.uniform(0, .002, size=(41, 17)).astype(np.float32)]
        for matrix in matrices:
            matrix[0] = 0
            x = torch.from_numpy(matrix)
            train = torch.as_tensor([0, 2, 5, 11, 19, 25, 38, 40])
            before = x.clone()
            for condition in CONDITIONS:
                expected, expected_metadata = fit_transform_features(x, train, condition)
                output, metadata = fit_transform_features_bounded(x, train, condition, block_bytes=280)
                self.assertTrue(torch.equal(output, expected))
                self.assertEqual(metadata, expected_metadata)
                self.assertTrue(torch.equal(x, before))

    def test_train_fitted_statistics_ignore_held_out_rows_when_global_min_fixed(self):
        x = torch.arange(48, dtype=torch.float32).reshape(12, 4)
        train = torch.arange(6)
        _, first = fit_transform_features_bounded(x, train, CONDITIONS[1])
        changed = x.clone()
        changed[6:] *= 4
        _, second = fit_transform_features_bounded(changed, train, CONDITIONS[1])
        self.assertEqual(first, second)

    def test_rejects_nonfinite_constant_sparse_and_bad_training_inputs(self):
        x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        invalid = x.clone()
        invalid[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite input"):
            fit_transform_features_bounded(invalid, torch.arange(3), CONDITIONS[1])
        with self.assertRaisesRegex(ValueError, "near-zero"):
            fit_transform_features_bounded(torch.ones_like(x), torch.arange(3), CONDITIONS[1])
        with self.assertRaisesRegex(ValueError, "no implicit densification"):
            fit_transform_features_bounded(x.to_sparse(), torch.arange(3), CONDITIONS[1])
        for indices in (torch.tensor([0, 0]), torch.tensor([6]), torch.tensor([.5])):
            with self.assertRaises(ValueError):
                fit_transform_features_bounded(x, indices, CONDITIONS[1])

    def test_memory_limit_refuses_before_normalization_or_dense_allocation(self):
        x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        with patch("scripts.input_robustness_data.transform_features") as transform:
            with self.assertRaises(MemoryError):
                fit_transform_features_bounded(x, torch.arange(3), CONDITIONS[1], max_feature_bytes=1)
            transform.assert_not_called()
        self.assertLess(estimate_transform_peak_bytes((18333, 6805), 10993), 4 * 1024**3)

    def test_array_hash_binds_dtype_shape_and_content(self):
        arr = np.arange(12, dtype=np.int64)
        self.assertNotEqual(array_sha256(arr), array_sha256(arr.reshape(3, 4)))
        self.assertNotEqual(array_sha256(arr), array_sha256(arr.astype(np.float64)))
        self.assertNotEqual(array_sha256(arr), array_sha256(arr + 1))
        self.assertEqual(array_sha256(arr), array_sha256(torch.from_numpy(arr)))

    def make_fixture(self, directory):
        x = torch.arange(36, dtype=torch.float32).reshape(12, 3)
        y = torch.tensor([0] * 6 + [1] * 6)
        edges = np.asarray([(a, b) for a in range(12) for b in range(a + 1, 12)])
        units = []
        for seed in range(10):
            split = make_stratified_split(y.numpy(), seed=seed)
            mask = np.zeros(12, dtype=bool)
            mask[split["train"]] = True
            eligible = edges[mask[edges[:, 0]] & mask[edges[:, 1]]]
            units.append({"dataset": "Cora", "seed": seed, "split_id": split_identifier(split),
                          "mean_degree": 11., "homophily": float(np.mean(y.numpy()[eligible[:, 0]] == y.numpy()[eligible[:, 1]]))})
        splits, checks, _ = check_original_binding("Cora", x, y, edges, units)
        arrays = {"x": x.numpy(), "y": y.numpy(), "edge_index": edges.T.copy()}
        for seed, (split, _) in splits.items():
            arrays.update({f"seed_{seed}_{part}": val for part, val in split.items()})
        target = Path(directory) / "Cora.npz"
        np.savez(target, **arrays)
        entry = {"dataset": "Cora", "status": "bound", "original_processed_byte_match": True,
                 "materialized_path": target.name, "materialized_sha256": file_sha256(target),
                 "tensor_sha256": {key: array_sha256(val) for key, val in arrays.items()},
                 "split_checks": checks}
        return {"datasets": {"Cora": entry}}, units, (x, y, edges)

    def test_binding_rejects_reordered_labels_or_old_diagnostic_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, units, (x, y, edges) = self.make_fixture(tmp)
            modified = copy.deepcopy(units)
            modified[0]["homophily"] += .01
            with self.assertRaisesRegex(ValueError, "diagnostic mismatch"):
                check_original_binding("Cora", x, y, edges, modified)
            with self.assertRaisesRegex(ValueError, "split mismatch"):
                check_original_binding("Cora", x, y.flip(0), edges, units)
            with self.assertRaisesRegex(ValueError, "exactly ten"):
                check_original_binding("Cora", x, y, edges, units[:-1])

    def test_read_only_load_and_file_tensor_split_corruption_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            binding, _, (x, y, _) = self.make_fixture(tmp)
            target = Path(tmp) / "Cora.npz"
            before = (target.stat().st_mtime_ns, file_sha256(target))
            loaded_x, loaded_y, _, splits, _ = load_bound_dataset("Cora", tmp, binding)
            self.assertTrue(torch.equal(loaded_x, x))
            self.assertTrue(torch.equal(loaded_y, y))
            self.assertEqual(set(splits), set(range(10)))
            self.assertEqual(before, (target.stat().st_mtime_ns, file_sha256(target)))
            bad = copy.deepcopy(binding)
            bad["datasets"]["Cora"]["tensor_sha256"]["x"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "tensor checksum"):
                load_bound_dataset("Cora", tmp, bad)
            bad = copy.deepcopy(binding)
            bad["datasets"]["Cora"]["split_checks"][0]["split_id"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "split mismatch"):
                load_bound_dataset("Cora", tmp, bad)
            target.write_bytes(target.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "materialized checksum"):
                load_bound_dataset("Cora", tmp, binding)

    def test_binding_never_allows_path_escape_or_unverified_original_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            binding, _, _ = self.make_fixture(tmp)
            binding["datasets"]["Cora"]["materialized_path"] = "../Cora.npz"
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_bound_dataset("Cora", tmp, binding)
            binding["datasets"]["Cora"]["original_processed_byte_match"] = False
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_bound_dataset("Cora", tmp, binding)

    def test_release_manifest_is_pinned_not_merely_self_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "MANIFEST.json").write_text('{"files": []}')
            with self.assertRaisesRegex(ValueError, "published v0.1.0 digest"):
                verify_release_manifest(tmp)

    def test_full_diagnostic_verification_rejects_two_hop_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            binding, _, _ = self.make_fixture(tmp)
            x, y, edges, splits, row = load_bound_dataset("Cora", tmp, binding)
            relative = "prospective/diagnostics/Cora/seed_000.json"
            path = Path(tmp) / relative
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"split_id": splits[0][1], "status": "success", "details": {"h_2": .5}}))
            manifest = {"files": [{"path": relative, "public_sha256": file_sha256(path)}]}
            with patch("scripts.input_robustness_data.verify_release_manifest", return_value=manifest), \
                 patch("scripts.input_robustness_data.train_only_diagnostics", return_value={"h_2": .6}) as diagnostic:
                with self.assertRaisesRegex(ValueError, "50k-walk diagnostic mismatch"):
                    enrich_dataset_binding(row, tmp, tmp)
                self.assertEqual(diagnostic.call_args.kwargs, {"sample_count": 50000, "seed": 200000})


if __name__ == "__main__":
    unittest.main()
