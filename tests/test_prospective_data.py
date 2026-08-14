import tempfile
import unittest
from pathlib import Path

import numpy as np
import experiments.prospective_data as prospective_data

from experiments.prospective_data import (
    canonical_undirected_edges,
    checksum_files,
    dataset_specifications,
    make_stratified_split,
    split_identifier,
    train_only_diagnostics,
)


class ProspectiveDataTests(unittest.TestCase):
    def test_dataset_eligibility_rejects_singleton_class_before_split(self):
        labels = np.asarray([0] * 33 + [1] + [2] * 18 + [3] * 101 + [4] * 30)
        with self.assertRaisesRegex(
            ValueError,
            r"minimum_class_count=3.*class_counts=.*'1': 1",
        ):
            prospective_data.validate_dataset_eligibility(
                labels,
                minimum_class_count=3,
            )

    def test_dataset_eligibility_returns_auditable_counts_when_eligible(self):
        counts = prospective_data.validate_dataset_eligibility(
            np.asarray([0, 0, 0, 1, 1, 1, 1]),
            minimum_class_count=3,
        )
        self.assertEqual(counts, {"0": 3, "1": 4})

    def test_dataset_registry_covers_the_frozen_scope(self):
        self.assertEqual(
            list(dataset_specifications()),
            [
                "Cora",
                "CiteSeer",
                "PubMed",
                "Texas",
                "Wisconsin",
                "Cornell",
                "Chameleon",
                "Squirrel",
                "Actor",
                "Roman-empire",
                "Amazon-ratings",
                "Coauthor-CS",
            ],
        )

    def test_stratified_split_is_deterministic_disjoint_and_complete(self):
        labels = np.repeat(np.arange(3), 10)
        first = make_stratified_split(labels, seed=7)
        second = make_stratified_split(labels, seed=7)
        for name in ("train", "validation", "test"):
            np.testing.assert_array_equal(first[name], second[name])
        combined = np.concatenate(list(first.values()))
        self.assertEqual(sorted(combined.tolist()), list(range(30)))
        self.assertEqual(len(np.unique(combined)), 30)
        for label in range(3):
            self.assertEqual(sum(labels[first["train"]] == label), 6)
            self.assertEqual(sum(labels[first["validation"]] == label), 2)
            self.assertEqual(sum(labels[first["test"]] == label), 2)
        self.assertNotEqual(split_identifier(first), split_identifier(make_stratified_split(labels, seed=8)))

    def test_canonical_edges_remove_loops_reverse_duplicates_and_sort(self):
        edge_index = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 2, 0]])
        edges = canonical_undirected_edges(edge_index, node_count=4)
        np.testing.assert_array_equal(edges, np.array([[0, 1], [0, 3], [1, 2]]))

    def test_diagnostics_never_read_nontraining_labels(self):
        edges = np.array(
            [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 4], [4, 5]],
            dtype=np.int64,
        )
        train = np.array([True, True, True, True, False, False])
        labels = np.array([0, 0, 1, 1, 0, 1])
        changed = labels.copy()
        changed[~train] = 99
        first = train_only_diagnostics(edges, labels, train, sample_count=500, seed=11)
        second = train_only_diagnostics(edges, changed, train, sample_count=500, seed=11)
        self.assertEqual(first, second)
        self.assertEqual(first["homophily_label_scope"], "train_only")
        self.assertEqual(first["two_hop_label_scope"], "train_only")
        self.assertFalse(first["uses_test_labels"])
        self.assertEqual(first["two_hop_requested_walks"], 500)
        self.assertEqual(first["two_hop_accepted_walks"], 500)

    def test_file_checksums_include_relative_path_size_and_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "nested" / "b.bin").write_bytes(b"beta")
            records = checksum_files(root)
        self.assertEqual([row["path"] for row in records], ["a.txt", "nested/b.bin"])
        self.assertEqual([row["size"] for row in records], [5, 4])
        self.assertTrue(all(len(row["sha256"]) == 64 for row in records))


if __name__ == "__main__":
    unittest.main()
