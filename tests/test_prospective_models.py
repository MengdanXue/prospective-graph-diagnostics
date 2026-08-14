import unittest

import torch

from experiments.prospective_models import (
    MODEL_IDS,
    build_model,
    prepare_h2_adjacencies,
    strict_two_hop_edge_index,
)


class ProspectiveModelTests(unittest.TestCase):
    def setUp(self):
        self.edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]],
            dtype=torch.long,
        )
        self.x = torch.randn(6, 5, generator=torch.Generator().manual_seed(1))

    def test_registry_contains_exact_frozen_model_set(self):
        self.assertEqual(
            list(MODEL_IDS),
            ["MLP", "GCN", "GAT", "GraphSAGE", "H2GCN", "LINKX", "GPR-GNN"],
        )

    def test_all_models_produce_finite_node_logits(self):
        h2 = prepare_h2_adjacencies(self.edge_index, num_nodes=6)
        for model_id in MODEL_IDS:
            with self.subTest(model=model_id):
                torch.manual_seed(4)
                model = build_model(
                    model_id,
                    num_nodes=6,
                    in_channels=5,
                    hidden_channels=8,
                    out_channels=3,
                    dropout=0.5,
                    edge_index=self.edge_index,
                    h2_adjacencies=h2,
                )
                model.eval()
                logits = model(self.x, self.edge_index)
                self.assertEqual(tuple(logits.shape), (6, 3))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 0)

    def test_strict_two_hop_edges_exclude_self_and_one_hop_edges(self):
        two_hop = strict_two_hop_edge_index(self.edge_index, num_nodes=6)
        one_hop = {tuple(edge) for edge in self.edge_index.t().tolist()}
        strict = {tuple(edge) for edge in two_hop.t().tolist()}
        self.assertTrue(strict)
        self.assertTrue(all(left != right for left, right in strict))
        self.assertFalse(one_hop & strict)
        self.assertIn((0, 2), strict)
        self.assertIn((2, 0), strict)


if __name__ == "__main__":
    unittest.main()
