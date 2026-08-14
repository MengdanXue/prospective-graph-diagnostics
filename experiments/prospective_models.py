#!/usr/bin/env python3
"""Frozen model registry for the Route A prospective benchmark."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import GATConv, GCNConv, LINKX, MessagePassing, SAGEConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm


MODEL_IDS = ("MLP", "GCN", "GAT", "GraphSAGE", "H2GCN", "LINKX", "GPR-GNN")


class MLPModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.first = nn.Linear(in_channels, hidden_channels)
        self.second = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        del edge_index
        return self.second(F.dropout(F.relu(self.first(x)), p=self.dropout, training=self.training))


class GCNModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.first = GCNConv(in_channels, hidden_channels)
        self.second = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.first(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.second(x, edge_index)


class GATModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
        heads: int = 4,
    ):
        super().__init__()
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by GAT heads")
        self.first = GATConv(
            in_channels,
            hidden_channels // heads,
            heads=heads,
            dropout=dropout,
        )
        self.second = GATConv(
            hidden_channels,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.first(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.second(x, edge_index)


class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.first = SAGEConv(in_channels, hidden_channels)
        self.second = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.first(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.second(x, edge_index)


class GPRPropagation(MessagePassing):
    def __init__(self, steps: int = 10, alpha: float = 0.1):
        super().__init__(aggr="add")
        self.steps = steps
        self.alpha = alpha
        coefficients = alpha * (1.0 - alpha) ** np.arange(steps + 1)
        coefficients[-1] = (1.0 - alpha) ** steps
        self.coefficients = Parameter(torch.tensor(coefficients, dtype=torch.float32))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        normalized_edges, weights = gcn_norm(
            edge_index,
            num_nodes=x.size(0),
            dtype=x.dtype,
            add_self_loops=True,
        )
        output = self.coefficients[0] * x
        for step in range(self.steps):
            x = self.propagate(normalized_edges, x=x, norm=weights)
            output = output + self.coefficients[step + 1] * x
        return output

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


class GPRGNNModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
        steps: int = 10,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.first = nn.Linear(in_channels, hidden_channels)
        self.second = nn.Linear(hidden_channels, out_channels)
        self.propagation = GPRPropagation(steps=steps, alpha=alpha)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.first(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.second(x)
        return self.propagation(x, edge_index)


def _scipy_adjacency(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    values = edge_index.detach().cpu().numpy()
    adjacency = sp.coo_matrix(
        (np.ones(values.shape[1], dtype=np.float32), (values[0], values[1])),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data[:] = 1.0
    return adjacency


def strict_two_hop_edge_index(edge_index: torch.Tensor, *, num_nodes: int) -> torch.Tensor:
    adjacency = _scipy_adjacency(edge_index, num_nodes)
    two_hop = adjacency @ adjacency
    two_hop.data[:] = 1.0
    two_hop.setdiag(0)
    two_hop = two_hop - two_hop.multiply(adjacency)
    two_hop.eliminate_zeros()
    two_hop = two_hop.tocoo()
    order = np.lexsort((two_hop.col, two_hop.row))
    indices = np.vstack((two_hop.row[order], two_hop.col[order])).astype(np.int64)
    return torch.from_numpy(indices)


def _normalized_sparse(adjacency: sp.csr_matrix) -> torch.Tensor:
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.zeros_like(degrees, dtype=np.float32)
    nonzero = degrees > 0
    inverse[nonzero] = np.power(degrees[nonzero], -0.5)
    normalized = adjacency.multiply(inverse[:, None]).multiply(inverse[None, :]).tocoo()
    indices = torch.from_numpy(
        np.vstack((normalized.row, normalized.col)).astype(np.int64)
    )
    values = torch.from_numpy(normalized.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, normalized.shape).coalesce()


def prepare_h2_adjacencies(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    one_hop = _scipy_adjacency(edge_index, num_nodes)
    strict_edges = strict_two_hop_edge_index(edge_index, num_nodes=num_nodes).numpy()
    two_hop = sp.coo_matrix(
        (
            np.ones(strict_edges.shape[1], dtype=np.float32),
            (strict_edges[0], strict_edges[1]),
        ),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    return _normalized_sparse(one_hop), _normalized_sparse(two_hop)


class H2GCNModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
        adjacencies: tuple[torch.Tensor, torch.Tensor],
        propagation_layers: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Linear(in_channels, hidden_channels)
        self.dropout = dropout
        self.propagation_layers = propagation_layers
        self.register_buffer("one_hop", adjacencies[0])
        self.register_buffer("two_hop", adjacencies[1])
        final_channels = hidden_channels * (2 ** (propagation_layers + 1) - 1)
        self.classifier = nn.Linear(final_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        del edge_index
        x = F.relu(self.embedding(x))
        representations = [x]
        for _ in range(self.propagation_layers):
            propagated = torch.cat(
                [torch.sparse.mm(self.one_hop, x), torch.sparse.mm(self.two_hop, x)],
                dim=1,
            )
            x = F.relu(propagated)
            representations.append(x)
        output = torch.cat(representations, dim=1)
        output = F.dropout(output, p=self.dropout, training=self.training)
        return self.classifier(output)


def build_model(
    model_id: str,
    *,
    num_nodes: int,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dropout: float,
    edge_index: torch.Tensor,
    h2_adjacencies: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> nn.Module:
    if model_id == "MLP":
        return MLPModel(in_channels, hidden_channels, out_channels, dropout)
    if model_id == "GCN":
        return GCNModel(in_channels, hidden_channels, out_channels, dropout)
    if model_id == "GAT":
        return GATModel(in_channels, hidden_channels, out_channels, dropout)
    if model_id == "GraphSAGE":
        return GraphSAGEModel(in_channels, hidden_channels, out_channels, dropout)
    if model_id == "H2GCN":
        if h2_adjacencies is None:
            h2_adjacencies = prepare_h2_adjacencies(edge_index, num_nodes=num_nodes)
        return H2GCNModel(
            in_channels,
            hidden_channels,
            out_channels,
            dropout,
            h2_adjacencies,
        )
    if model_id == "LINKX":
        return LINKX(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=2,
            num_edge_layers=1,
            num_node_layers=2,
            dropout=dropout,
        )
    if model_id == "GPR-GNN":
        return GPRGNNModel(in_channels, hidden_channels, out_channels, dropout)
    raise ValueError(f"unknown frozen model identifier: {model_id}")
