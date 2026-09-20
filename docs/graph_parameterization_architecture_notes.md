# Interpreting the affine input treatment across architectures

2026-09-08. These notes precede the graph transfer run and are based on the frozen model registry and the installed PyG 2.7.0 implementation. They constrain the interpretation of the planned results; they are not performance findings.

Write the treated input as Z = a(N - 1 m^T), with scalar a > 0. For a pointwise affine layer ZW + 1b^T, an equivalent layer on N uses W' = aW and b' = b - a m^T W. Thus the input treatment can be absorbed into a trainable affine layer that precedes all other operations. The L2 penalty, initialization, optimizer trajectory and some dropout distributions need not be preserved by this parameter mapping.

| Frozen model | What the first-layer argument supports |
|---|---|
| MLP | A biased linear layer precedes ReLU; evaluation functions admit the affine parameter mapping. |
| H2GCN | A biased pointwise embedding precedes graph propagation; mapping that embedding preserves its output and subsequent evaluation computations. |
| LINKX | The first node-MLP linear layer has a bias; mapping that layer preserves the node branch, while the adjacency branch is unchanged. |
| GPR-GNN | Evaluation has a biased pointwise linear layer before propagation. Input dropout occurs before that layer during training, so centering changes its stochastic training behavior even when evaluation functions can be mapped. |
| GraphSAGE | With the frozen mean aggregator and no isolated nodes, the neighbor and root linear maps can both be rescaled and their constant shifts absorbed in the neighbor-branch bias. Isolated nodes require separate treatment. Both selected graphs have zero isolated nodes in the canonical input graph. |
| GCN | The first linear projection has no bias, and a shared bias is added after symmetric graph aggregation. The MLP's constant-bias argument does not directly apply. |
| GAT | The first projection has no bias; projected features also enter attention before a leaky ReLU. A common feature shift can affect attention weights, so the MLP argument does not directly apply. |

For GCN, let S be the self-loop-augmented symmetric normalized adjacency. Then

\[
SZ W + \mathbf{1}b^T
= a SNW - a(S\mathbf{1})(m^TW) + \mathbf{1}b^T.
\]

When S1 is not constant, a shared bias cannot directly absorb the displayed shift. This does not prove that the entire multi-layer GCN function class is different; it identifies why the simple first-layer invariance proof is insufficient. A change in performance should not automatically be attributed solely to optimization.

Input-only checks using the original canonical edges and `gcn_norm` with self-loops found:

| Dataset | Isolated nodes | Minimum S1 | Maximum S1 | Sample SD of S1 |
|---|---:|---:|---:|---:|
| Roman-empire | 0 | 0.740603 | 1.825664 | 0.106530 |
| Amazon-ratings | 0 | 0.387698 | 4.363376 | 0.306325 |

These checks used no model training, validation scores or test scores. Data checksums and splits were verified by the existing input loader. All numerical performance conclusions must wait for the complete, paired 420-record study.

The graph runner retains the training-time validation values used to select each checkpoint separately from the final selected-state replay values reported for train and validation partitions. This distinction is necessary because CUDA message-passing kernels can be nondeterministic across replays; the final partition metrics are the canonical reported values and are required to agree internally with the saved row.
