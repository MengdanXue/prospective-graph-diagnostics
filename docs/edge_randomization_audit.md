# Edge-Randomization Provenance Audit

## Finding

The archived edge-randomization results are not degree-preserving evidence. The
legacy `random_edge_shuffle` implementation in the research archive used a
configuration-style stub pairing, but it discarded self-loops and inserted the
remaining pairs into a set. The first operation removes stubs from affected
nodes; the second collapses duplicate edges. The implementation did not retry
until the original edge count and node degrees were restored, and it did not
write before/after degree-sequence checksums.

The public manuscript therefore treats the archived numbers as an exploratory
topology perturbation. They cannot identify a degree-matched effect, and they do
not support the old "causal evidence for structure" wording.

## Verified replacement

`experiments/degree_preserving_edge_randomization.py` implements double-edge
swaps for simple undirected graphs. Every accepted swap replaces two disjoint
edges with two new non-loop, non-duplicate edges. The output records:

- node and edge counts before and after;
- SHA-256 checksums of the sorted degree sequence;
- SHA-256 checksums of the edge sets;
- the requested, attempted, and successful swap counts; and
- homophily, component count, largest-component fraction, degree assortativity,
  and average clustering before and after.

`tests/test_degree_preserving_shuffle.py` verifies deterministic output,
simple-graph constraints, identical node/edge counts, and an identical sorted
degree sequence. Invalid self-loops, reverse duplicates, and out-of-range node
identifiers are rejected.

## Interpretation boundary

Degree preservation does not isolate homophily. Double-edge swaps can change
label mixing, connected components, assortativity, clustering, path structure,
and other graph properties at the same time. A future intervention must report
these concurrent changes and be described as degree-matched randomization, not
as identification of a single structural mechanism.

## Result status

The archived Cora, CiteSeer, and PubMed numbers were not regenerated or relabeled.
A separate replacement experiment is complete with immutable paired per-seed
records, matched train/validation/test splits, invariant checksums, complete edge
lists, and concurrent structural statistics. The active manuscript reports that
replacement result. The legacy table and figure remain non-confirmatory and must
not be cited as degree-preserving evidence.
