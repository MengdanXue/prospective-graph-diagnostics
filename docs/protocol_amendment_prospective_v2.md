# Prospective Benchmark Protocol Amendment v2

Date frozen: 2026-08-11

Parent run: `route_a_prospective_v1`

Replacement run: `route_a_prospective_v2`

## Trigger

The v1 run completed Cora, CiteSeer, and PubMed. Its first Texas command then
failed before split creation because Texas has class counts
`33/1/18/101/30`. The singleton class cannot contribute one node to each of
train, validation, and test as required by the frozen per-class split. No Texas
model was trained, no Texas test accuracy was produced, and Wisconsin/Cornell
were not started in that command sequence.

The v1 directory is retained unchanged as a feasibility-aborted run containing
210 successful model records and 30 diagnostic records for the first three
datasets. Those records are not eligible for reuse in v2.

## Outcome-independent amendment

A dataset is eligible only when every class contains at least three nodes.
Applying this rule to labels before any v2 model outcome excludes Texas and no
other candidate. The v2 benchmark therefore contains eleven datasets. All
remaining scientific settings are byte-equivalent to v1: ten seeds, seven
models, four equal-budget trials, early stopping, train-only diagnostics,
selection rules, evaluator thresholds, bootstrap/permutation settings, and the
degree-matched intervention protocol.

## Reproducibility boundary

Version 2 uses a new run ID, configuration digest, code commit, and result
directory. It starts from zero. An assembler must reject v1 records, mixed
commits, mixed configurations, missing trial audits, or any failure artifact.
Raw caches remain read-only and processed data checksums remain mandatory.
