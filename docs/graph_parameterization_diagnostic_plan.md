# Graph input parameterization diagnostic: fixed next-stage design

2026-09-08. This is a post hoc, validation-only transfer study motivated by the completed MLP diagnostic. The composite centered/scaled condition was chosen after seeing those MLP results. It is not an independent confirmation or a new preregistration of the original benchmark.

## Question

Does the MLP improvement transfer to the six original graph architectures? Which normalized models actually concentrate predictions on the training majority class, and how do the full validation performance profiles change after the same joint centering/scale intervention?

All six graph architectures are retained, including H2GCN and LINKX, so the comparison is not restricted to architectures that previously performed poorly. MLP is rerun in every condition as a reference and implementation control. Do not combine newly trained MLP values with old graph outcomes.

## Fixed scope and budget

Use Roman-empire and Amazon-ratings, all ten original seeds, and MLP, GCN, GAT, GraphSAGE, H2GCN, LINKX and GPR-GNN. Compare exactly three feature inputs:

1. Raw features X.
2. Original PyG NormalizeFeatures output N.
3. a(N - m), where m and the **single global scalar** a use the unchanged train-feature fitting function from the MLP diagnostic. The scalar matches overall centered feature RMS, not each column separately.

Keep weight decay 0.0005 for all cells. Keep hidden width 64, the four original learning-rate/dropout trials, 500 maximum epochs, patience 100 and all existing checkpoint/trial selection rules. This phase tests transfer of the combined intervention, not a full graph-model scale/centering/decay factorial.

All 420 selected model units are trained anew: 2 datasets x 10 seeds x 7 architectures x 3 conditions. This is 1,680 trials. Run one CUDA worker. The previous 280-record run used 1.305 summed unit-hours; a roughly 2–3 hour training budget allows the additional condition and longer training after recovery. This is an estimate, not an outcome-dependent stop rule. Do not expand the grid or substitute favorable seeds after viewing results.

## Evaluation and provenance

Preserve the existing model registry, data/split construction, `_train_trial`, MLP runner and original benchmark files. A separate runner may reuse the validation-only unit execution and correct its returned family label for the requested architecture; the saved record and its validator must name the actual model. Do not pass graph records through validation by disguising them as MLP records.

Fit feature statistics on training features only, while retaining the original transductive NormalizeFeatures operation. Score only training and validation partitions. Full-node graph propagation uses the fixed graph as before; test nodes are never indexed for prediction metrics or selection. New test evaluations and test metrics remain zero.

Commit the design, config, new execution code and tests before evaluating these datasets. Bind every record and the run manifest to the execution source commit, executable/config fingerprints, environment, pinned NPZ hashes, original split ID, and exact transformed-feature hash and fit metadata. Record all four trial histories, selected trial/checkpoint, the training-time validation metrics used for selection, and the final selected-state train/validation metrics bound to their partitions. CUDA graph propagation can be nondeterministic across replays, so the two metric pairs remain separate and any discrepancy is disclosed rather than silently conflated. Also record train/validation label counts, prediction counts, class recalls, balanced accuracy, loss, majority baseline and dominant prediction fraction. Prepare H2GCN adjacency buffers once per dataset, using the original function.

Reuse strict selected-checkpoint and metric consistency checks. Check transformed inputs and actual partition class counts again when summarizing. Reject wrong provenance, incomplete or duplicate scope, stale/failure artifacts, invalid histories and unexpected test metrics. Writes are exclusive; an exact completed resume is read-only and idempotent. Before adding a missing unit during resume, validate every pre-existing record so corruption cannot be found only after new work has begun.

## Controls and descriptive comparisons

The primary analysis reports **each dataset and each architecture separately**; ten seeds are repeated runs, not ten independent datasets.

- For every architecture, report all three validation-accuracy means and sample SDs, train/validation loss and balanced accuracy, prediction concentration ranges, and the count of selected models predicting the training majority class for every validation example.
- For each architecture, report all ten paired differences for centered-scaled minus normalized (primary), centered-scaled minus raw (reference), and normalized minus raw (reproduction). Report means, sample SDs, ranges and positive/zero/negative counts; do not turn small datasets or repeated seeds into population significance claims.
- Compare the 280 raw/normalized controls with the earlier preprocessing run using validation accuracy, loss, selected trial, feature hashes and environment. Compare all 60 MLP cells with the preceding MLP diagnostic at weight decay 0.0005. Report exact mismatch counts and magnitudes. Investigate drift before interpreting results; numerical nondeterminism is not an automatic exemption.
- As a secondary development summary, select the graph architecture by its validation accuracy, then validation loss, then model ID within each condition. Report selection counts and graph-minus-MLP validation differences. The same validation set selects and reports the graph candidate, so this is selection-optimistic, uses 24 graph trials versus four MLP trials, and does not estimate held-out regret or a graph benefit on test data. Report the individual-architecture table alongside it.

If many models recover, describe sensitivity to the composite input treatment under this training recipe. If improvements are confined to particular models, retain that limitation. No condition replaces the frozen benchmark, and no new test-based target actions or policy regrets are computed.

## Interpretation limit specific to graph architectures

An invertible affine feature change preserves the input information, but it need not preserve the realized function family of every fixed graph architecture. For example, subtracting a constant feature vector before symmetric graph normalization can induce a node-dependent shift proportional to the normalized adjacency row sum; a shared post-aggregation bias cannot generally absorb it. Input dropout and attention can also interact with centering. Consequently, this study cannot automatically inherit the MLP's optimization-only interpretation. It tests empirical transfer and identifies which architecture-specific mechanisms deserve further work.

## Acceptance and next decision

Require exact 420/1,680 scope, zero new test evaluation, full record/input validation, disclosed control drift, meaningful CPU tests across all seven architectures, and a reviewed per-architecture report before interpreting the stage as complete. A resource or execution failure remains visible and is not replaced by a partial favorable summary.

After review, decide whether an independently designed dataset expansion is justified and whether the manuscript's scope should change. Keep the original manuscript unchanged during this diagnostic stage.
