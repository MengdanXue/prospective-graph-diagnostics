# Eleven-dataset input-treatment robustness: fixed post-hoc design

Design date: 2026-09-13. Configuration: [input_robustness_11_v1.json](../configs/input_robustness_11_v1.json).

Execution update, 2026-09-14: the [measured acceptance report](input_robustness_11_preflight_report.md)
records all 44 workers and 616 repeat pairs complete, with data binding,
short-repeat fingerprints and memory within the fixed requirements. The formal
time-budget gate did not pass: the conservative estimate is 12.910 days versus
seven days, and both Squirrel/H2GCN model units exceed their two-hour cap.
The scientific configuration below remains unchanged. No formal run or new
selection/calibration analysis has started; resource planning is the next gate.

## Purpose and stage boundary

The question is whether the original selection-loss findings, dependence on the
graph-model portfolio, and calibrated-threshold findings persist under one
candidate input treatment. The candidate was motivated by the earlier two-dataset
MLP and graph diagnostics. Its performance on the full set is unknown; it is not
designated the best preprocessing for these datasets. This entire study remains
**post hoc**, including its design freeze, resource preflight, retraining and
analysis. It is not a new independent confirmatory validation.

The current stage prepares and checks the design, input binding, deterministic
execution and resources. It permits bounded preflight work only. The configuration
sets `formal_training_enabled: false`; passing tests or a subset of probes does
not start the 6,160-trial experiment. Formal execution remains gated on complete
preflight evidence, tested formal recording/analysis code, a clean committed
source revision, verified remote synchronization, and a report of actual CI
status. A failed or unmeasured gate is reported as such. Enabling a later formal
stage requires a new committed stage/launch artifact that preserves this frozen
scientific specification and binds the accepted preflight; do not edit this
preflight artifact in place or imply its disabled flag authorizes training.

The user's new research instruction supersedes the September 9 freeze's earlier
restriction on new experiments. Preserve commit
`8847de65822552558144c4a472fd53fae4f021a0`, every historical result, manuscript and
configuration. Work in a new branch and output tree. Neither a new normalized
control nor the candidate replaces an old record. No historical MLP or graph
outcome enters a new condition's model portfolio.

## Fixed scope and input treatment

The inherited population is the eleven datasets in
`configs/prospective_benchmark_v2.json`, with its exact graph versions and node
ordering. Texas was excluded before that original v2 run and is not a newly
excluded observation here. All eleven datasets must pass applicability checks;
an unsuitable or unavailable dataset blocks complete execution rather than being
silently removed. The counts below are **reference expectations from the frozen
reviewer appendix**, not claims that the new local cache has passed verification.

| Dataset | Nodes | Undirected non-loop edges | Features | Classes | Applicability/resource focus |
|---|---:|---:|---:|---:|---|
| Cora | 2,708 | 5,278 | 1,433 | 7 | Preserve original features and class ordering |
| CiteSeer | 3,327 | 4,552 | 3,703 | 6 | Preserve isolated nodes and feature width |
| PubMed | 19,717 | 44,324 | 500 | 3 | Full-node tensor and H2GCN memory |
| Wisconsin | 251 | 450 | 1,703 | 5 | All classes populate each split; finite fitted RMS |
| Cornell | 183 | 277 | 1,703 | 5 | Small training partitions; finite fitted RMS |
| Chameleon | 2,277 | 31,371 | 2,325 | 5 | Original Wikipedia version, no cleaning substitution |
| Squirrel | 5,201 | 198,353 | 2,089 | 5 | Strict two-hop adjacency and attention memory/time |
| Actor | 7,600 | 26,659 | 932 | 5 | Input densification and split agreement |
| Roman-empire | 22,662 | 32,927 | 300 | 18 | Continuous-feature affine treatment; pinned NPZ |
| Amazon-ratings | 24,492 | 93,050 | 300 | 5 | Continuous-feature affine treatment; pinned NPZ |
| Coauthor-CS | 18,333 | 81,894 | 6,805 | 15 | Largest dense feature matrix; float64 transform copies |

Compare exactly two conditions on every dataset and split:

1. `normalize_features`: reuse the original PyG `NormalizeFeatures` operation.
2. `normalize_centered_scaled`: apply the existing train-fitted composite
   treatment to that same normalized matrix with unchanged numerical semantics.

Let the raw feature matrix be X and N = NormalizeFeatures(X). The original
normalization subtracts the full feature matrix minimum and divides rows by row
sums clamped below at one. It is the existing transductive feature operation; it
does not use labels. For training node indices T, compute normalized column means
m = mean(N[T], axis=0), raw column means r = mean(X[T], axis=0), and one scalar

`a = sqrt(mean((X[T] - r)^2)) / sqrt(mean((N[T] - m)^2))`.

The candidate is `a * (N - m)` for all nodes. Preserve the formula and fitting
semantics of `scripts.run_mlp_optimization_diagnostic.fit_transform_features`
through `scripts.input_robustness_data.fit_transform_features_bounded`, with
exact parity tests against the historical function: fit statistics
in float64 on **training features only**, output float32, and reject nonfinite or
at-most-`1e-12` train RMS. The scalar matches overall centered RMS; it is not
per-column standardization. Save fitted means, scalar, RMS values and final tensor
hashes for every dataset/seed/condition. Do not choose raw inputs, per-column
scaling, a different epsilon, zero decay, or an extra treatment after inspecting
results. Both existing conditions must be applicable to all ten splits.

Centering can densify sparse features, and the historical fitting function makes
large full-matrix float64 copies. Before resource measurement or new outcomes,
this design fixes a bounded-memory implementation that fits one float64 training
block at a time and produces the same float32 output in chunks. This changes
memory scheduling, not the input condition or statistic. Bind the new function's
source fingerprint and require parity tests before use; do not change reduction
precision, fit scope or arithmetic formula to meet the cap. Measure the complete
adapter's memory, especially on Coauthor-CS, and report failures rather than
claiming that equivalence implies feasibility. Affine input changes can interact with graph aggregation,
attention and dropout, so improved accuracy would not by itself establish an
optimization-only mechanism for every architecture.

## Pairing and training rules

Retain seeds 0 through 9, all seven original implementations (MLP, GCN, GAT,
GraphSAGE, H2GCN, LINKX and GPR-GNN), and four trials per model and condition:
learning rate/dropout `(0.01, 0.5)`, `(0.01, 0.7)`, `(0.005, 0.5)`,
`(0.005, 0.7)`. Keep width 64, Adam/weight decay 0.0005, 500 maximum epochs,
patience 100, and the original architecture parameters. Reuse the original
model registry, H2GCN adjacency constructor and training logic; do not redesign
an architecture to make a treatment pass.

For each trial, reset the same initialization/training seed across paired
conditions. Each condition independently selects its own checkpoint and trial by
the unchanged validation rules. Checkpoint selection follows `_train_trial`;
trial selection is validation accuracy descending, validation loss ascending,
then trial identifier ascending. Graph-architecture selection is **validation
accuracy descending then model identifier ascending**, as in the frozen
evaluator; it does not add a validation-loss tie breaker from a later diagnostic.
There is exactly one selected-state test evaluation per completed model unit,
after all four validation-selected trials. No test score controls training,
selection, scope, retry, grid or conditions.

The formal count is 11 datasets x 2 conditions x 7 architectures x 10 seeds =
**1,540 selected-model records and 6,160 trials**. Each condition contributes 770
records and 110 graph-versus-MLP decision units. The full graph action still has
24 trials compared with four for MLP. This study compares input treatments within
that budget; the single-architecture sensitivity separately matches four trials
against four. Neither comparison equalizes wall time, FLOPs or parameter count.

## Data and split binding before training

Acquire missing public source files explicitly into a dedicated new cache, then
require the reconstructed processed-file bytes to match the original release.
Preserve any old cache and failed download files. Acquisition is separate from
the read-only training-time loader and never selected using model outcomes.
Transport failures may receive at most three raw-file download attempts, with a
256 MiB per-file acquisition limit and preserved partial files. Acceptance still
requires the exact original processed bytes; these transport retries do not
authorize any training retry.
After binding, use the new verified cache/materialized tensors in read-only mode.
Bind the trusted original release
manifest/archive identity, all raw and processed file SHA-256 values, raw feature
tensor, labels, canonical undirected edge list and ordered bidirectional model
edge tensor. Record local paths only as provenance, not as identity. Loading must
not trigger a download, cache rewrite, graph cleaning, node reordering, feature
normalization twice or changed dataset constructor.

Recompute `make_stratified_split` with its original classwise floor/max rules and
seed for **all 110 dataset/seed pairs**. Require exact `split_identifier` equality
with the frozen record, complete and disjoint partition coverage, expected node
and feature counts, minimum class count three, and at least one member of every
class in each partition. Fixed split ratios are 0.6/0.2/0.2; realized small-graph
counts follow the original rounding. Bind all partition indices and class counts.

Recompute canonical mean degree and the original train-label-only homophily and
two-hop diagnostics (50,000 length-two walks, no identical endpoints) and require
agreement with all frozen unit diagnostics. This produces 110 unique diagnostic
units; a new condition's evaluator may reference their identical values through
a condition-specific provenance wrapper. Metadata may inspect labels to verify
dataset identity, class eligibility and split construction. That does not permit
preflight prediction metrics on validation or test labels. The two supplied NPZ
hashes must match the earlier pinned hashes as well as the original reference
binding; all other datasets need the same full tensor/split checks.

Seal a machine-readable binding manifest before formal training. Record its hash
with the configuration digest, execution source commit and executable file hashes
in every subsequent manifest and record. A mismatch is a hard stop; matching
rounded historical accuracies is not a substitute for identity verification.

## Determinism and bounded preflight

Use one worker at a time, four Torch CPU threads and one interop thread. The
device map is fixed before preflight: **H2GCN uses CPU for all eleven datasets
and both conditions; the other six models use CUDA**. The reviewed local
PyTorch/CUDA sparse-matrix multiplication path used by H2GCN lacks a strict
deterministic-dispatch guarantee. This explicit architecture-level execution
setting retains the original model implementation and is not a third input
condition or a fallback selected after an observed failure. Resource feasibility
for the CPU H2GCN path must still pass; no condition or dataset silently migrates
device. The version-pinned basis is the CSR algorithm dispatch in
[PyTorch 2.9.1 SparseCUDABlas.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/sparse/cuda/SparseCUDABlas.cpp)
and the algorithm-specific determinism qualifications in
[CUDA 12.8 cuSPARSE, SpMM](https://docs.nvidia.com/cuda/archive/12.8.0/cusparse/index.html).
This implementation review motivates the fixed CPU choice; it is not a claim
that new H2GCN GPU probes have already failed. Pair the same device/runtime within each architecture across both
conditions. Set Python/NumPy/Torch/CUDA seeds,
`PYTHONHASHSEED` before worker creation, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, strict
`torch.use_deterministic_algorithms(True, warn_only=False)`, deterministic cuDNN,
cuDNN benchmark off and TF32 off. Record package/CUDA/driver versions, device,
thread settings and determinism flags. An unsupported deterministic operation
fails the gate; do not silently use a warning, different model, different device
or a nondeterministic implementation. Different hardware/runtime results are a
new execution context, not an exact resume.

After data binding, run two fresh-process repeats for each of 11 datasets x two
conditions at seed 0: **44 workers**. Every worker probes all seven original
models and all four configured trials, with three training steps followed by an
evaluation-mode forward pass. Prepare H2GCN's original adjacency once within the
worker. Retain all four trial states for the current model during the probe,
including any H2GCN sparse buffers, so the measured state-retention peak reflects
the formal model unit. Reset seeds per model/trial as in the formal logic. Compare identical
condition/model/trial probes between repeats using feature, initial state, final
state, final gradient and evaluation-logit hashes. Require finite optimization
values and finite parameters/gradients/logits. Do not calculate or print any
validation/test accuracy, loss, targets, regret or comparative selection outcome.

This is 1,232 model/trial probes and 3,696 short optimization steps. Probe outputs
are timing/memory/finite-value/determinism evidence; **formal model records = 0,
validation evaluations = 0, test evaluations = 0**. Store them under a separate
preflight root with explicit mode, never a formal `records` tree, and never
resume their weights into formal training. Existing known historical outcomes
may motivate the design but do not select or rank probes. Seed 0/three-step hash
agreement is a bounded check, not a guarantee for all ten seeds or 500 epochs;
strict determinism and finite-value enforcement remain active during formal work.

## Resource ceilings and launch estimate

Initial hardware inventory reported for this host is an NVIDIA RTX 5060 Laptop
GPU with 8,151 MiB VRAM, approximately 31.42 GiB physical RAM and 20 logical CPUs.
Free memory and disk are transient; the preflight report must record fresh
observations. Hardware availability alone is not a successful preflight.

| Resource | Fixed ceiling or reserve |
|---|---:|
| Simultaneous CUDA workers | 1 |
| Torch CPU / interop threads | 4 / 1 |
| Worker peak resident memory | 8 GiB |
| System available memory reserve | at least 4 GiB |
| CUDA allocated memory | at most 6 GiB |
| CUDA reserved memory | at most 6 GiB |
| Free output-disk reserve | at least 20 GiB |
| Entire preflight wall time | 7,200 seconds |
| One preflight worker | 1,200 seconds |
| One formal model unit, all four trials | 7,200 seconds |
| Total formal worker wall time, including attempts | 604,800 seconds (seven days) |

Measure setup, transform, adjacency preparation, synchronized training-step and
evaluation-forward times, worker RSS, system available RAM, allocated/reserved
CUDA peaks and disk reserve. Worker RSS means the sum across the entire owned
process tree, including a Windows virtual-environment launcher, the real Python
interpreter and subprocesses. Include Windows resident-memory high-water marks;
terminate the entire owned tree on a cap. Launcher-only measurements cannot pass
acceptance. Project a conservative no-early-stopping runtime:
for each model/trial/condition/dataset use the maximum repeat's mean synchronized
epoch time, multiply by 500 epochs and ten seeds, then add conservative setup
allowances. For each dataset/condition, multiply maximum measured transform/setup
time plus maximum worker nontraining overhead by 70 model/seed units, and add
maximum H2GCN adjacency preparation time times ten seeds. Multiply the complete
sum by two. The repeated setup allowance deliberately overcounts reusable work.
Publish the exact arithmetic and each dataset's contribution. A three-step,
seed-0 probe is an estimate, **not a guaranteed upper bound**; wall/memory guards
remain active.

Formal launch is blocked if any probe, its determinism comparison, an all-seed
input check, a memory reserve or the projected seven-day total fails. Do not
shrink the scientific scope to fit. A resource-only amendment may be proposed
from timing/memory evidence without inspecting new comparative outcomes; it must
be documented, committed, tested and reported before using it. A timeout ends
the attempt as incomplete, not as a satisfactory reduced experiment.

### Preflight instrumentation correction, 2026-09-14

The first bounded attempt from `602b2f0` revealed that the Windows environment's
launcher spawned a separate Python interpreter. Launcher-only RSS understated
the worker footprint, so that attempt was stopped and its probes, logs and
explicit stop record were retained. It provides no valid resource acceptance.
Before a fresh attempt, the monitor was corrected to include and terminate the
whole owned process tree. Tests exercise a real child interpreter as well as
Windows high-water accounting and rejection of launcher-only artifacts. The
dataset/seed/model/grid/two-condition specification and resource ceilings are
unchanged. The replacement preflight uses a new output directory after tests,
commit and remote synchronization; it does not reuse stopped probe weights or
select a favorable probe subset.

## Failure, immutable writes and recovery

All outputs use atomic exclusive writes to a new run-owned directory. A run
manifest binds the exact configuration, source, runtime/device, data/splits,
transform statistics, preflight acceptance and resource ceilings. Every formal
record retains the actual model/family, all four trial configurations and
histories, selected trial/checkpoint, validation-selection metrics, test-once
counter, execution timing and provenance. Selected-state hashes/checkpoint
provenance make replay and consistency checks auditable.

Stop on the first data/provenance mismatch, unsupported deterministic operation,
hash replay mismatch, OOM/resource cap, nonfinite value, malformed/duplicate
record, invalid selection history, unresolved failure or nonzero worker exit.
Keep a failure/termination record with dataset, condition, model, seed, trial or
phase, exception, resource snapshot and elapsed time. No automatic retry, seed
replacement, grid edit, dataset deletion or appended condition is allowed.

Recovery resumes only complete valid model units under the identical committed
source, configuration, runtime/device and input binding. Validate **every**
existing record before starting any missing unit; reject foreign outputs or
mixed conditions. A fully completed exact resume is read-only and idempotent.
An externally interrupted unit has no valid selected record and remains
incomplete. A documented recovery attempt may rerun that whole unit from its
original seed after the cause is resolved, while retaining the earlier attempt
and failure evidence. Unresolved failures block a success report; a failure is
never resolved by deleting it. Code/environment/scientific changes cannot enter
an existing execution silently. Preserve partial attempts separately if a new
source-bound run is required.

The bounded preflight inventories the fixed probes even if an individual probe
fails, retaining each failure and blocking overall acceptance. It never retries
that probe or changes its condition. Resource guards may terminate a worker or
the remaining preflight. The first-failure stop rule above governs formal
execution; an incomplete preflight cannot authorize it.

During formal training, monitoring is restricted to progress, resource guards,
record integrity and provenance; do not aggregate or rank new test results until
the exact configured scope passes validation. No favorable completed subset can
stand in for the 1,540 required records.

## Fixed analysis from the new records

Build each condition's complete evaluator input independently from its **newly
trained** 770 model records and its 110 bound diagnostics. Recompute all
validation selections and policies; the changed MLP validation accuracy may
change Combined's explicit MLP/abstention counts and confidence. Reusing old
decisions would therefore corrupt coverage, even when effective fallback actions
remain equal. Report both historical normalized results as a separate reference
and the new normalized control. Report per-model and per-dataset historical
control drift, selected-trial/architecture changes, and environment differences.
The new condition contrast is paired within this execution; changes relative to
the historical run may also reflect its different execution context. Historical
discrepancies are disclosed; they do not justify selecting the closer or more
favorable run.

1. **Primary selection results.** Reuse all nine frozen policies from
   `experiments.evaluate_diagnostics`: always-MLP, always-graph, analytic random
   50/50, homophily-only, degree-only, homophily-plus-degree, validation selection,
   two-hop-only and historical Combined. Keep the one-percentage-point practical
   margin (graph only for a strictly greater test gap), all cutoffs/confidences,
   and common MLP fallback. Report raw MLP and selected-graph test accuracies,
   gap and target counts; policy action counts, coverage, selection accuracy,
   selective accuracy/risk, full/covered regret and risk-coverage curves. Regret
   uses that condition and portfolio's own selected graph/MLP oracle. Compute
   dataset means before equal-weight averaging across all eleven datasets.
2. **Uncertainty and pairing.** Preserve the original eight comparisons against
   Combined, dataset-level 10,000-draw bootstrap, seeds 20260808/20260809 and Holm
   family, with exact sign-flip enumeration where implemented. These are
   post-hoc robustness summaries, not a renewed confirmatory claim. For candidate
   minus normalized contrasts, retain all ten paired values per dataset/model,
   means, sample SDs, ranges and signs. Report equal-weight dataset paired
   changes in primary regret, Combined minus always-graph regret, and graph/MLP
   accuracies with 10,000-draw dataset-bootstrap intervals seeded 20260913.
   Eleven dataset clusters are not 110 independent replications; nonsignificance
   is not equivalence. Do not add a new significance claim across the sensitivity
   search.
3. **Portfolio reversal and equal trial budget.** Recompute all six
   single-architecture restrictions and all 63 nonempty graph subsets separately
   within each condition. For each, report four trials per architecture,
   validation-selected model counts, graph targets, Combined/always-graph/
   validation-selection regret, dataset wins/losses/ties, and
   `Delta = Combined regret - always-graph regret` in percentage points. Report
   sign transitions for every portfolio between conditions and whether the
   original GAT/GCN-versus-other-single-model pattern and GAT+GCN pair remain.
   Enumerate all subsets, including unfavorable ones; overlapping subsets are
   not independent replications. Preserve the oracle-cancellation identity check.
4. **Dataset and fallback sensitivity.** For every subset, report each
   leave-one-dataset-out influence range and the two fixed analysis-only
   exclusions: Cornell+Wisconsin and Chameleon+Squirrel. All eleven datasets are
   still trained and included in the primary analysis. Exclusions neither select
   the primary result nor imply evaluation of cleaned Wikipedia graphs. Report
   Combined's graph-on-abstention sensitivity alongside its MLP-fallback primary
   result, including all affected units and coverage. Recompute dataset loss
   contributions rather than carrying over old concentration percentages.
5. **Calibration.** Separately for each condition and each of seven fixed
   portfolios (six single architectures and full), refit homophily thresholds
   by leave-one-dataset-out (LODO). The grid is exactly
   `{-1, 0, 0.01, ..., 1, 2}`. Minimize equal-weight mean raw regret over only
   the other ten datasets; ties within `1e-12` select the smallest threshold.
   Their selected-model **test outcomes are explicit meta-training targets**.
   Neither records nor outcomes of the held-out dataset select its threshold.
   Evaluate that dataset's ten seeds, then average the eleven held-out means.
   Report every selected threshold, fold regret, graph-action count, constant
   policy count, calibrated-minus-frozen and calibrated-minus-always-graph
   differences, and dataset contributions to remaining regret.

The calibration excludes the outer dataset from threshold fitting, but it is not
a nested procedure that selects preprocessing, portfolio or grid on independent
inner folds. Those are fixed report strata; do not select a favored condition or
portfolio from their held-out performance and label that winner's score unbiased.
The candidate originated after earlier results, outer training folds overlap,
and some domains are related. Recomputing LODO therefore does not create external
validation or independent confirmation. No in-sample fitted threshold is
substituted for the held-out control, and no conditions are pooled in fitting.

The report will separate **maintained**, **changed** and **unresolved** conclusions
with exact new estimates and their scopes. “Maintained” describes the specified
direction/pattern on these records, not equivalence or universality. Include a
claim matrix covering full-portfolio excess selection loss; all single/subset
reversals; WebKB/Wikipedia influence; full and per-architecture calibration
relative to both frozen and always-graph rules; and any unresolved runtime or
reproducibility issue. Do not convert the candidate into a universally preferred
training recipe from these results.

## Code reuse, required adapter and acceptance order

The frozen implementations remain unchanged. Reuse their functions through a
new run-specific layer:

| Existing entrypoint | Role in the new layer |
|---|---|
| `experiments.prospective_data.make_stratified_split`, `split_identifier`, `canonical_undirected_edges`, `train_only_diagnostics` | Original input/split/diagnostic definitions |
| `experiments.prospective_models.build_model`, `prepare_h2_adjacencies` | Original seven models and strict two-hop buffers |
| `scripts.run_mlp_optimization_diagnostic.fit_transform_features` | Numerical reference for the bounded-memory adapter `scripts.input_robustness_data.fit_transform_features_bounded` |
| `experiments.run_prospective_benchmark._train_trial`, `select_trial`, `run_model_unit` | Original optimization and selection/test semantics |
| `experiments.evaluate_diagnostics.audit_payload` | New condition-specific complete nine-policy audit |
| `scripts.summarize_equal_budget_sensitivity.evaluate_portfolio` | New full/single/subset outcomes |
| `scripts.summarize_portfolio_robustness.summarize`, `leave_one_dataset_out`, `fit_threshold` | All 63 subsets and separately fitted seven-row LODO controls |
| `scripts.summarize_reviewer_appendix.summarize_audit` | Dataset/action/fallback descriptions from the new audit |

Existing `assemble_prospective_diagnostics.assemble_payload` and
`summarize_equal_budget_sensitivity.load_units` enforce the old run layout and
release manifest. Existing CLI entrypoints also default to the historical audit;
the equal-budget CLI checks specific published numbers. They must not be pointed
blindly at new outputs or weakened to accept mixed data. A new adapter must
validate the new manifest and full 1,540/6,160 scope, then generate standard
condition-specific evaluator payloads and in-memory unit maps. Cross-condition
binding and post-hoc metadata belong in a new wrapper; leave frozen functions and
historical tests intact. Reuse pure analysis functions after validation. This
adapter and a formal runner are launch prerequisites, not deliverables already
claimed complete by this design document.

Required tests cover the exact inherited dataset/model/seed/grid scope; transform
equivalence and train-only fitting; degeneracy rejection; complete split/tensor
binding; no held-out metrics or formal records in preflight; unsupported
determinism and resource failures; exclusive writes and rejected stale/mixed
resume; and original architecture/trial tie rules. Before formal launch, also
test condition-specific assembly, no new-MLP/old-graph mixing, exact record/trial
counts, test-once selection, corrupted histories, incomplete/duplicate scope,
oracle cancellation, and a LODO leakage test in which changing held-out outcomes
cannot change that fold's fitted threshold. Synthetic tests should verify these
invariants without generating new benchmark outcomes.

Acceptance order is: run and report appropriate tests; complete and inspect all
bounded preflight evidence; commit the design/config/code/tests and resource
report; synchronize the branch; report the actual local acceptance and remote CI
state (success, failure, pending or no matching CI); then advance to the next
gated stage. A clean commit, successful push or a CI job with no CUDA coverage is
not a substitute for the local GPU preflight. The present design provides no
claim that either input condition has already completed formal training.
