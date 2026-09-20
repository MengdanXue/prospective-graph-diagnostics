# Paired preprocessing sensitivity (post-hoc)

Design date: 2026-09-07. The original outcomes were already known. This is a new
post-hoc experiment, not a modification of the frozen benchmark or preregistration
of the original study. The configuration and code are committed before new model
outcomes are evaluated.

## Question and scope

On Roman-empire and Amazon-ratings, does the frozen `NormalizeFeatures` choice
materially change graph-vs-MLP gaps and the cost of the frozen homophily rule?
These two graphs are selected because the review questions preprocessing of their
continuous features; they are not a random sample. There are no population-wide
significance claims from these two datasets.

## Paired design

Both conditions are trained anew on the same machine/runtime: raw features and
PyG 2.7.0 `NormalizeFeatures` (subtract the matrix minimum and divide each row by
its sum clamped below at one). Every condition includes MLP, GCN, GAT, GraphSAGE,
H2GCN, LINKX and GPR-GNN, ten seeds, and the original four-trial training grid.
This is 280 selected-model records and 1,120 training trials, with up to 500 epochs
and patience 100. Neither validation nor test results change the design.

The original split function, model implementations, training/checkpoint selection
and single post-selection test evaluation are reused unchanged. Split IDs, training
homophily and mean degree must match the release for all ten seeds before training.
Raw NPZ SHA-256 values are pinned in the new config. Both conditions use identical
initialization seeds, data, graph structure, tuning grid, environment and device.
Hyperparameters are selected independently by validation within each condition.
The six-architecture graph action still has 24 trials versus four for MLP; this
experiment isolates a preprocessing choice within that design, not equal compute.

## Reporting

Report per-dataset/per-condition MLP accuracy, validation-selected graph accuracy,
graph-minus-MLP gap, graph target count, and Combined/always-graph/validation regrets.
Report seed-paired differences between conditions, and individual-architecture
accuracy changes to show whether any effect is specific to MLP. Use each condition's
own oracle for regret and report accuracies alongside it. Do not pool just the raw
MLP with the old normalized graph outcomes; do not silently replace any frozen row.
All failures and incomplete scope must be disclosed; no favorable seed/model subset
may stand in for the configured experiment. No outcome-based early stopping of the
experiment is permitted (the original per-model validation early stopping remains).

## Resource preflight and execution

`--preflight` runs three training steps per model on each graph to verify memory
and estimate wall time. It evaluates no validation/test outcomes, uses a separate
directory, and produces no manuscript accuracy results. The full experiment
requires a separate output root; a preflight cannot be resumed as a full run.

```bash
python scripts/run_preprocessing_sensitivity.py --data-root PATH_TO_NPZ --output-root tmp/preprocessing-preflight --preflight
python scripts/run_preprocessing_sensitivity.py --data-root PATH_TO_NPZ --output-root tmp/preprocessing-full
```

Completed model-unit records are written exclusively. Resume requires the exact
configuration, source commit, runtime, data/preprocessing provenance and trial
audit; use the same committed checkout for a resumed experiment. Prior failure
records remain visible. Outputs belong to the new run only.

Independent datasets and duplicate-filtered Wikipedia graph retraining remain
separate future work; this two-graph preprocessing test does not replace them.
