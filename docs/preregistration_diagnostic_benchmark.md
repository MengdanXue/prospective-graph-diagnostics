# Prospective Diagnostic Benchmark Specification

- Version: `route_a_diagnostic_v1`
- Frozen practical margin: `0.01` accuracy
- Implementation: `experiments/evaluate_diagnostics.py`

## Status and scope

This document is a prospective analysis specification for newly generated
records. It is not a claim that the historical experiments were preregistered:
the authors have already inspected several legacy aggregate result files. Those
files are excluded because they do not consistently preserve paired
dataset/model/seed outcomes or decision-time provenance. The specification
becomes confirmatory only for records generated after this document and the
scoring implementation are committed.

The benchmark asks one operational question: for a fixed dataset and seed,
does the graph model selected from the frozen graph-model set improve held-out
test accuracy over the validation-selected MLP by more than one percentage
point? The benchmark evaluates diagnostic decisions about that binary target;
it does not evaluate a new GNN.

## Frozen model set and selection

- Feature-only set: MLP.
- Graph-model set: GCN, GAT, GraphSAGE, H2GCN, LINKX, and GPR-GNN.
- Hyperparameter budgets, split identifiers, and early-stopping rules must be
  identical across methods wherever their parameterizations permit.
- Hyperparameter tuning occurs upstream within each architecture and retains all
  trial records. The evaluator input contains the validation-selected
  configuration for each named architecture. It then selects one graph
  architecture by validation accuracy; exact ties are broken by lexicographic
  model identifier. Test accuracy is read once after both selection stages.
- Every dataset/seed unit must retain one record for every frozen model. An
  incomplete model set is an error, not a missing value to be imputed.

For selected test accuracies `a_graph` and `a_mlp`, the target is `graph` only
when `a_graph - a_mlp > 0.01`. Every gap less than or equal to `0.01`, including
negative gaps, maps to `mlp` (no demonstrated practically material graph gain).
This single rule is used for every method.

## Decision-time information boundary

Full graph structure and node features are available because they are inputs at
deployment. Labels from the training split and validation accuracy are allowed.
Test labels, test accuracy, and statistics computed with test labels are
forbidden inputs to a diagnostic. Edge homophily and two-hop label agreement
must therefore be computed only from label pairs available in the training
split; the retained provenance must say `train_only`. A post-hoc statistic that
uses all labels may be reported descriptively, but it cannot enter this
benchmark. The evaluator rejects a record marked as using test labels.

## Frozen diagnostic baselines

All thresholds below are fixed sanity baselines, not values estimated from the
new benchmark outcomes.

1. `always_mlp`: always choose the validation-selected MLP.
2. `always_graph`: always choose the validation-selected graph model.
3. `random_50_50`: report the analytic expectation of choosing each action
   with probability one half; no favorable random draw is stored.
4. `homophily_only`: choose graph when train-only edge homophily is at least
   `0.50`; otherwise choose MLP.
5. `degree_only`: choose graph when mean degree is at least `10`; otherwise
   choose MLP. The score is `tanh(log(mean_degree / 10))`.
6. `homophily_plus_degree`: choose graph when
   `0.5 * (2 * homophily - 1) + 0.5 * tanh(log(mean_degree / 10)) >= 0`;
   otherwise choose MLP.
7. `validation_selection`: choose graph only when its validation advantage is
   greater than the same `0.01` practical margin.
8. `two_hop_only`: choose graph when train-only `delta_h = h_2 - h_1` is greater
   than `0.05`; otherwise choose MLP.
9. `historical_combined`: choose graph when homophily is at least `0.55`; when
   it is lower, choose MLP if MLP validation accuracy is at least `0.40`;
   otherwise abstain.

Missing inputs cause the affected diagnostic to abstain. They never receive a
correct prediction by default. For a full-set operational comparison, every
abstention falls back to MLP; coverage and selective metrics remain separately
reported.

## Records and immutable data contract

The evaluator consumes separate `model_records` and `diagnostic_records` keyed
by `(dataset, seed, split_id)`. Model records contain model identifier, family,
validation accuracy, and held-out test accuracy. Diagnostic records contain
homophily, mean degree, `delta_h`, and explicit label-scope provenance. The
input also freezes bootstrap/permutation seeds and resample counts. The
evaluator refuses to overwrite an existing audit file.

Training runs should write one immutable JSON file per dataset/model/seed. A
read-only assembly step may create the evaluator input, but must not replace the
source records. Each source record must include the Git commit, environment,
configuration, split identifier, seed, status, and exception details.

## Outcomes and missingness

For each method, report:

- coverage and number of abstentions;
- selection accuracy after the common MLP fallback;
- selective accuracy and selective risk among covered units;
- full-set mean regret relative to the better of the selected graph and MLP
  test accuracies;
- covered-set mean regret; and
- a risk--coverage curve ordered by predeclared confidence.

No method is credited twice on a near tie. Missing diagnostics remain visible
through coverage and do not remove the unit from full-set regret.

## Statistical analysis

Seeds remain paired within dataset. Regret is first averaged over seeds within
each dataset, and datasets—not dataset/seed rows—are the resampling unit. The
primary comparison is full-set regret for each baseline minus full-set regret
for `historical_combined`.

For every predeclared comparison, report the mean paired difference, a 95%
percentile bootstrap interval, a paired standardized mean difference, win rate,
and a two-sided paired sign-flip permutation p-value. Apply Holm family-wise
correction across the eight comparisons. Statistical non-significance is not
equivalence; equivalence would require a separately frozen equivalence margin
and test.

## Decision rule for the paper

The combined diagnostic earns an incremental-value claim only if it reduces
full-set regret relative to simple baselines without relying on lower coverage,
and the uncertainty interval excludes zero for the predeclared comparison. If
it does not, the negative result is the conclusion. Historical headline
selection accuracies are not substituted for this benchmark.
